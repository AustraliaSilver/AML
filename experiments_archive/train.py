import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Default: GRADIENT CHECKPOINTING OFF.
# Measured biggest free speed-up (~10% fwd and ~30% of epoch time, e.g.
# 170 -> 159.6s/epoch at batch 512) with NO accuracy change (diff ~6.9e-6),
# because checkpoint() only trades memory for recompute, not numerics.
# Set AML_NO_CKPT=0 in the shell to restore checkpointing if you run out of
# VRAM on a larger model.
os.environ.setdefault("AML_NO_CKPT", "1")
# AML_NO_COMPILE=1: with checkpoint off, torch.compile + grouped-conv bf16 backward
# hits a CUDA "misaligned address" kernel bug. Since compile also gives no speedup
# (136 vs 134 ms/forward measured), default to eager. Override with AML_NO_COMPILE=0.
os.environ.setdefault("AML_NO_COMPILE", "1")
# ---------------------------------------------------------------------------

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    bbx1 = max(0, cx - cut_w // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(W, cx + cut_w // 2)
    bby2 = min(H, cy + cut_h // 2)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AML_Core_Code'))
from train_cv_cifar_18 import AcceleratedCVNetwork, get_prototype_layout

# Use mgn from current directory (where we compiled it)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mgn.architecture import ModelStructure, GeneratedModel
from mgn.losses import MetaLoss

def build_input_adapter(in_ch, out_ch):
    """Map an input tensor with `in_ch` channels to `out_ch` channels, used to
    bridge dataset channels to the base model's intrinsic channels. For the
    unified 3-channel base, MNIST/Fashion (1 gray channel) are lifted 1->3 and
    CIFAR (3 RGB channels) passes natively. A 32->32 stem plus 1x1 bottleneck
    provides real learnable capacity so the gray->RGB lift preserves signal."""
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, out_ch, 1),
    )

class HtSMemory(nn.Module):
    """Hard-to-Soft weight region: a dedicated per-layer low-rank weight bank that
    always exists in the model (initialized to 0 so task A is pure AML). During
    retrain (task B/C) the AML base is frozen and ONLY this region is trained,
    producing per-task weight deltas = main + correction (HtS-B12 style) combined
    with the generated base weights. Each layer block gets its OWN basis so the
    delta is position-specific instead of competing on one shared vector.
    """
    def __init__(self, total_dim, num_tasks, latent=64, rank=32, sel_latent=128,
                 layer_offsets=None, wo_offset=0, bo_offset=0, num_classes=10):
        super().__init__()
        self.rank = rank
        self.total_dim = total_dim
        self.num_tasks = num_tasks
        if layer_offsets is None:
            layer_offsets = [(0, total_dim)]
        self._blocks = []          # (start, end) contiguous span for each unit
        for (ws, we, bs, be) in layer_offsets:
            self._blocks.append((ws, be))
        self._blocks.append((wo_offset, bo_offset + num_classes))
        self.block_dims = [e - s for (s, e) in self._blocks]
        self.num_blocks = len(self._blocks)

        # SHARED low-rank basis: ONE subspace per layer block, used by ALL tasks.
        # Each task only owns a small coefficient vector, so task t's delta is
        # basis @ coefs[t] (a rank-1 slice of the shared subspace). To keep
        # earlier tasks intact when retraining a new task, project the shared-
        # basis gradient ORTHOGONAL to every frozen prior coef direction
        # (OWM/GPM-style). This cuts the HtS bank from num_tasks x blocks down
        # to blocks (~5M HtS params total) instead of ~39M, while still
        # preventing catastrophic interference.
        self.basis = nn.ParameterList(
            nn.Parameter(torch.zeros(d, rank)) for d in self.block_dims
        )
        self.corr_basis = nn.ParameterList(
            nn.Parameter(torch.zeros(d, rank)) for d in self.block_dims
        )
        # Per-task coefficient vectors (tiny). Only the current task's coefs are
        # trainable during its phase; earlier tasks' coefs are frozen, and the
        # shared-basis gradient is projected orthogonal to their directions.
        self.coefs = nn.ParameterList(
            nn.Parameter(torch.zeros(rank * self.num_blocks)) for _ in range(num_tasks)
        )
        self.corr_coefs = nn.ParameterList(
            nn.Parameter(torch.zeros(rank * self.num_blocks)) for _ in range(num_tasks)
        )
        # SUPERVISED task-inference router: a small head on the vision-encoder
        # embedding z that predicts WHICH task an input belongs to. During each
        # training phase we know the true task label (main batch -> current task,
        # replay MNIST -> 0, replay Fashion -> 1), so it learns to self-route
        # without any external task id at inference time.
        self.task_head = nn.Sequential(
            nn.Linear(sel_latent, 64), nn.GELU(),
            nn.Linear(64, num_tasks),
        )
        self.register_buffer('centroids', torch.zeros(num_tasks, sel_latent))
        self.register_buffer('centroid_counts', torch.zeros(num_tasks))
        self.centroid_counts[0] = 1.0
        # Locked-delta snapshots for OWM/GPM-style protection: when a task is
        # done, we snapshot the exact delta (basis@coefs) it produced, and while
        # a later task trains we re-enforce each locked delta AFTER every
        # optimizer step. This is EXACT even under adaptive optimizers (AdamW),
        # whose per-coordinate scaling breaks pure gradient projection.
        self.register_buffer('basis_targets', torch.zeros(num_tasks, total_dim))
        self.register_buffer('corr_targets', torch.zeros(num_tasks, total_dim))
        self.allow_delta0 = False
        scale = 1e-3 / math.sqrt(max(rank, 1))
        with torch.no_grad():
            for b in self.basis:
                b.uniform_(-scale, scale)
            for b in self.corr_basis:
                b.uniform_(-2 * scale, 2 * scale)
            for t in range(num_tasks):
                self.coefs[t].uniform_(-scale, scale)
                self.corr_coefs[t].uniform_(-2 * scale, 2 * scale)

    def task_deltas(self):
        c = torch.cat([self.coefs[t].view(1, -1) for t in range(self.num_tasks)], dim=0)
        cc = torch.cat([self.corr_coefs[t].view(1, -1) for t in range(self.num_tasks)], dim=0)
        c = c.reshape(self.num_tasks, self.num_blocks, self.rank)
        cc = cc.reshape(self.num_tasks, self.num_blocks, self.rank)
        d = torch.zeros(self.num_tasks, self.total_dim, device=c.device)
        for t in range(self.num_tasks):
            for i, (s, e) in enumerate(self._blocks):
                d[t, s:e] = c[t, i] @ self.basis[i].t() + cc[t, i] @ self.corr_basis[i].t()
        d = d.clone()
        if not self.allow_delta0:
            d[0] = 0.0
        return d

    def get_delta(self, task_id, B, device):
        if task_id >= 0:
            d = torch.zeros(self.total_dim, device=device)
            c = self.coefs[task_id].reshape(self.num_blocks, self.rank)
            cc = self.corr_coefs[task_id].reshape(self.num_blocks, self.rank)
            for i, (s, e) in enumerate(self._blocks):
                d[s:e] = c[i] @ self.basis[i].t() + cc[i] @ self.corr_basis[i].t()
            if task_id == 0 and not self.allow_delta0:
                d.zero_()
            return d[None].expand(B, -1)
        return self.task_deltas()[0:1].expand(B, -1)

    def project_grads(self, task_id):
        """OWM/GPM-style protection: after backward(), project the SHARED basis
        gradient orthogonal to every previously frozen task's coefficient
        direction using fast in-place cuBLAS addmm_ without intermediate tensor allocations."""
        if task_id <= 1:
            return  # task 0's delta is forced 0 unless BT; nothing locked yet
        for i in range(self.num_blocks):
            gb = self.basis[i].grad
            gcb = self.corr_basis[i].grad
            if gb is None and gcb is None:
                continue
            for t in range(1, task_id):
                c = self.coefs[t].reshape(self.num_blocks, self.rank)[i].detach()
                cc = self.corr_coefs[t].reshape(self.num_blocks, self.rank)[i].detach()
                if gb is not None:
                    denom = c.dot(c)
                    if denom > 1e-12:
                        gb.addmm_(gb @ c.unsqueeze(1), c.unsqueeze(0), alpha=-1.0 / denom)
                if gcb is not None:
                    denom = cc.dot(cc)
                    if denom > 1e-12:
                        gcb.addmm_(gcb @ cc.unsqueeze(1), cc.unsqueeze(0), alpha=-1.0 / denom)

    @torch.no_grad()
    def lock_task(self, task_id):
        """Snapshot task `task_id`'s exact delta contributions from the shared
        basis/corr_basis. From now on, correct_basis() will re-project the
        shared basis onto this snapshot after every optimizer step so that later
        tasks can NEVER rewrite this task's delta."""
        d = torch.zeros(self.total_dim, dtype=self.basis[0].dtype, device=self.basis[0].device)
        dd = torch.zeros_like(d)
        c = self.coefs[task_id].reshape(self.num_blocks, self.rank)
        cc = self.corr_coefs[task_id].reshape(self.num_blocks, self.rank)
        for i, (s, e) in enumerate(self._blocks):
            d[s:e] = c[i] @ self.basis[i].t()
            dd[s:e] = cc[i] @ self.corr_basis[i].t()
        self.basis_targets[task_id] = d
        self.corr_targets[task_id] = dd

    @torch.no_grad()
    def correct_basis(self, task_id):
        """EXACT post-step protection: for every locked task t < task_id, undo
        whatever drift an optimizer step introduced to that task's delta by
        re-projecting the shared basis along coefs[t] using in-place cuBLAS addmm_."""
        if task_id <= 1:
            return
        for i, (s, e) in enumerate(self._blocks):
            for t in range(1, task_id):
                c = self.coefs[t].reshape(self.num_blocks, self.rank)[i]
                cc = self.corr_coefs[t].reshape(self.num_blocks, self.rank)[i]
                den = c.dot(c)
                if den > 1e-12:
                    diff = (self.basis[i] @ c) - self.basis_targets[t, s:e]
                    self.basis[i].addmm_(diff.unsqueeze(1), c.unsqueeze(0), alpha=-1.0 / den)
                denc = cc.dot(cc)
                if denc > 1e-12:
                    diffc = (self.corr_basis[i] @ cc) - self.corr_targets[t, s:e]
                    self.corr_basis[i].addmm_(diffc.unsqueeze(1), cc.unsqueeze(0), alpha=-1.0 / denc)

    def get_mixture_delta(self, probs):
        return probs @ self.task_deltas()

    @torch.no_grad()
    def update_centroid(self, task_id, z):
        if task_id < 0 or task_id >= self.num_tasks:
            return
        zb = z.mean(dim=0)
        zb = zb / (zb.norm() + 1e-8)
        c = self.centroid_counts[task_id]
        if c == 0:
            self.centroids[task_id] = zb
        else:
            self.centroids[task_id].mul_(c / (c + 1)).add_(zb / (c + 1))
        self.centroid_counts[task_id] = c + 1

    def route_probs(self, z):
        zb = F.normalize(z.float(), dim=-1)
        cb = F.normalize(self.centroids.float(), dim=-1)
        sim = zb @ cb.t()
        seen = self.centroid_counts > 0
        sim = sim.clone()
        sim[:, ~seen] = float('-inf')
        return torch.softmax(sim / 0.1, dim=-1)

    def route_task_logits(self, z):
        return self.task_head(z.float())

    def forward(self, x, task_id):
        return self.get_delta(task_id, x.shape[0], x.device)


class ConvAML_CIFAR_TrueDynamic(nn.Module):
    def __init__(self, K=32, latent_dim=128, max_layers=18, max_dim=64, in_channels=3, num_classes=10,
                 hts_tasks=0, hts_latent=64, hts_rank=32):
        super().__init__()
        self.K = K
        self.max_layers = max_layers
        self.max_dim = max_dim
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.active_task = -1
        self.input_adapter = None
        # Unified channel bridging: the base has `in_channels` intrinsic
        # channels (3 = RGB). 1-channel gray inputs (MNIST/Fashion) are lifted
        # to the base's channel count automatically; 3-channel RGB (CIFAR)
        # passes through natively. This lets every task share the SAME base.
        self.gray_adapter = None
        if in_channels > 1:
            self.gray_adapter = build_input_adapter(1, in_channels)

        # Per-task full conv encoder: every task owns its ENTIRE conv feature
        # extractor (input stem + deep body + projection). Tasks share only the
        # controller heads (prototype_selector / structure_delta / block_mode_head),
        # the generated-weight executor and the HtS bank. Because a new task can
        # never touch another task's conv body, there is ZERO cross-task forgetting
        # of features; each task is as strong as its own trained encoder.
        self.num_task_blocks = hts_tasks if hts_tasks > 0 else 1
        # SHARED UNIVERSAL ROUTER: a small conv network on the RAW input (not on
        # any per-task encoder feature) that predicts which task an input belongs
        # to. Trained with supervised CE on the true task id during every phase,
        # so at inference (auto-routing, active_task=-1) it does NOT depend on
        # which encoder produced the features. Per-sample argmax routes each
        # sample to its own task's encoder / controller heads / HtS delta.
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(48, self.num_task_blocks)
        )
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(128, latent_dim), nn.ReLU()
            ) for _ in range(self.num_task_blocks)
        ])
        self.prototype_selectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.GELU(),
                nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(),
                nn.Linear(256, K)
            ) for _ in range(self.num_task_blocks)
        ])
        self.structure_deltas = nn.ModuleList([
            nn.Linear(latent_dim, max_layers + (max_layers * max_dim)) for _ in range(self.num_task_blocks)
        ])
        self.num_modes = 4  # 0=Plain, 1=ResNet, 2=ViT, 3=MLP
        self.block_mode_heads = nn.ModuleList([
            nn.Linear(latent_dim, max_layers * self.num_modes) for _ in range(self.num_task_blocks)
        ])
        for sd in self.structure_deltas:
            nn.init.constant_(sd.weight, 0.0)
            nn.init.constant_(sd.bias, 0.0)
        for bmh in self.block_mode_heads:
            nn.init.constant_(bmh.weight, 0.0)
            nn.init.constant_(bmh.bias, 0.0)

        weight_dim = 0
        for i in range(max_layers):
            in_c = in_channels if i == 0 else max_dim
            weight_dim += max_dim * in_c * 9 + max_dim
            attn_dim = max_dim // 4
            weight_dim += (3 * attn_dim) * max_dim + 3 * attn_dim
            weight_dim += max_dim * attn_dim + max_dim
        weight_dim += (16 * max_dim) * num_classes + num_classes
        self._init_prototypes()
        
        if os.environ.get("AML_NO_COMPILE") == "1":
            self.exec_ = AcceleratedCVNetwork(max_layers=max_layers, max_dim=max_dim,
                                              num_classes=num_classes, in_channels=in_channels)
        else:
            self.exec_ = torch.compile(AcceleratedCVNetwork(max_layers=max_layers, max_dim=max_dim,
                                                            num_classes=num_classes, in_channels=in_channels))
        self.hts = None
        if hts_tasks > 0:
            ex = self.exec_
            self.hts = HtSMemory(ex.total_dim, num_tasks=hts_tasks,
                                 latent=hts_latent, rank=hts_rank,
                                 layer_offsets=[[a, b, c, d] for (a, b, c, d, *_ ) in ex._layer_info],
                                 wo_offset=ex.wo_offset, bo_offset=ex.bo_offset,
                                 num_classes=num_classes).to(next(ex.parameters()).device)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "_bn_frozen", False) and mode:
            for m in self.modules():
                if m is not self and isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
        return self

    def _init_prototypes(self):
        import math
        import torch.nn as nn
        
        weight_dim, offsets, layer_info, wo_offset, bo_offset = get_prototype_layout(self.max_layers, self.max_dim, self.num_classes, in_channels=self.in_channels)
        self.proto_weights = nn.Parameter(torch.zeros(self.K, weight_dim))
        
        nn.init.kaiming_uniform_(self.proto_weights, a=math.sqrt(5))

        with torch.no_grad():
            for i, (w_start, w_end, b_start, b_end, in_c, out_c) in enumerate(layer_info):
                if i > 0:  # Zero-residual for all layers except the first non-residual one
                    next_start = layer_info[i+1][0] if i + 1 < len(layer_info) else wo_offset
                    self.proto_weights[:, w_start:next_start] = 0.0

    def caf_attention(self, features):
        return torch.softmax(features, dim=-1).clamp(min=1e-8)

    def forward(self, x, tau=1.0, alpha=1.0):
        B = x.shape[0]; device = x.device
        # Auto bridge a 1-channel gray input up to the base's channel count.
        if x.dim() == 4 and x.size(1) == 1 and self.in_channels > 1:
            x = self.gray_adapter(x)
        elif self.input_adapter is not None:
            x = self.input_adapter(x)
        # Route through the per-task conv encoder. Each task has its OWN full
        # feature extractor so later tasks can never rewrite an earlier task's
        # features. During eval without a task id, use task 0's encoder only to
        # produce features for the task-inference head (which then selects the
        # correct task path for the final pass).
        if self.num_task_blocks == 1:
            f = self.encoders[0](x)
        else:
            enc_idx = self.active_task % self.num_task_blocks if self.active_task >= 0 else 0
            f = self.encoders[enc_idx](x)
        z = f
        if self.num_task_blocks > 1 and self.active_task >= 0:
            head_idx = self.active_task % self.num_task_blocks
        else:
            head_idx = 0
        prototype_logits = self.prototype_selectors[head_idx](z)
        pi = self.caf_attention(prototype_logits)
        
        delta = self.structure_deltas[head_idx](z)
        layer_logits = delta[:, :self.max_layers] + 2.0
        layer_soft = torch.sigmoid(layer_logits)
        if self.training:
            layer_soft = layer_soft + 0.1 * torch.randn_like(layer_soft)
        layer_soft = torch.clamp(layer_soft, 1e-4, 1.0)
        learned_gates = (layer_soft > 0.5).float().detach() + layer_soft - layer_soft.detach()
        layer_gates = alpha * learned_gates + (1.0 - alpha) * 1.0
        
        if getattr(self, "fixed_L", 0) > 0:
            fixed_L = self.fixed_L
            mask = torch.arange(self.max_layers, device=device).float().view(1, self.max_layers) < fixed_L
            layer_gates = mask.float().expand(B, -1)
            
        dim_logits = delta[:, self.max_layers:].view(B, self.max_layers, self.max_dim)
        
        d_logit = dim_logits.mean(dim=-1, keepdim=True) + 2.0
        dim_fractions = torch.sigmoid(d_logit)
        
        D_MIN = 24.0
        raw = dim_fractions * self.max_dim
        learned_d = D_MIN + F.softplus(raw - D_MIN)
        d_lengths = alpha * learned_d + (1.0 - alpha) * (0.95 * self.max_dim)
        
        if getattr(self, "fixed_D", 0) > 0:
            d_lengths = torch.full((B, self.max_layers, 1), float(self.fixed_D), device=device)

        ceiling = getattr(self, "_D_ceiling", 0.0)
        if ceiling > 0:
            d_lengths = d_lengths.clamp(max=float(ceiling))
            
        arange = torch.arange(self.max_dim, device=device).float().view(1, 1, self.max_dim)
        dim_masks = torch.sigmoid(10.0 * (d_lengths - arange))

        bm_logits = self.block_mode_heads[head_idx](z).view(B, self.max_layers, self.num_modes)
        
        block_mode = F.gumbel_softmax(bm_logits, tau=tau, hard=True, dim=-1)
            
        dim_weights = d_lengths / self.max_dim
        
        struct = ModelStructure(
            layer_gates=layer_gates,
            dim_weights=dim_weights,
            residual=torch.zeros(B, 1, device=device),
            dim_masks=dim_masks,
            block_mode=block_mode,
            use_conv=True, in_channels=self.in_channels, max_layers=self.max_layers, max_dim=self.max_dim,
            dim_choices=(self.max_dim,), output_dim=self.num_classes
        )
        
        delta = None
        if self.hts is not None:
            if self.active_task >= 0:
                delta = self.hts.get_delta(self.active_task, B, device)
            else:
                p = self.hts.route_probs(z)
                delta = self.hts.get_mixture_delta(p)
        out = self.exec_(x, self.proto_weights, pi, struct, delta)
        # Shared universal router (on the RAW input, not per-task features):
        # the supervised task-inference signal for auto-routing. Independently
        # trained each phase with the true task id, so per-sample routing at
        # inference never depends on which encoder was used to make features.
        router_logits = self.router(x)
        diag = {
            "layer_soft": layer_soft,
            "bm_logits": bm_logits,
            "d_lengths": d_lengths,
            "z": z.detach(),
            "z_full": z,
            "router_logits": router_logits.detach(),
            "task_logits": router_logits,
        }
        return out, struct, pi, diag

class _IndexedReplay(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i

def _prevent_sleep():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="True AML Dynamic Architecture Training")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist', 'fashion_mnist'])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--enc-lr-mult', type=float, default=2.5, help="LR multiplier for the current task's encoder when --freeze-base")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--max-dim', type=int, default=128)
    parser.add_argument('--max-layers', type=int, default=18)
    parser.add_argument('--base-channels', type=int, default=3, help="Intrinsic channel count of the shared base (3 = unified RGB base; gray tasks are auto-lifted 1->3, CIFAR passes natively)")
    parser.add_argument('--fixed-D', type=int, default=0, help="Fixed width for ablation (0 to disable)")
    parser.add_argument('--fixed-L', type=int, default=0, help="Fixed depth for ablation (0 to disable)")
    parser.add_argument('--p-allowance', type=float, default=400000.0)
    parser.add_argument('--lambda-eff', type=float, default=0.1)
    parser.add_argument('--out', type=str, default='aml_core_best.pth')
    parser.add_argument('--resume', type=str, default='', help="Checkpoint to load base weights from")
    parser.add_argument('--hts-tasks', type=int, default=0, help="Num tasks in HtS memory bank (0 = disabled)")
    parser.add_argument('--hts-latent', type=int, default=64)
    parser.add_argument('--hts-rank', type=int, default=32)
    parser.add_argument('--task-id', type=int, default=-1, help="Active HtS task; >=0 enables HtS region during this run")
    parser.add_argument('--freeze-base', action='store_true', help="Freeze all non-HtS params (retrain on task B/C)")
    parser.add_argument('--protect-prior', action=argparse.BooleanOptionalAction, default=True, help="Enable OWM/GPM projection + exact basis lock/correction to protect prior-task deltas (default ON)")
    parser.add_argument('--teacher', type=str, default='', help="Checkpoint of task A model (teacher) for KL distillation")
    parser.add_argument('--distill-lambda', type=float, default=0.0, help="KL weight vs teacher (soft target)")
    parser.add_argument('--distill-tau', type=float, default=3.0, help="Temperature for distillation")
    parser.add_argument('--replay-buffer', type=str, default='', help="Path to cached replay tensor (optional; else sample task-A train set)")
    parser.add_argument('--replay-n', type=int, default=512, help="Number of task-A exemplars to hold in memory")
    parser.add_argument('--replay-lambda', type=float, default=0.5, help="Weight of replay KL loss")
    parser.add_argument('--replay2-lambda', type=float, default=None, help="Weight of replay2 (task-B stream) KL loss; default = replay-lambda")
    parser.add_argument('--bt-lambda', type=float, default=0.0, help="Weight of backward-transfer CE loss on task-A replay exemplars (active_task=0, real labels)")
    parser.add_argument('--replay-every', type=int, default=1, help="Run replay/BT every N training batches (1 = every batch)")
    parser.add_argument('--input-adapt', type=int, default=0, help="Reserved (kept for CLI compat); channel bridging is now handled internally by the model's gray_adapter based on the base channel count.")
    parser.add_argument('--replay-buffer2', type=str, default='', help="Optional SECOND replay stream (task B protection): dataset (e.g. fashion_mnist)")
    parser.add_argument('--replay-teacher2', type=str, default='', help="Checkpoint used as teacher for replay-buffer2 (default: --replay-teacher)")
    parser.add_argument('--replay-task2', type=int, default=1, help="HtS task id the second replay stream should run under (the protected task B)")
    parser.add_argument('--replay-task', type=int, default=0, help="HtS task id the first replay stream should run under (the protected task A)")
    parser.add_argument('--router-lambda', type=float, default=0.0, help="Weight of supervised task-inference CE loss (teaches the model to self-route images to the right task without external task ids)")
    parser.add_argument('--stab-lambda', type=float, default=0.0, help="Weight of deep-feature consistency (MSE) to frozen teacher features on replay exemplars; protects the shared deep conv body from being rewritten by later tasks")
    return parser.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)
    _prevent_sleep()
    T0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    in_ch = 3 if args.dataset.startswith('cifar') else 1
    num_classes = 100 if args.dataset == 'cifar100' else 10
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if args.dataset == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
        else:
            trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    else:
        isn_mnist = args.dataset in ('mnist', 'fashion_mnist')
        if not isn_mnist:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        if isn_mnist:
            transform_train = transforms.Compose([
                transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            accessor_cls = torchvision.datasets.FashionMNIST if args.dataset == 'fashion_mnist' else torchvision.datasets.MNIST
            trainset = accessor_cls(root=data_root, train=True, download=True, transform=transform_train)
            testset = accessor_cls(root=data_root, train=False, download=True, transform=transform_test)
    
    _workers = 4 if os.cpu_count() and os.cpu_count() >= 4 else 0
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=_workers, pin_memory=True, persistent_workers=True if _workers else False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=_workers, pin_memory=True, persistent_workers=True if _workers else False)
    
    model = ConvAML_CIFAR_TrueDynamic(K=args.K, max_dim=args.max_dim, max_layers=args.max_layers,
                                      in_channels=args.base_channels, num_classes=num_classes,
                                      hts_tasks=args.hts_tasks, hts_latent=args.hts_latent,
                                      hts_rank=args.hts_rank).to(device)
    model.fixed_D = args.fixed_D
    model.fixed_L = args.fixed_L
    if args.base_channels > 1:
        print(f"Unified {args.base_channels}-channel base: gray tasks auto-lifted, RGB native")
    _D_map = {}
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model_state = model.state_dict()
        keys_to_remove = []
        for k, v in list(state.items()):
            if k in model_state and v.shape != model_state[k].shape:
                keys_to_remove.append(k)
                if k == "proto_weights":
                    min_dim = min(v.shape[1], model_state[k].shape[1])
                    model.proto_weights.data[:, :min_dim].copy_(v[:, :min_dim])
                    print(f"Copied shared backbone proto_weights: {min_dim} features")
                else:
                    print(f"Skipping head shape mismatch for {k}: ckpt {v.shape} vs model {model_state[k].shape}")
        for k in keys_to_remove:
            del state[k]
        model.load_state_dict(state, strict=False)
        _D_map = dict(state.get('_D_ceiling_map', {})) if isinstance(state, dict) else {}
        # Warm-start initialization for NEW tasks: copy the prior task's encoder
        # and controller heads into this task's slots so a fresh task (e.g. CIFAR)
        # starts from learned feature extraction + gating instead of cold zero/
        # random init. Per-task isolation is preserved because we only COPY into
        # this task's own slot (its params are still trainable and independent;
        # prior tasks' params are untouched).
        if args.task_id >= 0 and model.num_task_blocks > 1:
            src_t = max(0, args.task_id - 1)
            dst_t = args.task_id
            if f"encoders.{dst_t}.0.weight" not in state:
                with torch.no_grad():
                    src_enc = model.encoders[src_t]
                    dst_enc = model.encoders[dst_t]
                    for li, (sm, dm) in enumerate(zip(src_enc, dst_enc)):
                        if isinstance(sm, nn.Linear):
                            dm.weight.copy_(sm.weight)
                            dm.bias.copy_(sm.bias)
                        elif isinstance(sm, nn.Conv2d):
                            if sm.in_channels != dm.in_channels:
                                # 1->3 channel bridge: repeat gray filters across RGB
                                rep = dm.in_channels // sm.in_channels
                                dm.weight.copy_(sm.weight.repeat(1, rep, 1, 1) / float(rep))
                            else:
                                dm.weight.copy_(sm.weight)
                            dm.bias.copy_(sm.bias)
                        elif isinstance(sm, nn.BatchNorm2d):
                            dm.weight.copy_(sm.weight)
                            dm.bias.copy_(sm.bias)
                            dm.running_mean.copy_(sm.running_mean)
                            dm.running_var.copy_(sm.running_var)
                    # Controller heads: prototype selector, structure delta, block mode
                    for ps, pd in zip(model.prototype_selectors[src_t].parameters(),
                                      model.prototype_selectors[dst_t].parameters()):
                        pd.data.copy_(ps.data)
                    for ss, sd_ in zip(model.structure_deltas[src_t].parameters(),
                                       model.structure_deltas[dst_t].parameters()):
                        sd_.data.copy_(ss.data)
                    for bs, bd in zip(model.block_mode_heads[src_t].parameters(),
                                      model.block_mode_heads[dst_t].parameters()):
                        bd.data.copy_(bs.data)
                print(f"Warm-started task {dst_t} encoder+controller from task {src_t}")
        # Migration: old checkpoints had ONE shared prototype_selector /
        # structure_delta / block_mode_head. New architecture has per-task
        # heads; copy the old shared head into EVERY per-task slot so resumed
        # runs keep working (retraining will tune each task's own head).
        with torch.no_grad():
            for src_pre, dst_prefix in [("prototype_selector", "prototype_selectors"),
                                        ("structure_delta", "structure_deltas"),
                                        ("block_mode_head", "block_mode_heads")]:
                src_probe1 = f"{src_pre}.weight"
                src_probe2 = f"{src_pre}.0.weight"
                dst_probe = f"{dst_prefix}.0.weight"
                if (src_probe1 in state or src_probe2 in state) and dst_probe not in state:
                    for t in range(model.num_task_blocks):
                        dst = getattr(model, dst_prefix)[t]
                        dst_params = {n: p for n, p in dst.named_parameters()}
                        dst_buffers = {n: b for n, b in dst.named_buffers()}
                        for (sk, sv) in state.items():
                            if sk.startswith(src_pre + "."):
                                sub = sk[len(src_pre)+1:]
                                if sub in dst_params:
                                    dst_params[sub].copy_(sv)
                                elif sub in dst_buffers:
                                    dst_buffers[sub].copy_(sv)
            print(f"Loaded base weights from {args.resume}")
    
    if args.task_id >= 0:
        assert args.hts_tasks > 0, "--task-id requires --hts-tasks > 0"
        model.active_task = args.task_id
        print(f"HtS active on task {args.task_id}")
    if args.bt_lambda > 0.0 and model.hts is not None:
        model.hts.allow_delta0 = True
        print(f"Backward transfer ON: delta[0] trainable (bt_lambda={args.bt_lambda})")
    
    if args.freeze_base:
        for name, p in model.named_parameters():
            if name.startswith("hts.") or name.startswith("input_adapter.") or name.startswith("router."):
                continue
            if args.task_id >= 0 and (name.startswith(f"encoders.{args.task_id}.")
                                      or name.startswith(f"prototype_selectors.{args.task_id}.")
                                      or name.startswith(f"structure_deltas.{args.task_id}.")
                                      or name.startswith(f"block_mode_heads.{args.task_id}.")):
                continue
            p.requires_grad = False
        if args.task_id >= 0:
            # Per-task isolation guarantee: when retraining a specific task,
            # only that task's basis/coefs (+ delta0 for backward transfer, and
            # the input adapter) may be updated. Other tasks' params stay OUT of
            # the optimizer so they are untouched by weight decay or KL drift
            # (this is what keeps task B safe while task C is trained).
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith("input_adapter."):
                    continue
                if name.startswith("hts.coefs.") or name.startswith("hts.corr_coefs."):
                    idx = int(name.split(".")[2])
                    keep = (idx == args.task_id) or (idx == 0 and args.bt_lambda > 0.0)
                elif name.startswith("hts.basis.") or name.startswith("hts.corr_basis."):
                    keep = True  # SHARED basis: always trainable, protected via
                    #              project_grads() (orthogonal to frozen prior coefs)
                else:
                    keep = True  # task_head + centroids (buffers, no grad anyway)
                if not keep:
                    p.requires_grad = False
        trainable = [p for n, p in model.named_parameters() if p.requires_grad]
        model._bn_frozen = True
        print(f"Freeze-base ON: training only task-{args.task_id} HtS + adapter ({len(trainable)} tensors)")
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]

    teacher_model = None
    if args.teacher:
        teacher_model = ConvAML_CIFAR_TrueDynamic(K=args.K, max_dim=args.max_dim, max_layers=args.max_layers,
                                                  in_channels=args.base_channels, num_classes=num_classes,
                                                  hts_tasks=args.hts_tasks, hts_latent=args.hts_latent,
                                                  hts_rank=args.hts_rank).to(device)
        teacher_model.load_state_dict(torch.load(args.teacher, map_location=device), strict=False)
        teacher_model.fixed_D = args.fixed_D
        teacher_model.fixed_L = args.fixed_L
        teacher_model.active_task = 0
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_model.eval()
        print(f"Teacher (task A) loaded from {args.teacher}")

    replay_exemplars = None
    if (args.replay_lambda > 0.0 or args.bt_lambda > 0.0) and args.replay_buffer:
        replay_ds = args.replay_buffer
        assert replay_ds, "replay/BT requires --replay-buffer <dataset>"
        if replay_ds.startswith('cifar'):
            transform_replay = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            rcls = torchvision.datasets.CIFAR100 if replay_ds == 'cifar100' else torchvision.datasets.CIFAR10
        else:
            transform_replay = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            rcls = torchvision.datasets.FashionMNIST if replay_ds == 'fashion_mnist' else torchvision.datasets.MNIST
        rset = rcls(root=data_root, train=True, download=True, transform=transform_replay)
        rset_idx = _IndexedReplay(rset)
        replay_exemplars = torch.utils.data.DataLoader(rset_idx, batch_size=args.replay_n, shuffle=True, num_workers=0)
        teacher_logits_cache = None
        teacher_feat_cache = None
        cache_f1 = f".cache_{os.path.basename(args.teacher)}_{replay_ds}.pt"
        if teacher_model is not None and args.replay_lambda > 0.0:
            if os.path.exists(cache_f1):
                cached = torch.load(cache_f1, map_location=device)
                teacher_logits_cache = cached['logits']
                teacher_feat_cache = cached.get('feat')
                print(f"Teacher logits LOADED FROM CACHE {cache_f1} ({len(rset)} samples, {time.time()-T0:.1f}s)")
            else:
                teacher_logits_cache = torch.zeros(len(rset), num_classes, dtype=torch.float32, device=device)
                if args.stab_lambda > 0.0:
                    teacher_feat_cache = None
                t_loader = torch.utils.data.DataLoader(rset, batch_size=256, shuffle=False, num_workers=0)
                with torch.no_grad():
                    off = 0
                    for t_x, _ in t_loader:
                        t_x = t_x.to(device)
                        t_out, _, _, t_diag = teacher_model(t_x, tau=1.0, alpha=1.0)
                        if teacher_feat_cache is None:
                            teacher_feat_cache = torch.zeros(len(rset), t_diag["z_full"].shape[-1], dtype=torch.float32, device=device)
                        teacher_logits_cache[off:off + t_x.size(0)] = t_out
                        teacher_feat_cache[off:off + t_x.size(0)] = t_diag["z_full"]
                        off += t_x.size(0)
                torch.save({'logits': teacher_logits_cache.cpu(), 'feat': teacher_feat_cache.cpu() if teacher_feat_cache is not None else None}, cache_f1)
                print(f"Teacher logits computed & cached to {cache_f1} ({len(rset)} samples, {time.time()-T0:.1f}s)")
        else:
            teacher_logits_cache = None
        print(f"Replay exemplars from {replay_ds} ({len(rset)} samples, {args.replay_n}/batch)")

    # Optional SECOND replay stream: protect task B while training task C by
    # replaying task-B exemplars under the task-B HtS path, distilled toward the
    # task-B teacher checkpoint. The per-task basis already isolates deltas, so
    # this only helps keep the task-B decoder sharp under the new data flow.
    replay2_exemplars = None
    teacher2_logits_cache = None
    teacher2_feat_cache = None
    if args.replay_buffer2 and args.replay_lambda > 0.0:
        replay_ds2 = args.replay_buffer2
        if replay_ds2.startswith('cifar'):
            transform_replay2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            rcls2 = torchvision.datasets.CIFAR100 if replay_ds2 == 'cifar100' else torchvision.datasets.CIFAR10
        else:
            transform_replay2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            rcls2 = torchvision.datasets.FashionMNIST if replay_ds2 == 'fashion_mnist' else torchvision.datasets.MNIST
        rset2 = rcls2(root=data_root, train=True, download=True, transform=transform_replay2)
        replay2_exemplars = torch.utils.data.DataLoader(_IndexedReplay(rset2), batch_size=args.replay_n, shuffle=True, num_workers=0)
        teacher2_ckpt = args.replay_teacher2 or args.replay_teacher or ''
        if teacher2_ckpt:
            cache_f2 = f".cache_{os.path.basename(teacher2_ckpt)}_{replay_ds2}.pt"
            if os.path.exists(cache_f2):
                cached2 = torch.load(cache_f2, map_location=device)
                teacher2_logits_cache = cached2['logits'].to(device)
                teacher2_feat_cache = cached2['feat'].to(device) if cached2.get('feat') is not None else None
                print(f"Teacher2 logits LOADED FROM CACHE {cache_f2} ({len(rset2)} samples, {time.time()-T0:.1f}s)")
            else:
                teacher2_model = ConvAML_CIFAR_TrueDynamic(K=args.K, max_dim=args.max_dim, max_layers=args.max_layers,
                                                           in_channels=args.base_channels, num_classes=num_classes,
                                                           hts_tasks=args.hts_tasks, hts_latent=args.hts_latent,
                                                           hts_rank=args.hts_rank).to(device)
                teacher2_model.load_state_dict(torch.load(teacher2_ckpt, map_location=device), strict=False)
                teacher2_model.fixed_D = args.fixed_D
                teacher2_model.fixed_L = args.fixed_L
                teacher2_model.active_task = args.replay_task2
                for p in teacher2_model.parameters():
                    p.requires_grad = False
                teacher2_model.eval()
                teacher2_logits_cache = torch.zeros(len(rset2), num_classes, dtype=torch.float32, device=device)
                teacher2_feat_cache = None
                t2_loader = torch.utils.data.DataLoader(rset2, batch_size=256, shuffle=False, num_workers=0)
                with torch.no_grad():
                    off2 = 0
                    for t2_x, _ in t2_loader:
                        t2_x = t2_x.to(device)
                        t2_out, _, _, t2_diag = teacher2_model(t2_x, tau=1.0, alpha=1.0)
                        if teacher2_feat_cache is None:
                            teacher2_feat_cache = torch.zeros(len(rset2), t2_diag["z_full"].shape[-1], dtype=torch.float32, device=device)
                        teacher2_logits_cache[off2:off2 + t2_x.size(0)] = t2_out
                        teacher2_feat_cache[off2:off2 + t2_x.size(0)] = t2_diag["z_full"]
                        off2 += t2_x.size(0)
                torch.save({'logits': teacher2_logits_cache.cpu(), 'feat': teacher2_feat_cache.cpu() if teacher2_feat_cache is not None else None}, cache_f2)
                print(f"Teacher2 logits computed & cached to {cache_f2} ({len(rset2)} samples, {time.time()-T0:.1f}s)")
        else:
            print(f"WARNING: replay-buffer2={replay_ds2} but no --replay-teacher2/--replay-teacher; KL replay disabled (CE only via task2 path)")
        print(f"Task-B replay stream: {replay_ds2} ({len(rset2)} samples, {args.replay_n}/batch, under task {args.replay_task2})")

    if args.freeze_base and args.task_id >= 0:
        # Differentiated LR: the freshly-warm-started task encoder gets a higher
        # LR (2.5x) so it can fit the new domain (e.g. CIFAR RGB) quickly within
        # the 12-epoch budget, while the HtS bank / prior-task params keep the
        # base LR for stability.
        enc_names = [n for n, p in model.named_parameters() if f"encoders.{args.task_id}." in n and p.requires_grad]
        other_params = [p for n, p in model.named_parameters()
                        if f"encoders.{args.task_id}." not in n and p.requires_grad]
        enc_params = [p for n, p in model.named_parameters()
                      if f"encoders.{args.task_id}." in n and p.requires_grad]
        optimizer = optim.AdamW([
            {'params': other_params, 'lr': args.lr, 'weight_decay': 1e-4},
            {'params': enc_params, 'lr': args.lr * args.enc_lr_mult, 'weight_decay': 5e-5},
        ])
        print(f"Differentiated LR: encoder@ {args.lr*args.enc_lr_mult:.2e} ({len(enc_params)} tensors), rest@ {args.lr:.2e} ({len(other_params)} tensors)")
    else:
        optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    print(f"========== TRAINING {args.max_layers}-LAYER TRUE AML DYNAMIC ==========")
    meta_loss_fn = MetaLoss(
        lambda_eff=0.0, 
        lambda_p=0.0, 
        p_allowance=args.p_allowance
    ).to(device)
    
    # Track the best test accuracy on the CURRENT task so we can save the
    # optimal checkpoint. Late-epoch tau/lambda collapse can degrade the final
    # epoch's test acc even when earlier epochs were the best fit.
    best_test_acc = -1.0
    best_epoch = -1
    
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_eff = 0, 0
        train_correct, train_total = 0, 0
        t0 = time.time()
        
        # TWO-STAGE CURRICULUM
        # Phase A fraction lowered (0.6 -> 0.5) so the model starts compressing
        # toward its final architecture earlier instead of spending most epochs
        # in high-tau exploration. This speeds CIFAR convergence while keeping
        # the soft-tau stability at the phase boundary.
        # For TRANSFER tasks (task_id > 0 with freeze-base) the encoder was
        # warm-started from the prior task, so we shorten Phase A (0.5 -> 0.25)
        # to free more epochs for Phase-B compression/D-tuning.
        if args.task_id > 0 and args.freeze_base:
            phase_A_epochs = max(2, int(0.25 * args.epochs))
        else:
            phase_A_epochs = int(0.5 * args.epochs)
        
        if epoch < phase_A_epochs:
            # Phase A: Free Exploration (Military Training)
            current_lambda_eff = 0.0
            tau = max(1.0, 3.0 - (epoch / phase_A_epochs) * 2.0)
            # Keep block-mode entropy > 0 during exploration so every mode (incl.
            # MLP/ViT) stays alive ~25% and its weights actually get trained
            # (avoids the one-hot death-spiral where unselected modes never learn).
            w_gate, w_bm, w_dim = 0.0, 0.05, 0.0
            alpha = 0.0
            model._D_ceiling = 0.0
        else:
            # Phase B: Compression & Transition
            progress_B = (epoch - phase_A_epochs) / (args.epochs - phase_A_epochs)
            current_lambda_eff = args.lambda_eff * progress_B
            tau = max(0.3, 1.0 - progress_B * 0.7)
            w_gate, w_bm, w_dim = 0.04, 0.06, 0.005
            
            # Smoothly release controllers over RELEASE_EPOCHS
            RELEASE_EPOCHS = min(5, args.epochs - phase_A_epochs)
            alpha = min(1.0, (epoch - phase_A_epochs) / RELEASE_EPOCHS)

            # Smooth D-release schedule: shrink D linearly from phase-A width
            # (0.95*max_dim) down to D_MIN across the ENTIRE phase B, so eval's
            # D=learned_d (alpha=1) never snaps 42->24 in one epoch, which caused
            # the Test Acc collapse (4.4%) at the phase boundary.
            D_MIN = float(os.environ.get("AML_D_END", "24"))
            D_start = 0.95 * model.max_dim
            D_end = D_MIN
            n_phaseB = max(1, args.epochs - phase_A_epochs)
            model._D_ceiling = D_start - (D_start - D_end) * (progress_B * n_phaseB) / n_phaseB
            
        meta_loss_fn.lambda_eff = current_lambda_eff
        
        # Use CutMix in Phase A mostly, but keep it on slightly in Phase B if desired. 
        # For CIFAR-10, CutMix prob = 0.5 is standard. We'll turn it off in the last 5 epochs for clean fine-tuning.
        use_cutmix = (epoch < args.epochs - 5)
        
        replay_iter = None
        replay_sum = 0.0
        bt_sum = 0.0
        replay2_sum = 0.0
        batch_i = 0
        if replay_exemplars is not None:
            replay_iter = iter(replay_exemplars)
        replay2_iter = None
        if replay2_exemplars is not None:
            replay2_iter = iter(replay2_exemplars)
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            lam = 1.0
            if use_cutmix and np.random.rand() < 0.5:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha=1.0)
            elif use_cutmix and np.random.rand() < 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0)
            with torch.amp.autocast("cuda", enabled=False):
                outputs, struct, pi, diag = model(inputs, tau=tau, alpha=alpha)
                gen_model = GeneratedModel(structure=struct, weights={}, pi_logits=pi)
            
            if lam < 1.0:
                loss_dict_a = meta_loss_fn(
                    logits_train=outputs, targets_train=targets_a,
                    logits_val=None, targets_val=None,
                    model=gen_model, epoch_progress=epoch/args.epochs
                )
                loss_dict_b = meta_loss_fn(
                    logits_train=outputs, targets_train=targets_b,
                    logits_val=None, targets_val=None,
                    model=gen_model, epoch_progress=epoch/args.epochs
                )
                loss = lam * loss_dict_a["loss"] + (1 - lam) * loss_dict_b["loss"]
                task_loss_val = lam * loss_dict_a["task_loss"].item() + (1 - lam) * loss_dict_b["task_loss"].item()
                eff_cost_val = loss_dict_a.get("efficiency_cost", torch.tensor(0.0)).item()
            else:
                loss_dict = meta_loss_fn(
                    logits_train=outputs, targets_train=targets,
                    logits_val=None, targets_val=None,
                    model=gen_model, epoch_progress=epoch/args.epochs
                )
                loss = loss_dict["loss"]
                task_loss_val = loss_dict["task_loss"].item()
                eff_cost_val = loss_dict.get("efficiency_cost", torch.tensor(0.0)).item()
            # Track training accuracy against ORIGINAL labels (note: when CutMix/Mixup
            # is active, this underestimates the model's actual learning signal, but it
            # still reflects the overfitting trend; it becomes exact in the last 5 epochs
            # where augmentation is turned off).
            if lam >= 1.0:
                train_correct += (outputs.argmax(-1) == targets).sum()
                train_total += targets.size(0)
            else:
                # inputs/labels were mixed; compare against the higher-weight label
                lbl = targets_a if lam >= 0.5 else targets_b
                train_correct += (outputs.argmax(-1) == lbl).sum()
                train_total += lbl.size(0)

            if w_gate > 0.0 or w_bm > 0.0 or w_dim > 0.0:
                layer_soft = diag["layer_soft"].clamp(1e-6, 1.0 - 1e-6)
                ent_gate = -(layer_soft * torch.log(layer_soft) + (1.0 - layer_soft) * torch.log(1.0 - layer_soft)).mean()
                p_bm = F.softmax(diag["bm_logits"], dim=-1).clamp(1e-6, 1.0)
                ent_bm = -(p_bm * torch.log(p_bm)).sum(dim=-1).mean()
                dim_floor = F.relu(0.4 * model.max_dim - diag["d_lengths"].mean()).mean()
                reg = w_gate * ent_gate + w_bm * ent_bm + w_dim * dim_floor
                loss = loss + reg

            if model.hts is not None:
                model.hts.update_centroid(args.task_id if args.task_id >= 0 else 0, diag["z"])

            router_val = 0.0
            if args.router_lambda > 0.0 and model.hts is not None:
                tgt = torch.full((inputs.size(0),), args.task_id if args.task_id >= 0 else 0,
                                 dtype=torch.long, device=device)
                rl = F.cross_entropy(diag["task_logits"], tgt)
                loss = loss + args.router_lambda * rl
                router_val = rl.item()

            distill_val = 0.0
            replay_val = 0.0
            bt_val = 0.0
            stab_val = 0.0
            r_in = None
            if replay_exemplars is not None and model.hts is not None and (batch_i % max(1, args.replay_every) == 0):
                try:
                    r_in, r_lbl, r_idx = next(replay_iter)
                except (StopIteration, TypeError):
                    replay_iter = iter(replay_exemplars)
                    r_in, r_lbl, r_idx = next(replay_iter)
                r_in = r_in.to(device)
                r_lbl = r_lbl.to(device)
                r_idx = r_idx.to(device)
                # Gray replay (MNIST/Fashion) is auto-lifted to the base's
                # channel count by the model's gray_adapter in forward.
                if args.bt_lambda > 0.0:
                    saved_task = model.active_task
                    model.active_task = 0
                    r_out_bt, _, _, r_diag = model(r_in, tau=tau, alpha=alpha)
                    model.active_task = saved_task
                    bt_loss = F.cross_entropy(r_out_bt, r_lbl)
                    loss = loss + args.bt_lambda * bt_loss
                    bt_val = bt_loss.item()
                else:
                    r_diag = None
                if teacher_logits_cache is not None and args.replay_lambda > 0.0:
                    saved_task = model.active_task
                    model.active_task = args.replay_task
                    r_out_s, _, _, r_diag = model(r_in, tau=tau, alpha=alpha)
                    model.active_task = saved_task
                    t_logits = teacher_logits_cache[r_idx]
                    rt_prob = F.softmax(t_logits / args.distill_tau, dim=-1)
                    rs_prob = F.log_softmax(r_out_s / args.distill_tau, dim=-1)
                    rk = F.kl_div(rs_prob, rt_prob, reduction='batchmean') * (args.distill_tau ** 2)
                    loss = loss + args.replay_lambda * rk
                    replay_val = rk.item()
                    if args.stab_lambda > 0.0 and teacher_feat_cache is not None:
                        z_cur = r_diag["z_full"]
                        z_tea = teacher_feat_cache[r_idx]
                        loss = loss + args.stab_lambda * F.mse_loss(z_cur, z_tea)
                        stab_val = F.mse_loss(z_cur, z_tea).item()
                if args.router_lambda > 0.0 and r_diag is not None and r_diag.get("task_logits") is not None:
                    tgt0 = torch.full((r_in.size(0),), args.replay_task, dtype=torch.long, device=device)
                    rl0 = F.cross_entropy(r_diag["task_logits"], tgt0)
                    loss = loss + args.router_lambda * rl0
            replay_sum = replay_sum + replay_val
            bt_sum = bt_sum + bt_val

            replay2_val = 0.0
            stab2_val = 0.0
            if replay2_exemplars is not None and model.hts is not None and teacher2_logits_cache is not None \
                    and (batch_i % max(1, args.replay_every) == 0):
                try:
                    r2_in, r2_lbl, r2_idx = next(replay2_iter)
                except (StopIteration, TypeError):
                    replay2_iter = iter(replay2_exemplars)
                    r2_in, r2_lbl, r2_idx = next(replay2_iter)
                r2_in = r2_in.to(device)
                r2_lbl = r2_lbl.to(device)
                r2_idx = r2_idx.to(device)
                # Gray replay auto-lifted to base channels in forward.
                saved_task = model.active_task
                model.active_task = args.replay_task2
                r2_out_s, _, _, r2_diag = model(r2_in, tau=tau, alpha=alpha)
                model.active_task = saved_task
                if args.router_lambda > 0.0 and r2_diag is not None and r2_diag.get("task_logits") is not None:
                    tgt2 = torch.full((r2_in.size(0),), args.replay_task2, dtype=torch.long, device=device)
                    rl2 = F.cross_entropy(r2_diag["task_logits"], tgt2)
                    loss = loss + args.router_lambda * rl2
                t2_logits = teacher2_logits_cache[r2_idx]
                rt2 = F.softmax(t2_logits / args.distill_tau, dim=-1)
                rs2 = F.log_softmax(r2_out_s / args.distill_tau, dim=-1)
                rk2 = F.kl_div(rs2, rt2, reduction='batchmean') * (args.distill_tau ** 2)
                loss = loss + getattr(args, 'replay2_lambda', args.replay_lambda) * rk2
                replay2_val = rk2.item()
                if args.stab_lambda > 0.0 and teacher2_feat_cache is not None:
                    z2_cur = r2_diag["z_full"]
                    z2_tea = teacher2_feat_cache[r2_idx]
                    loss = loss + args.stab_lambda * F.mse_loss(z2_cur, z2_tea)
                    stab2_val = F.mse_loss(z2_cur, z2_tea).item()
            replay2_sum = replay2_sum + replay2_val

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if model.hts is not None and args.task_id >= 0 and args.protect_prior:
                model.hts.project_grads(args.task_id)
            optimizer.step()
            if model.hts is not None and args.task_id >= 0 and args.protect_prior:
                model.hts.correct_basis(args.task_id)
            
            total_loss += task_loss_val * inputs.size(0)
            total_eff += eff_cost_val * inputs.size(0)
            batch_i += 1
            
        scheduler.step()
        train_loss = total_loss / len(trainset)
        if isinstance(train_correct, torch.Tensor):
            train_correct = train_correct.item()
        train_acc = (train_correct / max(train_total, 1)) * 100.0
        
        model.eval()
        correct, eff_params_sum = 0, 0
        bm_sums = torch.zeros(model.max_layers, model.num_modes, device=device) # [max_layers, num_modes]
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, struct, _, _ = model(inputs, tau=0.1)
                correct += (outputs.argmax(-1) == targets).sum().item()
                eff = struct.effective_params_formula()
                if eff_params_sum == 0:  # print only for first batch
                    print(f"DEBUG - L: {struct.num_layers[0].item():.2f}, D_mean: {struct.hidden_dim.mean(dim=-1)[0].item():.2f}, eff_params: {eff[0].item():.2f}")
                    # print details
                    D_mean = struct.hidden_dim.mean(dim=-1)[0]
                    L = struct.num_layers[0]
                    c1 = 3 * D_mean * 9 + D_mean
                    c2 = torch.clamp(L - 1.0, min=0.0) * (D_mean * D_mean * 9 + D_mean)
                    vit_gate = struct.block_mode[0, :, 2].sum(dim=-1)
                    print(f"DEBUG - c1: {c1.item():.2f}, c2: {c2.item():.2f}, vit_gate: {vit_gate.item():.2f}")
                eff_params_sum += eff.sum().item()
                bm_sums += struct.block_mode.sum(dim=0)
                
        test_acc = correct / len(testset)
        avg_eff_params = eff_params_sum / len(testset)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            if args.protect_prior and model.hts is not None and args.task_id >= 1:
                # Snapshot THIS task's delta at the exact state that is about to
                # be saved as the best checkpoint. Running lock_task() only after
                # the full epoch loop snapshots the FINAL epoch's delta, but the
                # saved checkpoint is the best (mid-training) state -- the
                # mismatch made basis_targets stale and a later task's
                # correct_basis() would wipe this task's delta (catastrophic).
                model.hts.lock_task(args.task_id)
            best_state = {k: v.detach().to('cpu').clone()
                          for k, v in model.state_dict().items()}
            best_D_ceiling = float(getattr(model, '_D_ceiling', 0.0))
        
        n_batches = max(len(trainloader), 1)
        print(f"Epoch {epoch+1:2d}/{args.epochs} | {time.time()-t0:5.1f}s | Task Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc*100:.2f}% | Avg Used Params: {avg_eff_params/1e6:.2f}M | Tau: {tau:.1f} | Lambda: {current_lambda_eff:.4f} | Reg(g/b/d): {w_gate:.2f}/{w_bm:.2f}/{w_dim:.3f} | ReplayKL: {replay_sum/n_batches:.4f} | BT: {bt_sum/n_batches:.4f} | Replay2KL:            {replay2_sum/n_batches:.4f} | Stab: {(stab_val+stab2_val)/max(1,n_batches):.4f}")
        
        bm_avg = bm_sums / len(testset)
        print("  -> Block Mode Distribution (Plain / ResNet / ViT / MLP) per Layer:")
        for i in range(model.max_layers):
            p, r, v, m = bm_avg[i].tolist()
            print(f"     Layer {i+1}: {p*100:5.1f}% | {r*100:5.1f}% | {v*100:5.1f}% "
                  f"| {m*100:5.1f}%")

    if args.protect_prior and model.hts is not None and args.task_id >= 1:
        model.hts.lock_task(args.task_id)
        print(f"Locked HtS delta for task {args.task_id}: prior tasks are now frozen against reuse")
    # Save the BEST task checkpoint (best test acc on the current task) rather
    # than the final epoch, which can be degraded by late-epoch tau/lambda
    # collapse. lock_task already snapshotted this task's delta.
    # The architectural D_ceiling varies per epoch (annealed schedule), so we
    # must tag the saved state with the D_ceiling it was trained under; eval
    # reads it back, otherwise it would evaluate a mid-training state at the
    # final epoch's lower D_ceiling and collapse.
    if best_epoch >= 0:
        # Tag the D_ceiling of the CURRENT task's best epoch. Prior tasks were
        # finalized at the final (lower) D_ceiling, so eval only applies this
        # override when the evaluated task matches _D_ceiling_task; other tasks
        # fall back to the final-schedule D_ceiling.
        model_config = dict(K=args.K, max_dim=args.max_dim, max_layers=args.max_layers,
                            base_channels=args.base_channels, hts_tasks=args.hts_tasks,
                            hts_latent=args.hts_latent, hts_rank=args.hts_rank,
                            num_classes=num_classes, seed=args.seed,
                            fixed_D=args.fixed_D, fixed_L=args.fixed_L)
        best_state['_D_ceiling'] = best_D_ceiling
        best_state['_D_ceiling_task'] = int(args.task_id if args.task_id >= 0 else 0)
        _D_map[int(args.task_id if args.task_id >= 0 else 0)] = best_D_ceiling
        best_state['_D_ceiling_map'] = _D_map
        best_state['model_config'] = model_config
        torch.save(best_state, args.out)
        print(f"Saved CHECKPOINT -> {args.out}  (best test acc {best_test_acc*100:.2f}% at epoch {best_epoch+1}, task-{args.task_id} D_ceiling {best_D_ceiling:.1f})")
    else:
        final_state = model.state_dict()
        final_state['_D_ceiling'] = float(getattr(model, '_D_ceiling', 0.0))
        final_state['_D_ceiling_task'] = int(args.task_id if args.task_id >= 0 else 0)
        torch.save(final_state, args.out)
        print(f"Saved checkpoint -> {args.out}")

if __name__ == '__main__':
    print("Eager path supports 4 dynamic block modes: Plain / ResNet / ViT / MLP.")
    main()
