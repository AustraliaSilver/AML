import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AML_Core_Code'))
from train_cv_cifar_18 import AcceleratedCVNetwork, get_prototype_layout

# Use mgn from current directory (where we compiled it)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mgn.architecture import ModelStructure, GeneratedModel
from mgn.losses import MetaLoss
from mgn.generator import DistanceRouter

class TaskCentroidOOD(nn.Module):
    def __init__(self, feature_dim=128, max_tasks=10, K=32):
        super().__init__()
        self.K = K
        self.register_buffer('centroids', torch.zeros(max_tasks, feature_dim))
        self.register_buffer('task_counts', torch.zeros(max_tasks))
        self.current_task_id = 0
        
    def update_centroid(self, z):
        z_mean = z.mean(dim=0).detach()
        curr_count = self.task_counts[self.current_task_id]
        new_count = curr_count + 1
        self.centroids[self.current_task_id] = (self.centroids[self.current_task_id] * curr_count + z_mean) / new_count
        self.task_counts[self.current_task_id] = new_count
        
    def get_pi(self, z, active_rows_per_task):
        active_tasks = int((self.task_counts > 0).sum().item())
        if active_tasks == 0:
            pi = torch.ones(z.shape[0], self.K, device=z.device) / self.K
            return pi
            
        c = self.centroids[:active_tasks]
        z_norm = F.normalize(z, dim=-1)
        c_norm = F.normalize(c, dim=-1)
        sim = z_norm @ c_norm.T
        task_probs = torch.softmax(sim * 1000.0, dim=-1)
        
        pi = torch.zeros(z.shape[0], self.K, device=z.device)
        for t in range(active_tasks):
            start = t * active_rows_per_task
            end = start + active_rows_per_task
            pi[:, start:end] = (task_probs[:, t:t+1] / active_rows_per_task)
            
        return pi

class ConvAML_CIFAR_TrueDynamic(nn.Module):
    def __init__(self, K=32, latent_dim=1024, max_layers=18, max_dim=64, norm_mode="gn1", train_encoder=True):
        super().__init__()
        self.K = K
        self.max_layers = max_layers
        self.max_dim = max_dim

        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1), nn.GroupNorm(4, 128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.GroupNorm(16, 512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(512 * 4, latent_dim), nn.ReLU()
        )
        if not train_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        
        self.prototype_selector = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            DistanceRouter(256, K)
        )

        self.ood_router = TaskCentroidOOD(feature_dim=latent_dim, max_tasks=10, K=K)
        self.rows_per_task = 24
        self.structure_prototypes = nn.Parameter(torch.zeros(K, max_layers + (max_layers * max_dim)))
        self.block_mode_prototypes = nn.Parameter(torch.zeros(K, max_layers * 2))
        
        with torch.no_grad():
            # ResNet-heavy init: keeps residual paths dominant during warmup.
            # Balanced (0.5) init was tested and destabilized training: Plain
            # rewrites (relu(GN(conv(h)))) destroyed signal -> post-unfreeze
            # the router overshot into one-hot prototypes (-12% acc).
            self.block_mode_prototypes.data.view(K, max_layers, 2)[:, :, 1] = 2.0

        self._init_prototypes()

        self.exec_ = AcceleratedCVNetwork(max_layers=max_layers, max_dim=max_dim, num_classes=10, norm_mode=norm_mode, K=self.K)

    def _init_prototypes(self):
        import math
        import torch.nn as nn

        weight_dim, offsets, layer_info, wo_offset, bo_offset = get_prototype_layout(self.max_layers, self.max_dim, 10)
        self.proto_weights = nn.Parameter(torch.zeros(self.K, weight_dim))

        # Fix Kaiming Init bug: fan_in is incorrectly determined as weight_dim
        # We compute proper fan_in per spatial slice.
        with torch.no_grad():
            for i in range(self.max_layers):
                w_start, w_end, b_start, b_end, in_c = offsets[i]
                fan_in = in_c * 3 * 3
                bound = math.sqrt(6.0 / fan_in)
                self.proto_weights.data[:, w_start:w_end].uniform_(-bound, bound)
            # Second conv in ResNet blocks (w2)
            from mgn.layout import get_second_conv_layout
            offsets2 = get_second_conv_layout(self.max_layers, self.max_dim, 10)
            for (w2_start, w2_end, b2_start, b2_end) in offsets2:
                fan_in2 = self.max_dim * 3 * 3
                bound2 = math.sqrt(6.0 / fan_in2)
                self.proto_weights.data[:, w2_start:w2_end].uniform_(-bound2, bound2)
            # Head (linear)
            fan_in_head = 16 * self.max_dim
            bound_head = math.sqrt(6.0 / fan_in_head)
            self.proto_weights.data[:, wo_offset:bo_offset].uniform_(-bound_head, bound_head)

    def caf_attention(self, features):
        return torch.softmax(features, dim=-1).clamp(min=1e-8)

    def forward(self, x, tau=1.0, struct_override=None, pi_mask=None):
        B = x.shape[0]; device = x.device
        z = self.vision_encoder(x)
        
        if self.training:
            self.ood_router.current_task_id = getattr(self, 'train_task_id', 0)
            with torch.no_grad():
                self.ood_router.update_centroid(z)
        
        pi_logits = self.prototype_selector(z)
        if pi_mask is not None:
            if pi_mask.dtype == torch.bool:
                pi_logits = pi_logits.masked_fill(~pi_mask.unsqueeze(0), -1e4)
            else:
                pi_logits = pi_logits.masked_fill((pi_mask == 0).unsqueeze(0), -1e4)

        if self.training:
            pi = F.gumbel_softmax(pi_logits, tau=tau, hard=False, dim=-1)
        else:
            pi = F.softmax(pi_logits / tau, dim=-1)

        delta = pi @ self.structure_prototypes
        layer_logits = delta[:, :self.max_layers] + 2.0
        layer_soft = torch.sigmoid(layer_logits)
        if self.training:
            layer_soft = layer_soft + 0.1 * torch.randn_like(layer_soft)
        layer_soft = torch.clamp(layer_soft, 1e-4, 1.0)

        dim_logits = delta[:, self.max_layers:].view(B, self.max_layers, self.max_dim)
        d_logit = dim_logits.mean(dim=-1, keepdim=True) + 2.0
        dim_fractions = torch.sigmoid(d_logit)

        d_lengths = torch.clamp(dim_fractions * self.max_dim, min=24.0)
        arange = torch.arange(self.max_dim, device=device).float().view(1, 1, self.max_dim)
        dim_masks = torch.sigmoid(10.0 * (d_lengths - arange))

        bm_logits = (pi @ self.block_mode_prototypes).view(B, self.max_layers, 2)

        if struct_override is not None:
            # Continual mode: use the per-domain stored structure slot
            layer_gates, block_mode = struct_override
            layer_gates = layer_gates.to(device=device, dtype=torch.float32)
            block_mode = block_mode.to(device=device, dtype=torch.float32)
        else:
            layer_gates = (layer_soft > 0.5).float().detach() + layer_soft - layer_soft.detach()
            if self.training:
                block_mode = F.gumbel_softmax(bm_logits, tau=tau, hard=True, dim=-1)
            else:
                # Deterministic argmax mode at eval
                block_mode = F.one_hot(bm_logits.argmax(dim=-1), 2).float()

        dim_weights = d_lengths / self.max_dim
        
        struct = ModelStructure(
            layer_gates=layer_gates,
            dim_weights=dim_weights,
            residual=torch.zeros(B, 1, device=device),
            dim_masks=dim_masks,
            block_mode=block_mode,
            use_conv=True, in_channels=3, max_layers=self.max_layers, max_dim=self.max_dim,
            dim_choices=(self.max_dim,), output_dim=10
        )

        out = self.exec_(x, self.proto_weights, pi, struct)
        diag = {
            "layer_soft": layer_soft,
            "bm_logits": bm_logits,
            "d_lengths": d_lengths,
            "z": z,
        }
        return out, struct, pi, diag

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

def get_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="True AML Dynamic Architecture Training")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'mnist', 'fashion'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--max-dim', type=int, default=64)
    parser.add_argument('--max-layers', type=int, default=18)
    parser.add_argument('--p-allowance', type=float, default=1.5e6)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--data-root', type=str, default=None,
                        help='torchvision dataset root (default: <repo>/data)')
    parser.add_argument('--steps', type=int, default=0,
                        help='cap train/eval batches per epoch (0 = full pass); '
                             'for CI smoke runs and quick sanity checks')
    parser.add_argument('--out', type=str, default='aml_core_best.pth')
    parser.add_argument('--arch-reg', type=int, default=1,
                        help='1 (default): apply the w_gate/w_bm/w_dim/w_pi structure '
                             'regularizers to the ARCHITECTURE optimizer as well as the '
                             'main one. 0: legacy behaviour (arch step = pure CE; those '
                             'prune-phase terms reached no optimizer).')
    return parser.parse_args(argv)

@torch.no_grad()
def eval_structure(model, loader, n_total, use_amp, amp_dtype, diag=True, max_steps=0):
    """Deterministic eval pass; returns (acc, diagnostics dict). Used for
    BOTH validation-based model selection and test reporting. With
    max_steps>0 only that many batches are evaluated and accuracy is
    computed over the evaluated samples (CI smoke runs)."""
    device = next(model.parameters()).device
    model.eval()
    correct, seen, eff_params_sum = 0, 0, 0
    res_sum, res_n = 0.0, 0
    bm_sums = torch.zeros(model.max_layers, 2, device=device)
    pi_ent_sum, pi_ent_n = 0.0, 0
    gate_soft_list = []
    for step, (inputs, targets) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            outputs, struct, pi_eval, diag_eval = model(inputs, tau=0.1)
        correct += (outputs.argmax(-1) == targets).sum().item()
        seen += targets.numel()
        if diag:
            eff_params_sum += struct.effective_params_formula().sum().item()
            bm_sums += struct.block_mode.sum(dim=0)
            res_sum += struct.block_mode[:, :, 1].mean().item(); res_n += 1
            pi_f = pi_eval.float()
            pi_ent_sum += -(pi_f * torch.log(pi_f.clamp(min=1e-9))).sum(-1).sum().item()
            pi_ent_n += pi_f.shape[0]
            gate_soft_list.append(diag_eval["layer_soft"].float())
    n = seen if max_steps else n_total
    m = {"eff": eff_params_sum / max(n, 1),
         "pi_H": pi_ent_sum / max(pi_ent_n, 1),
         "res_frac": res_sum / max(res_n, 1),
         "gate_std": torch.cat(gate_soft_list, dim=0).std(0).mean().item() if gate_soft_list else 0.0,
         "bm_str": ""}
    if diag:
        bm_avg = bm_sums / max(n, 1)
        m["bm_str"] = " | ".join(
            f"L{i+1}: {bm_avg[i, 0] * 100:4.0f}/{bm_avg[i, 1] * 100:4.0f}"
            for i in range(model.max_layers))
    return correct / max(n, 1), m


def structure_reg(loss, diag, pi, model, w_gate, w_bm, w_dim, w_pi):
    """Structure-entropy / width-floor / routing-sharpening terms.

    MUST be applied to BOTH the main objective (shapes the weights so they
    adapt to whatever gates the arch side proposes) AND the architecture
    objective. It used to be attached only to the main loss, whose
    arch-parameter gradients are zero_grad()ed before optimizer_arch.step()
    -> every hand-designed structure regularizer silently never reached
    the architecture (width collapsed unopposed at unfreeze; block modes
    had no counter-pressure against pure val CE, which always prefers the
    residual branch)."""
    if not (w_gate > 0.0 or w_bm > 0.0 or w_dim > 0.0 or w_pi > 0.0):
        return loss
    # fp32 + log1p: in bf16, (1 - 1e-6) rounds to 1.0, so a
    # saturated gate (soft == 1.0 after noise/clamp) would give
    # log(0) = -inf and 0 * -inf = NaN, poisoning all weights.
    layer_soft = diag["layer_soft"].float().clamp(1e-6, 1.0 - 1e-6)
    ent_gate = -(layer_soft * torch.log(layer_soft)
                 + (1.0 - layer_soft) * torch.log1p(-layer_soft)).mean()
    p_bm = F.softmax(diag["bm_logits"].float(), dim=-1).clamp(1e-6, 1.0 - 1e-6)
    ent_bm = -(p_bm * torch.log(p_bm)).sum(dim=-1).mean()
    dim_floor = F.relu(0.4 * model.max_dim - diag["d_lengths"].float().mean()).mean()
    reg = -w_gate * ent_gate - w_bm * ent_bm + w_dim * dim_floor

    if w_pi > 0.0:
        # Per-sample specialization: sharpen each sample's
        # prototype routing (low per-sample entropy) while
        # keeping overall usage spread across prototypes
        # (high usage entropy) -> samples pick DIFFERENT
        # prototypes instead of all converging to uniform.
        pi_f = pi.float()
        ent_pi = -(pi_f * torch.log(pi_f.clamp(min=1e-9))).sum(-1).mean()
        usage = pi_f.mean(0)
        ent_usage = -(usage * torch.log(usage.clamp(min=1e-9))).sum()
        reg = reg + w_pi * (ent_pi - ent_usage)
    return loss + reg

def main(argv=None):
    args = get_args(argv)
    set_seed(args.seed)
    _prevent_sleep()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # AMP: bf16 preferred (Ampere+), fp16 with GradScaler as fallback
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda') if (use_amp and amp_dtype == torch.float16) else None

    # All datasets are consumed as 3x32x32 (grayscale replicated + resized),
    # matching the prototype layout (first-layer in_channels=3) and the
    # convention used by train_continual_aml.py.
    data_root = args.data_root or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    if args.dataset == 'cifar10':
        tf_clean = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            tf_clean.transforms[0],
            tf_clean.transforms[1],
        ])
        train_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_train)
        train_clean = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=tf_clean)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_clean)
    else:
        if args.dataset == 'mnist':
            mean, std, cls = (0.1307,) * 3, (0.3081,) * 3, torchvision.datasets.MNIST
        else:
            mean, std, cls = (0.2860,) * 3, (0.3530,) * 3, torchvision.datasets.FashionMNIST
        tf_clean = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        aug_ops = [transforms.Grayscale(num_output_channels=3),
                   transforms.Resize((32, 32)),
                   transforms.RandomCrop(32, padding=2)]
        if args.dataset == 'fashion':
            aug_ops.append(transforms.RandomHorizontalFlip())
        tf_train = transforms.Compose(aug_ops + [transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_full = cls(root=data_root, train=True, download=True, transform=tf_train)
        train_clean = cls(root=data_root, train=True, download=False, transform=tf_clean)
        testset = cls(root=data_root, train=False, download=True, transform=tf_clean)

    # HELD-OUT VALIDATION SPLIT (never used for weight updates) -- model
    # selection must not read the test set, otherwise every early-stopped
    # "best acc" number is optimistically biased.
    n_val = min(5000, len(train_full) // 10)
    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(args.seed))
    tr_idx, va_idx = perm[:-n_val].tolist(), perm[-n_val:].tolist()
    trainset = torch.utils.data.Subset(train_full, tr_idx)
    valset = torch.utils.data.Subset(train_clean, va_idx)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        drop_last=True, prefetch_factor=4 if args.workers > 0 else None)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=max(args.batch_size, 256), shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=max(args.batch_size, 256), shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    model = ConvAML_CIFAR_TrueDynamic(K=args.K, max_dim=args.max_dim, max_layers=args.max_layers).to(device)

    # Architecture parameters get their own optimizer fed by a VALIDATION
    # split of each batch (first-order DARTS bi-level approximation).
    # Rationale: the train-loss gradient systematically penalizes gated
    # branches early (untrained deep features look like noise), which
    # collapsed all depth gates within one epoch. The generalization
    # signal rewards capacity and keeps depth selection meaningful.
    arch_names = ("structure_prototypes", "block_mode_prototypes")
    arch_params = [p for n, p in model.named_parameters() if any(k in n for k in arch_names)]
    other_params = [p for n, p in model.named_parameters() if not any(k in n for k in arch_names)]
    optimizer = optim.AdamW(other_params, lr=args.lr, weight_decay=1e-4)
    optimizer_arch = optim.AdamW(arch_params, lr=args.lr * 0.2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler_arch = optim.lr_scheduler.CosineAnnealingLR(optimizer_arch, T_max=args.epochs)

    print(f"========== TRAINING {args.max_layers}-LAYER TRUE AML DYNAMIC on {args.dataset} ==========")
    meta_loss_fn = MetaLoss(
        lambda_eff=0.0,
        lambda_p=0.0,
        p_allowance=args.p_allowance
    ).to(device)

    best_val = 0.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        t0 = time.time()

        # WARMUP + EFFICIENCY/REGULARIZATION SCHEDULE
        if epoch < 5:
            current_lambda_eff = 0.0
            tau = 1.0
            w_gate = 0.0
            w_bm = 0.0
            w_dim = 0.0
            w_pi = 0.0
            # Dense supernet warmup: freeze architecture heads so gates stay
            # open (init logits +2 -> sigmoid ~0.88) while weights learn.
            for p in arch_params:
                p.requires_grad_(False)
        else:
            if epoch == 5:
                for p in arch_params:
                    p.requires_grad_(True)
                print("  [arch] unfroze structure_delta / block_mode_head")
            if epoch < 10:
                current_lambda_eff = 0.0
                tau = 0.8
                w_gate = 0.02
                w_bm = 0.02
                w_dim = 0.01
                # pi sharpening tested: on homogeneous CIFAR-10 the task
                # gradient fights it and it cost accuracy (runs #6/#7).
                # Kept available but off by default.
                w_pi = 0.0
            elif epoch < 15:
                current_lambda_eff = 0.01
                tau = 0.5
                w_gate = 0.05
                w_bm = 0.05
                w_dim = 0.05
                w_pi = 0.0
            else:
                current_lambda_eff = 0.05
                tau = 0.2
                w_gate = 0.1
                w_bm = 0.1
                w_dim = 0.1
                w_pi = 0.005

        meta_loss_fn.lambda_eff = current_lambda_eff

        for step, (inputs, targets) in enumerate(trainloader):
            if args.steps and step >= args.steps:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            optimizer_arch.zero_grad(set_to_none=True)

            # Second half of the batch is held out as the validation signal
            # for the architecture step below. The weight step sees ONLY the
            # first half: reusing the full batch and then "validating" on a
            # subset of it double-counts the same samples and contaminates
            # the DARTS bi-level approximation.
            n_v = inputs.size(0) // 2
            x_w, y_w = inputs[:n_v], targets[:n_v]

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                outputs, struct, pi, diag = model(x_w, tau=tau)
                gen_model = GeneratedModel(structure=struct, weights={}, pi_logits=pi)

                loss_dict = meta_loss_fn(
                    logits_train=outputs.float(), targets_train=y_w,
                    logits_val=None, targets_val=None,
                    model=gen_model, epoch_progress=epoch/args.epochs
                )

                loss = structure_reg(loss_dict["loss"], diag, pi, model,
                                     w_gate, w_bm, w_dim, w_pi)

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss at epoch {epoch+1}, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                optimizer_arch.zero_grad(set_to_none=True)
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(other_params, 5.0)
                scaler.step(optimizer)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(other_params, 5.0)
                optimizer.step()

            # ---- Architecture step: validation signal + the SAME structure
            # regularizers as the main loop. Without the reg here the arch
            # optimizer only sees cross-entropy (w_gate/w_bm/w_dim/w_pi from
            # the main loss are zeroed below), so the prune-phase schedule
            # never reaches the parameters it governs -> mode degeneracy and
            # width/routing collapse. Feeding them through lets dim_floor
            # cap width decay, bm entropy keep Plain/ResNet both alive, and
            # w_pi shape prototype routing.
            if epoch >= 5:
                optimizer_arch.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    outputs_v, _sv, pi_v, diag_v = model(inputs[n_v:], tau=tau)
                    loss_arch = F.cross_entropy(outputs_v.float(), targets[n_v:], label_smoothing=0.1)
                    if args.arch_reg:
                        loss_arch = structure_reg(loss_arch, diag_v, pi_v, model,
                                                  w_gate, w_bm, w_dim, w_pi)

                if torch.isfinite(loss_arch):
                    if scaler is not None:
                        scaler.scale(loss_arch).backward()
                        scaler.unscale_(optimizer_arch)
                        torch.nn.utils.clip_grad_norm_(arch_params, 5.0)
                        scaler.step(optimizer_arch)
                    else:
                        loss_arch.backward()
                        torch.nn.utils.clip_grad_norm_(arch_params, 5.0)
                        optimizer_arch.step()
                        
            if scaler is not None:
                scaler.update()

            total_loss += loss_dict["task_loss"].item() * x_w.size(0)


        scheduler.step()
        if epoch >= 5:
            if getattr(args, 'freeze_arch', False) == False:
                scheduler_arch.step()
        train_loss = total_loss / max(x_w.size(0) * (step + 1), 1)

        val_acc, vm = eval_structure(model, valloader, len(valset), use_amp, amp_dtype,
                                     max_steps=args.steps)
        test_acc, _ = eval_structure(model, testloader, len(testset), use_amp, amp_dtype,
                                     diag=False, max_steps=args.steps)

        # Model selection on the VALIDATION split only (see split note above).
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), args.out)
            print(f"  [*] Saved new best model to {args.out} (Val Acc: {val_acc*100:.2f}%)")

        print(f"Epoch {epoch+1:2d}/{args.epochs} | {time.time()-t0:5.1f}s | Task Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | Avg Used Params: {vm['eff']/1e6:.2f}M | pi_H: {vm['pi_H']:.3f}/{math.log(args.K):.3f} | resMode: {vm['res_frac']*100:4.1f}% | gate_std: {vm['gate_std']:.4f} | Tau: {tau:.1f} | archReg: {int(bool(args.arch_reg))}")
        print(f"  -> Plain/ResNet% per layer: {vm['bm_str']}")

    # ---- Report the final TEST accuracy of the best-validation model ----
    if best_state is not None:
        model.load_state_dict(best_state)
    best_test_acc, bvm = eval_structure(model, testloader, len(testset), use_amp, amp_dtype,
                                        max_steps=args.steps)
    print(f"\n========== FINAL: Best Val: {best_val*100:.2f}% | Test acc of best-val model: {best_test_acc*100:.2f}% | "
          f"Avg Used Params: {bvm['eff']/1e6:.2f}M | resMode: {bvm['res_frac']*100:.1f}% | gate_std: {bvm['gate_std']:.4f} ==========")
    print(f"  -> Plain/ResNet% per layer: {bvm['bm_str']}")

if __name__ == '__main__':
    main()
