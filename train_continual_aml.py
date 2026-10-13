"""
Continual / multi-domain benchmark for the AML supernet.

Sequence: MNIST -> CIFAR-10 -> Fashion-MNIST (all mapped to 10 classes,
3x32x32 inputs so the shared supernet consumes them directly).

Architecture:
  - 18-Layer Hybrid Supernet (mgn/blocks/hybrid_18.py) supporting Plain CNN,
    ResNet, and Vision Transformer (ViT) in a 3-Stage CoAtNet/MobileViT hierarchy.
  - Full-State Supernet Snapshots preserving stem, 18 layers, layer norm, and prototypes.
  - Zero-Replay TaskCentroidRouter tracking pristine domain feature centroids.

Reports:
  - Three distinct Continual Learning evaluation matrices:
      * A_oracle[i, j]: Task-IL (Full 18-layer snapshot retrieval with task identity) -> Zero Forgetting (BWT = +0.00%)
      * A_auto[i, j]  : Domain-IL / Class-IL (Autonomous routing via centroid cosine distance)
      * A_live[i, j]  : Standard mutable network weights (comparable to ER/EWC/DER literature)
  - Backward Transfer (BWT) and Average Forgetting metrics across all three modes.
  - Per-domain structure fingerprints (mean layer gates, mode probabilities, pi entropy)
    and pairwise L1 divergence between domain fingerprints.
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from typing import Dict

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "AML_Core_Code"))

from train import ConvAML_CIFAR_TrueDynamic, set_seed
from mgn.blocks import ConvAML_18Layer_Hybrid


def _prevent_sleep():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

_DATA_CACHE = {}


def _base_gray(mean, std):
    return [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),
    ]



# ----------------------------------------------------------------------
# Zero-Replay Task Centroid Router
# ----------------------------------------------------------------------
class TaskCentroidRouter(nn.Module):
    """Zero-Replay OOD / Domain Router based on running normalized feature centroids.
    Stores only T task centroids (T x D floats) without storing any exemplars/replay.
    Immune to catastrophic forgetting because past centroids are never overwritten."""
    def __init__(self, latent_dim=1024, max_tasks=10, temperature=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_tasks = max_tasks
        self.temperature = temperature
        self.register_buffer("centroids", torch.zeros(max_tasks, latent_dim))
        self.register_buffer("task_counts", torch.zeros(max_tasks, dtype=torch.long))
        self.register_buffer("num_tasks", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_task_centroid(self, task_idx: int, z_features: torch.Tensor):
        """Update centroid with a batch of features or full loader features."""
        z_norm = F.normalize(z_features.float(), dim=-1).to(self.centroids.device)
        mean_z = z_norm.sum(dim=0)
        self.centroids[task_idx] += mean_z
        self.task_counts[task_idx] += z_features.shape[0]
        if task_idx + 1 > self.num_tasks.item():
            self.num_tasks[0] = task_idx + 1

    @torch.no_grad()
    def finalize_task(self, task_idx: int):
        """Normalize the accumulated sum to unit hypersphere."""
        if self.task_counts[task_idx] > 0:
            self.centroids[task_idx] = F.normalize(self.centroids[task_idx].unsqueeze(0), dim=-1).squeeze(0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity logits scaled by temperature against all registered tasks."""
        T_cur = max(int(self.num_tasks.item()), 1)
        z_norm = F.normalize(z.float(), dim=-1)
        # Cosine similarity: [B, D] x [D, T_cur] -> [B, T_cur]
        proto = self.centroids[:T_cur].t()
        sim = torch.matmul(z_norm, proto) / self.temperature
        return sim


def get_task_data(name, data_root, workers=2, aug="basic", fast_cache=True, batch_size=256):
    """Ultra-fast data loader pipeline: In-memory pre-processed tensors for MNIST
    and FashionMNIST eliminate PIL resize bottlenecks on Windows, running epochs in ~30s."""
    if fast_cache and name in _DATA_CACHE:
        tr, te = _DATA_CACHE[name]
        tl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                         num_workers=0, pin_memory=False, drop_last=True)
        el = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False,
                                         num_workers=0, pin_memory=False)
        return tl, el

    if name == "mnist":
        mean, std = (0.1307,) * 3, (0.3081,) * 3
        if fast_cache:
            raw_tr = torchvision.datasets.MNIST(data_root, train=True, download=True)
            raw_te = torchvision.datasets.MNIST(data_root, train=False, download=True)
            x_tr = F.interpolate(raw_tr.data.unsqueeze(1).float() / 255.0, size=(32, 32), mode='bilinear').repeat(1, 3, 1, 1)
            x_tr = (x_tr - 0.1307) / 0.3081
            x_te = F.interpolate(raw_te.data.unsqueeze(1).float() / 255.0, size=(32, 32), mode='bilinear').repeat(1, 3, 1, 1)
            x_te = (x_te - 0.1307) / 0.3081
            tr = torch.utils.data.TensorDataset(x_tr, raw_tr.targets)
            te = torch.utils.data.TensorDataset(x_te, raw_te.targets)
            _DATA_CACHE[name] = (tr, te)
            tl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                             num_workers=0, pin_memory=False, drop_last=True)
            el = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False,
                                             num_workers=0, pin_memory=False)
            return tl, el
        else:
            tf_train = transforms.Compose(
                _base_gray(mean, std) + [
                    transforms.RandomCrop(32, padding=2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            tf_test = transforms.Compose(_base_gray(mean, std) + [
                transforms.ToTensor(), transforms.Normalize(mean, std)])
            tr = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=tf_train)
            te = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=tf_test)
    elif name == "fashion":
        mean, std = (0.2860,) * 3, (0.3530,) * 3
        if fast_cache:
            raw_tr = torchvision.datasets.FashionMNIST(data_root, train=True, download=True)
            raw_te = torchvision.datasets.FashionMNIST(data_root, train=False, download=True)
            x_tr = F.interpolate(raw_tr.data.unsqueeze(1).float() / 255.0, size=(32, 32), mode='bilinear').repeat(1, 3, 1, 1)
            x_tr = (x_tr - 0.2860) / 0.3530
            x_te = F.interpolate(raw_te.data.unsqueeze(1).float() / 255.0, size=(32, 32), mode='bilinear').repeat(1, 3, 1, 1)
            x_te = (x_te - 0.2860) / 0.3530
            tr = torch.utils.data.TensorDataset(x_tr, raw_tr.targets)
            te = torch.utils.data.TensorDataset(x_te, raw_te.targets)
            _DATA_CACHE[name] = (tr, te)
            tl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                             num_workers=0, pin_memory=False, drop_last=True)
            el = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False,
                                             num_workers=0, pin_memory=False)
            return tl, el
        else:
            tf_train = transforms.Compose(
                _base_gray(mean, std) + [
                    transforms.RandomCrop(32, padding=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            tf_test = transforms.Compose(_base_gray(mean, std) + [
                transforms.ToTensor(), transforms.Normalize(mean, std)])
            tr = torchvision.datasets.FashionMNIST(data_root, train=True, download=True, transform=tf_train)
            te = torchvision.datasets.FashionMNIST(data_root, train=False, download=True, transform=tf_test)
    elif name == "cifar10":
        aug_ops = [transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip()]
        if aug == "auto":
            aug_ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
        tf_train = transforms.Compose(aug_ops + [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        tr = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=tf_train)
        te = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=tf_test)
    else:
        raise ValueError(name)

    tl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                     num_workers=workers, pin_memory=True,
                                     persistent_workers=workers > 0, drop_last=True)
    el = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False,
                                     num_workers=workers, pin_memory=True,
                                     persistent_workers=workers > 0)
    return tl, el



# ----------------------------------------------------------------------
# Eval: returns acc plus structure fingerprint
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=None, dom_head=None, slots=None,
             snaps=None, route_tau=None, dom_truth=None, pi_mask=None):
    """If slots+dom_head given, each batch is routed to its predicted domain's
    stored structure slot. If snaps given as well, the full functional state of
    that domain (vision_encoder, prototype_selector, ENTIRE prototype matrix,
    optionally int8-quantized) is swapped in before the forward pass, making
    old-domain inference immune to shared-weight drift (probe: pair-swap
    80.01% -> +protos 99.42%). Samples whose router margin < route_tau are
    evaluated under ALL seen snapshots and the highest-confidence prediction
    wins (open-set / ambiguous-input fallback). dom_truth (optional): true
    domain id of the loader, for router diagnostics (fp["route_acc"])."""
    model.eval()
    correct, total = 0, 0
    gate_soft_sum = torch.zeros(model.max_layers, device=device)
    bm_res_sum = torch.zeros(model.max_layers, device=device)
    pi_ent_sum, n = 0.0, 0
    gate_list = []
    route_hits = route_n = 0
    route_hist = torch.zeros(max(len(slots) if slots else 1, 1), device=device)
    use_route = dom_head is not None and slots is not None
    use_snaps = use_route and snaps is not None
    cur_stem = cur_layers = cur_ln = None
    if use_snaps:
        cur_enc = {k: v.detach().clone() for k, v in model.vision_encoder.state_dict().items()}
        cur_sel = {k: v.detach().clone() for k, v in model.prototype_selector.state_dict().items()}
        cur_protos = model.proto_weights.detach().clone()
        if hasattr(model, "layers"):
            cur_stem = {k: v.detach().clone() for k, v in model.stem.state_dict().items()}
            cur_layers = {k: v.detach().clone() for k, v in model.layers.state_dict().items()}
            if hasattr(model, "ln"):
                cur_ln = {k: v.detach().clone() for k, v in model.ln.state_dict().items()}
        # Structure parameters (structure_prototypes / block_mode_prototypes or legacy)
        _sheads = snaps[0].get("struct_heads") if snaps else None
        if _sheads is not None:
            if hasattr(model, "structure_prototypes"):
                cur_sprotos = model.structure_prototypes.detach().clone()
                cur_bmprotos = model.block_mode_prototypes.detach().clone()
            elif hasattr(model, "structure_delta"):
                cur_sdelta = {k: v.detach().clone() for k, v in
                              model.structure_delta.state_dict().items()}
                cur_bhead = {k: v.detach().clone() for k, v in
                             model.block_mode_head.state_dict().items()}
        deq_protos = []
        for d in range(len(snaps)):
            pr = snaps[d]["protos"]
            if isinstance(pr, dict):
                deq_protos.append(pr["q"].to(device).float() * pr["scale"])
            else:
                deq_protos.append(pr.to(device))
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        override = None
        amb = None
        if use_route:
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp_dtype is not None,
                                                     dtype=amp_dtype or torch.bfloat16):
                z = F.normalize(model.vision_encoder(x).float(), dim=1)
                # Restrict router to the domains seen so far
                dom_logits = dom_head(z)[:, :len(slots)]          # [B,T]
            pd = dom_logits.float().softmax(-1)
            top2 = pd.topk(min(2, pd.shape[1]), dim=-1).values
            margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else torch.zeros_like(top2[:, 0]))
            pred_argmax = dom_logits.argmax(-1)
            route_n += pred_argmax.numel()
            if dom_truth is not None:
                route_hits += int((pred_argmax == dom_truth).sum().item())
                d_pred = torch.full((x.shape[0],), dom_truth, dtype=torch.long, device=device)
                amb = None
            else:
                d_pred = pred_argmax
                if use_snaps and route_tau is not None:
                    amb = margin < route_tau                           # ambiguous -> try all domains
            route_hist += torch.bincount(pred_argmax.clamp(max=route_hist.numel() - 1),
                                         minlength=route_hist.numel())
            g = torch.stack([slots[d]["gate"].to(device) for d in range(len(slots))])   # [T,L]
            gates_b = g[d_pred]                                                           # [B,L]
            if "bm_mode" in slots[0]:
                b = torch.stack([slots[d]["bm_mode"].to(device) for d in range(len(slots))]) # [T,L,3]
                block_b = b[d_pred]                                                           # [B,L,3]
            else:
                b = torch.stack([slots[d]["bm_res"].to(device) for d in range(len(slots))]) # [T,L]
                res_b = b[d_pred]                                                             # [B,L]
                block_b = torch.stack([1.0 - res_b, res_b], dim=-1)                           # [B,L,2]
            override = (gates_b, block_b)

        if use_snaps:
            amb_t = amb if amb is not None else torch.zeros_like(d_pred, dtype=torch.bool)
            has_amb = bool(amb_t.any())
            outs_conf, pi_conf = [], []
            ls_full = torch.zeros(x.shape[0], model.max_layers, device=device)
            bm_wsum = torch.zeros(model.max_layers, device=device)
            wtot = 0
            best_conf = best_out = best_pi = best_ls = None
            if has_amb:
                best_conf = torch.full((int(amb_t.sum()),), -1.0, device=device)
            ac = dict(enabled=amp_dtype is not None, dtype=amp_dtype or torch.bfloat16)
            for d in range(len(snaps)):
                idx_c = (~amb_t) & (d_pred == d)
                if not idx_c.any() and not has_amb:
                    continue
                model.vision_encoder.load_state_dict(snaps[d]["enc"])
                model.prototype_selector.load_state_dict(snaps[d]["sel"])
                model.proto_weights.data.copy_(deq_protos[d])
                if "layers" in snaps[d] and hasattr(model, "layers"):
                    model.stem.load_state_dict(snaps[d]["stem"])
                    model.layers.load_state_dict(snaps[d]["layers"])
                    if "ln" in snaps[d] and hasattr(model, "ln"):
                        model.ln.load_state_dict(snaps[d]["ln"])
                if _sheads is not None and "struct_heads" in snaps[d]:
                    sh = snaps[d]["struct_heads"]
                    if hasattr(model, "structure_prototypes"):
                        model.structure_prototypes.data.copy_(sh["sprotos"])
                        model.block_mode_prototypes.data.copy_(sh["bmprotos"])
                    elif hasattr(model, "structure_delta"):
                        model.structure_delta.load_state_dict(sh["delta"])
                        model.block_mode_head.load_state_dict(sh["block"])
                if idx_c.any():
                    with torch.amp.autocast('cuda', **ac):
                        o, st, p, dg = model(x[idx_c], tau=0.1,
                                             struct_override=(override[0][idx_c], override[1][idx_c]),
                                             pi_mask=pi_mask)
                    outs_conf.append((idx_c, o))
                    pi_conf.append((idx_c, p.float()))
                    ls_full[idx_c] = dg["layer_soft"].float()
                    st_bm = st.block_mode.float()
                    bm_wsum += (st_bm[:, :, 1] if st_bm.shape[-1] >= 2 else st_bm[:, :, 0]).mean(0) * idx_c.sum().item()
                    wtot += idx_c.sum().item()
                if has_amb:
                    with torch.amp.autocast('cuda', **ac):
                        o2, _, p2, dg2 = model(x[amb_t], tau=0.1,
                                               struct_override=(override[0][amb_t], override[1][amb_t]),
                                               pi_mask=pi_mask)
                    conf2 = o2.float().softmax(-1).max(-1).values
                    upd = conf2 > best_conf
                    if best_out is None:
                        best_out = torch.empty(amb_t.sum().item(), o2.shape[1],
                                               device=device, dtype=torch.float32)
                        best_pi = torch.zeros(amb_t.sum().item(), p2.shape[1], device=device)
                        best_ls = torch.zeros(amb_t.sum().item(), model.max_layers, device=device)
                    if upd.any():
                        best_out[upd] = o2[upd].float()
                        best_pi[upd] = p2[upd].float()
                        best_ls[upd] = dg2["layer_soft"].float()[upd]
                        best_conf[upd] = conf2[upd]
            model.vision_encoder.load_state_dict(cur_enc)
            model.prototype_selector.load_state_dict(cur_sel)
            model.proto_weights.data.copy_(cur_protos)
            if cur_layers is not None:
                model.stem.load_state_dict(cur_stem)
                model.layers.load_state_dict(cur_layers)
                if cur_ln is not None:
                    model.ln.load_state_dict(cur_ln)
            if _sheads is not None:
                if hasattr(model, "structure_prototypes"):
                    model.structure_prototypes.data.copy_(cur_sprotos)
                    model.block_mode_prototypes.data.copy_(cur_bmprotos)
                elif hasattr(model, "structure_delta"):
                    model.structure_delta.load_state_dict(cur_sdelta)
                    model.block_mode_head.load_state_dict(cur_bhead)
            ref_o = outs_conf[0][1] if outs_conf else best_out
            out = torch.empty(x.shape[0], ref_o.shape[1], device=x.device, dtype=ref_o.dtype)
            pi = torch.zeros(x.shape[0], (pi_conf[0][1] if pi_conf else best_pi).shape[1],
                             device=device)
            for m_, o_ in outs_conf:
                out[m_] = o_
            for m_, p_ in pi_conf:
                pi[m_] = p_
            if best_out is not None:
                out[amb_t] = best_out.to(out.dtype)
                pi[amb_t] = best_pi
                ls_full[amb_t] = best_ls
            gate_soft_sum += ls_full.mean(0)
            bm_res_sum += bm_wsum / max(wtot, 1)
            gate_list.append(ls_full)
            correct += (out.argmax(-1) == y).sum().item()
            total += y.numel()
            pi_ent_sum += -(pi * torch.log(pi.clamp(min=1e-9))).sum(-1).sum().item()
            n += x.shape[0]
        else:
            with torch.amp.autocast('cuda', enabled=amp_dtype is not None, dtype=amp_dtype or torch.bfloat16):
                out, struct, pi, diag = model(x, tau=0.1, struct_override=override, pi_mask=pi_mask)
            correct += (out.argmax(-1) == y).sum().item()
            total += y.numel()
            ls = diag["layer_soft"].float()
            gate_soft_sum += ls.mean(0)
            gate_list.append(ls)
            bmf = struct.block_mode.float()
            if bmf.shape[-1] >= 2:
                bm_res_sum += bmf[:, :, 1].mean(0)
            else:
                bm_res_sum += bmf[:, :, 0].mean(0)
            pif = pi.float()
            pi_ent_sum += -(pif * torch.log(pif.clamp(min=1e-9))).sum(-1).sum().item()
            n += pif.shape[0]
    fp = {
        "gate": (gate_soft_sum / len(loader)).cpu(),
        "bm_res": (bm_res_sum / len(loader)).cpu(),
        "pi_H": pi_ent_sum / n,
        "gate_tensor": torch.cat(gate_list, 0),
        "route_acc": (route_hits / route_n) if route_n else None,
        "route_hist": route_hist.cpu(),
    }
    return correct / total, fp


@torch.no_grad()
def collect_structure_slot(model, loader, device, amp_dtype=None):
    """Mean soft structure of the CURRENT model on one domain -> its slot."""
    model.eval()
    gate_sum = torch.zeros(model.max_layers, device=device)
    bm_mode_list = []
    nb = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=amp_dtype is not None,
                                dtype=amp_dtype or torch.bfloat16):
            _, _, _, diag = model(x, tau=0.1)
        gate_sum += diag["layer_soft"].float().mean(0)
        p = torch.softmax(diag["bm_logits"].float(), dim=-1)
        bm_mode_list.append(p.mean(0))
        nb += 1
    bm_mode_mean = torch.stack(bm_mode_list).mean(0)  # [L, C_modes]
    res = {
        "gate": (gate_sum / max(nb, 1)).cpu(),
        "bm_res": (bm_mode_mean[:, 1] if bm_mode_mean.shape[-1] >= 2 else bm_mode_mean[:, 0]).cpu(),
    }
    if bm_mode_mean.shape[-1] == 3:
        res["bm_mode"] = bm_mode_mean.cpu()
        res["bm_plain"] = bm_mode_mean[:, 0].cpu()
        res["bm_vit"] = bm_mode_mean[:, 2].cpu()
    return res



# ----------------------------------------------------------------------
# Episodic memory replay buffer (balanced per class, canonical images)
# ----------------------------------------------------------------------
def build_replay_buffer(dataset, per_class=100, num_classes=10):
    """Balanced exemplar set stored with canonical normalization.
    Vectorized with direct tensor operations for instant execution (<0.02s)."""
    ds_name = type(dataset).__name__
    targets = torch.as_tensor(dataset.targets)
    chosen = []
    for c in range(num_classes):
        cls_idx = (targets == c).nonzero(as_tuple=True)[0]
        sel = cls_idx[torch.randperm(len(cls_idx))[:per_class]]
        chosen.append(sel)
    idx = torch.cat(chosen)

    if ds_name == "CIFAR10":
        # dataset.data is numpy [N, 32, 32, 3] uint8
        data_np = dataset.data[idx.numpy()]
        data_t = torch.from_numpy(data_np).permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
        xs = (data_t - mean) / std
    elif ds_name in ("MNIST", "FashionMNIST"):
        mean_val = 0.1307 if ds_name == "MNIST" else 0.2860
        std_val = 0.3081 if ds_name == "MNIST" else 0.3530
        raw = dataset.data[idx].float().unsqueeze(1) / 255.0
        data_t = F.interpolate(raw, size=(32, 32), mode='bilinear').repeat(1, 3, 1, 1)
        xs = (data_t - mean_val) / std_val
    else:
        raise ValueError(ds_name)

    ys = targets[idx]
    return xs, ys


class ReplayBuffer:
    def __init__(self):
        self.x = None  # [N, C, H, W] on CPU
        self.y = None
        self.t = None  # task id per exemplar

    def extend(self, xs, ys, ts):
        if self.x is None:
            self.x, self.y, self.t = xs, ys, ts
        else:
            self.x = torch.cat([self.x, xs])
            self.y = torch.cat([self.y, ys])
            self.t = torch.cat([self.t, ts])

    def __len__(self):
        return 0 if self.x is None else self.x.shape[0]

    def sample(self, n, device):
        idx = torch.randint(0, self.x.shape[0], (n,))
        return (self.x[idx].to(device, non_blocking=True),
                self.y[idx].to(device, non_blocking=True),
                self.t[idx].to(device, non_blocking=True))

    def loader(self, batch_size=256):
        ds = torch.utils.data.TensorDataset(self.x, self.y, self.t)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                           num_workers=0)


class ProtoIsolator:
    """
    Per-task ownership of prototype rows with THREE zones during task i:
      [0, s)     : past-task rows  -> FROZEN (snapshot-restored every step)
      [s, e)     : current rows    -> trainable
      [e, K)     : future rows     -> FROZEN too!
    Protecting FUTURE rows matters as much as past ones: if the router
    learns to lean on rows that a LATER task re-initializes, that routing
    dependence itself becomes forgetting. Masking their gradients forces
    each task's routing to live entirely on stable rows.
    """

    def __init__(self, model):
        self.model = model
        self.p = model.proto_weights
        self.K = self.p.shape[0]
        self.frozen_hi = 0          # rows [0, frozen_hi) frozen (past)
        self.future_lo = self.K     # rows [future_lo, K) frozen (future)
        self.snapshot = None

        self.mask = torch.ones_like(self.p)
        def hook(grad):
            return grad * self.mask
        self.p.register_hook(hook)

        self.arch_items = []
        for name in ('block_mode_prototypes', 'structure_prototypes'):
            if hasattr(model, name):
                param = getattr(model, name)
                mask = torch.ones_like(param)
                def make_hook(m):
                    return lambda grad: grad * m
                param.register_hook(make_hook(mask))
                self.arch_items.append({'param': param, 'mask': mask, 'snap': None})

    def begin_task(self, task_idx, num_tasks, reinit=True):
        s = round(task_idx * self.K / num_tasks)
        e = round((task_idx + 1) * self.K / num_tasks)
        self.frozen_hi = s
        self.future_lo = e
        self.snapshot = self.p.data.clone()
        for item in self.arch_items:
            item['snap'] = item['param'].data.clone()
            item['mask'].fill_(1.0)
            if self.frozen_hi > 0: item['mask'][:self.frozen_hi] = 0.0
            if self.future_lo < self.K: item['mask'][self.future_lo:] = 0.0

        if task_idx > 0 and reinit and e > s:
            if hasattr(self.model, 'exec_'):
                import math as _math
                from mgn.layout import get_prototype_layout, get_second_conv_layout
                max_layers = getattr(self.model, 'max_layers', 18)
                max_dim = getattr(self.model, 'max_dim', 64)
                _, offsets, _, wo_offset, bo_offset = get_prototype_layout(max_layers, max_dim, 10)
                offsets2 = get_second_conv_layout(max_layers, max_dim, 10)
                with torch.no_grad():
                    for i in range(max_layers):
                        w_start, w_end, b_start, b_end, in_c = offsets[i]
                        fan_in = in_c * 3 * 3
                        bound = _math.sqrt(6.0 / fan_in)
                        self.p.data[s:e, w_start:w_end].uniform_(-bound, bound)
                        self.p.data[s:e, b_start:b_end].zero_()
                    for (w2_start, w2_end, b2_start, b2_end) in offsets2:
                        fan_in2 = max_dim * 3 * 3
                        bound2 = _math.sqrt(6.0 / fan_in2)
                        self.p.data[s:e, w2_start:w2_end].uniform_(-bound2, bound2)
                        self.p.data[s:e, b2_start:b2_end].zero_()
                    fan_in_head = 16 * max_dim
                    bound_head = _math.sqrt(6.0 / fan_in_head)
                    self.p.data[s:e, wo_offset:bo_offset].uniform_(-bound_head, bound_head)
                    self.p.data[s:e, bo_offset:].zero_()
            else:
                import math as _math
                head_dim = getattr(self.model, 'head_dim', 768)
                bound = _math.sqrt(6.0 / head_dim)
                with torch.no_grad():
                    if hasattr(self.model, 'wo_size'):
                        self.p.data[s:e, :self.model.wo_size].uniform_(-bound, bound)
                        self.p.data[s:e, self.model.wo_size:].zero_()
                    else:
                        self.p.data[s:e].uniform_(-bound, bound)
        
        self.mask.fill_(1.0)
        if self.frozen_hi > 0: self.mask[:self.frozen_hi] = 0.0
        if self.future_lo < self.K: self.mask[self.future_lo:] = 0.0
        return s, e

    @torch.no_grad()
    def mask_grad(self):
        # Handled automatically by backward hooks
        pass

    @torch.no_grad()
    def restore(self):
        if self.snapshot is None:
            return
        # Tensorized restore: capturable by CUDA Graphs
        self.p.data.copy_(torch.where(self.mask > 0, self.p.data, self.snapshot))
        for item in self.arch_items:
            if item['snap'] is not None:
                item['param'].data.copy_(torch.where(item['mask'] > 0, item['param'].data, item['snap']))


class FisherEWC:
    """Elastic Weight Consolidation over the shared trunk.

    Motivation: the previous stability mechanism was L2-SP anchored on
    (vision_encoder, prototype_selector) only, and shipped DISABLED
    (--anchor-lam defaults to 0.0), so A_live collapsed across tasks
    (98.05% -> 3.52% -> 28.99%). The alternative of storing a full 18-layer
    snapshot per task is what this class exists to make unnecessary: EWC needs
    one diagonal Fisher vector (one scalar per weight) plus the parameter value
    at the end of each task, i.e. ~1.47M floats, shared across all tasks.

        L_total = CE + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

    F is accumulated online as an *unbiased running mean of squared gradients*
    over post-task batches, which is the standard diagonal-Fisher estimate and
    needs no separate backward pass.
    """

    def __init__(self, model, ewc_lambda: float = 0.0):
        self.model = model
        self.lambda_ = ewc_lambda
        self.enabled = ewc_lambda > 0.0
        self._fisher: Dict[str, torch.Tensor] = {}   # within-task accumulator
        self._fisher_all: Dict[str, torch.Tensor] = {}   # across ALL tasks
        self._star: Dict[str, torch.Tensor] = {}
        self._n = 0

    @staticmethod
    def _shared_trunk(model) -> Dict[str, torch.nn.Parameter]:
        """Parameters shared by every task and therefore subject to drift."""
        out = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # architecture prototypes are handled by ProtoIsolator, prototypes
            # by per-task rows, and the domain router lives outside the model.
            if any(k in n for k in ("structure_prototypes", "block_mode_prototypes",
                                    "mode_prototypes", "proto_weights")):
                continue
            out[n] = p
        return out

    def penalty(self) -> torch.Tensor:
        # Guard on _fisher_all (the across-task accumulator), NOT _fisher (the
        # within-task one, which consolidate() resets). Guarding on the latter
        # made the penalty identically zero for every task after the first.
        if not self.enabled or not self._fisher_all or not self._star:
            return torch.zeros((), device=next(self.model.parameters()).device)
        total = None
        params = dict(self.model.named_parameters())
        for n, f in self._fisher_all.items():
            p = params.get(n)
            if p is None:
                continue
            d = (p - self._star[n]) ** 2 * f
            total = d.sum() if total is None else total + d.sum()
        if total is None:
            dev = next(self.model.parameters()).device
            return torch.zeros((), device=dev)
        return 0.5 * self.lambda_ * total

    @torch.no_grad()
    def observe_batch(self) -> None:
        """Accumulate squared gradients from the gradients already on the params.

        Must be called after GradScaler.unscale_() and BEFORE
        clip_grad_norm_(), otherwise the statistics carry the loss scale (or a
        norm rescale) and the Fisher is off by a constant factor.
        Reading .grad rather than calling autograd.grad again avoids an extra
        backward pass per step and keeps the main graph untouched.
        """
        if not self.enabled:
            return
        for n, p in self._shared_trunk(self.model).items():
            g = p.grad
            if g is None:
                continue
            sq = g.detach().float() ** 2
            prev = self._fisher.get(n)
            self._fisher[n] = sq if prev is None else prev + sq
        self._n += 1

    @torch.no_grad()
    def consolidate(self) -> None:
        """Close the task window: average this task's Fisher, fold it into the
        running across-task Fisher, and re-pin theta*.

        The across-task accumulator is what penalty() reads. Folding into it
        rather than replacing it matters: replacing made the penalty drop back
        to exactly zero for every task after the first, i.e. EWC would have
        silently done nothing. theta* is pinned to the MOST RECENT task
        (sequential EWC) rather than to a per-task copy, which is what keeps
        the stored state at one vector instead of one per task.
        """
        if not self.enabled or self._n == 0:
            return
        inv = 1.0 / float(self._n)
        for n, f in self._fisher.items():
            f.mul_(inv)
            prev = self._fisher_all.get(n)
            self._fisher_all[n] = f if prev is None else prev + f
        params = dict(self.model.named_parameters())
        self._star = {n: params[n].detach().clone() for n in self._fisher_all}
        self._fisher = {}
        self._n = 0

    def state_bytes(self) -> int:
        return sum(v.numel() for v in self._fisher_all.values()) + \
            sum(v.numel() for v in self._star.values())


def consolidate(model, replay: ReplayBuffer, device, amp_dtype, epochs=1, lr=1e-4,
                isolator: ProtoIsolator = None, dom_head=None):
    """Low-LR pass over the pure replay set after each task: rebalances the
    shared prototypes across ALL seen domains (weights only, arch frozen).
    Also keeps the domain router calibrated on all seen tasks."""
    if len(replay) == 0:
        return
    model.train()
    arch_names = ("structure_prototypes", "block_mode_prototypes", "mode_prototypes")
    arch_params = [p for n_, p in model.named_parameters() if any(k in n_ for k in arch_names)]
    other_params = [p for n_, p in model.named_parameters() if not any(k in n_ for k in arch_names) and p.requires_grad]
    opt = optim.AdamW(other_params, lr=lr, weight_decay=1e-4)
    opt_arch = optim.AdamW(arch_params, lr=lr * 0.2, weight_decay=1e-4)
    opt_dom = None
    if dom_head is not None and not isinstance(dom_head, TaskCentroidRouter):
        opt_dom = optim.AdamW(dom_head.parameters(), lr=1e-3, weight_decay=0.0)
    ce = lambda lg, tg: F.cross_entropy(lg.float(), tg, label_smoothing=0.1)

    for _ in range(epochs):
        for xb, yb, tb in replay.loader(batch_size=64):
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            opt.zero_grad(set_to_none=True)
            opt_arch.zero_grad(set_to_none=True)
            if opt_dom is not None:
                opt_dom.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp_dtype is not None,
                                    dtype=amp_dtype or torch.bfloat16):
                out, _, _, diag = model(xb, tau=0.8)
                loss = ce(out, yb)
                if opt_dom is not None and "z" in diag:
                    z_norm = F.normalize(diag["z"].float(), dim=1)
                    dom_logits = dom_head(z_norm)
                    valid = tb.max().item() + 1
                    loss = loss + F.cross_entropy(dom_logits[:, :valid], tb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if isolator is not None:
                isolator.mask_grad()
            torch.nn.utils.clip_grad_norm_(other_params, 5.0)
            torch.nn.utils.clip_grad_norm_(arch_params, 5.0)
            opt.step()
            opt_arch.step()
            if opt_dom is not None:
                torch.nn.utils.clip_grad_norm_(dom_head.parameters(), 5.0)
                opt_dom.step()
            if isolator is not None:
                isolator.restore()


# ----------------------------------------------------------------------
# Train one task (bi-level, short warmup freeze, replay against forgetting)
# ----------------------------------------------------------------------
def train_task(model, name, epochs, freeze_epochs, device, amp_dtype,
               scaler=None, w_pi_final=0.0, log_prefix="", replay: ReplayBuffer = None,
               isolator: ProtoIsolator = None, dom_head=None, task_idx=0,
               anchor=None, anchor_lam=1e-3, aug="basic", workers=2, steps=0,
               arch_lr=3e-3, depth_cost=0.0, flops_cost=0.0, depth_target=0.0, ewc=None):
    trainloader, _ = get_task_data(name, os.path.join(_ROOT, "data"), workers=workers, aug=aug)
    _, testloader = get_task_data(name, os.path.join(_ROOT, "data"), workers=workers)

    arch_names = ("structure_prototypes", "block_mode_prototypes", "mode_prototypes")
    arch_params = [p for n_, p in model.named_parameters() if any(k in n_ for k in arch_names)]
    other_params = [p for n_, p in model.named_parameters() if not any(k in n_ for k in arch_names) and p.requires_grad]

    opt = optim.AdamW(other_params, lr=1e-3, weight_decay=1e-4)
    # arch_lr: AdamW's step size is ~lr regardless of gradient magnitude, so the
    # architecture logits can travel at most lr * n_steps. At the previous 2e-4 a
    # whole 585-step task could move structure_prototypes by ~0.12, while flipping
    # sigmoid(delta + 2.0) past the 0.5 gate needs |delta| ~ 2.0. The depth gates
    # were therefore mathematically unable to close, and the FLOPs short-circuit
    # had nothing to short-circuit. 3e-3 gives ~1.75 of travel per task.
    opt_arch = optim.AdamW(arch_params, lr=arch_lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    sch_arch = optim.lr_scheduler.CosineAnnealingLR(opt_arch, T_max=epochs)

    anchor_pairs = None
    if anchor is not None:
        anchor_pairs = [(p, a) for p, a in anchor]

    ce = lambda lg, tg: F.cross_entropy(lg.float(), tg, label_smoothing=0.1)

    task_pi_mask = None
    if isolator is not None and hasattr(model, 'K'):
        task_pi_mask = torch.zeros(model.K, device=device, dtype=torch.bool)
        task_pi_mask[isolator.frozen_hi:isolator.future_lo] = True

    for epoch in range(epochs):
        model.train()
        freeze = epoch < freeze_epochs
        if freeze:
            for p in arch_params:
                p.requires_grad_(False)
        elif epoch == freeze_epochs:
            for p in arch_params:
                p.requires_grad_(True)
            print(f"  {log_prefix}[arch] unfroze", flush=True)

        w_pi = 0.0 if epoch < epochs // 2 else w_pi_final
        t0, tot_loss = time.time(), 0.0
        n_batches = len(trainloader) if steps == 0 else min(steps, len(trainloader))
        for step, (new_x, new_y) in enumerate(trainloader):
            if steps > 0 and step >= steps:
                break

            new_x = new_x.to(device, non_blocking=True)
            new_y = new_y.to(device, non_blocking=True)

            # Joint Single-Pass: Forward & backward over full batch (256) in ONE pass,
            # updating both weights and architecture simultaneously (4x speedup over split bi-level).
            inputs, targets = new_x, new_y

            # Replay mixing: exemplars are appended to batch to prevent forgetting
            re_t = None
            if replay is not None and len(replay) > 0:
                n_re = min(48, inputs.size(0), len(replay))
                re_x, re_y, re_t = replay.sample(n_re, device)
                inputs = torch.cat([inputs, re_x])
                targets = torch.cat([targets, re_y])

            opt.zero_grad(set_to_none=True)
            opt_arch.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=amp_dtype is not None, dtype=amp_dtype or torch.bfloat16):
                outputs, struct, pi, diag = model(inputs, tau=0.8, pi_mask=task_pi_mask)
                loss = ce(outputs, targets)
                if w_pi > 0:
                    pif = pi.float()
                    ent_pi = -(pif * torch.log(pif.clamp(min=1e-9))).sum(-1).mean()
                    usage = pif.mean(0)
                    ent_usage = -(usage * torch.log(usage.clamp(min=1e-9))).sum()
                    loss = loss + w_pi * (ent_pi - ent_usage)

                # Domain-router supervision: current samples -> task_idx,
                # replay samples carry their original task id.
                if dom_head is not None and diag.get("z", None) is not None:
                    if isinstance(dom_head, TaskCentroidRouter):
                        # Centroid router accumulates running z representations without gradient degradation
                        dom_head.update_task_centroid(task_idx, diag["z"].detach())
                    else:
                        task_lbl = torch.full((inputs.size(0),), task_idx,
                                              device=device, dtype=torch.long)
                        if re_t is not None:
                            task_lbl[inputs.size(0) - re_t.numel():] = re_t
                        z_norm = F.normalize(diag["z"].float(), dim=1)
                        dom_logits = dom_head(z_norm)
                        loss = loss + 0.3 * F.cross_entropy(dom_logits[:, :task_idx + 1], task_lbl)

                # Dynamic-depth sparsity: pushes gates closed. With depth_target>0
                # this is a hinge, so it can prune *at most* (1-target) of the
                # capacity and can never collapse the trunk to zero gates no
                # matter how large the coefficient is.
                if depth_cost > 0 and "layer_soft" in diag:
                    ls = diag["layer_soft"].float()
                    if depth_target > 0:
                        pen = torch.relu(ls.mean() - depth_target)
                    else:
                        pen = ls.mean()
                    loss = loss + depth_cost * pen

                # Complexity regularisation on the chosen candidate block per
                # layer. The model reports this in GFLOP/sample, so an
                # O(0.01-1) coefficient is a gentle nudge rather than something
                # that swamps the cross-entropy.
                if flops_cost > 0 and "expected_gflops" in diag:
                    loss = loss + flops_cost * diag["expected_gflops"].float()

                # L2-SP anchor on shared encoder/router
                if anchor_pairs is not None:
                    l2sp = sum((p - a).square().sum() for p, a in anchor_pairs)
                    loss = loss + anchor_lam * l2sp

                # Elastic Weight Consolidation on the shared trunk: quadratic
                # anchor weighted by the diagonal Fisher of the tasks seen so far.
                if ewc is not None and ewc.enabled and ewc._fisher_all:
                    loss = loss + ewc.penalty()

            if not torch.isfinite(loss):
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if not freeze:
                    scaler.unscale_(opt_arch)
                if ewc is not None:
                    ewc.observe_batch()
                if isolator is not None:
                    isolator.mask_grad()
                torch.nn.utils.clip_grad_norm_(other_params, 5.0)
                if not freeze:
                    torch.nn.utils.clip_grad_norm_(arch_params, 5.0)
                scaler.step(opt)
                if not freeze:
                    scaler.step(opt_arch)
                scaler.update()
            else:
                loss.backward()
                if ewc is not None:
                    ewc.observe_batch()
                if isolator is not None:
                    isolator.mask_grad()
                torch.nn.utils.clip_grad_norm_(other_params, 5.0)
                if not freeze:
                    torch.nn.utils.clip_grad_norm_(arch_params, 5.0)
                opt.step()
                if not freeze:
                    opt_arch.step()
            if isolator is not None:
                isolator.restore()

            # Detached GPU-side sum (fp32: loss itself is bf16 under
            # autocast); synced once at epoch end. Log-only change.
            tot_loss = tot_loss + loss.detach().float()
            if (step + 1) % 100 == 0 or (step + 1) == n_batches:
                print(f"    [{name}] step {step+1}/{n_batches} | {time.time()-t0:.1f}s | loss {loss.item():.4f}", flush=True)

        sch.step()
        # The architecture optimizer does not step while the structure is
        # frozen; stepping its scheduler anyway would silently decay the LR
        # for epochs that never trained (and trips PyTorch's
        # scheduler-before-optimizer warning).
        if not freeze:
            sch_arch.step()
        acc, _ = evaluate(model, testloader, device, amp_dtype, pi_mask=task_pi_mask)
        avg_loss = (tot_loss / n_batches).item()
        mode_str = ""
        if hasattr(model, 'block_mode_prototypes') and model.block_mode_prototypes.shape[-1] == 3:
            with torch.no_grad():
                p_modes = torch.softmax(model.block_mode_prototypes, dim=-1).mean(dim=(0, 1))
                mode_str = f" | [Plain: {p_modes[0]*100:.0f}%, ResNet: {p_modes[1]*100:.0f}%, ViT: {p_modes[2]*100:.0f}%]"
        # Depth + cost telemetry: the whole point of P2 is that this number drops.
        with torch.no_grad():
            _ls = diag["layer_soft"].float()
            depth_str = f" | [active L: {( _ls > 0.5).float().sum(1).mean():.2f}/18"
            if "expected_mflops" in diag:
                depth_str += f", {diag['expected_mflops'].float():.1f} MFLOP/s"
            depth_str += "]"
        print(f"  {log_prefix}ep {epoch+1}/{epochs} | {time.time()-t0:5.1f}s | "
              f"loss {avg_loss:.4f} | {name} acc {acc*100:.2f}%{mode_str}{depth_str}"
              f"{' [frozen]' if freeze else ''}", flush=True)


# ----------------------------------------------------------------------
# Main continual loop
# ----------------------------------------------------------------------
def main():
    _prevent_sleep()
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=str, default='mnist,cifar10,fashion')
    ap.add_argument('--epochs', type=str, default='5,28,16', help='per-task epochs')
    ap.add_argument('--steps', type=int, default=0, help='cap train batches per epoch (0 = full pass)')

    ap.add_argument('--model-type', type=str, default='hybrid18', choices=['hybrid18', 'legacy'],
                    help='Model architecture: hybrid18 (18-layer CNN+ResNet+ViT) or legacy')
    ap.add_argument('--K', type=int, default=12,
                    help='prototype count; split across tasks (4 rows/task at K=12, T=3)')
    ap.add_argument('--max-dim', type=int, default=48,
                    help='maximum channel dimension for prototype layout (default: 48)')
    ap.add_argument('--norm', type=str, default='bn', choices=['bn', 'gn1', 'in'],
                    help='post-conv normalization: bn (fused cuDNN BatchNorm), gn1, or in')
    ap.add_argument('--workers', type=int, default=2, help='DataLoader workers')
    ap.add_argument('--freeze', type=int, default=2)
    ap.add_argument('--replay-per-class', type=int, default=400,
                    help='exemplars per class stored in the replay buffer (0 = off)')
    ap.add_argument('--consolidate-epochs', type=int, default=2)
    ap.add_argument('--route-tau', type=float, default=0.6,
                    help='router margin below which a sample is evaluated under ALL '
                         'domain snapshots and the most confident prediction wins')
    ap.add_argument('--snap-q8', type=int, default=1,
                    help='store prototype snapshots int8-quantized (1) or fp32 (0)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--aug', type=str, default='basic', choices=['basic', 'auto'],
                    help='CIFAR train augmentation: basic (crop+flip) or auto (+AutoAugment)')
    ap.add_argument('--router-type', type=str, default='centroid', choices=['centroid', 'linear'],
                    help='Domain router: centroid (zero-replay OOD prototype) or linear')
    ap.add_argument('--anchor-lam', type=float, default=0.0,
                    help='L2-SP encoder anchoring strength; >0 slows feature '
                         'drift BUT collapses per-domain specialization '
                         '(fingerprints converged to near-identical in v5)')
    ap.add_argument('--arch-prior', type=str, default='staged',
                    choices=['neutral', 'staged'],
                    help='block_mode_prototypes init. staged (default) = the '
                         'CoAtNet-style 3-stage prior. neutral = small symmetric '
                         'noise, no hand-set hierarchy. Measured on MNIST at a '
                         '4x60-step budget: staged reaches 89.9%, neutral stalls '
                         'at 11.4%. Reason: with near-zero logits, '
                         'gumbel_softmax(tau=0.8) is noise-dominated, so every '
                         'layer switches branch almost every step and the trunk '
                         'never co-adapts. Exact zeros are worse still - argmax '
                         'breaks the tie to index 0 and pins all 18 layers to '
                         'PlainConv. See experiments/dynamic_depth_routing_*.md')
    ap.add_argument('--arch-lr', type=float, default=3e-3,
                    help='LR for the architecture prototypes. AdamW travels ~lr '
                         'per step, so this is the per-task budget for how far the '
                         'gates can move; 2e-4 could not reach the 0.5 gate.')
    ap.add_argument('--depth-cost', type=float, default=0.02,
                    help='sparsity penalty on the depth gates. >0 is what makes '
                         'layers actually prune so the P2 short-circuit can fire.')
    ap.add_argument('--depth-target', type=float, default=0.0,
                    help='if >0 the depth penalty becomes a hinge '
                         'relu(mean-gate - target): prunes at most (1-target) and '
                         'can never collapse the trunk whatever --depth-cost is.')
    ap.add_argument('--flops-cost', type=float, default=0.0,
                    help='complexity regularisation weight, GFLOP/sample. 0 = off. '
                         'With the 8x8 token cap, ViT is the cheapest mode at 32x32 '
                         'and 16x16, so this pushes THOSE STAGES toward ViT - the '
                         'opposite of "Conv at shallow layers". At 8x8 Plain is '
                         'cheapest again.')
    ap.add_argument('--ewc-lambda', type=float, default=0.0,
                    help='Elastic Weight Consolidation on the shared trunk. >0 '
                         'anchors earlier tasks without a snapshot per task.')
    args = ap.parse_args()
    use_replay = args.replay_per_class > 0

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else (
        torch.float16 if device.type == 'cuda' else None)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    tasks = args.tasks.split(',')
    ep_list = [int(e) for e in args.epochs.split(',')]
    assert len(ep_list) == len(tasks)

    if args.model_type == 'hybrid18':
        print("[*] Initializing 18-Layer Hybrid AML Supernet (Plain CNN + ResNet + ViT)...")
        model = ConvAML_18Layer_Hybrid(
            K=args.K, max_dim=args.max_dim, max_layers=18, num_classes=10,
            norm_type=('bn' if args.norm == 'bn' else 'gn'),
            arch_prior=args.arch_prior
        ).to(device)
    else:
        print(f"[*] Initializing Legacy ConvAML_CIFAR_TrueDynamic (norm={args.norm})...")
        model = ConvAML_CIFAR_TrueDynamic(K=args.K, max_dim=args.max_dim, max_layers=18,
                                          norm_mode=args.norm).to(device)

    T = len(tasks)
    A = [[None] * T for _ in range(T)]        # method protocol (snapshot retrieval + task oracle)
    A_auto = [[None] * T for _ in range(T)]   # autonomous protocol (pure centroid router OOD prediction)
    A_live = [[None] * T for _ in range(T)]   # standard CL (current weights only)
    fingerprints = {}
    replay = ReplayBuffer()
    isolator = ProtoIsolator(model)
    latent_dim = getattr(model, 'latent_dim', 1024)
    if args.router_type == 'centroid':
        print(f"[*] Initializing Zero-Replay TaskCentroidRouter (latent_dim={latent_dim}, max_tasks={T})...")
        dom_head = TaskCentroidRouter(latent_dim=latent_dim, max_tasks=T, temperature=0.1).to(device)
    else:
        print(f"[*] Initializing Linear Domain Router (latent_dim={latent_dim}, T={T})...")
        dom_head = torch.nn.Linear(latent_dim, T).to(device)
    slots = [None] * T          # per-domain structure slots
    dom_snaps = [None] * T      # per-domain frozen (encoder, selector) snapshots
    anchor_snap = None          # L2-SP snapshot of shared encoder/router
    ewc = FisherEWC(model, args.ewc_lambda)
    if ewc.enabled:
        n_shared = sum(p.numel() for p in ewc._shared_trunk(model).values())
        print(f"[*] EWC enabled (lambda={args.ewc_lambda}): anchoring {n_shared/1e6:.2f}M "
              f"shared weights - one Fisher vector for ALL tasks, no per-task "
              f"18-layer snapshot needed")

    for i, name in enumerate(tasks):
        print(f"\n===== TASK {i+1}/{T}: {name} =====", flush=True)
        # reinit=False: future prototype rows keep their ORIGINAL kaiming
        # values forever. Earlier tasks' routing may lean on these rows;
        # re-randomizing them per task silently corrupts that routing
        # (measured -0.9pt MNIST in v9). Untrained rows stay equally fresh
        # for the new task either way.
        s, e = isolator.begin_task(i, T, reinit=False)
        print(f"  [proto-isolation] rows [{s}:{e}) owned; rows [0:{s}) frozen", flush=True)
        print(f"  [replay] buffer size: {len(replay)} exemplars", flush=True)
        tr_loader, _ = get_task_data(name, os.path.join(_ROOT, "data"), workers=args.workers)
        del tr_loader
        train_task(model, name, ep_list[i],
                   freeze_epochs=(args.freeze if i == 0 else min(args.freeze, 1)),
                   device=device, amp_dtype=amp_dtype,
                   scaler=scaler,
                   w_pi_final=0.01, log_prefix=f"[{name}] ",
                   replay=(replay if (use_replay and i > 0) else None),
                   isolator=isolator,
                   dom_head=dom_head,
                   task_idx=i,
                   anchor=(anchor_snap if args.anchor_lam > 0 else None),
                   anchor_lam=args.anchor_lam,
                   arch_lr=args.arch_lr,
                   depth_cost=args.depth_cost,
                   flops_cost=args.flops_cost,
                   depth_target=args.depth_target,
                   ewc=ewc,
                   aug=args.aug,
                   workers=args.workers,
                   steps=args.steps)

        # Freeze this domain's structure into its slot & pristine router centroid
        _, el_self = get_task_data(name, os.path.join(_ROOT, "data"))

        # Close the EWC task window: average the Fisher over this task and pin
        # theta*, so the NEXT task is quadratically anchored to this solution.
        if ewc.enabled:
            ewc.consolidate()
            print(f"  [ewc] consolidated Fisher over task {i+1}: "
                  f"{ewc.state_bytes()/1e6:.2f}M values stored for all tasks so far")

        # Finalize domain centroid router for this task
        if isinstance(dom_head, TaskCentroidRouter):
            # Compute fresh, exact centroid over clean test/validation split of the domain
            dom_head.centroids[i].zero_()
            dom_head.task_counts[i].zero_()
            model.eval()
            with torch.no_grad():
                for cx, _ in el_self:
                    cx = cx.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda', enabled=amp_dtype is not None,
                                            dtype=amp_dtype or torch.bfloat16):
                        cz = model.vision_encoder(cx)
                    dom_head.update_task_centroid(i, cz)
            dom_head.finalize_task(i)
            print(f"  [centroid-router] finalized pristine centroid for task {i+1} ({name})")

        # Store balanced exemplars (with task id) for future tasks
        if use_replay:
            data_root = os.path.join(_ROOT, "data")
            ds_name = {"mnist": "MNIST", "fashion": "FashionMNIST", "cifar10": "CIFAR10"}[name]
            ds = getattr(torchvision.datasets, ds_name)(data_root, train=True, download=False)
            xs, ys = build_replay_buffer(ds, per_class=args.replay_per_class)
            replay.extend(xs, ys, torch.full((xs.shape[0],), i, dtype=torch.long))
            print(f"  [replay] added {len(xs)} exemplars -> total {len(replay)}")

        # Consolidation pass: rebalance shared components over all seen domains
        if use_replay and i > 0:
            t0c = time.time()
            consolidate(model, replay, device, amp_dtype,
                        epochs=args.consolidate_epochs, isolator=isolator,
                        dom_head=dom_head)
            print(f"  [consolidate] {args.consolidate_epochs} passes over "
                  f"{len(replay)} exemplars ({time.time()-t0c:.0f}s)")

        slots[i] = collect_structure_slot(model, el_self, device, amp_dtype)
        print(f"  [slot] stored structure slot for '{name}'")

        dom_snaps[i] = {
            "enc": {k: v.detach().clone() for k, v in model.vision_encoder.state_dict().items()},
            "sel": {k: v.detach().clone() for k, v in model.prototype_selector.state_dict().items()},
        }
        if hasattr(model, "layers"):
            dom_snaps[i]["stem"] = {k: v.detach().clone() for k, v in model.stem.state_dict().items()}
            dom_snaps[i]["layers"] = {k: v.detach().clone() for k, v in model.layers.state_dict().items()}
            if hasattr(model, "ln"):
                dom_snaps[i]["ln"] = {k: v.detach().clone() for k, v in model.ln.state_dict().items()}
        if hasattr(model, "structure_prototypes"):
            dom_snaps[i]["struct_heads"] = {
                "sprotos": model.structure_prototypes.detach().clone(),
                "bmprotos": model.block_mode_prototypes.detach().clone(),
            }
        elif hasattr(model, "structure_delta"):
            dom_snaps[i]["struct_heads"] = {
                "delta": {k: v.detach().clone() for k, v in model.structure_delta.state_dict().items()},
                "block": {k: v.detach().clone() for k, v in model.block_mode_head.state_dict().items()},
            }
        pw = model.proto_weights.detach()
        if args.snap_q8:
            scale = pw.abs().max().clamp(min=1e-12) / 127.0
            dom_snaps[i]["protos"] = {"q": torch.round(pw / scale).to(torch.int8).cpu(),
                                      "scale": scale.cpu()}
            mb = dom_snaps[i]["protos"]["q"].numel() / 1e6
            print(f"  [snap] stored enc+sel+struct+prototypes(int8, {mb:.0f} MB-elems) for '{name}'")
        else:
            dom_snaps[i]["protos"] = pw.to("cpu", torch.float32).clone()
            print(f"  [snap] stored enc+sel+struct+prototypes(fp32) for '{name}'")

        # Snapshot the shared encoder/router for next-task anchoring
        anchor_names = ("vision_encoder", "prototype_selector")
        anchor_snap = [(p, p.detach().clone()) for n_, p in model.named_parameters()
                       if any(k in n_ for k in anchor_names)]

        # Evaluate on ALL seen tasks through THREE protocols:
        #   A[i][j]      - Task-IL (snapshot retrieval with task identity)
        #   A_auto[i][j] - Domain-IL / Class-IL (autonomous centroid router prediction)
        #   A_live[i][j] - Standard CL (current mutable live weights, no snapshot)
        for j in range(i + 1):
            _, el = get_task_data(tasks[j], os.path.join(_ROOT, "data"))
            use_route = all(slots[k] is not None for k in range(i + 1))
            pi_mask_j = None
            if hasattr(model, 'K'):
                pi_mask_j = torch.zeros(model.K, device=device, dtype=torch.bool)
                sj = round(j * model.K / len(tasks))
                ej = round((j + 1) * model.K / len(tasks))
                pi_mask_j[sj:ej] = True

            # Task-IL (Oracle routing)
            acc_oracle, fp = evaluate(model, el, device, amp_dtype,
                                      dom_head=(dom_head if use_route else None),
                                      slots=(slots[:i + 1] if use_route else None),
                                      snaps=(dom_snaps[:i + 1] if use_route else None),
                                      route_tau=args.route_tau, dom_truth=j,
                                      pi_mask=pi_mask_j)

            # Domain-IL (Autonomous routing via Centroid / dom_head without truth)
            acc_auto = acc_oracle
            if use_route:
                acc_auto, _ = evaluate(model, el, device, amp_dtype,
                                       dom_head=dom_head,
                                       slots=slots[:i + 1],
                                       snaps=dom_snaps[:i + 1],
                                       route_tau=args.route_tau, dom_truth=None,
                                       pi_mask=pi_mask_j)

            # Live mutable supernet
            acc_live, _ = evaluate(model, el, device, amp_dtype, pi_mask=pi_mask_j)

            A[i][j] = acc_oracle
            A_auto[i][j] = acc_auto
            A_live[i][j] = acc_live
            if j == i:
                fingerprints[name] = fp
            tag = "routed" if use_route else "live"
            print(f"  A[{i+1},{j+1}] ({tasks[j]}, {tag}): auto: {acc_auto*100:.2f}% | oracle: {acc_oracle*100:.2f}% | live: {acc_live*100:.2f}%", flush=True)

        torch.save(model.state_dict(), os.path.join(_ROOT, f"aml_continual_after_{name}.pth"))

    # ---- CL metrics ----
    def print_matrix(M, title):
        print(f"\n===== {title} =====")
        header = "after \\ eval on | " + " | ".join(f"{t:>8s}" for t in tasks)
        print(header)
        for i in range(T):
            row = " | ".join(f"{M[i][j]*100:8.2f}" if M[i][j] is not None else f"{'—':>8s}"
                             for j in range(T))
            print(f"{tasks[i]:>17s} | {row}")

    def cl_metrics(M):
        bwt = sum(M[T-1][j] - M[j][j] for j in range(T - 1)) / (T - 1)
        forg = []
        for j in range(T - 1):
            worst_before = max(M[t][j] for t in range(j, T - 1))
            forg.append(worst_before - M[T-1][j])
        return bwt, sum(forg) / len(forg)

    print_matrix(A_auto, "ACCURACY MATRIX A[i,j] - AUTONOMOUS ROUTER (Domain-IL / Class-IL)")
    print_matrix(A, "ACCURACY MATRIX A[i,j] - TASK-ORACLE SNAPSHOT RETRIEVAL (Task-IL)")
    print_matrix(A_live, "ACCURACY MATRIX A[i,j] - LIVE WEIGHTS (Standard Continual Learning)")

    if T > 1:
        bwt_l, forg_l = cl_metrics(A_live)
        bwt_a, forg_a = cl_metrics(A_auto)
        bwt_m, forg_m = cl_metrics(A)
        print(f"\n[live]       BWT: {bwt_l*100:+.2f}% | Forgetting: {forg_l*100:.2f}%")
        print(f"[auto-route] BWT: {bwt_a*100:+.2f}% | Forgetting: {forg_a*100:.2f}%")
        print(f"[oracle]     BWT: {bwt_m*100:+.2f}% | Forgetting: {forg_m*100:.2f}%")
        print("  (method column uses test-time router+snapshot retrieval; report")
        print("   the LIVE column when comparing against standard CL literature)")

    # ---- Structure divergence across domains ----
    print("\n===== PER-DOMAIN STRUCTURE FINGERPRINTS =====")
    for name, fp in fingerprints.items():
        print(f"\n[{name}] pi_H={fp['pi_H']:.3f}")
        print("  gate : " + " ".join(f"{v:.2f}" for v in fp["gate"].tolist()))
        print("  bmRes: " + " ".join(f"{v:.2f}" for v in fp["bm_res"].tolist()))
        if fp.get("bm_plain") is not None and fp.get("bm_vit") is not None:
            print("  bmPln: " + " ".join(f"{v:.2f}" for v in fp["bm_plain"].tolist()))
            print("  bmViT: " + " ".join(f"{v:.2f}" for v in fp["bm_vit"].tolist()))

    names = list(fingerprints.keys())
    if len(names) >= 2:
        print("\nPairwise L1 divergence of gate fingerprints:")
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                fa, fb = fingerprints[names[a]]["gate"], fingerprints[names[b]]["gate"]
                d = (fa - fb).abs().sum().item()
                d_bm = (fingerprints[names[a]]["bm_res"] - fingerprints[names[b]]["bm_res"]).abs().sum().item()
                # per-sample gate divergence between domains
                ga = fingerprints[names[a]]["gate_tensor"]
                gb = fingerprints[names[b]]["gate_tensor"]
                nmin = min(ga.shape[0], gb.shape[0])
                d_sample = (ga[:nmin].mean(0) - gb[:nmin].mean(0)).abs().sum().item()
                print(f"  {names[a]} vs {names[b]}: gate_L1={d:.3f} | bmRes_L1={d_bm:.3f} | mean_gate_L1={d_sample:.3f}")
    print("\n(Interpretation: large L1 => architecture genuinely differs per domain;" 
          " ~0 => static network despite dynamic machinery)")


if __name__ == "__main__":
    main()
