"""
Continual / multi-domain benchmark for the AML supernet.

Sequence: MNIST -> CIFAR-10 -> Fashion-MNIST (all mapped to 10 classes,
3x32x32 inputs so the shared supernet consumes them directly).

Purpose: test whether the dynamic machinery (depth gates, block modes,
prototype routing) actually SPECIALIZES per domain when the input
distribution shifts - the setting where per-sample/per-domain adaptive
architecture has a reason to exist.

Reports:
  - Accuracy matrix A[i,j], BWT, Forgetting (academic CL metrics)
  - Per-domain structure fingerprints (mean layer gates, ResNet prob,
    pi entropy) + pairwise L1 divergence between domain fingerprints.
"""

import os
import sys
import time
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "AML_Core_Code"))

from train import ConvAML_CIFAR_TrueDynamic, set_seed


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def _base_gray(mean, std):
    return [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),
    ]


def get_task_data(name, data_root, workers=0, aug="basic"):
    if name == "mnist":
        mean, std = (0.1307,) * 3, (0.3081,) * 3
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

    tl = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True,
                                     num_workers=workers, pin_memory=True,
                                     persistent_workers=workers > 0, drop_last=True)
    el = torch.utils.data.DataLoader(te, batch_size=256, shuffle=False,
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
    cur_enc = cur_sel = cur_protos = deq_protos = None
    cur_sdelta = cur_bhead = None
    if use_snaps:
        cur_enc = {k: v.detach().clone() for k, v in model.vision_encoder.state_dict().items()}
        cur_sel = {k: v.detach().clone() for k, v in model.prototype_selector.state_dict().items()}
        cur_protos = model.proto_weights.detach().clone()
        # Structure heads (structure_delta / block_mode_head) set layer gates,
        # width masks and block modes. Without restoring them the LIVE
        # (post-later-task) heads reinterpret the old rows -> the 26.4% ->
        # 11.9% CIFAR drop measured under METHOD. Optional key so existing
        # callers that never captured them keep working.
        _sheads = snaps[0].get("struct_heads") if snaps else None
        if not _sheads or not hasattr(model, "structure_delta"):
            _sheads = None
        else:
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
            if dom_truth is not None:
                d_pred = torch.full((x.shape[0],), dom_truth, dtype=torch.long, device=device)
                route_hits += d_pred.numel()
                route_n += d_pred.numel()
                amb = None
            else:
                d_pred = dom_logits.argmax(-1)
                route_n += d_pred.numel()
                if use_snaps and route_tau is not None:
                    amb = margin < route_tau                           # ambiguous -> try all domains
            route_hist += torch.bincount(d_pred.clamp(max=route_hist.numel() - 1),
                                         minlength=route_hist.numel())
            g = torch.stack([slots[d]["gate"] for d in range(len(slots))])   # [T,L]
            b = torch.stack([slots[d]["bm_res"] for d in range(len(slots))]) # [T,L]
            gates_b = g[d_pred]                                   # [B,L]
            res_b = b[d_pred]                                     # [B,L]
            block_b = torch.stack([1.0 - res_b, res_b], dim=-1)   # [B,L,2]
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
                if _sheads is not None:
                    model.structure_delta.load_state_dict(snaps[d]["struct_heads"]["delta"])
                    model.block_mode_head.load_state_dict(snaps[d]["struct_heads"]["block"])
                if idx_c.any():
                    with torch.amp.autocast('cuda', **ac):
                        o, st, p, dg = model(x[idx_c], tau=0.1,
                                             struct_override=(override[0][idx_c], override[1][idx_c]),
                                             pi_mask=pi_mask)
                    outs_conf.append((idx_c, o))
                    pi_conf.append((idx_c, p.float()))
                    ls_full[idx_c] = dg["layer_soft"].float()
                    bm_wsum += st.block_mode.float()[:, :, 1].mean(0) * idx_c.sum().item()
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
            if _sheads is not None:
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
            bm_res_sum += bmf[:, :, 1].mean(0)
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
    bmres_sum = torch.zeros(model.max_layers, device=device)
    nb = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=amp_dtype is not None,
                                dtype=amp_dtype or torch.bfloat16):
            _, _, _, diag = model(x, tau=0.1)
        gate_sum += diag["layer_soft"].float().mean(0)
        # soft ResNet prob from logits (deterministic expectation)
        p = torch.softmax(diag["bm_logits"].float(), dim=-1)[:, :, 1]
        bmres_sum += p.mean(0)
        nb += 1
    return {"gate": gate_sum / max(nb, 1), "bm_res": bmres_sum / max(nb, 1)}


# ----------------------------------------------------------------------
# Episodic memory replay buffer (balanced per class, canonical images)
# ----------------------------------------------------------------------
def build_replay_buffer(dataset, per_class=100, num_classes=10):
    """Balanced exemplar set stored with TEST transform (canonical images,
    no augmentation baked in)."""
    ds_name = type(dataset).__name__
    root = dataset.root
    kw = dict(train=True, download=False)
    if ds_name == "MNIST":
        mean, std = (0.1307,) * 3, (0.3081,) * 3
        canon = torchvision.datasets.MNIST(root, transform=transforms.Compose(
            _base_gray(mean, std) + [transforms.ToTensor(), transforms.Normalize(mean, std)]), **kw)
    elif ds_name == "FashionMNIST":
        mean, std = (0.2860,) * 3, (0.3530,) * 3
        canon = torchvision.datasets.FashionMNIST(root, transform=transforms.Compose(
            _base_gray(mean, std) + [transforms.ToTensor(), transforms.Normalize(mean, std)]), **kw)
    else:
        canon = torchvision.datasets.CIFAR10(root, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]), **kw)

    targets = torch.as_tensor(canon.targets)
    chosen = []
    for c in range(num_classes):
        cls_idx = (targets == c).nonzero(as_tuple=True)[0]
        sel = cls_idx[torch.randperm(len(cls_idx))[:per_class]]
        chosen.append(sel)
    idx = torch.cat(chosen).tolist()
    xs = torch.stack([canon[i][0] for i in idx])
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

    def loader(self, batch_size=64):
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

    def begin_task(self, task_idx, num_tasks, reinit=True):
        s = round(task_idx * self.K / num_tasks)
        e = round((task_idx + 1) * self.K / num_tasks)
        self.frozen_hi = s
        self.future_lo = e
        self.snapshot = self.p.data.clone()
        if task_idx > 0 and reinit and e > s:
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
        
        self.mask.fill_(1.0)
        if self.frozen_hi > 0: self.mask[:self.frozen_hi] = 0.0
        if self.future_lo < self.K: self.mask[self.future_lo:] = 0.0
        return s, e

    @torch.no_grad()
    def mask_grad(self):
        # Now handled automatically by the backward hook.
        pass

    @torch.no_grad()
    def restore(self):
        if self.snapshot is None:
            return
        # Tensorized restore: capturable by CUDA Graphs
        self.p.data.copy_(torch.where(self.mask > 0, self.p.data, self.snapshot))


def consolidate(model, replay: ReplayBuffer, device, amp_dtype, epochs=1, lr=1e-4,
                isolator: ProtoIsolator = None, dom_head=None):
    """Low-LR pass over the pure replay set after each task: rebalances the
    shared prototypes across ALL seen domains (weights only, arch frozen).
    Also keeps the domain router calibrated on all seen tasks."""
    if len(replay) == 0:
        return
    model.train()
    arch_names = ("structure_prototypes", "block_mode_prototypes")
    arch_params = [p for n_, p in model.named_parameters() if any(k in n_ for k in arch_names)]
    other_params = [p for n_, p in model.named_parameters() if not any(k in n_ for k in arch_names) and p.requires_grad]
    opt = optim.AdamW(other_params, lr=lr, weight_decay=1e-4)
    opt_arch = optim.AdamW(arch_params, lr=lr * 0.2, weight_decay=1e-4)
    if dom_head is not None:
        opt_dom = optim.AdamW(dom_head.parameters(), lr=1e-3, weight_decay=0.0)
    ce = lambda lg, tg: F.cross_entropy(lg.float(), tg, label_smoothing=0.1)

    for _ in range(epochs):
        for xb, yb, tb in replay.loader(batch_size=64):
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            opt.zero_grad(set_to_none=True)
            opt_arch.zero_grad(set_to_none=True)
            if dom_head is not None:
                opt_dom.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp_dtype is not None,
                                    dtype=amp_dtype or torch.bfloat16):
                out, _, _, diag = model(xb, tau=0.8)
                loss = ce(out, yb)
                if dom_head is not None and "z" in diag:
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
            if dom_head is not None:
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
               anchor=None, anchor_lam=1e-3, aug="basic"):
    trainloader, _ = get_task_data(name, os.path.join(_ROOT, "data"), aug=aug)
    _, testloader = get_task_data(name, os.path.join(_ROOT, "data"))

    arch_names = ("structure_prototypes", "block_mode_prototypes")
    arch_params = [p for n_, p in model.named_parameters() if any(k in n_ for k in arch_names)]
    other_params = [p for n_, p in model.named_parameters() if not any(k in n_ for k in arch_names) and p.requires_grad]

    opt = optim.AdamW(other_params, lr=1e-3, weight_decay=1e-4)
    opt_arch = optim.AdamW(arch_params, lr=2e-4, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    sch_arch = optim.lr_scheduler.CosineAnnealingLR(opt_arch, T_max=epochs)

    anchor_pairs = None
    if anchor is not None:
        anchor_pairs = [(p, a) for p, a in anchor]

    ce = lambda lg, tg: F.cross_entropy(lg.float(), tg, label_smoothing=0.1)

    for epoch in range(epochs):
        model.train()
        freeze = epoch < freeze_epochs
        if freeze:
            for p in arch_params:
                p.requires_grad_(False)
        elif epoch == freeze_epochs:
            for p in arch_params:
                p.requires_grad_(True)
            print(f"  {log_prefix}[arch] unfroze")

        w_pi = 0.0 if epoch < epochs // 2 else w_pi_final
        t0, tot_loss = time.time(), 0.0
        for new_x, new_y in trainloader:
            new_x = new_x.to(device, non_blocking=True)
            new_y = new_y.to(device, non_blocking=True)

            # Split NEW data: first half (+ replay) updates weights, second
            # half supervises ONLY the architecture step. Feeding the weight
            # update the same samples the arch step "validates" on double
            # counts the data and contaminates the DARTS bi-level signal.
            n_v = new_x.size(0) // 2
            x_a, y_a = new_x[n_v:], new_y[n_v:]
            inputs, targets = new_x[:n_v], new_y[:n_v]

            # Replay mixing: exemplars are appended to both the WEIGHT half
            # and the ARCHITECTURE half to prevent architectural forgetting
            re_t = None
            if replay is not None and len(replay) > 0:
                n_re = min(48, inputs.size(0), len(replay))
                re_x, re_y, re_t = replay.sample(n_re, device)
                inputs = torch.cat([inputs, re_x])
                targets = torch.cat([targets, re_y])
                
                n_re_a = min(48, x_a.size(0), len(replay))
                re_x_a, re_y_a, _ = replay.sample(n_re_a, device)
                x_a = torch.cat([x_a, re_x_a])
                y_a = torch.cat([y_a, re_y_a])

            opt.zero_grad(set_to_none=True)
            opt_arch.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=amp_dtype is not None, dtype=amp_dtype or torch.bfloat16):
                outputs, struct, pi, diag = model(inputs, tau=0.8)
                loss = ce(outputs, targets)
                if w_pi > 0:
                    pif = pi.float()
                    ent_pi = -(pif * torch.log(pif.clamp(min=1e-9))).sum(-1).mean()
                    usage = pif.mean(0)
                    ent_usage = -(usage * torch.log(usage.clamp(min=1e-9))).sum()
                    loss = loss + w_pi * (ent_pi - ent_usage)

                # Domain-router supervision: current samples -> task_idx,
                # replay samples carry their original task id.
                # z is L2-normalized so the random-init router can't blow up
                # the loss scale (raw-z logits started ~10x too large).
                # NOTE: the valid-column count is constant within a task
                # (new samples carry task_idx, replay ids are older), so it
                # is hoisted: the old per-batch task_lbl.max().item() was a
                # needless device sync on every iteration.
                if dom_head is not None and diag.get("z", None) is not None:
                    task_lbl = torch.full((inputs.size(0),), task_idx,
                                          device=device, dtype=torch.long)
                    if re_t is not None:
                        task_lbl[inputs.size(0) - re_t.numel():] = re_t
                    z_norm = F.normalize(diag["z"].float(), dim=1)
                    dom_logits = dom_head(z_norm)
                    loss = loss + 0.3 * F.cross_entropy(dom_logits[:, :task_idx + 1], task_lbl)

                # L2-SP anchor on shared encoder/router
                if anchor_pairs is not None:
                    l2sp = sum((p - a).square().sum() for p, a in anchor_pairs)
                    loss = loss + anchor_lam * l2sp

            if not torch.isfinite(loss):
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if isolator is not None:
                    isolator.mask_grad()
                torch.nn.utils.clip_grad_norm_(other_params, 5.0)
                scaler.step(opt)
            else:
                loss.backward()
                if isolator is not None:
                    isolator.mask_grad()
                torch.nn.utils.clip_grad_norm_(other_params, 5.0)
                opt.step()
            if isolator is not None:
                isolator.restore()

            if not freeze:
                opt_arch.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=amp_dtype is not None, dtype=amp_dtype or torch.bfloat16):
                    out_v, _, _, _ = model(x_a, tau=0.8)
                    loss_a = ce(out_v, y_a)
                if torch.isfinite(loss_a):
                    if scaler is not None:
                        scaler.scale(loss_a).backward()
                        scaler.unscale_(opt_arch)
                        torch.nn.utils.clip_grad_norm_(arch_params, 5.0)
                        scaler.step(opt_arch)
                    else:
                        loss_a.backward()
                        torch.nn.utils.clip_grad_norm_(arch_params, 5.0)
                        opt_arch.step()
                        
            if scaler is not None:
                scaler.update()

            # Detached GPU-side sum (fp32: loss itself is bf16 under
            # autocast); synced once at epoch end. Log-only change.
            tot_loss = tot_loss + loss.detach().float()

        sch.step()
        # The architecture optimizer does not step while the structure is
        # frozen; stepping its scheduler anyway would silently decay the LR
        # for epochs that never trained (and trips PyTorch's
        # scheduler-before-optimizer warning).
        if not freeze:
            sch_arch.step()
        acc, _ = evaluate(model, testloader, device, amp_dtype)
        avg_loss = (tot_loss / len(trainloader)).item()
        print(f"  {log_prefix}ep {epoch+1}/{epochs} | {time.time()-t0:5.1f}s | "
              f"loss {avg_loss:.4f} | {name} acc {acc*100:.2f}%"
              f"{' [frozen]' if freeze else ''}", flush=True)


# ----------------------------------------------------------------------
# Main continual loop
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=str, default='mnist,cifar10,fashion')
    ap.add_argument('--epochs', type=str, default='5,28,16', help='per-task epochs')
    ap.add_argument('--K', type=int, default=48,
                    help='prototype count; split across tasks (16 rows/task at K=48,T=3)')
    ap.add_argument('--norm', type=str, default='gn1', choices=['gn1', 'in'],
                    help='post-conv normalization: gn1 (volume stats, legacy) or in (per-channel InstanceNorm)')
    ap.add_argument('--freeze', type=int, default=2)
    ap.add_argument('--replay-per-class', type=int, default=400,
                    help='exemplars per class stored in the replay buffer (0 = off)')
    ap.add_argument('--consolidate-epochs', type=int, default=8)
    ap.add_argument('--route-tau', type=float, default=0.6,
                    help='router margin below which a sample is evaluated under ALL '
                         'domain snapshots and the most confident prediction wins')
    ap.add_argument('--snap-q8', type=int, default=1,
                    help='store prototype snapshots int8-quantized (1) or fp32 (0)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--aug', type=str, default='basic', choices=['basic', 'auto'],
                    help='CIFAR train augmentation: basic (crop+flip) or auto (+AutoAugment)')
    ap.add_argument('--anchor-lam', type=float, default=0.0,
                    help='L2-SP encoder anchoring strength; >0 slows feature '
                         'drift BUT collapses per-domain specialization '
                         '(fingerprints converged to near-identical in v5)')
    args = ap.parse_args()
    use_replay = args.replay_per_class > 0

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else (
        torch.float16 if device.type == 'cuda' else None)

    tasks = args.tasks.split(',')
    ep_list = [int(e) for e in args.epochs.split(',')]
    assert len(ep_list) == len(tasks)

    model = ConvAML_CIFAR_TrueDynamic(K=args.K, max_dim=64, max_layers=18,
                                      norm_mode=args.norm).to(device)

    T = len(tasks)
    A = [[None] * T for _ in range(T)]        # method protocol (snapshot retrieval)
    A_live = [[None] * T for _ in range(T)]   # standard CL (current weights only)
    fingerprints = {}
    replay = ReplayBuffer()
    isolator = ProtoIsolator(model)
    dom_head = torch.nn.Linear(1024, T).to(device)
    slots = [None] * T          # per-domain structure slots
    dom_snaps = [None] * T      # per-domain frozen (encoder, selector) snapshots
    anchor_snap = None          # L2-SP snapshot of shared encoder/router

    for i, name in enumerate(tasks):
        print(f"\n===== TASK {i+1}/{T}: {name} =====", flush=True)
        # reinit=False: future prototype rows keep their ORIGINAL kaiming
        # values forever. Earlier tasks' routing may lean on these rows;
        # re-randomizing them per task silently corrupts that routing
        # (measured -0.9pt MNIST in v9). Untrained rows stay equally fresh
        # for the new task either way.
        s, e = isolator.begin_task(i, T, reinit=False)
        print(f"  [proto-isolation] rows [{s}:{e}) owned; rows [0:{s}) frozen")
        print(f"  [replay] buffer size: {len(replay)} exemplars")
        tr_loader, _ = get_task_data(name, os.path.join(_ROOT, "data"))
        del tr_loader
        train_task(model, name, ep_list[i],
                   freeze_epochs=(args.freeze if i == 0 else min(args.freeze, 1)),
                   device=device, amp_dtype=amp_dtype,
                   w_pi_final=0.01, log_prefix=f"[{name}] ",
                   replay=(replay if (use_replay and i > 0) else None),
                   isolator=isolator,
                   dom_head=dom_head,
                   task_idx=i,
                    anchor=(anchor_snap if args.anchor_lam > 0 else None),
                    anchor_lam=args.anchor_lam,
                    aug=args.aug)

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

        # Freeze this domain's structure into its slot (post-consolidation)
        _, el_self = get_task_data(name, os.path.join(_ROOT, "data"))
        slots[i] = collect_structure_slot(model, el_self, device, amp_dtype)
        print(f"  [slot] stored structure slot for '{name}'")

        dom_snaps[i] = {
            "enc": {k: v.detach().clone() for k, v in model.vision_encoder.state_dict().items()},
            "sel": {k: v.detach().clone() for k, v in model.prototype_selector.state_dict().items()},
        }
        pw = model.proto_weights.detach()
        if args.snap_q8:
            scale = pw.abs().max().clamp(min=1e-12) / 127.0
            dom_snaps[i]["protos"] = {"q": torch.round(pw / scale).to(torch.int8).cpu(),
                                      "scale": scale.cpu()}
            mb = dom_snaps[i]["protos"]["q"].numel() / 1e6
            print(f"  [snap] stored enc+sel+prototypes(int8, {mb:.0f} MB-elems) for '{name}'")
        else:
            dom_snaps[i]["protos"] = pw.to("cpu", torch.float32).clone()
            print(f"  [snap] stored encoder+selector+prototypes(fp32) for '{name}'")

        # Snapshot the shared encoder/router for next-task anchoring
        anchor_names = ("vision_encoder", "prototype_selector")
        anchor_snap = [(p, p.detach().clone()) for n_, p in model.named_parameters()
                       if any(k in n_ for k in anchor_names)]

        # Evaluate on ALL seen tasks through BOTH protocols:
        #   A[i][j]      - method protocol: per-domain snapshot + router
        #                  (incl. test-time max-confidence retrieval of
        #                  ambiguous samples). Reported by design.
        #   A_live[i][j] - standard CL protocol: current mutable weights,
        #                  no snapshot swap, no test-time selection. This is
        #                  the column comparable to ER/EWC/DER literature.
        for j in range(i + 1):
            _, el = get_task_data(tasks[j], os.path.join(_ROOT, "data"))
            use_route = all(slots[k] is not None for k in range(i + 1))
            acc, fp = evaluate(model, el, device, amp_dtype,
                               dom_head=(dom_head if use_route else None),
                               slots=(slots[:i + 1] if use_route else None),
                               snaps=(dom_snaps[:i + 1] if use_route else None),
                               route_tau=args.route_tau, dom_truth=j)
            acc_live, _ = evaluate(model, el, device, amp_dtype)
            A[i][j] = acc
            A_live[i][j] = acc_live
            if j == i:
                fingerprints[name] = fp
            tag = "routed" if use_route else "live"
            print(f"  A[{i+1},{j+1}] ({tasks[j]}, {tag}): {acc*100:.2f}% | live: {acc_live*100:.2f}%")

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

    print_matrix(A, "ACCURACY MATRIX A[i,j] - METHOD (snapshot retrieval)")
    print_matrix(A_live, "ACCURACY MATRIX A[i,j] - LIVE WEIGHTS (standard CL)")

    if T > 1:
        bwt, forg = cl_metrics(A_live)
        bwt_m, forg_m = cl_metrics(A)
        print(f"\n[live]   BWT: {bwt*100:+.2f}% | Forgetting (avg): {forg*100:.2f}%")
        print(f"[method] BWT: {bwt_m*100:+.2f}% | Forgetting (avg): {forg_m*100:.2f}%")
        print("  (method column uses test-time router+snapshot retrieval; report")
        print("   the LIVE column when comparing against standard CL literature)")

    # ---- Structure divergence across domains ----
    print("\n===== PER-DOMAIN STRUCTURE FINGERPRINTS =====")
    for name, fp in fingerprints.items():
        print(f"\n[{name}] pi_H={fp['pi_H']:.3f}")
        print("  gate : " + " ".join(f"{v:.2f}" for v in fp["gate"].tolist()))
        print("  bmRes: " + " ".join(f"{v:.2f}" for v in fp["bm_res"].tolist()))

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
