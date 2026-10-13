"""
[PHASE 3 PILOT / ARCHIVAL EXPERIMENT]
No-Replay Continual Learning for AML Supernet: Encoder-Stability Research
===============================================================================
Historical Context & Purpose:
This script contains the Phase 3 pilot exploration that tested weight-space
regularization (EWC, Feature Distillation / KD, and bottom-layer parameter freezing)
on the shared vision encoder without episodic replay buffers.

Key Scientific Finding from this Pilot:
Shared weight constraints alone cannot prevent representation drift under new domains
(A_live forgetting remained at ~43.7% in results_noreplay_full_pilot4.json). This key
insight motivated the architectural upgrade in Phase 4: Full Supernet Snapshotting
combined with TaskCentroidRouter and Prototype Isolation.

Active Production Entry Point:
For current 18-Layer 3-Stage Hybrid Continual Learning (Zero-Forgetting 99.42% with
zero replay exemplars), use the top-level production script:
    `train_continual_aml.py` (with `mgn/blocks/hybrid_18.py`).
===============================================================================
"""

import argparse
import copy
import itertools
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "AML_Core_Code"))

from train_cv_cifar_18 import AcceleratedCVNetwork
from mgn.architecture import ModelStructure
from mgn.layout import get_prototype_layout
from train import set_seed
from train_continual_aml import (ProtoIsolator, collect_structure_slot, evaluate,
                                  get_task_data)

DATA_ROOT = os.path.join(ROOT, "data")


class AMLNet(nn.Module):
    def __init__(self, K=48, latent_dim=128, max_layers=18, max_dim=64, norm_mode="gn1"):
        super().__init__()
        self.K = K
        self.max_layers = max_layers
        self.max_dim = max_dim

        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(128, latent_dim), nn.ReLU()
        )
        self.prototype_selector = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, K)
        )
        self.structure_delta = nn.Linear(latent_dim, max_layers + (max_layers * max_dim))
        self.block_mode_head = nn.Linear(latent_dim, max_layers * 2)
        nn.init.constant_(self.structure_delta.weight, 0.0)
        nn.init.constant_(self.structure_delta.bias, 0.0)
        nn.init.constant_(self.block_mode_head.weight, 0.0)
        nn.init.constant_(self.block_mode_head.bias, 0.0)
        with torch.no_grad():
            self.block_mode_head.bias.data.view(max_layers, 2)[:, 1] = 2.0

        total_dim, _offs, _info, _wo, _bo = get_prototype_layout(max_layers, max_dim, 10, 3)
        self.proto_weights = nn.Parameter(torch.empty(K, total_dim))
        nn.init.kaiming_uniform_(self.proto_weights, a=math.sqrt(5))

        self.exec_ = AcceleratedCVNetwork(max_layers=max_layers, max_dim=max_dim,
                                          num_classes=10, norm_mode=norm_mode, K=self.K)

    def caf_attention(self, features):
        return torch.softmax(features, dim=-1).clamp(min=1e-8)

    def forward(self, x, tau=1.0, struct_override=None, pi_mask=None):
        B = x.shape[0]
        device = x.device
        z = self.vision_encoder(x)
        sel_logits = self.prototype_selector(z)
        if pi_mask is not None:
            sel_logits = sel_logits.float().masked_fill(~pi_mask, float("-inf"))
        pi = self.caf_attention(sel_logits)

        delta = self.structure_delta(z)
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

        bm_logits = self.block_mode_head(z).view(B, self.max_layers, 2)

        if struct_override is not None:
            layer_gates, block_mode = struct_override
            layer_gates = layer_gates.to(device=device, dtype=torch.float32)
            block_mode = block_mode.to(device=device, dtype=torch.float32)
        else:
            layer_gates = (layer_soft > 0.5).float().detach() + layer_soft - layer_soft.detach()
            if self.training:
                block_mode = F.gumbel_softmax(bm_logits, tau=tau, hard=True, dim=-1)
            else:
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
        diag = {"layer_soft": layer_soft, "bm_logits": bm_logits,
                "d_lengths": d_lengths, "z": z}
        return out, struct, pi, diag


class CentroidRouter(nn.Module):
    """Nearest-centroid domain router: one EMA-updated 128-d centroid per task
    on the L2-normalized encoder output. Replaces the learned Linear router,
    which collapsed to one-hot-on-latest-task when trained without replay
    (measured: route_acc 0%/0%/100% for mnist/cifar/fashion at full size).
    Memory cost: T x latent_dim floats - no old data needed."""

    def __init__(self, dim, T, scale=20.0):
        super().__init__()
        self.register_buffer("cent", torch.zeros(T, dim))
        self.register_buffer("seen", torch.zeros(1))
        self.scale = scale
    def set_seen(self, n):
        self.seen.fill_(n)

    def forward(self, z_norm):
        n = int(self.seen.item())
        if n == 0:
            n = 1
        return (z_norm @ self.cent[:n].T) * self.scale


class OnlineEWC:
    def __init__(self, model, scope=("vision_encoder",)):
        self.scope = scope
        self.params = {n: p for n, p in model.named_parameters()
                       if any(k in n for k in scope)}
        self.theta_star = {n: p.detach().clone() for n, p in self.params.items()}
        self.fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.tasks = 0

    def refresh(self, model, batches, device, amp_dtype, n_batches=64):
        model.eval()
        acc = {n: torch.zeros_like(p) for n, p in self.params.items()}
        n_seen = 0
        for x, y in itertools.islice(batches, n_batches):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp_dtype is not None,
                                    dtype=amp_dtype or torch.bfloat16):
                out, _, _, _ = model(x, tau=0.1)
                loss = F.cross_entropy(out.float(), y)
            loss.backward()
            for n, p in self.params.items():
                if p.grad is not None:
                    acc[n] += p.grad.detach().float() ** 2
            n_seen += 1
        with torch.no_grad():
            for n, p in self.params.items():
                f_new = acc[n] / max(n_seen, 1)
                mean = f_new.mean()
                if mean > 0:
                    f_new = f_new / mean
                self.fisher[n] = self.fisher[n] + f_new
            self.theta_star = {n: p.detach().clone() for n, p in self.params.items()}
        self.tasks += 1
        model.train()
        return float(sum(f.mean().item() for f in self.fisher.values()))

    def penalty(self, model, lambda_):
        if lambda_ <= 0 or self.tasks == 0:
            return None
        named = dict(model.named_parameters())
        total = None
        for n, p in self.params.items():
            d = p - self.theta_star[n]
            term = (self.fisher[n].to(d.device) * d * d).sum()
            total = term if total is None else total + term
        return lambda_ * total


@torch.no_grad()
def encoder_fingerprint(model, probe_x, device):
    z = model.vision_encoder(probe_x).float()
    return F.normalize(z, dim=-1)


def train_task(model, name, epochs, freeze_epochs, device, amp_dtype, args,
               trainloader, isolator=None, task_idx=0, ewc=None, teacher=None,
               router=None, log_prefix=""):
    _, testloader = get_task_data(name, DATA_ROOT, workers=args.workers)
    arch = [p for n, p in model.named_parameters()
            if "structure_delta" in n or "block_mode_head" in n]
    other = [p for n, p in model.named_parameters()
             if not ("structure_delta" in n or "block_mode_head" in n) and p.requires_grad]
    opt = optim.AdamW(other, lr=args.lr, weight_decay=1e-4)
    opt_arch = optim.AdamW(arch, lr=args.lr * 0.2, weight_decay=1e-4)
    is_centroid = router is not None and hasattr(router, "cent")
    opt_dom = (optim.AdamW(router.parameters(), lr=1e-3, weight_decay=0.0)
               if (router is not None and not is_centroid) else None)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    sch_arch = optim.lr_scheduler.CosineAnnealingLR(opt_arch, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        freeze = epoch < freeze_epochs
        for p in arch:
            p.requires_grad_(not freeze)
        w_pi = 0.0 if epoch < epochs // 2 else args.w_pi
        t0, tot_loss, tot_kd, tot_reg, nb = time.time(), 0.0, 0.0, 0.0, 0
        for step, (new_x, new_y) in enumerate(trainloader):
            if args.max_batches and step >= args.max_batches:
                break
            new_x = new_x.to(device, non_blocking=True)
            new_y = new_y.to(device, non_blocking=True)

            n_v = new_x.size(0) // 2
            x_w, y_w = new_x[:n_v], new_y[:n_v]
            x_a, y_a = new_x[n_v:], new_y[n_v:]

            opt.zero_grad(set_to_none=True)
            opt_arch.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp_dtype is not None,
                                    dtype=amp_dtype or torch.bfloat16):
                outputs, struct, pi, diag = model(x_w, tau=args.tau)
                loss = F.cross_entropy(outputs.float(), y_w, label_smoothing=0.1)
                if w_pi > 0:
                    pif = pi.float()
                    ent_pi = -(pif * pif.clamp(min=1e-9).log()).sum(-1).mean()
                    usage = pif.mean(0)
                    ent_usage = -(usage * usage.clamp(min=1e-9).log()).sum()
                    loss = loss + w_pi * (ent_pi - ent_usage)
                reg = ewc.penalty(model, args.ewc_lambda) if ewc is not None else None
                if reg is not None:
                    loss = loss + reg
                    tot_reg += float(reg)
                kd = None
                if teacher is not None and args.kd_lambda > 0:
                    with torch.no_grad():
                        z_t = teacher.vision_encoder(x_w).float()
                    z_s = diag["z"].float()
                    kd = (1.0 - F.cosine_similarity(z_s, z_t, dim=-1)).mean()
                    loss = loss + args.kd_lambda * kd
                    tot_kd += float(kd)
                if is_centroid and diag.get("z", None) is not None:
                    with torch.no_grad():
                        cm = F.normalize(diag["z"].float().mean(0), dim=0)
                        router.cent[task_idx] = (0.9 * router.cent[task_idx] + 0.1 * cm)
                elif router is not None and diag.get("z", None) is not None:
                    z_norm = F.normalize(diag["z"].float(), dim=1)
                    dom_logits = router(z_norm)
                    task_lbl = torch.full((x_w.size(0),), task_idx,
                                          device=device, dtype=torch.long)
                    loss = loss + 0.3 * F.cross_entropy(dom_logits[:, :task_idx + 1], task_lbl)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if isolator is not None:
                isolator.mask_grad()
            torch.nn.utils.clip_grad_norm_(other, 5.0)
            opt.step()
            if opt_dom is not None:
                torch.nn.utils.clip_grad_norm_(router.parameters(), 5.0)
                opt_dom.step()
            if isolator is not None:
                isolator.restore()

            if not freeze:
                opt_arch.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=amp_dtype is not None,
                                        dtype=amp_dtype or torch.bfloat16):
                    out_v, _, _, _ = model(x_a, tau=args.tau)
                    loss_a = F.cross_entropy(out_v.float(), y_a, label_smoothing=0.1)
                if torch.isfinite(loss_a):
                    loss_a.backward()
                    torch.nn.utils.clip_grad_norm_(arch, 5.0)
                    opt_arch.step()

            tot_loss += float(loss.detach())
            nb += 1

        sch.step()
        if not freeze:
            sch_arch.step()
        acc, _ = evaluate(model, testloader, device, amp_dtype)
        print(f"  {log_prefix}ep {epoch+1}/{epochs} | {time.time()-t0:5.1f}s | "
              f"loss {tot_loss/max(nb,1):.4f} | kd {tot_kd/max(nb,1):.4f} | "
              f"ewc {tot_reg/max(nb,1):.4f} | {name} acc {acc*100:.2f}%"
              f"{' [frozen-arch]' if freeze else ''}", flush=True)


def cl_metrics(M, T):
    bwt = sum(M[T-1][j] - M[j][j] for j in range(T - 1)) / (T - 1)
    forg = []
    for j in range(T - 1):
        worst_before = max(M[t][j] for t in range(j, T - 1))
        forg.append(worst_before - M[T-1][j])
    return bwt, sum(forg) / len(forg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=str, default='mnist,cifar10,fashion')
    ap.add_argument('--epochs', type=str, default='2,2,2')
    ap.add_argument('--method', type=str, default='ewckd',
                    choices=['naive', 'ewc', 'kd', 'ewckd'])
    ap.add_argument('--ewc-lambda', type=float, default=5.0)
    ap.add_argument('--kd-lambda', type=float, default=1.0)
    ap.add_argument('--ewc-sel', action='store_true',
                    help='include prototype_selector in the EWC scope')
    ap.add_argument('--freeze-enc-bottom', type=int, default=0)
    ap.add_argument('--K', type=int, default=48)
    ap.add_argument('--max-layers', type=int, default=18)
    ap.add_argument('--max-dim', type=int, default=64)
    ap.add_argument('--latent-dim', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--tau', type=float, default=0.8)
    ap.add_argument('--w-pi', type=float, default=0.01)
    ap.add_argument('--freeze-arch-epochs', type=int, default=1)
    ap.add_argument('--max-batches', type=int, default=0)
    ap.add_argument('--fisher-batches', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--aug', type=str, default='basic', choices=['basic'])
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--tag', type=str, default=None)
    ap.add_argument('--method-eval', action='store_true',
                    help='also report the METHOD protocol (domain router + '
                         'per-domain weight snapshots; model memory, no data)')
    ap.add_argument('--route-tau', type=float, default=0.6)
    ap.add_argument('--router-scale', type=float, default=20.0,
                    help='logit scale of the centroid router; lower values make '
                         'more samples "ambiguous" so the try-all-domains '
                         'fallback fires (route_tau)')
    ap.add_argument('--router', type=str, default='centroid', choices=['linear', 'centroid'],
                    help='domain router for the METHOD protocol: learned Linear '
                         '(recency-biased when trained without replay) or EMA '
                         'nearest-centroid on normalized z (default)')
    ap.add_argument('--no-snap-struct', action='store_true',
                    help='do NOT snapshot the structure heads with each domain '
                         '(old buggy behaviour; use to A/B the snapshot leak)')
    ap.add_argument('--exec-no-hts', action='store_true',
                    help='executor skips the per-sample HtS low-rank adapters and '
                         'runs the two-conv path (eager or fused) instead: '
                         '~5-7x faster at full size, fewer synthable params')
    args = ap.parse_args()

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
    T = len(tasks)

    model = AMLNet(K=args.K, latent_dim=args.latent_dim, max_layers=args.max_layers,
                   max_dim=args.max_dim).to(device)
    if args.exec_no_hts:
        model.exec_.use_hts_low_rank = False
    isolator = ProtoIsolator(model)
    ewc = None
    if args.method in ("ewc", "ewckd"):
        scope = ("vision_encoder", "prototype_selector") if args.ewc_sel else ("vision_encoder",)
        ewc = OnlineEWC(model, scope=scope)

    _, el0 = get_task_data(tasks[0], DATA_ROOT, workers=0)
    probe_x = next(iter(el0))[0][:512].to(device)
    baseline_z = None
    drift_log = []

    A = [[None] * T for _ in range(T)]
    M = [[None] * T for _ in range(T)] if args.method_eval else None
    R = [[None] * T for _ in range(T)] if args.method_eval else None
    fingerprints = {}
    router = (CentroidRouter(args.latent_dim, T, scale=args.router_scale).to(device) if args.router == 'centroid' else torch.nn.Linear(args.latent_dim, T).to(device)) if args.method_eval else None
    slots = [None] * T
    snaps = [None] * T

    print(f"===== NO-REPLAY CL | method={args.method} | tasks={tasks} | "
          f"epochs={ep_list} | K={args.K} L={args.max_layers} D={args.max_dim} | "
          f"ewc_l={args.ewc_lambda} kd_l={args.kd_lambda} | "
          f"max_batches={args.max_batches} =====", flush=True)

    for i, name in enumerate(tasks):
        print(f"\n===== TASK {i+1}/{T}: {name} =====", flush=True)
        s, e = isolator.begin_task(i, T, reinit=False)
        print(f"  [proto-isolation] rows [{s}:{e}) owned; rows [0:{s}) frozen")

        if i == 0 and args.freeze_enc_bottom > 0:
            for m in list(model.vision_encoder.children())[:args.freeze_enc_bottom]:
                for p in m.parameters():
                    p.requires_grad_(False)
            print(f"  [enc-freeze] froze first {args.freeze_enc_bottom} children of vision_encoder")

        trainloader, _ = get_task_data(name, DATA_ROOT, workers=args.workers, aug=args.aug)
        teacher = None
        if args.method in ("kd", "ewckd") and i > 0:
            teacher = copy.deepcopy(model)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

        train_task(model, name, ep_list[i],
                   freeze_epochs=(args.freeze_arch_epochs if i == 0 else 0),
                   device=device, amp_dtype=amp_dtype, args=args,
                   trainloader=trainloader, isolator=isolator, task_idx=i,
                   ewc=ewc, teacher=teacher, router=router,
                   log_prefix=f"[{name}] ")

        if args.method_eval:
            _, el_self = get_task_data(name, DATA_ROOT, workers=0)
            slots[i] = collect_structure_slot(model, el_self, device, amp_dtype)
            snaps[i] = {
                "enc": {k: v.detach().clone() for k, v in
                        model.vision_encoder.state_dict().items()},
                "sel": {k: v.detach().clone() for k, v in
                        model.prototype_selector.state_dict().items()},
            }
            pw = model.proto_weights.detach()
            scale = pw.abs().max().clamp(min=1e-12) / 127.0
            snaps[i]["protos"] = {"q": torch.round(pw / scale).to(torch.int8).cpu(),
                                  "scale": scale.cpu()}
            snaps[i]["struct_heads"] = ({} if args.no_snap_struct else {
                "delta": {k: v.detach().clone() for k, v in
                          model.structure_delta.state_dict().items()},
                "block": {k: v.detach().clone() for k, v in
                          model.block_mode_head.state_dict().items()},
            })
            print(f"  [snap] stored structure slot + enc/sel/protos(int8)/struct-heads for '{name}'")

        for j in range(i + 1):
            _, elj = get_task_data(tasks[j], DATA_ROOT, workers=0)
            acc, fp = evaluate(model, elj, device, amp_dtype)
            A[i][j] = acc
            if j == i:
                fingerprints[name] = fp
            if args.method_eval and all(slots[k] is not None for k in range(i + 1)):
                if hasattr(router, "set_seen"):
                    router.set_seen(i + 1)
                acc_m, fpm = evaluate(model, elj, device, amp_dtype,
                                      dom_head=router, slots=slots[:i + 1],
                                      snaps=snaps[:i + 1], route_tau=args.route_tau,
                                      dom_truth=j)
                M[i][j] = acc_m
                ra = fpm["route_acc"]
                R[i][j] = ra
                print(f"  A[{i+1},{j+1}] ({tasks[j]}, live): {acc*100:.2f}% | "
                      f"method: {acc_m*100:.2f}% | route_acc: "
                      f"{ra*100:.1f}%" if ra is not None else
                      f"  A[{i+1},{j+1}] ({tasks[j]}, live): {acc*100:.2f}% | "
                      f"method: {acc_m*100:.2f}%")
            else:
                print(f"  A[{i+1},{j+1}] ({tasks[j]}, live): {acc*100:.2f}%")

        if ewc is not None:
            fbar = ewc.refresh(model, trainloader, device, amp_dtype,
                               n_batches=args.fisher_batches)
            print(f"  [ewc] fisher refreshed ({fbar:.3f} mean) + theta_star updated")

        if i == 0:
            baseline_z = encoder_fingerprint(model, probe_x, device)
            drift_log.append((1, 0.0))
            print("  [bo-mat drift] baseline = encoder right after task 0")

        z_now = encoder_fingerprint(model, probe_x, device)
        drift = float(1.0 - (z_now * baseline_z).sum(-1).mean())
        if i > 0:
            drift_log.append((i + 1, drift))
            print(f"  [bo-mat drift] cosine distance vs post-task-0 encoder: {drift:.4f}")

    print("\n===== ACCURACY MATRIX A[i,j] - LIVE (no replay) =====")
    header = "after \\ eval on | " + " | ".join(f"{t:>10s}" for t in tasks)
    print(header)
    for i in range(T):
        row = " | ".join(f"{A[i][j]*100:10.2f}" if A[i][j] is not None else f"{'—':>10s}"
                         for j in range(T))
        print(f"{tasks[i]:>17s} | {row}")

    if T > 1:
        bwt, forg = cl_metrics(A, T)
        print(f"\n[live] BWT: {bwt*100:+.2f}% | Forgetting (avg): {forg*100:.2f}%")
    print("[bo-mat drift] " + " -> ".join(f"T{t}:{d:.3f}" for t, d in drift_log))

    if args.method_eval:
        print("\n===== ACCURACY MATRIX A[i,j] - METHOD (router + snapshot retrieval) =====")
        print(header)
        for i in range(T):
            row = " | ".join(f"{M[i][j]*100:10.2f}" if M[i][j] is not None else f"{'—':>10s}"
                             for j in range(T))
            print(f"{tasks[i]:>17s} | {row}")
        if T > 1:
            bwt_m, forg_m = cl_metrics(M, T)
            print(f"\n[method] BWT: {bwt_m*100:+.2f}% | Forgetting (avg): {forg_m*100:.2f}%")

    names = list(fingerprints.keys())
    if len(names) >= 2:
        print("\nPairwise L1 divergence of gate fingerprints (specialization check):")
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                fa = fingerprints[names[a]]["gate"]
                fb = fingerprints[names[b]]["gate"]
                print(f"  {names[a]} vs {names[b]}: gate_L1={(fa-fb).abs().sum().item():.3f}")

    tag = args.tag or f"{args.method}_{'_'.join(tasks)}"
    out = {
        "config": vars(args),
        "A_live": A,
        "A_method": M,
        "route_acc": R,
        "bwt": (bwt if T > 1 else None),
        "forgetting": (forg if T > 1 else None),
        "drift": drift_log,
        "fingerprint_pi_H": {k: v["pi_H"] for k, v in fingerprints.items()},
    }
    path = os.path.join(ROOT, "experiments", f"results_noreplay_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
