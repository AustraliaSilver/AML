"""
experiments/exp_basis_ogp_corrected.py
======================================
Honest head-to-head on one question:

    can orthogonal gradient projection remove the per-task snapshot?

Protocol: MNIST -> Fashion-MNIST on the HtS dual-residual basis
(W = base + sum_i alpha_i(x) * priv_i), task-IL with the ground-truth task id
supplied at evaluation -- i.e. the oracle row A_oracle of the AML paper, not
A_auto.

Every difference from experiments/exp_verify_basis_ogp.py is deliberate and
listed here so the two scripts can be diffed in intent:

  1. STORAGE IS MEASURED FOR EVERY METHOD.
     The previous script printed "0 KB (True Single Model)" for the OGP row.
     But the per-block basis U_k returned by compute_activation_subspaces has
     to survive the task boundary, so it is state.  It is counted here.

  2. THE SNAPSHOT ROW IS MEASURED.
     The previous script wrote `acc_m_snap = acc_m_initial` under the comment
     "Snapshot restores exactly 100%" and never evaluated MNIST for it.  Here
     the snapshot is really written with torch.save, really reloaded and
     really evaluated.

  3. THE SUBSPACE COMES FROM THE TENSOR THE KERNEL ACTUALLY MULTIPLIES.
     HtSResidualBlock.forward does

         res = gelu(gn1(x));  out = conv2d(res, kernel)

     so U must span the *conv input* `res`.  The previous script unfolded the
     *block input* `x`.  Both are run here (`ogp_wrong` vs `ogp_fixed`) so the
     difference is measurable rather than argued.

  4. THE THRESHOLD IS A KNOB, NOT A CLAIM.
     `--thresholds` reports the full storage <-> rank <-> forgetting curve
     instead of asserting a single point.

  5. CONTROLS.
     `random_proj` uses a random orthonormal basis of identical rank and
     identical storage, which answers "does the orthogonality of U matter at
     all?".  `freeze_basis` removes the basis gradient entirely: the ceiling
     on retention and the floor on plasticity.  `ogp_fixed_locked` also freezes
     the GroupNorm affine params, the 1x1 shortcut and task_ctrl, which are
     gradient paths the previous script never touched.

  6. A DIAGNOSTIC INSTEAD OF A NARRATIVE.
     `kernel_drift` measures, after task 2, how much of each block's effective
     kernel changed *inside* the protected subspace and how much of that came
     from the alpha path alone.  That is the number that decides whether the
     projection did anything.

Output: a table plus JSON at --json-out.
"""

import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_continual_aml import get_task_data, set_seed          # noqa: E402
from hts_lib.models.hypernetwork import HtSContinualModel         # noqa: E402


STAGE_PARAMS = {
    "b1": ("base1", "b1_priv"),
    "b1_2": ("base1_2", "b1_2_priv"),
    "b2": ("base2", "b2_priv"),
    "b2_3": ("base2_3", "b2_3_priv"),
    "b3": ("base3", "b3_priv"),
}


def mib(nbytes):
    return nbytes / (1024 * 1024)


def state_bytes(module):
    return sum(p.numel() * p.element_size() for p in module.parameters())


def subspace_bytes(subspaces):
    return sum(u.numel() * u.element_size() for u in subspaces.values())


# ---------------------------------------------------------------------------
# 1. ACTIVATION SUBSPACE, TAKEN FROM THE RIGHT TENSOR
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_subspaces(model, loader, device, task="mnist", threshold=0.98,
                      max_batches=15, source="conv_input", rank_cap=None):
    """Correlation of the tensor the dynamic kernel multiplies (or, with
    source='block_input', of the block input -- the historical behaviour).

    Returns ({block_i: U_k with orthonormal columns}, {block_i: k}).
    """
    model.eval()
    corr, handles = {}, []

    def make_hook(name, stride):
        def hook(module, inp, outp):
            # gn1 output -> gelu  ==  what conv2d(res, kernel) consumes
            z = F.gelu(outp) if source == "conv_input" else inp[0]
            b, c, h, w = z.shape
            u = F.unfold(z, kernel_size=3, padding=1, stride=stride)
            u = u.transpose(1, 2).reshape(-1, c * 9)
            r = (u.t() @ u).double()
            corr[name] = r.cpu() if name not in corr else corr[name] + r.cpu()
        return hook

    for idx, blk in enumerate(model.blocks):
        target = blk.gn1 if source == "conv_input" else blk
        handles.append(target.register_forward_hook(make_hook(f"block_{idx}", blk.stride)))

    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        model(x.to(device), task=task, tau=0.5)

    for h in handles:
        h.remove()

    subspaces, ranks = {}, {}
    for name, R in corr.items():
        R = 0.5 * (R + R.t())
        U, S, _ = torch.linalg.svd(R)
        U = U.float()
        S = S.float().clamp_min(0.0)
        cum = torch.cumsum(S, 0) / S.sum()
        k = int((cum < threshold).sum().item()) + 1
        k = max(1, min(k, U.shape[1]))
        if rank_cap is not None:
            k = max(1, min(k, rank_cap))
        # NOTE: deliberately NOT capped at D_in - 1.  The previous script did
        # that "to preserve plasticity", which makes exact preservation
        # unreachable by construction and hides it inside a 98% threshold.
        subspaces[name] = U[:, :k].to(device).contiguous()
        ranks[name] = k
    return subspaces, ranks


def random_subspaces_like(subspaces, seed=0):
    """Identical rank, identical storage, no semantics."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for name, U in subspaces.items():
        d, k = U.shape
        a = torch.randn(d, k, generator=g)
        q, _ = torch.linalg.qr(a)
        out[name] = q[:, :k].to(U.device).contiguous()
    return out


# ---------------------------------------------------------------------------
# 2. GRADIENT SURGERY
# ---------------------------------------------------------------------------
def _project_inplace(p, U):
    """g <- g (I - U U^T) so that the weight update satisfies dW U = 0."""
    if p.grad is None:
        return
    d_in = U.shape[0]
    gf = p.grad.reshape(-1, d_in)
    if gf.shape[1] != d_in:
        return
    p.grad.copy_((gf - (gf @ U) @ U.t()).reshape(p.grad.shape))


class GradPolicy:
    """Applied after loss.backward(), before clip/step."""

    def __init__(self, kind="none", subspaces=None, freezes=()):
        self.kind = kind                      # none | ogp | freeze_basis
        self.subspaces = subspaces or {}
        self.freezes = set(freezes)           # norm | shortcut | ctrl

    def apply(self, model):
        for idx, blk in enumerate(model.blocks):
            base_name, priv_name = STAGE_PARAMS[model.block_kernels[idx]]
            base = getattr(model.basis_sys, base_name)
            priv = getattr(model.basis_sys, priv_name)
            if self.kind == "freeze_basis":
                base.grad = None
                priv.grad = None
            elif self.kind == "ogp":
                U = self.subspaces.get(f"block_{idx}")
                if U is not None:
                    _project_inplace(base, U)
                    _project_inplace(priv, U)

        if "norm" in self.freezes:
            for blk in model.blocks:
                for p in list(blk.gn1.parameters()) + list(blk.gn2.parameters()):
                    p.grad = None
        if "shortcut" in self.freezes:
            for blk in model.blocks:
                if not isinstance(blk.shortcut, nn.Identity):
                    for p in blk.shortcut.parameters():
                        p.grad = None
        if "ctrl" in self.freezes:
            for p in model.task_ctrl.parameters():
                p.grad = None


# ---------------------------------------------------------------------------
# 3. TRAIN / EVAL
# ---------------------------------------------------------------------------
def train_task(model, loader, device, task, epochs, lr, policy=None, verbose=True):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(1, epochs + 1):
        tot, corr, n = 0.0, 0, 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, alphas = model(x, task=task, tau=0.8)
            loss = F.cross_entropy(logits, y)
            eps = 1e-8
            mean_alpha = alphas.mean(dim=0)
            loss = loss + 0.01 * (mean_alpha * torch.log(mean_alpha + eps)).sum()
            loss.backward()
            if policy is not None:
                policy.apply(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * y.size(0)
            corr += (logits.argmax(-1) == y).sum().item()
            n += y.size(0)
        if verbose:
            print(f"        epoch {ep}/{epochs} [{task}] "
                  f"loss {tot / n:.4f} | acc {100.0 * corr / n:.2f}%")
    return opt


@torch.no_grad()
def evaluate(model, loader, device, task):
    model.eval()
    corr, n = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x, task=task, tau=0.5)
        corr += (logits.argmax(-1) == y).sum().item()
        n += y.size(0)
    return 100.0 * corr / n


# ---------------------------------------------------------------------------
# 4. DIAGNOSTIC: HOW MUCH OF THE KERNEL REALLY MOVED
# ---------------------------------------------------------------------------
@torch.no_grad()
def kernel_drift(model_before, model_after, loader, device, task, subspaces,
                 n_batches=3):
    """For each block, fraction of the effective-kernel change that lies in
    the protected subspace U, plus the same quantity for the basis-only change
    (alpha held at its pre-task-2 value).  alpha is the path OGP never sees.
    """
    model_before.eval()
    model_after.eval()
    acc = {}

    def add(idx, key, val):
        acc.setdefault(idx, {}).setdefault(key, []).append(val)

    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(device)
        _, ab = model_before(x, task=task, tau=0.5)
        _, aa = model_after(x, task=task, tau=0.5)
        for idx in range(model_after.n_blocks):
            kname = model_after.block_kernels[idx]
            kb = model_before.basis_sys.get_kernel(kname, ab[:, idx])
            ka = model_after.basis_sys.get_kernel(kname, aa[:, idx])
            # same alphas, so this isolates the basis parameters
            kb_same_alpha = model_after.basis_sys.get_kernel(kname, ab[:, idx])
            U = subspaces.get(f"block_{idx}")
            if U is None:
                continue

            def frac_in_U(d):
                df = d.reshape(d.shape[0], d.shape[1], -1)
                du = (df @ U) @ U.t()
                num = du.flatten(1).norm(dim=1)
                den = df.flatten(1).norm(dim=1).clamp_min(1e-12)
                return (num / den).mean().item()

            def rel_size(d, ref):
                return (d.flatten(1).norm(dim=1).mean()
                        / ref.flatten(1).norm(dim=1).mean().clamp_min(1e-12)).item()

            add(idx, "dW_rel", rel_size(ka - kb, kb))
            add(idx, "fracU_dW", frac_in_U(ka - kb))
            add(idx, "fracU_basis_only", frac_in_U(kb_same_alpha - kb))
            add(idx, "dW_basis_only_rel", rel_size(kb_same_alpha - kb, kb))

    out = {}
    for idx, d in acc.items():
        out[idx] = {k: sum(v) / len(v) for k, v in d.items()}
    return out


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-task1", type=float, default=1.5e-3)
    ap.add_argument("--lr-task2", type=float, default=1.2e-3)
    ap.add_argument("--threshold", type=float, default=0.98,
                    help="variance kept for the method rows")
    ap.add_argument("--thresholds", type=str, default="",
                    help="comma list -> additionally sweep ogp_fixed over these")
    ap.add_argument("--methods", type=str,
                    default="naive,snapshot,ogp_wrong,ogp_fixed,random_proj,"
                            "freeze_basis,ogp_fixed_locked")
    ap.add_argument("--dims", type=str, default="24,48,96")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-batches", type=int, default=15)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    d1, d2, d3 = (int(v) for v in args.dims.split(","))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print(f" BASIS + OGP vs PER-TASK SNAPSHOT | device={device} | dims={d1},{d2},{d3} "
          f"| K={args.experts} | epochs={args.epochs}")
    print(" Protocol: MNIST -> Fashion-MNIST, task-IL ORACLE (task id given at eval)")
    print("=" * 100)

    data_root = os.path.join(ROOT, "data")
    tr_m, te_m = get_task_data("mnist", data_root, fast_cache=True, batch_size=args.batch_size)
    tr_f, te_f = get_task_data("fashion", data_root, fast_cache=True, batch_size=args.batch_size)

    def new_model():
        return HtSContinualModel(d1=d1, d2=d2, d3=d3, K=args.experts, num_classes=10,
                                 supported_tasks=["mnist", "fashion"]).to(device)

    # ---------------- task 1 ----------------
    base = new_model()
    print("\n[task 1] MNIST")
    train_task(base, tr_m, device, "mnist", args.epochs, args.lr_task1)
    acc_m_init = evaluate(base, te_m, device, "mnist")
    print(f"  A[1,1] MNIST after task 1: {acc_m_init:.2f}%")

    live_bytes = state_bytes(base)
    stem_head_bytes = state_bytes(base.stems) + state_bytes(base.heads)
    print(f"  live model state: {mib(live_bytes):.3f} MiB "
          f"(backbone {mib(live_bytes - stem_head_bytes):.3f} + per-task stubs "
          f"{mib(stem_head_bytes):.3f})")

    # subspaces collected once from the task-1 model, as in the original
    sub_fixed, ranks_fixed = collect_subspaces(base, tr_m, device, "mnist",
                                               args.threshold, args.max_batches,
                                               source="conv_input")
    sub_wrong, ranks_wrong = collect_subspaces(base, tr_m, device, "mnist",
                                               args.threshold, args.max_batches,
                                               source="block_input")
    sub_rand = random_subspaces_like(sub_fixed, seed=0)
    print(f"  subspace rank (conv-input U): {list(ranks_fixed.values())} "
          f"-> {mib(subspace_bytes(sub_fixed)):.3f} MiB")

    results = []          # dicts
    drifts = {}

    # ---------------- reference: no protection, no storage ----------------
    if "naive" in methods:
        print("\n[naive] no protection")
        m = copy.deepcopy(base)
        train_task(m, tr_f, device, "fashion", args.epochs, args.lr_task2)
        am, af = evaluate(m, te_m, device, "mnist"), evaluate(m, te_f, device, "fashion")
        results.append(dict(method="naive", extra_bytes=0, acc_m=am, acc_f=af,
                            forget=acc_m_init - am))
        drifts["naive"] = kernel_drift(base, m, tr_m, device, "mnist", sub_fixed)

    # ---------------- reference: the snapshot, actually measured -----------
    if "snapshot" in methods:
        print("\n[snapshot] full weight snapshot, MEASURED with a real save/load")
        m = copy.deepcopy(base)
        snap_path = os.path.join(os.environ.get("TEMP", "."), "aml_snap_probe.pt")
        torch.save(base.state_dict(), snap_path)          # snapshot taken after task 1
        snap_file = os.path.getsize(snap_path)
        train_task(m, tr_f, device, "fashion", args.epochs, args.lr_task2)
        af = evaluate(m, te_f, device, "fashion")
        m.load_state_dict(torch.load(snap_path, map_location=device))  # serve task 1
        am = evaluate(m, te_m, device, "mnist")
        os.remove(snap_path)
        results.append(dict(method="snapshot", extra_bytes=snap_file, acc_m=am, acc_f=af,
                            forget=acc_m_init - am))
        print(f"  snapshot file on disk: {mib(snap_file):.3f} MiB")

    # ---------------- the methods under test -------------------------------
    plan = []
    if "ogp_wrong" in methods:
        plan.append(("ogp_wrong", "ogp", sub_wrong, ()))
    if "ogp_fixed" in methods:
        plan.append(("ogp_fixed", "ogp", sub_fixed, ()))
    if "random_proj" in methods:
        plan.append(("random_proj", "ogp", sub_rand, ()))
    if "freeze_basis" in methods:
        plan.append(("freeze_basis", "freeze_basis", None, ()))
    if "ogp_fixed_locked" in methods:
        plan.append(("ogp_fixed_locked", "ogp", sub_fixed, ("norm", "shortcut", "ctrl")))

    for label, kind, subs, freezes in plan:
        print(f"\n[{label}] subspace={'yes' if subs else 'no'} freezes={freezes or '-'}")
        m = copy.deepcopy(base)
        policy = GradPolicy(kind=kind, subspaces=subs, freezes=freezes)
        train_task(m, tr_f, device, "fashion", args.epochs, args.lr_task2, policy=policy)
        am, af = evaluate(m, te_m, device, "mnist"), evaluate(m, te_f, device, "fashion")
        extra = subspace_bytes(subs) if subs else 0
        results.append(dict(method=label, extra_bytes=extra, acc_m=am, acc_f=af,
                            forget=acc_m_init - am))
        if subs is not None:
            drifts[label] = kernel_drift(base, m, tr_m, device, "mnist", subs)

    # ---------------- threshold sweep --------------------------------------
    sweep = []
    if args.thresholds:
        for t in [float(v) for v in args.thresholds.split(",")]:
            print(f"\n[sweep] ogp_fixed @ threshold {t}")
            subs, ranks = collect_subspaces(base, tr_m, device, "mnist", t,
                                            args.max_batches, source="conv_input")
            m = copy.deepcopy(base)
            train_task(m, tr_f, device, "fashion", args.epochs, args.lr_task2,
                       policy=GradPolicy("ogp", subs), verbose=False)
            am = evaluate(m, te_m, device, "mnist")
            af = evaluate(m, te_f, device, "fashion")
            sweep.append(dict(threshold=t, extra_bytes=subspace_bytes(subs),
                              ranks=list(ranks.values()), acc_m=am, acc_f=af,
                              forget=acc_m_init - am))
            print(f"  ranks {list(ranks.values())} | extra {mib(subspace_bytes(subs)):.3f} MiB "
                  f"| MNIST {am:.2f}% | FASHION {af:.2f}%")

    # ---------------- report ----------------------------------------------
    print("\n" + "=" * 100)
    print(" RESULTS  (task-IL oracle; storage = bytes that must persist per past task)")
    print("=" * 100)
    hdr = (f"{'method':<18} | {'extra/state':>11} | {'MNIST A[2,1]':>12} | "
           f"{'FASHION A[2,2]':>14} | {'forgetting':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['method']:<18} | {mib(r['extra_bytes']):>8.3f} MiB | "
              f"{r['acc_m']:>11.2f}% | {r['acc_f']:>13.2f}% | {r['forget']:>9.2f}%")
    print("-" * len(hdr))
    print(f"{'reference: initial':<18} | {'-':>11} | {acc_m_init:>11.2f}% | "
          f"{'-':>13} | {'-':>10}")
    print(f"{'reference: full-rank':<18} | "
          f"{mib(sum(u.shape[0] ** 2 * 4 for u in sub_fixed.values())):>8.3f} MiB | "
          f"{'(cost only)':>12} | {'-':>13} | {'-':>10}")
    print("  NOTE: a full-rank basis is NOT exact here. The rank clamp "
          "k = min(k, D_in - 1) makes exact preservation unreachable by\n"
          "  construction, so this row is a COST reference only. The measured "
          "threshold=1.0 row of the sweep below is the honest figure at that cost.")

    if sweep:
        print("\n THRESHOLD SWEEP (ogp_fixed)")
        print(f"{'threshold':>9} | {'ranks':>26} | {'extra':>11} | {'MNIST':>7} | "
              f"{'FASHION':>8} | {'forget':>7}")
        print("-" * 90)
        for s in sweep:
            print(f"{s['threshold']:>9.3f} | {str(s['ranks']):>26} | "
                  f"{mib(s['extra_bytes']):>8.3f} MiB | {s['acc_m']:>6.2f}% | "
                  f"{s['acc_f']:>7.2f}% | {s['forget']:>6.2f}%")

    if drifts:
        print("\n KERNEL DRIFT DIAGNOSTIC  (mean over blocks, after task 2)")
        print(f"{'method':<18} | {'|dW|/|W|':>9} | {'|dW U|/|dW|':>12} | "
              f"{'basis-only |dW U|/|dW|':>25}")
        print("-" * 78)
        for k, d in drifts.items():
            dw = sum(v["dW_rel"] for v in d.values()) / len(d)
            fu = sum(v["fracU_dW"] for v in d.values()) / len(d)
            fb = sum(v["fracU_basis_only"] for v in d.values()) / len(d)
            print(f"{k:<18} | {dw:>9.4f} | {fu:>11.4f} | {fb:>25.4f}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(dict(acc_m_init=acc_m_init, live_bytes=live_bytes,
                           stem_head_bytes=stem_head_bytes,
                           subspace_bytes=subspace_bytes(sub_fixed),
                           ranks=ranks_fixed,
                           exact_bytes=sum(u.shape[0] ** 2 * 4 for u in sub_fixed.values()),
                           results=results, sweep=sweep, drift=drifts), fh, indent=2)
        print(f"\n[json] {args.json_out}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[done] {time.time() - t0:.1f}s")
