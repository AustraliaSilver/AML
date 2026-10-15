"""
experiments/exp_cheap_constraint.py
===================================
Storage-matched comparison of orthogonal-constraint mechanisms.

The question is NOT "does OGP work at a 98% variance threshold".  It is:

    can an orthogonal constraint be made cheap enough to beat the mechanisms
    the framework already ships -- 3A (`--shared-backbone`, 1.355 MB/task) and
    EWC (two trunk-sized accumulators, O(1) in the number of tasks)?

Two levers control the cost of the constraint, and neither was explored:

  1. RANK CAP.  The constraint is U_k in R^{D_in x k} with D_in = C_in*9, so it
     costs  sum_b D_in,b * k_b * 4 bytes  per task.  The 98% threshold lands at
     k/D_in ~ 0.68, i.e. the constraint costs about as much as the kernel it
     protects.  A hard cap on k makes the cost linear and directly settable.

  2. int8 + QR.  U is orthonormal, so it quantises well: store round(U/s) as
     int8 (4x smaller), dequantise on load and re-orthonormalise with a
     deterministic QR.  Costs 1 byte per element instead of 4.

References measured in the same run, on the same model and data:

  * naive           - no retention at all (lower bound)
  * snapshot        - real torch.save / load_state_dict round trip (exact)
  * freeze          - basis gradient removed entirely: 0 extra bytes
  * EWC             - 2 trunk-sized accumulators, O(1) in T

Reported: bytes/task, forgetting, and the number of tasks T at which each
constraint overtakes EWC's fixed cost.
"""

import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from train_continual_aml import get_task_data, set_seed            # noqa: E402
from hts_lib.models.hypernetwork import HtSContinualModel           # noqa: E402
from exp_basis_ogp_corrected import GradPolicy, state_bytes           # noqa: E402


def mib_(n):
    return n / (1024 * 1024)


# ---------------------------------------------------------------------------
# full-rank basis, so the rank cap can be swept without recomputing the SVD
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_full_basis(model, loader, device, task="mnist", max_batches=15):
    """Top-D_in eigenvectors of the conv-input correlation, per block.

    U is returned at full width so any rank cap can be taken from it.
    """
    model.eval()
    corr, handles = {}, []

    def make_hook(name, stride):
        def hook(module, inp, outp):
            z = F.gelu(outp)                       # what conv2d(res, kernel) eats
            b, c, h, w = z.shape
            u = F.unfold(z, kernel_size=3, padding=1, stride=stride)
            u = u.transpose(1, 2).reshape(-1, c * 9)
            r = (u.t() @ u).double()
            corr[name] = r.cpu() if name not in corr else corr[name] + r.cpu()
        return hook

    for idx, blk in enumerate(model.blocks):
        handles.append(blk.gn1.register_forward_hook(make_hook(f"block_{idx}", blk.stride)))

    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        model(x.to(device), task=task, tau=0.5)
    for h in handles:
        h.remove()

    basis = {}
    for name, R in corr.items():
        R = 0.5 * (R + R.t())
        U, S, _ = torch.linalg.svd(R)
        basis[name] = (U.float().contiguous(), S.float().clamp_min(0.0))
    return basis


def cut(basis, rank):
    """[D_in, min(rank, D_in)] from the full basis."""
    return {n: U[:, :max(1, min(rank, U.shape[1]))].contiguous() for n, (U, _) in basis.items()}


def quantize_basis(subspaces):
    """int8 storage + deterministic QR re-orthonormalisation.

    Returns (orthonormal U on the original device, stored_bytes).
    """
    out, nbytes = {}, 0
    for name, U in subspaces.items():
        s = U.abs().max().clamp_min(1e-12) / 127.0
        q = torch.round(U / s).to(torch.int8)
        nbytes += q.numel() * 1 + 4                      # int8 codes + one fp32 scale
        Q, _ = torch.linalg.qr(q.float().to(U.device) * s)
        out[name] = Q[:, :U.shape[1]].contiguous()
    return out, nbytes


def subspace_bytes(subspaces, quantized):
    if quantized:
        return sum(u.shape[0] * u.shape[1] + 4 for u in subspaces.values())
    return sum(u.numel() * 4 for u in subspaces.values())


# ---------------------------------------------------------------------------
# EWC on the HtS trunk (same quadratic AML's FisherEWC implements)
# ---------------------------------------------------------------------------
class TrunkEWC:
    def __init__(self, model, lam):
        self.model = model
        self.lam = lam
        self.enabled = lam > 0.0
        self.F, self.n = {}, 0
        self.A, self.B = {}, {}

    @torch.no_grad()
    def observe(self):
        if not self.enabled:
            return
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            sq = p.grad.detach().float() ** 2
            self.F[n] = sq if n not in self.F else self.F[n] + sq
        self.n += 1

    @torch.no_grad()
    def consolidate(self):
        if not self.enabled or self.n == 0:
            return
        inv = 1.0 / float(self.n)
        params = dict(self.model.named_parameters())
        for n, f in self.F.items():
            f.mul_(inv)
            p = params[n]
            self.A[n] = f.clone() if n not in self.A else self.A[n] + f
            contrib = f * p.detach().float()
            self.B[n] = contrib if n not in self.B else self.B[n] + contrib
        self.F, self.n = {}, 0

    def penalty(self):
        if not self.enabled or not self.A:
            dev = next(self.model.parameters()).device
            return torch.zeros((), device=dev)
        params = dict(self.model.named_parameters())
        total = None
        for n, a in self.A.items():
            p = params[n]
            mu = self.B[n] / a.clamp_min(1e-12)
            d = (a * (p - mu) ** 2).sum()
            total = d if total is None else total + d
        return 0.5 * self.lam * total

    def state_bytes(self):
        return 4 * (sum(v.numel() for v in self.A.values())
                    + sum(v.numel() for v in self.B.values()))


# ---------------------------------------------------------------------------
# train / eval (EWC-aware)
# ---------------------------------------------------------------------------
def train_task(model, loader, device, task, epochs, lr, policy=None,
               ewc=None, verbose=True):
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
            ma = alphas.mean(dim=0)
            loss = loss + 0.01 * (ma * torch.log(ma + eps)).sum()
            if ewc is not None and ewc.A:
                loss = loss + ewc.penalty()
            loss.backward()
            if ewc is not None:
                ewc.observe()
            if policy is not None:
                policy.apply(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * y.size(0)
            corr += (logits.argmax(-1) == y).sum().item()
            n += y.size(0)
        if verbose:
            print(f"        epoch {ep}/{epochs} [{task}] loss {tot / n:.4f} "
                  f"| acc {100.0 * corr / n:.2f}%")
    if ewc is not None:
        ewc.consolidate()


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
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-task1", type=float, default=1.5e-3)
    ap.add_argument("--lr-task2", type=float, default=1.2e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-batches", type=int, default=15)
    ap.add_argument("--ranks", type=str, default="8,32,128,286")
    ap.add_argument("--quant-ranks", type=str, default="32,128")
    ap.add_argument("--ewc-lambdas", type=str, default="100,1000")
    ap.add_argument("--dims", type=str, default="24,48,96")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    d1, d2, d3 = (int(v) for v in args.dims.split(","))
    ranks = [int(v) for v in args.ranks.split(",") if v.strip()]
    qranks = [int(v) for v in args.quant_ranks.split(",") if v.strip()]
    lambdas = [float(v) for v in args.ewc_lambdas.split(",") if v.strip()]

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 104)
    print(f" CHEAP CONSTRAINT vs EWC vs SNAPSHOT | device={device} | epochs={args.epochs}")
    print(" Protocol: MNIST -> Fashion-MNIST, task-IL ORACLE (task id given at eval)")
    print("=" * 104)

    data_root = os.path.join(ROOT, "data")
    tr_m, te_m = get_task_data("mnist", data_root, fast_cache=True, batch_size=args.batch_size)
    tr_f, te_f = get_task_data("fashion", data_root, fast_cache=True, batch_size=args.batch_size)

    def new_model():
        return HtSContinualModel(d1=d1, d2=d2, d3=d3, K=args.experts, num_classes=10,
                                 supported_tasks=["mnist", "fashion"]).to(device)

    # ---- task 1 ----
    base = new_model()
    trunk_ewc = TrunkEWC(base, lam=0.0)          # observe-only during task 1
    trunk_ewc.enabled = True
    print("\n[task 1] MNIST")
    train_task(base, tr_m, device, "mnist", args.epochs, args.lr_task1, ewc=trunk_ewc)
    acc_m_init = evaluate(base, te_m, device, "mnist")
    print(f"  A[1,1] MNIST after task 1: {acc_m_init:.2f}%")
    print(f"  EWC accumulators after task 1: {mib_(trunk_ewc.state_bytes()):.3f} MiB "
          f"(O(1) in T)")

    live = state_bytes(base)
    basis = collect_full_basis(base, tr_m, device, "mnist", args.max_batches)
    print(f"  basis collected for {len(basis)} blocks; D_in = "
          f"{[U.shape[0] for U, _ in basis.values()]}")

    results = []
    ewc_bytes = trunk_ewc.state_bytes()

    def run(label, policy=None, subspaces=None, quantized=False, ewc_lam=0.0,
            extra=0, note=""):
        m = copy.deepcopy(base)
        e = None
        if ewc_lam > 0:
            e = TrunkEWC(m, ewc_lam)
            # carry task 1's Fisher over: replay it onto the copy
            e.A = {k: v.clone() for k, v in trunk_ewc.A.items()}
            e.B = {k: v.clone() for k, v in trunk_ewc.B.items()}
        print(f"\n[{label}]")
        train_task(m, tr_f, device, "fashion", args.epochs, args.lr_task2,
                   policy=policy, ewc=e)
        am = evaluate(m, te_m, device, "mnist")
        af = evaluate(m, te_f, device, "fashion")
        per_task = subspace_bytes(subspaces, quantized) if subspaces else 0
        row = dict(method=label, per_task=per_task, fixed=extra, acc_m=am, acc_f=af,
                   forget=acc_m_init - am, note=note)
        results.append(row)
        print(f"  MNIST {am:.2f}% | FASHION {af:.2f}% | forget {row['forget']:.2f}% "
              f"| bytes/task {mib_(per_task):.3f} MiB" + (f" | fixed {mib_(extra):.3f} MiB" if extra else ""))
        return row

    # ---- references ----
    run("naive")
    m = copy.deepcopy(base)
    snap_path = os.path.join(os.environ.get("TEMP", "."), "aml_snap_cheap.pt")
    torch.save(base.state_dict(), snap_path)
    snap_file = os.path.getsize(snap_path)
    train_task(m, tr_f, device, "fashion", args.epochs, args.lr_task2)
    af = evaluate(m, te_f, device, "fashion")
    m.load_state_dict(torch.load(snap_path, map_location=device))
    am = evaluate(m, te_m, device, "mnist")
    os.remove(snap_path)
    results.append(dict(method="snapshot", per_task=snap_file, fixed=0, acc_m=am,
                        acc_f=af, forget=acc_m_init - am, note="exact"))
    print(f"\n[snapshot] measured save/load | MNIST {am:.2f}% | FASHION {af:.2f}% "
          f"| bytes/task {mib_(snap_file):.3f} MiB")
    run("freeze", policy=GradPolicy("freeze_basis"), note="0 extra bytes")

    # ---- EWC ----
    for lam in lambdas:
        run(f"ewc_l{lam:g}", ewc_lam=lam, extra=ewc_bytes, note="O(1) in T")

    # ---- constrained, rank sweep ----
    for r in ranks:
        sub = cut(basis, r)
        run(f"ogp_r{r}", policy=GradPolicy("ogp", sub), subspaces=sub,
            note=f"k=min({r},D_in)")

    # ---- constrained, int8 + QR ----
    for r in qranks:
        sub = cut(basis, r)
        qsub, qbytes = quantize_basis(sub)
        m = copy.deepcopy(base)
        print(f"\n[ogp_r{r}_q8]")
        train_task(m, tr_f, device, "fashion", args.epochs, args.lr_task2,
                   policy=GradPolicy("ogp", qsub))
        am = evaluate(m, te_m, device, "mnist")
        af = evaluate(m, te_f, device, "fashion")
        results.append(dict(method=f"ogp_r{r}_q8", per_task=qbytes, fixed=0, acc_m=am,
                            acc_f=af, forget=acc_m_init - am, note="int8+QR"))
        print(f"  MNIST {am:.2f}% | FASHION {af:.2f}% | forget {acc_m_init - am:.2f}% "
              f"| bytes/task {mib_(qbytes):.3f} MiB")

    # ---- report ----
    print("\n" + "=" * 104)
    print(" STORAGE-MATCHED TABLE  (task-IL oracle; extra bytes = state that must "
          "persist for each PAST task)")
    print("=" * 104)
    hdr = (f"{'method':<16} | {'bytes/task':>12} | {'fixed':>10} | {'MNIST':>8} | "
           f"{'FASHION':>8} | {'forget':>8} | note")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda d: (d["per_task"] + d["fixed"])):
        print(f"{r['method']:<16} | {mib_(r['per_task']):>9.3f} MiB | "
              f"{mib_(r['fixed']):>7.3f} MiB | {r['acc_m']:>7.2f}% | "
              f"{r['acc_f']:>7.2f}% | {r['forget']:>7.2f}% | {r['note']}")

    # ---- crossover vs EWC ----
    ewc_rows = [r for r in results if r["method"].startswith("ewc_")]
    ewc_best = min(ewc_rows, key=lambda d: d["forget"]) if ewc_rows else None
    print("\n" + "=" * 104)
    print(" CROSSOVER vs EWC  (T = number of tasks at which a per-task cost overtakes "
          "EWC's fixed cost)")
    print("=" * 104)
    if ewc_best:
        print(f" best EWC: {ewc_best['method']}  forget {ewc_best['forget']:.2f}%  "
              f"fixed {mib_(ewc_best['fixed']):.3f} MiB (O(1) in T)")
        print(f" {'method':<16} | {'forget':>7} | {'bytes/task':>11} | "
              f"{'T_overtake':>10} | verdict")
        print(" " + "-" * 82)
        for r in sorted(results, key=lambda d: d["per_task"]):
            if r["per_task"] <= 0:
                continue
            thr = ewc_best["fixed"] / r["per_task"]
            if r["forget"] <= ewc_best["forget"]:
                verdict = "better retention; storage overtakes EWC past that T"
            else:
                verdict = "DOMINATED (worse retention AND more bytes past that T)"
            print(f" {r['method']:<16} | {r['forget']:>6.2f}% | "
                  f"{mib_(r['per_task']):>8.3f} MiB | {thr:>10.1f} | {verdict}")

    print(f"\n live model {mib_(live):.3f} MiB | full-rank basis (what the 98% "
          f"threshold effectively costs) "
          f"{mib_(sum(u.numel() * 4 for u, _ in basis.values())):.3f} MiB")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(dict(acc_m_init=acc_m_init, live_bytes=live,
                           ewc_bytes=ewc_bytes, full_basis_bytes=sum(u.numel() * 4 for u, _ in basis.values()),
                           D_in=[U.shape[0] for U, _ in basis.values()],
                           results=results), fh, indent=2)
        print(f"[json] {args.json_out}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[done] {time.time() - t0:.1f}s")
