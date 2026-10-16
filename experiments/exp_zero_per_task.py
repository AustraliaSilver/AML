"""
experiments/exp_zero_per_task.py
================================
Can per-task storage be **0 bytes**?

The framework currently pays 7.209 MB per past task, 81 % of which is a copy of
the 18-layer trunk.  ``--shared-backbone 1`` (mode 3A) already removes that copy,
but still snapshots the encoder, the prototype selector and the prototypes, so it
lands at 1.355 MB/task -- not zero.

There is a structural reason this is hard, and it is worth stating before the
numbers:

    Forgetting is caused by the shared trunk drifting.
    Undoing drift requires knowing WHAT to preserve.
    That knowledge is either per-task (-> bytes per task) or stored once over
    the shared parameters (-> bytes that do NOT grow with T).

A per-task object such as an OGP basis U_k (D_in x k per block) is therefore
*structurally incompatible* with a 0-byte/task target: it is per-task by
construction, and it was measured at 4.20 MB/task at a 98 % threshold.  The only
two shapes of mechanism that can hit 0 B/task are:

    (a) freeze  -- store nothing, accept lost plasticity;
    (b) one O(1) anchor over the shared weights -- EWC-style, or a frozen-weight
        mask.  A mask releases the values it protects IF AND ONLY IF the
        decoupled weight decay is masked along with the gradient: with grad == 0
        and no decay the optimizer cannot write those elements, so only the
        SELECTION is carried (4 bytes for a seed, one bit per weight for
        Fisher).  Restoring them from a saved copy -- which is what AdamW's
        global decay forces -- silently costs frac x |basis| instead, i.e. a
        snapshot rather than a selection.  This script takes the honest path
        (see build_mask/train_task) and the difference is easy to get wrong.

This script measures (a), (b) and their variants in a test bed where the
per-task byte count is EXACTLY zero by construction:

    one shared stem + one shared 20-wide head, MNIST -> Fashion-MNIST, labels
    0..9 and 10..19, class-range cross-entropy (so the rows of the previous
    task receive no gradient at all and cannot drift).

Mechanisms, all with 0 bytes/task:

    naive          no protection at all (lower bound)
    freeze_all     every parameter frozen after task 1 (upper bound on retention,
                   lower bound on plasticity)
    freeze_trunk   basis + controller frozen, norms/head free
    ewc_l*         quadratic anchor over the shared weights (O(1) in T)
    ewc_int8       the same anchor quantised to int8 (0.5x the bytes)
    mask_<sel>_p   freeze a fraction p of the basis weights.  The decoupled
                   weight decay is masked together with the gradient, so the
                   frozen VALUES are never written and only the SELECTION is
                   stored: 4 bytes if it is a random seed, or one bit per weight
                   if it is chosen by Fisher importance.  (The reported byte
                   count is the size the selection is PERSISTED at -- a seed, or
                   a packed bitmask -- not the unpacked in-RAM bool tensor.)

Reported: MNIST retention, Fashion plasticity, and the fixed O(1) bytes that
buy them -- with the snapshot cost quoted alongside for contrast.
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

from train_continual_aml import get_task_data, set_seed                # noqa: E402
from hts_lib.models.hypernetwork import HtSContinualModel              # noqa: E402


def mib(n):
    return n / (1024 * 1024)


def human(n):
    if n == 0:
        return "0"
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{mib(n):.3f} MiB"


# ---------------------------------------------------------------------------
# data: two domains, one shared label space of width 20
# ---------------------------------------------------------------------------
class OffsetLoader:
    """Adds a constant to the labels of an existing DataLoader.

    With ``gray=True`` it also collapses 3-channel inputs to 1 channel, so a
    colour dataset can drive the same 1-channel shared stem.
    """

    def __init__(self, loader, offset, gray=False):
        self.loader = loader
        self.offset = offset
        self.gray = gray

    def __iter__(self):
        for x, y in self.loader:
            if self.gray and x.shape[1] == 3:
                x = x.mean(dim=1, keepdim=True)
            yield x, y + self.offset

    def __len__(self):
        return len(self.loader)


# ---------------------------------------------------------------------------
# EWC over the shared weights -- stores (A, mu), i.e. the diagonal Fisher and
# the anchor point it is centred on.  O(1) in the number of tasks.
# ---------------------------------------------------------------------------
class SharedEWC:
    def __init__(self, model, lam=0.0, quantize=False):
        self.model = model
        self.lam = lam
        self.quantize = quantize
        self.enabled = lam > 0.0
        self.Acc, self.Bcc, self.n = {}, {}, 0
        self.A, self.mu = {}, {}
        self.locked = False          # once loaded, never move the anchor again

    @torch.no_grad()
    def observe(self):
        if not self.enabled:
            return
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            sq = p.grad.detach().float() ** 2
            self.Acc[n] = sq if n not in self.Acc else self.Acc[n] + sq
        self.n += 1

    @torch.no_grad()
    def consolidate(self):
        if not self.enabled or self.n == 0 or self.locked:
            self.Acc, self.n = {}, 0
            return
        inv = 1.0 / float(self.n)
        params = dict(self.model.named_parameters())
        for n, acc in self.Acc.items():
            a = acc * inv
            p = params[n].detach().float()
            if n in self.A:                       # accumulate across tasks
                a = self.A[n] + a
            self.A[n] = a
            self.mu[n] = p.clone()
        self.Acc, self.n = {}, 0
        if self.quantize:
            for n in list(self.A):
                self.A[n], self.mu[n] = self._q(self.A[n]), self._q(self.mu[n])

    @staticmethod
    def _q(t):
        s = t.abs().max().clamp_min(1e-12) / 127.0
        return torch.round(t / s).to(torch.int8), s

    def penalty(self):
        if not self.enabled or not self.A:
            return torch.zeros((), device=next(self.model.parameters()).device)
        params = dict(self.model.named_parameters())
        total = None
        for n, a in self.A.items():
            if self.quantize:
                a, sa = a
                mu, sm = self.mu[n]
                a, mu = a.float() * sa, mu.float() * sm
            else:
                mu = self.mu[n]
            d = (a * (params[n].float() - mu) ** 2).sum()
            total = d if total is None else total + d
        return 0.5 * self.lam * total

    def state_bytes(self):
        tot = 0
        for n, a in self.A.items():
            if self.quantize:
                tot += a[0].numel() * 1 + 4          # int8 codes + scale
                tot += self.mu[n][0].numel() * 1 + 4
            else:
                tot += a.numel() * 4 + self.mu[n].numel() * 4
        return tot


# ---------------------------------------------------------------------------
# frozen-weight mask: the decay is masked too, so the frozen values are never
# written and only the selection is stored (4-byte seed, or one bit per weight)
# ---------------------------------------------------------------------------
def build_mask(model, frac, select, fisher=None, seed=0):
    """Selection of the fraction `frac` of the basis weights to freeze.

    Only the SELECTION is returned; the values are pinned at update time by
    masking the gradient and the decoupled decay, so no copy of them is held.
    Returning a restore copy instead would cost frac x |basis| here -- exactly
    the difference between a selection and a snapshot.
    """
    frozen = {}
    gen = torch.Generator().manual_seed(seed)
    nbits = 0
    for n, p in model.named_parameters():
        if "basis_sys" not in n:
            continue
        numel = p.numel()
        k = int(round(numel * frac))
        if k <= 0:
            continue
        if select == "random":
            idx = torch.randperm(numel, generator=gen)[:k]
        else:
            f = fisher[n].flatten() if fisher and n in fisher else torch.ones(numel)
            idx = torch.topk(f, k).indices
        m = torch.zeros(numel, dtype=torch.bool)
        m[idx] = True
        frozen[n] = m.view(p.shape).to(p.device)
        nbits += numel
    # random selection -> a 4-byte seed; Fisher selection -> one bit per weight
    nbytes = 4 if select == "random" else (nbits + 7) // 8
    return {"frozen": frozen}, nbytes


# ---------------------------------------------------------------------------
# train / evaluate
# ---------------------------------------------------------------------------
def range_ce(logits, y, lo, hi):
    """Cross-entropy restricted to the task's own class range.

    Rows outside [lo, hi) receive no gradient, which is exactly the property
    that makes a shared head cost 0 bytes of per-task protection.
    """
    return F.cross_entropy(logits[:, lo:hi], y - lo)


def train_task(model, loader, device, lo, hi, epochs, lr,
               ewc=None, mask=None, verbose=True, wd=1e-4):
    model.train()
    # AdamW's decoupled decay multiplies EVERY element of a tensor, so a frozen
    # element would move and would then have to be restored from a saved copy --
    # storage the mask is supposed to avoid.  The decay is applied by hand
    # instead, with a factor of EXACTLY 1.0 on frozen elements (x * 1.0 is
    # exact), so those values are never written and the mask costs a selection
    # rather than a copy.  Multiplying by (1 - lr*wd) before opt.step() is
    # AdamW's own order, so every non-masked row is unchanged.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    frozen = mask["frozen"] if mask else None
    decay = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if frozen is not None and n in frozen:
            keep = torch.ones_like(p.data)
            move = torch.full_like(p.data, 1.0 - lr * wd)
            decay[n] = torch.where(frozen[n], keep, move)
        else:
            decay[n] = 1.0 - lr * wd
    for ep in range(1, epochs + 1):
        tot, corr, n = 0.0, 0, 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, alphas = model(x, task="mnist", tau=0.8)
            loss = range_ce(logits, y, lo, hi)
            ma = alphas.mean(dim=0)
            loss = loss + 0.01 * (ma * torch.log(ma + 1e-8)).sum()
            if ewc is not None and ewc.A:
                loss = loss + ewc.penalty()
            loss.backward()
            if ewc is not None:
                ewc.observe()
            if frozen is not None:
                for nm, p in model.named_parameters():
                    if nm in frozen and p.grad is not None:
                        p.grad[frozen[nm]] = 0.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            with torch.no_grad():
                # decoupled decay, before the Adam update (AdamW's own order)
                for nm, p in model.named_parameters():
                    if nm in decay:
                        p.data.mul_(decay[nm])
            opt.step()
            tot += loss.item() * y.size(0)
            corr += (logits[:, lo:hi].argmax(-1) == (y - lo)).sum().item()
            n += y.size(0)
        if verbose:
            print(f"        epoch {ep}/{epochs} loss {tot / n:.4f} | acc {100.0 * corr / n:.2f}%")
    if ewc is not None:
        ewc.consolidate()


def self_check(verbose=True):
    """Verify the invariant the mask's byte accounting rests on (CPU only).

    AdamW's global decay multiplies every element, so a masked element would
    move and would have to be restored from a saved copy -- a copy, not a
    selection.  Masking the decay too (factor exactly 1.0 on frozen elements)
    leaves them untouched.  On a tiny tensor, with the same weights and the same
    gradient sequence for every variant:

      ref        AdamW(weight_decay=1e-4)                    (old behaviour)
      manual     AdamW(wd=0) + p *= (1 - lr*wd), unmasked    (must be identical)
      naive      ... applied to the masked tensor as well    (values move)
      honest     ... with a 1.0 factor where frozen          (values never move)

    Returns 0 if every claim holds, 1 otherwise.
    """
    steps, lr, wd = 8, 1e-3, 1e-4
    gen = torch.Generator().manual_seed(7)
    grads = [torch.randn(64, 7, generator=gen) for _ in range(steps)]
    mask = torch.zeros(64, 7, dtype=torch.bool)
    sel = torch.randperm(64 * 7, generator=torch.Generator().manual_seed(3))
    mask.view(-1)[sel[: int(0.9 * 64 * 7)]] = True

    def run(mode):
        torch.manual_seed(1)
        p = torch.nn.Parameter(torch.randn(64, 7))
        init = p.detach().clone()
        opt = torch.optim.AdamW([p], lr=lr, weight_decay=wd if mode == "ref" else 0.0)
        for g in grads:
            p.grad = g.clone()
            if mode in ("naive", "honest"):
                p.grad[mask] = 0.0
            if mode != "ref":
                with torch.no_grad():
                    if mode == "honest":
                        p.data.mul_(torch.where(mask, torch.ones_like(p.data),
                                                torch.full_like(p.data, 1.0 - lr * wd)))
                    else:
                        p.data.mul_(1.0 - lr * wd)
            opt.step()
        return p.detach().clone(), init

    ref, init = run("ref")
    manual, _ = run("manual")
    naive, _ = run("naive")
    honest, _ = run("honest")
    checks = [
        ("manual decay is bit-identical to AdamW's own decay", torch.equal(ref, manual)),
        ("frozen values are never written (bit-equal to init)",
         torch.equal(honest[mask], init[mask])),
        ("free values match the unmasked reference exactly",
         torch.equal(honest[~mask], ref[~mask])),
        ("an unmasked decay would move the frozen values",
         not torch.equal(naive[mask], init[mask])),
    ]
    if verbose:
        for name, ok in checks:
            print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in checks) else 1


@torch.no_grad()
def evaluate(model, loader, device, lo, hi):
    model.eval()
    corr, n = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x, task="mnist", tau=0.5)
        corr += (logits[:, lo:hi].argmax(-1) == (y - lo)).sum().item()
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
    ap.add_argument("--ewc-lambdas", type=str, default="100,1000")
    ap.add_argument("--mask-fracs", type=str, default="0.5,0.9")
    ap.add_argument("--methods", type=str, default="all",
                    help="comma list of method labels to run, or 'all'. Task 1 and "
                         "the Fisher are always computed.")
    ap.add_argument("--task2", type=str, default="fashion",
                    choices=["fashion", "cifar10gray"],
                    help="fashion = easy pair (same modality/resolution); "
                         "cifar10gray = hard pair (colour, 32x32, natural images, "
                         "collapsed to 1 channel so the shared stem still applies)")
    ap.add_argument("--dims", type=str, default="24,48,96")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for every mechanism; run the script once per seed and "
                         "aggregate the JSONs to get error bars")
    ap.add_argument("--json-out", type=str, default="")
    ap.add_argument("--self-check", action="store_true",
                    help="run the CPU-only mask-accounting invariant check and exit")
    args = ap.parse_args()

    if args.self_check:
        print(" mask accounting self-check (no GPU needed)")
        return self_check()

    d1, d2, d3 = (int(v) for v in args.dims.split(","))
    lambdas = [float(v) for v in args.ewc_lambdas.split(",") if v.strip()]
    fracs = [float(v) for v in args.mask_fracs.split(",") if v.strip()]

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 104)
    print(f" ZERO BYTES PER TASK | device={device} | epochs={args.epochs} | seed={args.seed}")
    print(f" Shared stem + shared 20-wide head; MNIST 0..9 -> {args.task2} 10..19")
    print(" Per-task parameter count is 0 BY CONSTRUCTION (one stem, one head, one kernel set)")
    print("=" * 104)

    data_root = os.path.join(ROOT, "data")
    tr_m, te_m = get_task_data("mnist", data_root, fast_cache=True, batch_size=args.batch_size)
    if args.task2 == "fashion":
        tr_f, te_f = get_task_data("fashion", data_root, fast_cache=True,
                                   batch_size=args.batch_size)
        tr_f, te_f = OffsetLoader(tr_f, 10), OffsetLoader(te_f, 10)
    else:
        tr_f, te_f = get_task_data("cifar10", data_root, fast_cache=True,
                                   batch_size=args.batch_size)
        tr_f, te_f = OffsetLoader(tr_f, 10, gray=True), OffsetLoader(te_f, 10, gray=True)

    def new_model():
        return HtSContinualModel(d1=d1, d2=d2, d3=d3, K=args.experts, num_classes=20,
                                 supported_tasks=["mnist"]).to(device)

    base = new_model()
    n_per_task = 0
    print(f"\n params: total {sum(p.numel() for p in base.parameters()):,} "
          f"| per-task stubs {n_per_task} (shared stem + shared head)")

    # ---- task 1 ----
    fisher = SharedEWC(base, lam=1e-4)          # observe-only during task 1
    fisher.enabled = True
    print("\n[task 1] MNIST (classes 0..9)")
    train_task(base, tr_m, device, 0, 10, args.epochs, args.lr_task1, ewc=fisher)
    acc_m_init = evaluate(base, te_m, device, 0, 10)
    print(f"  MNIST after task 1: {acc_m_init:.2f}%")
    # train_task already consolidated, so the Fisher lives in .A (Acc is cleared)
    fisher_A = {k: v.clone() for k, v in fisher.A.items()}

    trunk_bytes = 4 * sum(v.numel() for v in fisher_A.values())
    print(f"  Fisher observed over {mib(trunk_bytes):.3f} MiB of shared weights")

    def load_anchor(obj):
        """Seed an EWC object with task 1's Fisher and freeze its anchor point."""
        obj.Acc = {k: v.clone() for k, v in fisher_A.items()}
        obj.n = 1
        obj.consolidate()
        obj.locked = True

    results = []

    def record(label, model, per_task=0, fixed=0, note=""):
        am = evaluate(model, te_m, device, 0, 10)
        af = evaluate(model, te_f, device, 10, 20)
        row = dict(method=label, per_task=per_task, fixed=fixed, acc_m=am,
                   acc_f=af, forget=acc_m_init - am, note=note)
        results.append(row)
        print(f"  MNIST {am:.2f}% | FASHION {af:.2f}% | forget {row['forget']:.2f}% "
              f"| bytes/task {per_task} | fixed {mib(fixed):.3f} MiB")
        return row

    def fresh():
        # Re-seed per mechanism: without this the global RNG state carries over
        # between runs and every row after the first inherits different Gumbel
        # noise, which is enough to move accuracies by a point or two.
        set_seed(args.seed)
        return copy.deepcopy(base)

    want = None if args.methods == "all" else {s.strip() for s in args.methods.split(",") if s.strip()}

    def on(name):
        return want is None or name in want

    # ---- naive ----
    if on("naive"):
        print("\n[naive] 0 bytes/task, 0 fixed")
        m = fresh()
        train_task(m, tr_f, device, 10, 20, args.epochs, args.lr_task2)
        record("naive", m, note="no protection")

    # ---- freeze_all ----
    if on("freeze_all"):
        print("\n[freeze_all] 0 bytes/task, 0 fixed")
        record("freeze_all", fresh(), note="nothing can learn task 2 at all")

    # ---- freeze_trunk (basis + controller frozen, norms/head free) ----
    if on("freeze_trunk"):
        print("\n[freeze_trunk] 0 bytes/task, 0 fixed")
        m = fresh()
        for n, p in m.named_parameters():
            if "basis_sys" in n or "task_ctrl" in n:
                p.requires_grad_(False)
        train_task(m, tr_f, device, 10, 20, args.epochs, args.lr_task2)
        record("freeze_trunk", m, note="0 bytes, uses only norms/head plasticity")

    # ---- EWC (fp32 and int8) ----
    for lam in lambdas:
        if not on(f"ewc_l{lam:g}"):
            continue
        print(f"\n[ewc_l{lam:g}] 0 bytes/task, fixed = 2 x shared weights")
        m = fresh()
        e = SharedEWC(m, lam=lam)
        e.enabled = True
        load_anchor(e)
        train_task(m, tr_f, device, 10, 20, args.epochs, args.lr_task2, ewc=e)
        record(f"ewc_l{lam:g}", m, fixed=e.state_bytes(), note="O(1) in T")

    if on("ewc_int8"):
        print("\n[ewc_int8] 0 bytes/task, fixed = 0.5 x shared weights")
        m = fresh()
        e = SharedEWC(m, lam=1000.0, quantize=True)
        e.enabled = True
        load_anchor(e)
        train_task(m, tr_f, device, 10, 20, args.epochs, args.lr_task2, ewc=e)
        record("ewc_int8", m, fixed=e.state_bytes(), note="int8 anchor")

    # ---- frozen-weight masks ----
    for select in ("random", "fisher"):
        for frac in fracs:
            label = f"mask_{select}_p{frac:g}"
            if not on(label):
                continue
            print(f"\n[{label}] 0 bytes/task, fixed = selection only "
                  f"(decay masked too, so no copy of the values is held)")
            m = fresh()
            mask, mbytes = build_mask(m, frac, select, fisher=fisher_A, seed=args.seed)
            train_task(m, tr_f, device, 10, 20, args.epochs, args.lr_task2, mask=mask)
            record(label, m, fixed=mbytes,
                   note=f"freeze {frac:.0%} of basis; {'seed' if select == 'random' else 'bitmask'}")

    # ---- the reference we are trying to avoid paying ----
    if on("snapshot"):
        # Task-IL protocol: a snapshot is per-task, so task 2 is measured on the
        # trained model and task 1 only AFTER restoring.  Measuring both from the
        # restored model would erase task 2 and understate the baseline.
        print("\n[snapshot] reference: exact, but NOT 0 bytes/task")
        m = fresh()
        snap = os.path.join(os.environ.get("TEMP", "."), "zero_task_snap.pt")
        torch.save(base.state_dict(), snap)
        snap_bytes = os.path.getsize(snap)
        train_task(m, tr_f, device, 10, 20, args.epochs, args.lr_task2)
        af = evaluate(m, te_f, device, 10, 20)               # BEFORE the restore
        m.load_state_dict(torch.load(snap, map_location=device))
        am = evaluate(m, te_m, device, 0, 10)                # AFTER the restore
        results.append(dict(method="snapshot", per_task=snap_bytes, fixed=0,
                            acc_m=am, acc_f=af, forget=acc_m_init - am,
                            note="exact; per-task restore"))
        print(f"  MNIST {am:.2f}% | FASHION {af:.2f}% | forget {acc_m_init - am:.2f}% "
              f"| bytes/task {human(snap_bytes)}")
        os.remove(snap)

    # ---- report ----
    print("\n" + "=" * 104)
    print(" 0-BYTE/TASK TABLE  (per-task columns MUST be 0; the question is what the")
    print(" O(1) anchor costs and how much retention/plasticity it buys)")
    print("=" * 104)
    hdr = (f"{'mechanism':<20} | {'bytes/task':>10} | {'fixed O(1)':>12} | "
           f"{'MNIST':>7} | {'FASHION':>7} | {'forget':>7} | note")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda d: (d["per_task"], d["fixed"])):
        print(f"{r['method']:<20} | {human(r['per_task']):>10} | "
              f"{human(r['fixed']):>12} | "
              f"{r['acc_m']:>6.2f}% | {r['acc_f']:>6.2f}% | {r['forget']:>6.2f}% | {r['note']}")

    zero = [r for r in results if r["per_task"] == 0]
    if any(r["method"].startswith("mask_") for r in results):
        print("\n note: a mask row's fixed bytes are the SELECTION it persists -- 4 bytes for a")
        print(" random seed, or one bit per protected weight if Fisher-ranked.  The values it")
        print(" pins are never written, because the decoupled decay is masked along with the")
        print(" gradient (x * 1.0 is exact).  An implementation that instead restores them from")
        print(" a saved copy would hold frac x |basis| of extra state -- the difference between")
        print(" a selection and a snapshot.")
    best = max(zero, key=lambda d: (d["acc_f"] - d["forget"])) if zero else None
    if best:
        print(f"\n best 0-byte/task option: {best['method']}  "
              f"MNIST {best['acc_m']:.2f}%  FASHION {best['acc_f']:.2f}%  "
              f"forget {best['forget']:.2f}%  fixed {mib(best['fixed']):.3f} MiB")
        print(f" it costs {best['per_task']} bytes per task vs the snapshot's "
              f"{mib(snap_bytes):.3f} MiB -- i.e. the snapshot is "
              f"{'infinitely' if best['per_task'] == 0 else ''} more expensive per task")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(dict(acc_m_init=acc_m_init, snapshot_bytes=snap_bytes,
                           shared_weight_bytes=trunk_bytes, seed=args.seed,
                           task2=args.task2, results=results),
                      fh, indent=2)
        print(f"[json] {args.json_out}")


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"\n[done] {time.time() - t0:.1f}s")
    sys.exit(rc or 0)
