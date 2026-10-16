"""A/B the hoisted routing flags against the old two-syncs-per-layer path.

Comparing two *runs* on a laptop GPU measures the machine's clock drift as much as
the change under test: the first sweep put batch 16 at 232 ms after the change and
187 ms before, then 196 ms at batch 64 -- i.e. a 24 % "regression" that no
mechanism explains.  So this script measures both variants **in one process,
interleaved**, re-seeding the RNG before every single step.  The seed makes the
sampled architecture identical across variants and repetitions, and interleaving
makes any thermal or clock drift hit both variants equally.

Variant A = current code (flags hoisted, one transfer per forward).
Variant B = monkeypatched `HybridBlock.forward` that ignores `flags`, i.e. the
            old two-synchronisations-per-layer path.

    python -u benchmarks/ab_router_flag_hoist.py
    python -u benchmarks/ab_router_flag_hoist.py --reps 21 --res 64 --batch 128
"""

import argparse
import os
import statistics as st
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mgn.blocks import ConvAML_18Layer_Hybrid  # noqa: E402
from mgn.blocks.hybrid_router import HybridBlock  # noqa: E402

_ORIG = HybridBlock.forward


def _old_path(self, x, mode_weights=None, dim_mask=None, flags=None):
    return _ORIG(self, x, mode_weights=mode_weights, dim_mask=dim_mask, flags=None)


def measure(model, opt, x, y, pi_mask, reps, seed, amp_dtype):
    """Median ms/step over `reps` steps, each started from the same RNG state."""
    ts = []
    for _ in range(reps):
        torch.manual_seed(seed)                    # same sampled arch every step
        opt.zero_grad(set_to_none=True)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        torch.cuda.synchronize()
        ev[0].record()
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None,
                                dtype=amp_dtype or torch.bfloat16):
            out, _, _, _ = model(x, tau=0.8, pi_mask=pi_mask)
            loss = F.cross_entropy(out.float(), y)
        loss.backward()
        opt.step()
        ev[1].record()
        torch.cuda.synchronize()
        ts.append(ev[0].elapsed_time(ev[1]))
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--max-dim", type=int, default=48)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if dev.type == "cuda" else None

    torch.manual_seed(args.seed)
    model = ConvAML_18Layer_Hybrid(
        K=args.K, max_dim=args.max_dim, max_layers=18, num_classes=10,
        norm_type="in", encoder_tail_act="none").to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(args.batch, 3, args.res, args.res, device=dev)
    y = torch.randint(0, 10, (args.batch,), device=dev)
    pi_mask = torch.zeros(args.K, dtype=torch.bool, device=dev)
    pi_mask[:4] = True

    # Warm up both variants so lazy init and cuDNN autotuning are not charged to
    # whichever variant happens to run first.
    for fn in (_ORIG, _old_path):
        HybridBlock.forward = fn
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_dtype is not None,
                                    dtype=amp_dtype or torch.bfloat16):
                out, _, _, _ = model(x, tau=0.8, pi_mask=pi_mask)
                loss = F.cross_entropy(out.float(), y)
            loss.backward()
            opt.step()
    torch.cuda.synchronize()

    a, b = [], []
    for _ in range(args.reps):
        HybridBlock.forward = _ORIG
        a += measure(model, opt, x, y, pi_mask, 1, args.seed, amp_dtype)
        HybridBlock.forward = _old_path
        b += measure(model, opt, x, y, pi_mask, 1, args.seed, amp_dtype)
    HybridBlock.forward = _ORIG

    ma, mb = st.median(a), st.median(b)
    print(f"{args.res}x{args.res} batch {args.batch} | {args.reps} interleaved reps")
    print(f"  hoisted flags (new): {ma:8.2f} ms  [min {min(a):7.2f}, "
          f"max {max(a):7.2f}, sd {st.pstdev(a):5.2f}]")
    print(f"  per-layer sync (old): {mb:8.2f} ms  [min {min(b):7.2f}, "
          f"max {max(b):7.2f}, sd {st.pstdev(b):5.2f}]")
    print(f"  change: {mb - ma:+.2f} ms  ({100.0 * (mb - ma) / mb:+.2f} %)")
    print(f"  means, same data:     {st.mean(a):8.2f} vs {st.mean(b):8.2f} "
          f"({100.0 * (st.mean(b) - st.mean(a)) / st.mean(b):+.2f} %)")


if __name__ == "__main__":
    main()
