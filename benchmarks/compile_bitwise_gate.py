"""Does `torch.compile` make the 18-layer AML training step bitwise identical?

The profile blames casts: `aten::to` runs a few hundred times per step, and fusing
those boundaries is the textbook case for Inductor. Measured on an RTX 3050
(sm_86, torch 2.5.1+cu124), 18-layer AML, batch 256, 32x32, `--norm in`, bf16
autocast, median of 9 seeded steps:

    eager    ~409-415 ms/step | 452-599 aten::to | peak 2704 MiB
    compiled ~175-177 ms/step |  34-145 aten::to | peak 1864 MiB

i.e. ~2.3x faster, ~31 % less peak memory, and most of the casts fused away.

The `aten::to` counts vary that much **between runs of this same script** because
the count is a property of the sampled architecture, not of the model: `pi`
decides `delta`, `delta` decides `dim_masks`, and `dim_masks` decides how many
channels are actually live, hence how many dtype boundaries get crossed. Quote
the ratio, not the count. The verdict below, by contrast, reproduced on every
run.

This is a large, real win **as a different computation**. It is not a speed knob,
because it is not the same computation:

  TEST 1  PASS  A fixed `manual_seed` does pin the sampled architecture in eager,
                 bitwise. So a seeded comparison of anything is meaningful.
                 (This is the premise `benchmarks/profile_hybrid18.py` relies on.)

  TEST 2  FAIL  `torch.compile` does NOT preserve the architecture RNG stream.
                 Even compiled-and-warm with the same seed, the sampled `pi`
                 differs by ~0.99 on a 1.0 scale and argmax agreement lands at
                 chance -- 17.2 %, 16.8 % and 9.4 % on three runs, against 10 %
                 expected for 10 classes. Different network, not a rounding
                 difference. So compiling the training path changes WHICH
                 architecture is trained.

  TEST 3  FAIL  With the architecture held fixed -- `model.eval()`, where this
                 model uses no RNG at all (`pi` and `block_mode` are both
                 `one_hot(argmax)` and the `0.05*randn` depth perturbation sits
                 behind `if self.training`) -- Inductor is still not bitwise:
                 logits differ by ~5 % of their scale, **0 of 126 parameter
                 gradients** match bitwise, worst off by ~20-24 % of the gradient
                 scale. Inductor fuses and reorders reductions, and 18 layers of
                 that accumulates.

                 Worse, the autograd graph differs *structurally*, not just
                 numerically: the compiled path returns gradients for parameters
                 the eager path leaves untouched, e.g. `block_mode_prototypes`
                 and `vision_encoder.0.weight`. In eval mode
                 `block_mode = F.one_hot(bm_logits.argmax(-1), 3).float()`, and
                 `argmax` is not differentiable, so eager has no gradient path
                 there at all. The set of parameters that get updated would
                 therefore differ, which changes the optimisation itself.

Both failures matter numerically. TEST 3's ~96-97 % argmax agreement flips
~3-4 % of predictions, which on these tasks is a 1-3 point accuracy move -- the
same order as, or larger than, the whole `cudnn.benchmark` run-to-run drift
measured in `REPRODUCIBILITY.md` 3Z.1. Enabling it would invalidate published
numbers.

Note the compiled model is *stable*: its argmax agreement came out at 97.27 %
on two independent runs. It is deterministic, just not the same function.

    python -u benchmarks/compile_bitwise_gate.py              # the three tests
    python -u benchmarks/compile_bitwise_gate.py --speed      # + the table above
    python -u benchmarks/compile_bitwise_gate.py --speed --reps 15

WINDOWS CACHE BUG PATCHED BELOW
-------------------------------
Inductor's `codecache.write_atomic` writes a temp file then calls
`tmp_path.rename(path)`. `Path.rename` is `os.rename`, which FAILS if the
destination exists on Windows and silently overwrites on POSIX, so the second
write of one cache key raises `FileExistsError: [WinError 183]`. A function
named for an atomic write should not have that property; `os.replace` is the
overwrite-atomically primitive on both platforms. This looks like a genuine
upstream bug worth reporting.

The alternatives were rejected rather than used: `TORCHINDUCTOR_FX_GRAPH_CACHE=0`
disables a different cache, `force_disable_caches = True` does not cover
`pad_mm`'s separate `get_pad_cache()`, and turning off `shape_padding` would
avoid the crash but also changes *what Inductor does*, which is the thing under
test. Patching `write_atomic` fixes the platform bug and leaves every Inductor
setting at its default, so the numbers above describe the configuration someone
would actually run.
"""

import argparse
import os
import pathlib
import sys
import threading

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import torch  # noqa: E402
import torch._dynamo as dynamo  # noqa: E402
import torch._inductor.codecache as codecache  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from mgn.blocks import ConvAML_18Layer_Hybrid  # noqa: E402


# --------------------------------------------------------------------------
# Windows-only: make Inductor's atomic cache write overwrite instead of raise.
# A no-op on POSIX, where os.replace and os.rename agree.
# --------------------------------------------------------------------------
def _windows_safe_write_atomic(path_, content, make_dirs=False, encode_utf_8=False):
    assert isinstance(content, (str, bytes)), "unexpected cache content type"
    path = pathlib.Path(path_)
    if make_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"
    mode = "w" if isinstance(content, str) else "wb"
    with tmp_path.open(mode, encoding="utf-8" if encode_utf_8 else None) as f:
        f.write(content)
    os.replace(tmp_path, path)          # not tmp_path.rename(path)


def patch_inductor_cache():
    if os.name == "nt":
        codecache.write_atomic = _windows_safe_write_atomic
    return os.name == "nt"


# --------------------------------------------------------------------------
def build(args, device):
    return ConvAML_18Layer_Hybrid(
        K=args.K, max_dim=args.max_dim, max_layers=18, num_classes=args.classes,
        norm_type=args.norm, encoder_tail_act="none",
    ).to(device)


def make_batch(args, device):
    """Fixed input, independent of the `seed` used to pin architecture sampling."""
    g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    x = torch.randn(args.batch, 3, args.res, args.res, generator=g).to(device)
    y = torch.randint(0, args.classes, (args.batch,), generator=g).to(device)
    return x, y


def fwd_bwd(fn, x, y, amp_dtype, params_from, seed=None):
    """Run `fn(x)`; collect gradients by name from `params_from`.

    `params_from` must be the ORIGINAL module, never the `torch.compile` wrapper:
    the wrapper is an `OptimizedModule` whose `named_parameters()` yields
    `_orig_mod.stem.0.weight` instead of `stem.0.weight`. Collecting from the
    wrapper compares two dicts with disjoint keys and reports every gradient as
    missing rather than as different -- which reads exactly like "no gradients
    were produced", i.e. a false FAIL.
    """
    params_from.zero_grad(set_to_none=True)
    if seed is not None:
        torch.manual_seed(seed)
    with torch.amp.autocast("cuda", enabled=amp_dtype is not None,
                            dtype=amp_dtype or torch.bfloat16):
        out, _struct, pi, _diag = fn(x, tau=0.8)
        loss = F.cross_entropy(out.float(), y)
    loss.backward()
    grads = {n: (p.grad.detach().float().clone() if p.grad is not None else None)
             for n, p in params_from.named_parameters()}
    return out.detach().float(), pi.detach().clone(), grads


def cmp(tag, a, b):
    eq = torch.equal(a, b)
    md = (a.float() - b.float()).abs().max().item()
    print(f"    {tag:<26} bitwise={'YES' if eq else 'NO':<3} max|diff|={md:.6e}"
          f"  max|ref|={a.float().abs().max().item():.4e}")
    return eq


def cmp_grads(ge, gc):
    both = [n for n in ge if ge[n] is not None and gc.get(n) is not None]
    only_e = [n for n in ge if ge[n] is not None and gc.get(n) is None]
    only_c = [n for n in ge if ge[n] is None and gc.get(n) is not None]
    if not both:
        print("    no comparable gradients -- the collection is broken, not the model")
        return False, False
    n_bit = sum(1 for n in both if torch.equal(ge[n], gc[n]))
    worst = max(((ge[n] - gc[n]).abs().max().item() for n in both), default=0.0)
    scale = max((ge[n].abs().max().item() for n in both), default=0.0)
    print(f"    {n_bit}/{len(both)} gradients bitwise | worst max|diff|="
          f"{worst:.6e}  max|g|={scale:.4e}  ({100 * worst / max(scale, 1e-30):.1f} % of scale)")
    if only_e:
        print(f"    !! eager-only gradients: {only_e[:4]}")
    if only_c:
        print(f"    !! compiled-only gradients: {only_c[:4]}")
    return n_bit == len(both), bool(only_e or only_c)


# --------------------------------------------------------------------------
def speed_table(args, model, compiled, x, y, amp_dtype):
    """ms/step, `aten::to` calls/step and peak memory, eager then compiled.

    The RNG is re-seeded before every measured step so each step of the
    comparison runs the same sampled architecture, and peak memory is read
    AFTER the measured repetitions -- reading it straight after
    `reset_peak_memory_stats()` reports the current allocation, not a peak.
    """
    def step_ms(fn, reps):
        times = []
        for _ in range(reps):
            torch.manual_seed(args.seed)
            e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            torch.cuda.synchronize()
            e0.record()
            fwd_bwd(fn, x, y, amp_dtype, model)
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1))
        return sorted(times)[len(times) // 2]

    def to_calls(fn, n=5):
        for _ in range(2):
            torch.manual_seed(args.seed)
            fwd_bwd(fn, x, y, amp_dtype, model)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                torch.manual_seed(args.seed)
                fwd_bwd(fn, x, y, amp_dtype, model)
            torch.cuda.synchronize()
        return sum(e.count for e in prof.key_averages()
                   if e.key.startswith("aten::to")) / n

    print("\n=== speed / memory ===")
    rows = []
    for name, fn, reps in (("eager", model, args.reps), ("compiled", compiled, args.reps)):
        for _ in range(args.warmup):
            torch.manual_seed(args.seed)
            fwd_bwd(fn, x, y, amp_dtype, model)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()        # reset BEFORE the reps...
        ms = step_ms(fn, reps)
        peak = torch.cuda.max_memory_allocated() / 2**20   # ...read AFTER
        calls = to_calls(fn)
        rows.append((name, ms, calls, peak))
        print(f"    {name:<10} {ms:8.2f} ms/step | aten::to {calls:6.1f} "
              f"| peak {peak:7.1f} MiB")
    (_, e_ms, e_to, e_pk), (_, c_ms, c_to, c_pk) = rows
    print(f"    -> {e_ms / c_ms:.2f}x | aten::to {e_to:.0f} -> {c_to:.0f} "
          f"({100 * (1 - c_to / max(e_to, 1e-9)):.1f} % removed) | peak "
          f"{e_pk:.1f} -> {c_pk:.1f} MiB ({c_pk - e_pk:+.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--max-dim", type=int, default=48)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--norm", type=str, default="in")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--speed", action="store_true",
                    help="also report ms/step, aten::to calls and peak memory")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device: this compares two GPU kernels.")
    device = torch.device("cuda")
    amp_dtype = None if args.no_bf16 else torch.bfloat16
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    patched = patch_inductor_cache()
    cap = torch.cuda.get_device_capability(device)
    print(f"device={torch.cuda.get_device_name(device)} sm_{cap[0]}{cap[1]} "
          f"torch={torch.__version__} bf16={amp_dtype} batch={args.batch} "
          f"{args.res}x{args.res} cache_patched={patched}")

    model = build(args, device)
    x, y = make_batch(args, device)

    results = {}

    # ---------------- TEST 1 ---------------------------------------------
    print("\n=== TEST 1  eager, train mode, same seed twice ===")
    model.train()
    o_a, pi_a, _ = fwd_bwd(model, x, y, amp_dtype, model, seed=args.seed)
    o_b, pi_b, _ = fwd_bwd(model, x, y, amp_dtype, model, seed=args.seed)
    results["t1"] = cmp("pi (1st vs 2nd)", pi_a, pi_b) and cmp("logits", o_a, o_b)
    print(f"    -> premise {'HOLDS' if results['t1'] else 'FAILS'}: a fixed seed "
          f"{'does' if results['t1'] else 'does NOT'} pin the sampled architecture")

    # ---------------- compile, then warm it up before measuring -------------
    dynamo.reset()
    dynamo.config.cache_size_limit = 256
    compiled = torch.compile(model)
    print("\n  compiling + warming up (result discarded)...")
    fwd_bwd(compiled, x, y, amp_dtype, model, seed=args.seed)
    c = dynamo.utils.counters
    print(f"    recompiles={sum(c.get('recompiles', {}).values())} "
          f"graph-break sites={len(c.get('graph_break', {}))}")

    # ---------------- TEST 2 ---------------------------------------------
    print("\n=== TEST 2  train mode, compiled (warm) vs eager, same seed ===")
    o_c, pi_c, _ = fwd_bwd(compiled, x, y, amp_dtype, model, seed=args.seed)
    results["t2"] = cmp("pi (eager vs comp)", pi_a, pi_c) and cmp("logits", o_a, o_c)
    print(f"    argmax agreement "
          f"{(o_a.argmax(-1) == o_c.argmax(-1)).float().mean().item() * 100:.2f} %")
    print(f"    -> compile {'preserves' if results['t2'] else 'does NOT preserve'} "
          f"the architecture RNG stream")

    # ---------------- TEST 3 ---------------------------------------------
    print("\n=== TEST 3  eval mode (no RNG at all), eager vs compiled ===")
    model.eval()
    o_e, _pi_e, ge = fwd_bwd(model, x, y, amp_dtype, model)
    o_ce, _pi_ce, gce = fwd_bwd(compiled, x, y, amp_dtype, model)
    results["t3"] = cmp("logits", o_e, o_ce)
    g_ok, g_missing = cmp_grads(ge, gce)
    results["t3"] = results["t3"] and g_ok and not g_missing
    print(f"    argmax agreement "
          f"{(o_e.argmax(-1) == o_ce.argmax(-1)).float().mean().item() * 100:.2f} %")

    if args.speed:
        model.train()
        speed_table(args, model, compiled, x, y, amp_dtype)

    print("\n=== VERDICT ===")
    for key, label in (("t1", "eager seeded reproduction   "),
                       ("t2", "compile keeps RNG stream    "),
                       ("t3", "compiled == eager (bitwise) ")):
        print(f"  TEST {key[1]}  {label}: {'PASS' if results[key] else 'FAIL'}")
    print()
    if results["t3"] and results["t2"]:
        print("  PASS - bitwise identical, so a pure speed/memory opt-in knob.")
    elif results["t3"]:
        print("  Inductor is arithmetically faithful but compiling changes WHICH")
        print("  architecture gets sampled: a different protocol, and every published")
        print("  number would need re-measuring.")
    else:
        print("  FAIL - not bitwise, and not only because of the RNG stream.")
        print("  TEST 3 already holds the architecture fixed, so this is Inductor's")
        print("  codegen, which means hoisting the sampler out of the compiled")
        print("  region cannot rescue it.")
        if g_missing:
            print("  Note the gradient SETS differ, not just the gradient values:")
            print("  the compiled path reaches parameters the eager path does not.")
            print("  That changes which tensors get updated, so it changes the")
            print("  optimisation, not only its rounding. The likely mechanism is")
            print("  the single graph break splitting the trace differently, which")
            print("  this measurement observes but does not itself confirm.")
        print("  Do not enable torch.compile by default: it would move every")
        print("  published number.")
    return 0 if results["t3"] and results["t2"] else 1


if __name__ == "__main__":
    sys.exit(main())