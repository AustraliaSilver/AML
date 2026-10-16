"""Where does a production training step actually spend its time?

The existing `benchmarks/profile_training_step.py` profiles
`ConvAML_CIFAR_TrueDynamic` from `train.py`, which is the *legacy* model and is
not what `train_continual_aml.py` trains by default.  This profiles the shipped
one: `ConvAML_18Layer_Hybrid` at the CLI defaults (K=12, max_dim=48,
max_layers=18, `--norm in`), bf16 autocast, batch 256, and reports

  * ms/step split into forward / backward / optimizer,
  * peak memory,
  * per-layer forward time (a hook on each of the 18 blocks), and
  * the top CUDA kernels from torch.profiler at the biggest resolution.

Motivated by a standing anomaly: 64x64 measured ~15x a 32x32 step while having
only 4x the pixels.  Something is super-linear, and guessing which part of the
model it is has been wrong before, so measure it.

    python -u benchmarks/profile_hybrid18.py            # 28/32/64, then kernels
    python -u benchmarks/profile_hybrid18.py --res 64 --layers
"""

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mgn.blocks import ConvAML_18Layer_Hybrid  # noqa: E402


def build(args, device):
    m = ConvAML_18Layer_Hybrid(
        K=args.K, max_dim=args.max_dim, max_layers=18, num_classes=args.classes,
        norm_type=args.norm, encoder_tail_act="none",
    ).to(device)
    return m


def step_ms(model, opt, x, y, pi_mask, amp_dtype, reps, device, split=False, seed=0):
    """Return (total, fwd, bwd, opt) in ms/step, median of `reps`.

    The RNG is re-seeded before each repetition.  This is not cosmetic: the
    forward samples its architecture per step (Gumbel over `pi_logits`, Gumbel
    over the block-mode logits, and a `0.05 * randn_like` perturbation of the
    depth gates), so two steps of the *same* configuration can execute different
    modes and cost 2x each other.  Without a fixed seed, a batch sweep is not a
    batch sweep -- it is a sweep over sampled architectures, and it hid the
    effect it was meant to measure (batch 16 measured slower than batch 64).
    """
    times = {"tot": [], "fwd": [], "bwd": [], "opt": []}
    for r in range(reps):
        torch.manual_seed(seed)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        ev[0].record()
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None,
                                dtype=amp_dtype or torch.bfloat16):
            out, struct, pi, diag = model(x, tau=0.8, pi_mask=pi_mask)
            loss = F.cross_entropy(out.float(), y)
        ev[1].record()
        loss.backward()
        ev[2].record()
        opt.step()
        ev[3].record()
        torch.cuda.synchronize()
        times["fwd"].append(ev[0].elapsed_time(ev[1]))
        times["bwd"].append(ev[1].elapsed_time(ev[2]))
        times["opt"].append(ev[2].elapsed_time(ev[3]))
        times["tot"].append(ev[0].elapsed_time(ev[3]))
    med = lambda a: sorted(a)[len(a) // 2]  # noqa: E731
    if split:
        return med(times["tot"]), med(times["fwd"]), med(times["bwd"]), med(times["opt"])
    return med(times["tot"]), None, None, None


def layer_profile(model, x, pi_mask, amp_dtype, args, device):
    """Forward time attributed to each of the 18 blocks, in ms, one pass.

    CUDA events, not `perf_counter`, and recorded from hooks rather than by
    synchronising: a per-layer `synchronize` would serialise the pipeline and
    attribute the resulting stalls to whichever layer happens to follow.  An
    event pair per layer measures GPU time between two points in the stream,
    which is what "this layer costs X ms" means.  Run in training mode with the
    same `tau`/`pi_mask` as the real step, under `no_grad` so the backwards pass
    of the previous step is not still in flight.
    """
    L = len(model.layers)
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(L)]
    pre = [lyr.register_forward_pre_hook(
        (lambda i: lambda _m, _inp: ev[i][0].record())(i)) for i, lyr in enumerate(model.layers)]
    post = [lyr.register_forward_hook(
        (lambda i: lambda _m, _inp, _out: ev[i][1].record())(i)) for i, lyr in enumerate(model.layers)]
    outer = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    torch.cuda.synchronize()
    outer[0].record()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None,
                                dtype=amp_dtype or torch.bfloat16):
            model(x, tau=0.8, pi_mask=pi_mask)
    outer[1].record()
    torch.cuda.synchronize()
    for h in pre + post:
        h.remove()
    per = [a.elapsed_time(b) for a, b in ev]
    tot = sum(per)
    print(f"  per-layer forward (ms): total-in-layers {tot:7.2f}, "
          f"whole forward {outer[0].elapsed_time(outer[1]):7.2f}")
    for i, t in enumerate(per):
        sp = getattr(model, "_stage_spatial", None)
        tag = f" {sp[i]}" if sp is not None and i < len(sp) else ""
        bar = "#" * max(1, int(round(t * 2)))
        print(f"    layer {i:>2}{tag:>10} {t:6.2f} {bar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--max-dim", type=int, default=48)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--norm", type=str, default="in")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234,
                    help="RNG seed re-applied before each measured step, so every "
                         "step of a comparison runs the same sampled architecture")
    ap.add_argument("--res", type=int, default=0, help="single resolution; 0 = all")
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--no-benchmark", action="store_true",
                    help="leave cudnn.benchmark at its default (False)")
    ap.add_argument("--kernels", action="store_true", help="profiler kernel table")
    ap.add_argument("--memory", action="store_true", help="profiler memory table")
    ap.add_argument("--layers", action="store_true", help="per-layer breakdown")
    args = ap.parse_args()

    if args.no_benchmark:
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else None
    res_list = [args.res] if args.res else [28, 32, 64]

    print(f"device={device} batch={args.batch} K={args.K} max_dim={args.max_dim} "
          f"norm={args.norm} bf16={bool(args.bf16)} "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark}")

    model = build(args, device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"parameters {n_par:,} | layers {len(model.layers)} | "
          f"stage spatial {getattr(model, '_stage_spatial', '?')}")

    for R in res_list:
        x = torch.randn(args.batch, 3, R, R, device=device)
        y = torch.randint(0, args.classes, (args.batch,), device=device)
        pi_mask = torch.zeros(args.K, dtype=torch.bool, device=device)
        pi_mask[:max(1, args.K // 3)] = True
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(args.warmup):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_dtype is not None,
                                    dtype=amp_dtype or torch.bfloat16):
                out, _, _, _ = model(x, tau=0.8, pi_mask=pi_mask)
                loss = F.cross_entropy(out.float(), y)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tot, fwd, bwd, ost = step_ms(model, opt, x, y, pi_mask, amp_dtype,
                                     args.reps, device, split=True, seed=args.seed)
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
        print(f"  {R:>3}x{R:<3} | total {tot:8.2f} ms | fwd {fwd:7.2f} | "
              f"bwd {bwd:7.2f} | opt {ost:6.2f} | peak {peak:7.1f} MiB")

        if args.layers:
            layer_profile(model, x, pi_mask, amp_dtype, args, device)

        if args.memory:
            opt.zero_grad(set_to_none=True)
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA,
                                torch.profiler.ProfilerActivity.CPU],
                    profile_memory=True) as prof:
                with torch.amp.autocast("cuda", enabled=amp_dtype is not None,
                                        dtype=amp_dtype or torch.bfloat16):
                    out, _, _, _ = model(x, tau=0.8, pi_mask=pi_mask)
                    loss = F.cross_entropy(out.float(), y)
                loss.backward()
                torch.cuda.synchronize()
            print(f"\n=== largest single allocations at {R}x{R} batch {args.batch} "
                  f"({args.batch} samples; divide by batch for MiB/sample) ===")
            print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

        if args.kernels:
            x2 = x
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA,
                                torch.profiler.ProfilerActivity.CPU],
                    record_shapes=False) as prof:
                for _ in range(5):
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=amp_dtype is not None,
                                            dtype=amp_dtype or torch.bfloat16):
                        out, _, _, _ = model(x2, tau=0.8, pi_mask=pi_mask)
                        loss = F.cross_entropy(out.float(), y)
                    loss.backward()
                    opt.step()
                torch.cuda.synchronize()
            print(f"\n=== top CUDA kernels at {R}x{R} (5 steps) ===")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))
            print(f"=== top CPU ops at {R}x{R} ===")
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=12))
            del x2


if __name__ == "__main__":
    main()
