"""Capture / compare the exact outputs of `ConvAML_18Layer_Hybrid`.

Why this exists
---------------
An optimisation to `mgn/blocks/hybrid_router.py` moved a device->host
synchronisation out of the per-layer loop and into `ConvAML_18Layer_Hybrid.forward`
(one transfer for all 18 layers instead of two per layer).  That kind of change is
supposed to be a pure latency change: the *values* it computes must be identical,
including the sampled architecture, because the sampled architecture is drawn from
the global RNG and the RNG must be consumed in the same order.

"Should be" is not evidence.  This script captures the forward output, the routing
distribution `pi`, every parameter and a representative gradient from a fixed seed,
and can compare a later run against that capture with `torch.equal` -- exact
bitwise equality, not `allclose`.  Run it before the change with `--save`, then
again after with `--compare`.

The configuration is the production one: `K=12`, `max_dim=48`, `--norm in`,
bf16 autocast, batch 64 (small so it runs on any GPU), in *both* training and eval
mode, because the two modes take different branches inside `HybridBlock.forward`
(training: batch-consistent fast path; eval: per-sample MoE dispatch).

    python -u benchmarks/capture_router_identity.py --save  %TEMP%\\router_before.pt
    python -u benchmarks/capture_router_identity.py --compare %TEMP%\\router_before.pt
"""

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mgn.blocks import ConvAML_18Layer_Hybrid  # noqa: E402


def build(seed, device, K=12, max_dim=48, classes=10, norm="in"):
    """Build the model from `seed`, so two runs get identical weights."""
    torch.manual_seed(seed)
    model = ConvAML_18Layer_Hybrid(
        K=K, max_dim=max_dim, max_layers=18, num_classes=classes,
        norm_type=norm, encoder_tail_act="none",
    ).to(device)
    return model


def capture(seed, device, batch=64, res=32, K=12, amp_dtype=torch.bfloat16):
    model = build(seed, device, K=K)
    model.train()

    # Training-mode pass with gradients.  A per-sample pi_mask exercises the
    # row-masked routing path, which is what the continual trainer uses.
    x = torch.randn(batch, 3, res, res, device=device)
    y = torch.randint(0, 10, (batch,), device=device)
    pi_mask = torch.zeros(K, dtype=torch.bool, device=device)
    pi_mask[:4] = True

    torch.manual_seed(seed + 1)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=amp_dtype):
        out, struct, pi, diag = model(x, tau=0.8, pi_mask=pi_mask)
        loss = F.cross_entropy(out.float(), y)
    loss.backward()

    rec = {
        "out_train": out.detach().float().cpu().clone(),
        "pi_train": pi.detach().float().cpu().clone(),
        "loss": loss.detach().float().cpu().clone(),
        "layer_soft": diag["layer_soft"].detach().float().cpu().clone(),
        "bm_logits": diag["bm_logits"].detach().float().cpu().clone(),
        "n_executed": diag["n_executed"],
        "params": {k: v.detach().float().cpu().clone()
                   for k, v in model.state_dict().items()},
        "grads": {k: p.grad.detach().float().cpu().clone()
                  for k, p in model.named_parameters() if p.grad is not None},
    }

    # Eval-mode pass: deterministic argmax modes, per-sample dispatch possible.
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=amp_dtype):
            out_e, _, pi_e, _ = model(x, tau=0.8, pi_mask=pi_mask)
    rec["out_eval"] = out_e.detach().float().cpu().clone()
    rec["pi_eval"] = pi_e.detach().float().cpu().clone()
    return model, rec


def compare(a, b):
    """Exact equality over every recorded tensor; returns list of mismatches."""
    bad = []
    if set(a) != set(b):
        bad.append(f"key sets differ: {sorted(set(a) ^ set(b))}")
        return bad

    def walk(path, xa, xb):
        if isinstance(xa, dict):
            if set(xa) != set(xb):
                bad.append(f"{path}: keys differ")
                return
            for k in xa:
                walk(f"{path}.{k}", xa[k], xb[k])
        elif isinstance(xa, torch.Tensor):
            if xa.shape != xb.shape:
                bad.append(f"{path}: shape {tuple(xa.shape)} != {tuple(xb.shape)}")
            elif not torch.equal(xa, xb):
                d = (xa - xb).abs().max().item()
                bad.append(f"{path}: not bitwise equal (max|diff| = {d:.3e})")
        else:
            if xa != xb:
                bad.append(f"{path}: {xa!r} != {xb!r}")

    walk("rec", a, b)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--compare", type=str, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else None
    _, rec = capture(args.seed, device, batch=args.batch, res=args.res,
                     amp_dtype=amp_dtype)
    n_par = len(rec["params"])
    n_grad = len(rec["grads"])
    print(f"device={device} seed={args.seed} batch={args.batch} res={args.res} "
          f"bf16={bool(args.bf16)} | loss={rec['loss'].item():.6f} "
          f"n_executed={rec['n_executed']} | {n_par} params, {n_grad} grads")

    if args.save:
        torch.save(rec, args.save)
        print(f"saved {args.save}")

    if args.compare:
        ref = torch.load(args.compare, map_location="cpu", weights_only=False)
        bad = compare(ref, rec)
        if bad:
            print(f"IDENTITY CHECK FAILED ({len(bad)} mismatches):")
            for line in bad[:20]:
                print("   ", line)
            raise SystemExit(1)
        print("IDENTITY CHECK PASSED: every recorded tensor is bitwise identical "
              "to the capture")


if __name__ == "__main__":
    main()
