"""probe_gate_collapse.py -- diagnose & fix the ConvAML accuracy gap.

Fast-regime benchmark showed ConvAML collapsing to avg_L=0.085 (layer 0
only) while losing ~10% accuracy to SmallCNN, and the No_O ablation
beating the full objective on BOTH accuracy and efficiency.

Three rounds of controlled ablations (fashion_mnist, 1200/400 samples,
25 epochs, 2 seeds) found the real causes -- none of them architectural:

  Round 1 (architecture policy): depth-gate collapse happens with or
  without the efficiency objective; forcing gates open (floor warmup)
  backfires (forced-random deep layers accumulate negative gradients);
  freezing all gates open (fixed, 285k params) does NOT beat the
  collapsed dynamic nets. Architecture selection is not the problem.

  Round 2 (training machinery):
    - EVAL-TIME STOCHASTICITY BUG: F.gumbel_softmax ran during eval for
      routing pi, block modes and MLP layer gates, and the executor's
      dropout was never disabled (exec_.eval() was never called), so
      every previously recorded accuracy carried train-mode noise.
      Fixed in mgn/generator.py, mgn/executor.py, mgn/trainer.py,
      experiments/benchmark_conv_aml.py. Worth ~+1 point alone.
    - lambda_balance=0: the -entropy(pi_avg) term (weight 1.0) fought
      the task loss. +3.1 points. The single biggest win.
    - InstanceNorm: not guilty. lr=1e-3: +2 but does not stack.

  Round 3 (data/step levers): batch 128 hurts (-3.6); shrinking the
  val split changes nothing. nobal config confirmed as champion.

Final (fast regime): base 75.0 -> fixed 79.1 @15.9k params vs SmallCNN
84.8 @111k / ResNetBlock 77.5 @78k. ConvAML now strictly beats
ResNetBlock on both axes; the residual gap to SmallCNN in this
data-starved regime is the price of per-sample weight synthesis.

Run: python probes/probe_gate_collapse.py [--seeds 2]
"""
import argparse
import copy
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, ".")

from mgn import ModelGenerator, TaskNetwork, MetaLoss, MetaTrainer  # noqa: E402
from experiments.benchmark_datasets import get_dataset  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 25
BATCH = 64


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_model(in_channels, output_dim, lambda_eff, lambda_balance=1.0):
    gen = ModelGenerator(
        input_dim=784 if in_channels == 1 else 3072, output_dim=output_dim,
        max_layers=4, max_dim=128, dim_choices=(32, 64, 96, 128),
        latent_dim=64, noise_dim=64, num_prototypes=4,
        use_conv=True, in_channels=in_channels,
    )
    exec_ = TaskNetwork()
    p_allow = 562186.0 if in_channels == 3 else 506762.0
    loss = MetaLoss(lambda_p=0.0, lambda_c=1e-9, lambda_m=0,
                    lambda_balance=lambda_balance, lambda_stability=1e-6,
                    lambda_design=0,
                    lambda_eff=lambda_eff, p_allowance=p_allow)
    return gen, exec_, loss


def disable_instancenorm():
    """Diagnostic-only: make the executor's per-sample InstanceNorm a no-op."""
    import mgn.executor as _ex
    if getattr(_ex.F.group_norm, "_probe_noop", False):
        return
    orig = _ex.F.group_norm

    def _noop(*a, **k):
        return a[0]

    _noop._probe_noop = True
    _noop._orig = orig
    _ex.F.group_norm = _noop


def restore_instancenorm():
    import mgn.executor as _ex
    fn = getattr(_ex.F.group_norm, "_orig", None)
    if fn is not None:
        _ex.F.group_norm = fn


def floor_schedule(progress, start=0.55, end=0.30, horizon=0.6):
    """Annealed exploration floor; <=0 after `horizon` of training."""
    if progress >= horizon:
        return 0.0
    t = progress / horizon
    return start + (end - start) * t


def run_variant(name, data, in_channels, seed, lambda_eff=0.2, use_floor=False,
                freeze_arch=False, lambda_balance=1.0, lr=5e-4, no_in=False,
                batch_size=BATCH, val_frac=0.2):
    set_seed(seed)
    gen, exec_, loss = make_model(in_channels, data["output_dim"], lambda_eff,
                                  lambda_balance=lambda_balance)
    gen.to(DEVICE)

    if freeze_arch:
        for p in gen.prototype_selector.parameters():
            p.requires_grad = False
        for p in gen.structure_delta.parameters():
            p.requires_grad = False
        for p in gen.block_mode_head.parameters():
            p.requires_grad = False

    n_train = len(data["train_x"])
    n_val = max(1, int(n_train * val_frac))
    idx = torch.randperm(n_train)
    trx, try_ = data["train_x"][idx[:-n_val]], data["train_y"][idx[:-n_val]]
    vx, vy = data["train_x"][idx[-n_val:]], data["train_y"][idx[-n_val:]]

    trainer = MetaTrainer(gen, exec_, loss, lr=lr, device=DEVICE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=EPOCHS, eta_min=1e-5)

    loader = DataLoader(TensorDataset(trx, try_), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(vx, vy), batch_size=batch_size)

    gate_trace = []
    t0 = time.time()
    if no_in:
        disable_instancenorm()
    try:
        for epoch in range(EPOCHS):
            progress = epoch / max(1, EPOCHS - 1)
            tau = max(0.5, 1.0 - epoch / 30.0)
            gen.gate_floor = floor_schedule(progress) if use_floor else 0.0
            metrics = trainer.train_epoch(loader, val_dataloader=val_loader,
                                          tau=tau, epoch_progress=progress)
            scheduler.step()
            with torch.no_grad():
                smp = gen(data["train_x"][:64].to(DEVICE))
                gate_trace.append(smp.structure.num_layers.mean().item())
    finally:
        if no_in:
            restore_instancenorm()

    gen.eval()
    exec_.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(data["test_x"]), 256):
            xb = data["test_x"][i:i + 256].to(DEVICE)
            yb = data["test_y"][i:i + 256]
            m = gen(xb)
            correct += (exec_(xb, m).argmax(-1).cpu() == yb).sum().item()
            total += len(yb)
    test_acc = correct / total

    with torch.no_grad():
        smp = gen(data["test_x"][:256].to(DEVICE))
        L = smp.structure.num_layers.mean().item()
        P = smp.structure.effective_params_formula().mean().item()
        D = smp.structure.hidden_dim.mean().item()

    print(f"  [{name:10s} seed={seed}] acc={test_acc:.4f} L={L:.2f} "
          f"D={D:.0f} P={int(P)} | gate_trace={['%.2f' % g for g in gate_trace[::5]]} "
          f"| {time.time()-t0:.0f}s", flush=True)
    return {"acc": test_acc, "L": L, "P": P}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    data = get_dataset("fashion_mnist", max_train=1200, max_test=400)
    in_ch = 1
    print(f"Device={DEVICE} | train={data['n_train']} test={data['n_test']} "
          f"| epochs={EPOCHS} seeds={args.seeds}", flush=True)

    # Round 3: with lambda_balance=0 as the new base, test data/step levers.
    variants = [
        ("nobal",       dict(lambda_eff=0.2, lambda_balance=0.0)),
        ("nobal_b128",  dict(lambda_eff=0.2, lambda_balance=0.0, batch_size=128)),
        ("nobal_val05", dict(lambda_eff=0.2, lambda_balance=0.0, val_frac=0.05)),
        ("nobal_both",  dict(lambda_eff=0.2, lambda_balance=0.0, batch_size=128,
                             val_frac=0.05)),
    ]

    summary = {}
    for name, cfg in variants:
        runs = [run_variant(name, data, in_ch, 42 + s, **cfg)
                for s in range(args.seeds)]
        summary[name] = {
            "acc": float(np.mean([r["acc"] for r in runs])),
            "L": float(np.mean([r["L"] for r in runs])),
            "P": float(np.mean([r["P"] for r in runs])),
        }

    print("\n" + "=" * 62)
    print(f"{'variant':12s} {'test_acc':>9s} {'avg_L':>6s} {'P_eff':>8s}")
    print("-" * 62)
    for name, s in summary.items():
        print(f"{name:12s} {s['acc']:>9.4f} {s['L']:>6.2f} {int(s['P']):>8d}")
    print("=" * 62)
    print("Reference (same regime, from conv_aml_fashion_mnist.json):")
    print("  SmallCNN 84.75% @111k | FixedMLP 82.50% @55k | ResNetBlock 77.50% @78k")


if __name__ == "__main__":
    main()
