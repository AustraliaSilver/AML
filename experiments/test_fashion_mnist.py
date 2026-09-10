import sys, os, json, time, warnings, io
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from mgn import (
    ModelGenerator, TaskNetwork, MetaLoss, MetaTrainer,
    compute_aml_efficiency,
)
from experiments.benchmark_datasets import get_dataset
from experiments.benchmark_baselines import (
    FixedMLP, SmallCNN, ResNetBlockBaseline,
    train_torch,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = 1
EPOCHS = 25
MAX_TRAIN = 1200
MAX_TEST = 400
BATCH_SIZE = 128
DATASET = "fashion_mnist"
P_ALLOW = 100000.0


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_conv_aml(input_dim, output_dim, in_channels):
    gen = ModelGenerator(
        input_dim=input_dim, output_dim=output_dim,
        max_layers=4, max_dim=128, dim_choices=(32, 64, 96, 128),
        latent_dim=64, noise_dim=64, num_prototypes=2,
        use_conv=True, in_channels=in_channels,
    )
    exec_ = TaskNetwork()
    loss = MetaLoss(
        lambda_p=0.0, lambda_c=1e-9, lambda_m=0,
        lambda_balance=1.0, lambda_stability=1e-6, lambda_design=0,
        lambda_eff=0.5, p_allowance=P_ALLOW,
    )
    return gen, exec_, loss


def eval_conv_acc(gen, exec_, x_tensor, y_tensor, device=DEVICE):
    gen.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(x_tensor), 256):
            xb = x_tensor[i:i+256].to(device)
            yb = y_tensor[i:i+256].to(device)
            m = gen(xb)
            correct += (exec_(xb, m).argmax(-1) == yb).sum().item()
            total += len(yb)
    return correct / total if total > 0 else 0.0


def eval_conv_details(gen, exec_, x_tensor, y_tensor, device=DEVICE):
    gen.eval()
    all_P, all_L, all_D = [], [], []
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(x_tensor), 256):
            xb = x_tensor[i:i+256].to(device)
            yb = y_tensor[i:i+256].to(device)
            m = gen(xb)
            logits = exec_(xb, m)
            correct += (logits.argmax(-1) == yb).sum().item()
            total += len(yb)
            all_P.append(m.structure.effective_params_formula().mean().item())
            all_L.append(m.structure.num_layers.mean().item())
            all_D.append(m.structure.hidden_dim.mean().item())
    acc = correct / total if total > 0 else 0.0
    return acc, float(np.mean(all_P)), float(np.mean(all_L)), float(np.mean(all_D))


def train_conv_aml_manual(gen, exec_, loss, train_x, train_y, val_x, val_y,
                           epochs, batch_size, device, seed_label=""):
    t0 = time.time()
    trainer = MetaTrainer(gen, exec_, loss, lr=5e-4, device=device)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    epoch_logs = []
    for epoch in range(1, epochs + 1):
        tau = max(0.5, 1.0 - epoch / 30.0)
        epoch_progress = (epoch - 1) / max(1, epochs - 1)
        metrics = trainer.train_epoch(train_loader, val_dataloader=val_loader,
                                      tau=tau, epoch_progress=epoch_progress)
        log_entry = {
            "epoch": epoch,
            "loss": metrics["loss"],
            "task_loss": metrics["task_loss"],
            "acc": metrics["acc"],
            "resource_cost": metrics["resource_cost"],
            "aml_objective": metrics.get("aml_objective", 0.0),
        }
        epoch_logs.append(log_entry)
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  [{seed_label}] Epoch {epoch:>3d}/{epochs} | "
                f"loss={metrics['loss']:.4f} | task={metrics['task_loss']:.4f} | "
                f"acc={metrics['acc']:.4f} | cost={metrics['resource_cost']:.4f} | "
                f"O={metrics.get('aml_objective', 0.0):.4f} | tau={tau:.3f}"
            )
            sys.stdout.flush()

    elapsed = time.time() - t0
    return trainer, epoch_logs, elapsed


def evaluate_conv_aml(gen, exec_, trainer, train_x, train_y, test_x, test_y, device):
    gen.eval()
    with torch.no_grad():
        test_acc, test_P, test_L, test_D = eval_conv_details(gen, exec_, test_x, test_y, device)
        train_acc = eval_conv_acc(gen, exec_, train_x, train_y, device)
        overfit = train_acc - test_acc

        sample_x = test_x[:512].to(device)
        sample_m = gen(sample_x)
        S = sample_m.structure

        P_eff = S.effective_params_formula().mean().item()
        L_eff = S.num_layers.mean().item()
        D_eff = S.hidden_dim.mean().item()

        block_mode_pct = {"resnet": 0.0, "vit": 0.0, "plain": 0.0}
        if S.block_mode is not None:
            block_mode_pct["resnet"] = S.block_mode[:, :, 1].mean().item() * 100.0
            if S.block_mode.shape[-1] > 2:
                block_mode_pct["vit"] = S.block_mode[:, :, 2].mean().item() * 100.0
            block_mode_pct["plain"] = S.block_mode[:, :, 0].mean().item() * 100.0

        proto_info = trainer.prototype_usage(train_x[:500].to(device))
        entropy = proto_info["entropy"]
        n_used = proto_info["num_prototypes_used"]

        E_aml = compute_aml_efficiency(test_acc, P_ALLOW, P_eff)

    return {
        "test_acc": test_acc, "train_acc": train_acc, "overfit_gap": overfit,
        "P_eff": P_eff, "L_eff": L_eff, "D_eff": D_eff,
        "resnet_pct": block_mode_pct["resnet"],
        "vit_pct": block_mode_pct["vit"],
        "plain_pct": block_mode_pct["plain"],
        "n_prototypes_used": n_used, "entropy": entropy,
        "E_aml": E_aml,
    }


def main():
    print("=" * 70)
    print("Fashion-MNIST ConvAML Benchmark")
    print(f"Device: {DEVICE} | Seeds: {SEEDS} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print("=" * 70)
    sys.stdout.flush()

    data = get_dataset(DATASET, max_train=MAX_TRAIN, max_test=MAX_TEST)
    in_channels = 1
    input_dim = data["input_dim"]
    output_dim = data["output_dim"]
    print(f"Dataset: {DATASET} | Input: {input_dim} | Output: {output_dim}")
    print(f"Train: {data['n_train']} | Test: {data['n_test']}")
    sys.stdout.flush()

    all_results = {}

    # ── ConvAML ──
    print(f"\n{'='*70}")
    print(f"ConvAML ({SEEDS} seeds)")
    print(f"{'='*70}")
    sys.stdout.flush()
    conv_results = []
    for seed in range(SEEDS):
        set_seed(42 + seed)
        gen, exec_, loss_fn = make_conv_aml(input_dim, output_dim, in_channels)

        n_train = len(data["train_x"])
        n_val = int(n_train * 0.2)
        indices = torch.randperm(n_train)
        train_idx = indices[:-n_val]
        val_idx = indices[-n_val:]
        trx = data["train_x"][train_idx]
        try_ = data["train_y"][train_idx]
        vx = data["train_x"][val_idx]
        vy = data["train_y"][val_idx]

        print(f"\n  Seed {seed} | Train: {len(trx)} | Val: {len(vx)}")
        trainer, epoch_logs, elapsed = train_conv_aml_manual(
            gen, exec_, loss_fn, trx, try_, vx, vy,
            epochs=EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
            seed_label=f"ConvAML s{seed}",
        )
        print(f"  Training time: {elapsed:.1f}s")

        metrics = evaluate_conv_aml(gen, exec_, trainer, trx, try_, data["test_x"], data["test_y"], DEVICE)
        metrics["time"] = elapsed
        metrics["epoch_logs"] = epoch_logs
        conv_results.append(metrics)

        print(f"  [Seed {seed} Result] test={metrics['test_acc']:.4f} train={metrics['train_acc']:.4f} "
              f"P_eff={metrics['P_eff']:.0f} L={metrics['L_eff']:.1f} D={metrics['D_eff']:.1f} "
              f"ResNet={metrics['resnet_pct']:.1f}% ViT={metrics['vit_pct']:.1f}% "
              f"H={metrics['entropy']:.3f} E_aml={metrics['E_aml']:.5f} t={elapsed:.0f}s")
        sys.stdout.flush()

    all_results["ConvAML"] = conv_results

    # ── SmallCNN ──
    print(f"\n{'='*70}")
    print(f"SmallCNN ({SEEDS} seeds)")
    print(f"{'='*70}")
    cnn_results = []
    for seed in range(SEEDS):
        set_seed(42 + seed)
        cnn = SmallCNN(in_channels, input_dim, output_dim)
        res = train_torch(cnn, data["train_x"], data["train_y"],
                          data["test_x"], data["test_y"],
                          epochs=EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
                          label=f"SmallCNN s{seed}")
        cnn_results.append(res)
        print(f"  [Seed {seed}] test={res['test_acc']:.4f} train={res['train_acc']:.4f} P={res['P']} t={res['time']:.0f}s")
        sys.stdout.flush()
    all_results["SmallCNN"] = cnn_results

    # ── FixedMLP ──
    print(f"\n{'='*70}")
    print(f"FixedMLP (L=2, D=64) ({SEEDS} seeds)")
    print(f"{'='*70}")
    mlp_results = []
    for seed in range(SEEDS):
        set_seed(42 + seed)
        mlp = FixedMLP(input_dim, 64, 2, output_dim)
        res = train_torch(mlp, data["train_x"], data["train_y"],
                          data["test_x"], data["test_y"],
                          epochs=EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
                          label=f"FixedMLP s{seed}")
        mlp_results.append(res)
        print(f"  [Seed {seed}] test={res['test_acc']:.4f} train={res['train_acc']:.4f} P={res['P']} t={res['time']:.0f}s")
        sys.stdout.flush()
    all_results["FixedMLP"] = mlp_results

    # ── ResNetBlock ──
    print(f"\n{'='*70}")
    print(f"ResNetBlock (base_ch=32) ({SEEDS} seeds)")
    print(f"{'='*70}")
    resnet_results = []
    for seed in range(SEEDS):
        set_seed(42 + seed)
        rn = ResNetBlockBaseline(in_channels, input_dim, output_dim, base_channels=32)
        res = train_torch(rn, data["train_x"], data["train_y"],
                          data["test_x"], data["test_y"],
                          epochs=EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
                          label=f"ResNet s{seed}")
        resnet_results.append(res)
        print(f"  [Seed {seed}] test={res['test_acc']:.4f} train={res['train_acc']:.4f} P={res['P']} t={res['time']:.0f}s")
        sys.stdout.flush()
    all_results["ResNetBlock"] = resnet_results

    # ── Summary ──
    print(f"\n{'='*70}")
    print("COMPREHENSIVE SUMMARY")
    print(f"{'='*70}")

    def mean_std(vals):
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return m, s

    summary_rows = []

    # ConvAML
    c_test = [r["test_acc"] for r in conv_results]
    c_train = [r["train_acc"] for r in conv_results]
    c_P = [r["P_eff"] for r in conv_results]
    c_L = [r["L_eff"] for r in conv_results]
    c_D = [r["D_eff"] for r in conv_results]
    c_resnet = [r["resnet_pct"] for r in conv_results]
    c_vit = [r["vit_pct"] for r in conv_results]
    c_entropy = [r["entropy"] for r in conv_results]
    c_E = [r["E_aml"] for r in conv_results]
    c_time = [r["time"] for r in conv_results]

    c_ts, c_tss = mean_std(c_test)
    c_trs, c_trss = mean_std(c_train)
    c_ps, c_pss = mean_std(c_P)
    c_ls, c_lss = mean_std(c_L)
    c_ds, c_dss = mean_std(c_D)
    c_rns, c_rnss = mean_std(c_resnet)
    c_vs, c_vss = mean_std(c_vit)
    c_es, c_ess = mean_std(c_entropy)
    c_Es, c_Ess = mean_std(c_E)
    c_times = float(np.mean(c_time))

    summary_rows.append(
        f"  {'ConvAML':<14} {c_ts:.4f}±{c_tss:.4f}  {c_trs:.4f}±{c_trss:.4f}  "
        f"{c_ts - c_trs:+.4f}  {c_ps:>8.0f}±{c_pss:.0f}  {c_ls:.1f}±{c_lss:.1f}  {c_ds:.1f}±{c_dss:.1f}  "
        f"{c_rns:>5.1f}%  {c_vs:>5.1f}%  {c_es:.3f}  {c_Es:.5f}  {c_times:>5.0f}s"
    )

    # Baselines
    for name, results in [("SmallCNN", cnn_results), ("FixedMLP", mlp_results), ("ResNetBlock", resnet_results)]:
        b_test = [r["test_acc"] for r in results]
        b_train = [r["train_acc"] for r in results]
        b_P = [r["P"] for r in results]
        b_time = [r["time"] for r in results]
        b_ts, b_tss = mean_std(b_test)
        b_trs, b_trss = mean_std(b_train)
        b_ps, b_pss = mean_std(b_P)
        b_Es = compute_aml_efficiency(b_ts, P_ALLOW, b_ps)
        b_times = float(np.mean(b_time))
        summary_rows.append(
            f"  {name:<14} {b_ts:.4f}+/-{b_tss:.4f}  {b_trs:.4f}+/-{b_trss:.4f}  "
            f"{b_ts - b_trs:+.4f}  {b_ps:>8.0f}+/-{b_pss:.0f}  {'NA':>5}    {'NA':>5}    "
            f"{'NA':>5}  {'NA':>5}  {'NA':>4}  {b_Es:.5f}  {b_times:>5.0f}s"
        )

    hdr = (f"  {'Method':<14} {'Test Acc':<16} {'Train Acc':<16} {'Gap':<8} "
           f"{'P_eff':<14} {'L_eff':<12} {'D_eff':<12} "
           f"{'ResNet%':<8} {'ViT%':<7} {'H':<6} {'E_aml':<8} {'Time':<7}")
    print(f"\n{hdr}")
    print(f"  {'-'*130}")
    for row in summary_rows:
        print(row)

    # ── Save JSON ──
    save_data = {}
    for method, results in all_results.items():
        if method == "ConvAML":
            save_data[method] = {
                "test_acc_mean": c_ts, "test_acc_std": c_tss,
                "train_acc_mean": c_trs, "train_acc_std": c_trss,
                "P_eff_mean": c_ps, "P_eff_std": c_pss,
                "L_eff_mean": c_ls, "L_eff_std": c_lss,
                "D_eff_mean": c_ds, "D_eff_std": c_dss,
                "resnet_pct_mean": c_rns, "resnet_pct_std": c_rnss,
                "vit_pct_mean": c_vs, "vit_pct_std": c_vss,
                "entropy_mean": c_es, "entropy_std": c_ess,
                "E_aml_mean": c_Es, "E_aml_std": c_Ess,
                "time_mean": c_times,
                "per_seed": [
                    {
                        "seed": i,
                        "test_acc": results[i]["test_acc"],
                        "train_acc": results[i]["train_acc"],
                        "overfit_gap": results[i]["overfit_gap"],
                        "P_eff": results[i]["P_eff"],
                        "L_eff": results[i]["L_eff"],
                        "D_eff": results[i]["D_eff"],
                        "resnet_pct": results[i]["resnet_pct"],
                        "vit_pct": results[i]["vit_pct"],
                        "plain_pct": results[i]["plain_pct"],
                        "n_prototypes_used": results[i]["n_prototypes_used"],
                        "entropy": results[i]["entropy"],
                        "E_aml": results[i]["E_aml"],
                        "time": results[i]["time"],
                    }
                    for i in range(len(results))
                ],
                "epoch_logs": results[0]["epoch_logs"],
            }
        else:
            save_data[method] = {
                "test_acc_mean": float(np.mean([r["test_acc"] for r in results])),
                "test_acc_std": float(np.std([r["test_acc"] for r in results], ddof=1)) if SEEDS > 1 else 0.0,
                "train_acc_mean": float(np.mean([r["train_acc"] for r in results])),
                "train_acc_std": float(np.std([r["train_acc"] for r in results], ddof=1)) if SEEDS > 1 else 0.0,
                "P_mean": float(np.mean([r["P"] for r in results])),
                "P_std": float(np.std([r["P"] for r in results], ddof=1)) if SEEDS > 1 else 0.0,
                "time_mean": float(np.mean([r["time"] for r in results])),
                "per_seed": [
                    {"seed": i, "test_acc": r["test_acc"], "train_acc": r["train_acc"],
                     "P": r["P"], "time": r["time"]}
                    for i, r in enumerate(results)
                ],
            }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_fashion_mnist.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
