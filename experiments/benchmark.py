"""
[PHASE 1 HISTORICAL BENCHMARK]
MGN vs Baselines on Tabular / PCA-Reduced Real Datasets
=====================================================================================
Evaluates the baseline Model Generation Network (MGN) on tabular and PCA feature
vectors against standard static baselines (FixedMoE, SmallCNN, Random Forest).

Usage:
    python experiments/benchmark.py [--datasets mnist,fashion_mnist,cifar10] [--seeds 5] [--quick]
=====================================================================================
"""

import sys, os, json, time, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mgn import (
    ModelGenerator, TaskNetwork, MetaLoss, MetaTrainer,
)
from experiments.benchmark_datasets import get_dataset, cifar_hflip_flat
from experiments.benchmark_baselines import (
    FixedMoE, SmallCNN,
    train_torch, train_rf, search_best_mlp,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


def make_mgn(input_dim, output_dim, latent_dim=128, num_prototypes=8):
    if input_dim > 2048:
        max_dim = 128
        dim_choices = (32, 64, 128)
    elif input_dim > 512:
        max_dim = 64
        dim_choices = (16, 32, 64)
    else:
        max_dim = 32
        dim_choices = (8, 16, 32)
    max_layers = 3

    generator = ModelGenerator(
        input_dim=input_dim,
        output_dim=output_dim,
        max_layers=max_layers,
        max_dim=max_dim,
        dim_choices=dim_choices,
        latent_dim=latent_dim,
        noise_dim=latent_dim,
        num_prototypes=num_prototypes,
    )
    executor = TaskNetwork()
    meta_loss = MetaLoss(
        lambda_p=5e-5,
        lambda_c=1e-7,
        lambda_m=1e-8,
        lambda_balance=0.1,
        lambda_stability=0.01,
        lambda_design=1e-4,
    )
    return generator, executor, meta_loss


def train_mgn(generator, executor, meta_loss, train_x, train_y,
              test_x, test_y, epochs=100, batch_size=128, device=DEVICE, label=""):
    t0 = time.time()
    trainer = MetaTrainer(generator, executor, meta_loss, lr=1e-3, device=device,
                          preprocess_fn=cifar_hflip_flat)
    trainer.train(train_x, train_y, epochs=epochs, batch_size=batch_size)
    elapsed = time.time() - t0

    generator.eval()
    with torch.no_grad():
        test_model = generator(test_x.to(device))
        test_logits = executor(test_x.to(device), test_model)
        test_acc = (test_logits.argmax(-1) == test_y.to(device)).float().mean().item()

        train_model = generator(train_x.to(device))
        train_logits = executor(train_x.to(device), train_model)
        train_acc = (train_logits.argmax(-1) == train_y.to(device)).float().mean().item()

        P = test_model.structure.effective_params_formula().mean().item()
        L = test_model.structure.num_layers.mean().item()
        D = test_model.structure.hidden_dim.mean().item()

        pi = torch.softmax(test_model.pi_logits, dim=-1)
        pi_avg = pi.mean(dim=0)
        entropy = -(pi_avg * torch.log(pi_avg + 1e-8)).sum().item()
        n_used = (pi_avg > 0.01).sum().item()

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "P": P,
        "L": L,
        "D": D,
        "entropy": entropy,
        "prototypes_used": n_used,
        "time": elapsed,
    }


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_benchmark(dataset_name, seeds=5, quick=False):
    print(f"\n{'='*70}")
    print(f"Benchmark: {dataset_name}")
    print(f"{'='*70}")

    max_train = 5000 if quick else 10000 if dataset_name in ("imdb",) else 20000 if dataset_name in ("ag_news",) else None
    max_test = 1000 if quick else 2000

    print(f"Loading {dataset_name}... (train={max_train or 'full'}, test={max_test})")
    data = get_dataset(dataset_name, max_train=max_train, max_test=max_test, max_features=2000)
    print(f"  Input dim: {data['input_dim']}  Output dim: {data['output_dim']}")
    print(f"  Train: {data['n_train']}  Test: {data['n_test']}")

    train_x, train_y = data["train_x"], data["train_y"]
    test_x, test_y = data["test_x"], data["test_y"]

    results = {"dataset": dataset_name, "seeds": seeds, "methods": {}}

    epochs = 50 if quick else 100

    # --- MGN ---
    print(f"\n--- MGN ({seeds} seeds) ---")
    mgn_train_accs, mgn_test_accs, mgn_Ps, mgn_times = [], [], [], []
    mgn_Ls, mgn_Ds, mgn_entropies = [], [], []
    for seed in range(seeds):
        set_seed(42 + seed)
        gen, exec_, loss = make_mgn(data["input_dim"], data["output_dim"])
        res = train_mgn(gen, exec_, loss, train_x, train_y, test_x, test_y,
                        epochs=epochs, label=f"MGN seed={seed}")
        mgn_test_accs.append(res["test_acc"])
        mgn_train_accs.append(res["train_acc"])
        mgn_Ps.append(res["P"])
        mgn_times.append(res["time"])
        mgn_Ls.append(res["L"])
        mgn_Ds.append(res["D"])
        mgn_entropies.append(res["entropy"])
        print(f"  Seed {seed}: test={res['test_acc']:.4f} P={res['P']:.0f} L={res['L']:.2f} D={res['D']:.1f}")

    results["methods"]["MGN"] = {
        "test_acc_mean": float(np.mean(mgn_test_accs)),
        "test_acc_std": float(np.std(mgn_test_accs, ddof=1)),
        "train_acc_mean": float(np.mean(mgn_train_accs)),
        "train_acc_std": float(np.std(mgn_train_accs, ddof=1)),
        "P_mean": float(np.mean(mgn_Ps)),
        "P_std": float(np.std(mgn_Ps, ddof=1)),
        "L_mean": float(np.mean(mgn_Ls)),
        "entropy_mean": float(np.mean(mgn_entropies)),
        "time_mean": float(np.mean(mgn_times)),
        "all_test_accs": mgn_test_accs,
        "all_Ps": mgn_Ps,
    }

    # --- Best Fixed MLP ---
    print(f"\n--- Best MLP (grid search, {seeds} seeds) ---")
    mlp_test_accs, mlp_Ps, mlp_times = [], [], []
    for seed in range(seeds):
        set_seed(42 + seed)
        best = search_best_mlp(train_x, train_y, test_x, test_y,
                               data["input_dim"], data["output_dim"],
                               device=DEVICE, epochs=epochs//2, verbose=False)
        mlp_test_accs.append(best["test_acc"])
        mlp_Ps.append(best["P"])
        mlp_times.append(best["time"])
        print(f"  Seed {seed}: test={best['test_acc']:.4f} P={best['P']} L={best['L']} D={best['D']}")

    results["methods"]["BestMLP"] = {
        "test_acc_mean": float(np.mean(mlp_test_accs)),
        "test_acc_std": float(np.std(mlp_test_accs, ddof=1)),
        "P_mean": float(np.mean(mlp_Ps)),
        "P_std": float(np.std(mlp_Ps, ddof=1)),
        "time_mean": float(np.mean(mlp_times)),
        "all_test_accs": mlp_test_accs,
        "all_Ps": mlp_Ps,
    }

    # --- Random Forest ---
    print("\n--- Random Forest ---")
    rf_res = train_rf(train_x, train_y, test_x, test_y, n_estimators=200)
    print(f"  test={rf_res['test_acc']:.4f} P≈{rf_res['P']}")
    results["methods"]["RandomForest"] = {
        "test_acc_mean": rf_res["test_acc"],
        "test_acc_std": 0.0,
        "P_mean": rf_res["P"],
        "P_std": 0.0,
        "time_mean": rf_res["time"],
    }

    # --- Small CNN (image datasets only) ---
    if dataset_name in ("mnist", "fashion_mnist", "cifar10"):
        print(f"\n--- Small CNN ({seeds} seeds) ---")
        cnn_test_accs, cnn_Ps, cnn_times = [], [], []
        in_channels = 1 if data["input_dim"] in (784,) else 3
        for seed in range(seeds):
            set_seed(42 + seed)
            cnn = SmallCNN(in_channels, data["input_dim"], data["output_dim"])
            res = train_torch(cnn, train_x, train_y, test_x, test_y,
                              epochs=epochs, device=DEVICE, label=f"CNN seed={seed}")
            cnn_test_accs.append(res["test_acc"])
            cnn_Ps.append(res["P"])
            cnn_times.append(res["time"])
            print(f"  Seed {seed}: test={res['test_acc']:.4f} P={res['P']}")

        results["methods"]["SmallCNN"] = {
            "test_acc_mean": float(np.mean(cnn_test_accs)),
            "test_acc_std": float(np.std(cnn_test_accs, ddof=1)),
            "P_mean": float(np.mean(cnn_Ps)),
            "P_std": float(np.std(cnn_Ps, ddof=1)),
            "time_mean": float(np.mean(cnn_times)),
            "all_test_accs": cnn_test_accs,
            "all_Ps": cnn_Ps,
        }

    # --- MoE ---
    print(f"\n--- MoE (4 experts, {seeds} seeds) ---")
    moe_test_accs, moe_Ps, moe_times = [], [], []
    for seed in range(seeds):
        set_seed(42 + seed)
        hidden = min(128, data["input_dim"] // 4)
        moe = FixedMoE(data["input_dim"], hidden, 4, data["output_dim"])
        res = train_torch(moe, train_x, train_y, test_x, test_y,
                          epochs=epochs, device=DEVICE, label=f"MoE seed={seed}")
        moe_test_accs.append(res["test_acc"])
        moe_Ps.append(res["P"])
        moe_times.append(res["time"])
        print(f"  Seed {seed}: test={res['test_acc']:.4f} P={res['P']}")

    results["methods"]["MoE"] = {
        "test_acc_mean": float(np.mean(moe_test_accs)),
        "test_acc_std": float(np.std(moe_test_accs, ddof=1)),
        "P_mean": float(np.mean(moe_Ps)),
        "P_std": float(np.std(moe_Ps, ddof=1)),
        "time_mean": float(np.mean(moe_times)),
        "all_test_accs": moe_test_accs,
        "all_Ps": moe_Ps,
    }

    return results


def summarize(results_list):
    print(f"\n{'='*70}")
    print("Summary: Test Accuracy (mean ± std)  |  Parameters (mean ± std)")
    print(f"{'='*70}")

    for res in results_list:
        ds = res["dataset"]
        print(f"\n{ds}:")
        print(f"  {'Method':<16} {'Test Acc':<16} {'Params':<16} {'Time (s)':<10}")
        print(f"  {'-'*16} {'-'*16} {'-'*16} {'-'*10}")
        for method, metrics in res["methods"].items():
            ta = metrics["test_acc_mean"]
            ts = metrics.get("test_acc_std", 0)
            pm = metrics["P_mean"]
            ps = metrics.get("P_std", 0)
            tm = metrics["time_mean"]
            print(f"  {method:<16} {ta:.4f}±{ts:.4f}   {pm:>8.0f}±{ps:.0f}   {tm:.1f}")

    print(f"\n{'='*70}")
    print("Statistical Significance (Welch's t-test, p<0.05 vs MGN)")
    print(f"{'='*70}")
    for res in results_list:
        ds = res["dataset"]
        if "MGN" not in res["methods"]:
            continue
        mgn_accs = np.array(res["methods"]["MGN"]["all_test_accs"])
        print(f"\n{ds}:")
        for method, metrics in res["methods"].items():
            if method == "MGN" or "all_test_accs" not in metrics:
                continue
            other_accs = np.array(metrics["all_test_accs"])
            if len(other_accs) < 2:
                continue
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(mgn_accs, other_accs, equal_var=False)
            sig = "SIGNIFICANT" if p_val < 0.05 else "n.s."
            direction = "BETTER" if metrics["test_acc_mean"] > res["methods"]["MGN"]["test_acc_mean"] else "WORSE"
            print(f"  MGN vs {method:<12}: {sig:>13} (Δ={metrics['test_acc_mean']-res['methods']['MGN']['test_acc_mean']:+.4f}, {direction}, p={p_val:.4f})")


def plot_comparison(results_list, save_dir="experiments"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    ax = axes[0]
    datasets = [r["dataset"] for r in results_list]
    methods = list(results_list[0]["methods"].keys())
    x = np.arange(len(datasets))
    n_methods = len(methods)
    width = 0.8 / n_methods

    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
    for idx, method in enumerate(methods):
        means = [r["methods"].get(method, {}).get("test_acc_mean", 0) for r in results_list]
        stds = [r["methods"].get(method, {}).get("test_acc_std", 0) for r in results_list]
        offset = (idx - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=method, color=colors[idx], alpha=0.8, edgecolor="k")
        ax.errorbar(x + offset, means, yerr=stds, fmt="none", capsize=3, color="black", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Phase 2: Real Dataset Benchmark")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Parameters comparison
    ax = axes[1]
    for idx, method in enumerate(methods):
        means = [r["methods"].get(method, {}).get("P_mean", 0) for r in results_list]
        offset = (idx - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=method, color=colors[idx], alpha=0.8, edgecolor="k")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylabel("Parameters (P)")
    ax.set_title("Model Size Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")

    plt.tight_layout()
    path = os.path.join(save_dir, "benchmark_comparison.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n[Saved {path}]")

    # Efficiency plot (accuracy vs params)
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ["o", "s", "^", "D", "v", "<", ">"]
    for idx, method in enumerate(methods):
        for r_idx, res in enumerate(results_list):
            ds = res["dataset"]
            if method not in res["methods"]:
                continue
            ta = res["methods"][method]["test_acc_mean"]
            pm = res["methods"][method]["P_mean"]
            ax.scatter(pm, ta, marker=markers[idx % len(markers)],
                       color=colors[idx], s=100, alpha=0.7, edgecolor="k",
                       label=f"{method}-{ds}" if r_idx == 0 else None)
            ax.annotate(ds, (pm, ta), xytext=(4, 4), textcoords="offset points", fontsize=7, alpha=0.7)

    ax.set_xlabel("Parameters (P)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Efficiency: Accuracy vs Model Size")
    ax.set_xscale("log")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "benchmark_efficiency.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[Saved {path}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="mnist,fashion_mnist,cifar10")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Smaller subsets, fewer epochs")
    args = parser.parse_args()

    dataset_names = [d.strip() for d in args.datasets.split(",")]
    all_results = []

    for ds in dataset_names:
        result = run_benchmark(ds, seeds=args.seeds, quick=args.quick)
        all_results.append(result)

        with open(f"experiments/benchmark_{ds}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  [Saved experiments/benchmark_{ds}.json]")

    with open("experiments/benchmark_all.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("  [Saved experiments/benchmark_all.json]")

    summarize(all_results)
    plot_comparison(all_results)
