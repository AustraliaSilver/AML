"""
Phase 2 Benchmark: MGN vs Baselines with PCA-reduced inputs.
MGN and MLP baselines use PCA(dim_reduce) features.
CNN baselines use raw images.

Usage:
    python experiments/benchmark_pca.py [--datasets mnist,fashion_mnist,cifar10] [--seeds 5] [--pca_dim 32]
"""

import sys, os, json, time, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from mgn import ModelGenerator, TaskNetwork, MetaLoss, MetaTrainer
from experiments.benchmark_datasets import get_dataset
from experiments.benchmark_baselines import FixedMLP, FixedMoE, SmallCNN, train_torch, search_best_mlp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


def apply_pca(train_x, test_x, n_components):
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_x.numpy())
    test_pca = pca.transform(test_x.numpy())
    tr = torch.tensor(train_pca, dtype=torch.float32)
    te = torch.tensor(test_pca, dtype=torch.float32)
    xm = tr.mean(0, keepdim=True)
    xs = tr.std(0, keepdim=True) + 1e-8
    return (tr - xm) / xs, (te - xm) / xs


def make_mgn(input_dim, output_dim):
    gen = ModelGenerator(
        input_dim=input_dim, output_dim=output_dim,
        max_layers=3, max_dim=32, dim_choices=(8, 16, 32),
        latent_dim=64, noise_dim=64, num_prototypes=8,
    )
    exec_ = TaskNetwork()
    loss = MetaLoss(lambda_p=5e-5, lambda_c=1e-7, lambda_m=1e-8,
                    lambda_balance=0.1, lambda_stability=0.01, lambda_design=1e-4)
    return gen, exec_, loss


def train_mgn(gen, exec_, loss, train_x, train_y, test_x, test_y,
              epochs=100, batch_size=128, device=DEVICE):
    t0 = time.time()
    trainer = MetaTrainer(gen, exec_, loss, lr=1e-3, device=device)
    trainer.train(train_x, train_y, epochs=epochs, batch_size=batch_size)
    elapsed = time.time() - t0
    gen.eval()
    with torch.no_grad():
        model = gen(test_x.to(device))
        logits = exec_(test_x.to(device), model)
        test_acc = (logits.argmax(-1) == test_y.to(device)).float().mean().item()
        train_model = gen(train_x.to(device))
        train_logits = exec_(train_x.to(device), train_model)
        train_acc = (train_logits.argmax(-1) == train_y.to(device)).float().mean().item()
        P = model.structure.effective_params_formula().mean().item()
        L = model.structure.num_layers.mean().item()
        D = model.structure.hidden_dim.mean().item()
        pi = torch.softmax(model.pi_logits, dim=-1)
        entropy = -(pi.mean(0) * torch.log(pi.mean(0) + 1e-8)).sum().item()
        n_used = (pi.mean(0) > 0.01).sum().item()
    return {"test_acc": test_acc, "train_acc": train_acc, "P": P, "L": L, "D": D,
            "entropy": entropy, "prototypes_used": n_used, "time": elapsed}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_benchmark(dataset_name, seeds=5, pca_dim=32, quick=False):
    print(f"\n{'='*70}")
    print(f"Benchmark: {dataset_name} (PCA={pca_dim}, seeds={seeds})")
    print(f"{'='*70}")

    max_train = 3000 if quick else None
    max_test = 500 if quick else 2000
    epochs = 80 if quick else 150

    data = get_dataset(dataset_name, max_train=max_train, max_test=max_test)
    print(f"  Raw: {data['input_dim']} -> PCA({pca_dim})  Output={data['output_dim']}")
    print(f"  Train: {data['n_train']}  Test: {data['n_test']}")

    # PCA
    train_x_pca, test_x_pca = apply_pca(data["train_x"], data["test_x"], pca_dim)
    train_y, test_y = data["train_y"], data["test_y"]

    results = {"dataset": dataset_name, "pca_dim": pca_dim, "seeds": seeds, "methods": {}}

    # ======== MGN ========
    print(f"\n--- MGN ({'GPU' if DEVICE=='cuda' else 'CPU'}, {seeds} seeds) ---")
    mgn_list = []
    for seed in range(seeds):
        set_seed(42 + seed)
        gen, exec_, loss = make_mgn(pca_dim, data["output_dim"])
        res = train_mgn(gen, exec_, loss, train_x_pca, train_y, test_x_pca, test_y, epochs=epochs)
        mgn_list.append(res)
        print(f"  Seed {seed}: test={res['test_acc']:.4f} P={res['P']:.0f} L={res['L']:.2f} D={res['D']:.1f} H={res['entropy']:.3f} t={res['time']:.0f}s")

    results["methods"]["MGN"] = {
        "test_acc_mean": float(np.mean([r["test_acc"] for r in mgn_list])),
        "test_acc_std": float(np.std([r["test_acc"] for r in mgn_list], ddof=1)),
        "P_mean": float(np.mean([r["P"] for r in mgn_list])),
        "P_std": float(np.std([r["P"] for r in mgn_list], ddof=1)),
        "time_mean": float(np.mean([r["time"] for r in mgn_list])),
        "all_test_accs": [r["test_acc"] for r in mgn_list],
        "all_Ps": [r["P"] for r in mgn_list],
        "avg_L": float(np.mean([r["L"] for r in mgn_list])),
        "avg_entropy": float(np.mean([r["entropy"] for r in mgn_list])),
        "avg_prototypes_used": float(np.mean([r["prototypes_used"] for r in mgn_list])),
    }

    # ======== Best Fixed MLP (on PCA features) ========
    print(f"\n--- BestMLP (grid on PCA, {seeds} seeds) ---")
    mlp_list = []
    for seed in range(seeds):
        set_seed(42 + seed)
        best = search_best_mlp(train_x_pca, train_y, test_x_pca, test_y,
                               pca_dim, data["output_dim"], device=DEVICE, epochs=epochs // 2, verbose=False)
        mlp_list.append(best)
        print(f"  Seed {seed}: test={best['test_acc']:.4f} P={best['P']} L={best['L']} D={best['D']} t={best['time']:.0f}s")

    results["methods"]["BestMLP"] = {
        "test_acc_mean": float(np.mean([r["test_acc"] for r in mlp_list])),
        "test_acc_std": float(np.std([r["test_acc"] for r in mlp_list], ddof=1)),
        "P_mean": float(np.mean([r["P"] for r in mlp_list])),
        "P_std": float(np.std([r["P"] for r in mlp_list], ddof=1)),
        "time_mean": float(np.mean([r["time"] for r in mlp_list])),
        "all_test_accs": [r["test_acc"] for r in mlp_list],
        "all_Ps": [r["P"] for r in mlp_list],
    }

    # ======== MoE (on PCA features) ========
    print(f"\n--- MoE (on PCA, {seeds} seeds) ---")
    moe_list = []
    for seed in range(seeds):
        set_seed(42 + seed)
        moe = FixedMoE(pca_dim, 32, 4, data["output_dim"])
        res = train_torch(moe, train_x_pca, train_y, test_x_pca, test_y,
                          epochs=epochs, device=DEVICE)
        moe_list.append(res)
        print(f"  Seed {seed}: test={res['test_acc']:.4f} P={res['P']} t={res['time']:.0f}s")

    results["methods"]["MoE"] = {
        "test_acc_mean": float(np.mean([r["test_acc"] for r in moe_list])),
        "test_acc_std": float(np.std([r["test_acc"] for r in moe_list], ddof=1)),
        "P_mean": float(np.mean([r["P"] for r in moe_list])),
        "P_std": float(np.std([r["P"] for r in moe_list], ddof=1)),
        "time_mean": float(np.mean([r["time"] for r in moe_list])),
        "all_test_accs": [r["test_acc"] for r in moe_list],
        "all_Ps": [r["P"] for r in moe_list],
    }

    # ======== CNN (raw images) ========
    if dataset_name in ("mnist", "fashion_mnist", "cifar10"):
        in_channels = 1 if data["input_dim"] in (784,) else 3
        print(f"\n--- SmallCNN (raw images, {seeds} seeds) ---")
        cnn_list = []
        for seed in range(seeds):
            set_seed(42 + seed)
            cnn = SmallCNN(in_channels, data["input_dim"], data["output_dim"])
            res = train_torch(cnn, data["train_x"], data["train_y"], data["test_x"], data["test_y"],
                              epochs=epochs, device=DEVICE)
            cnn_list.append(res)
            print(f"  Seed {seed}: test={res['test_acc']:.4f} P={res['P']} t={res['time']:.0f}s")

        results["methods"]["SmallCNN"] = {
            "test_acc_mean": float(np.mean([r["test_acc"] for r in cnn_list])),
            "test_acc_std": float(np.std([r["test_acc"] for r in cnn_list], ddof=1)),
            "P_mean": float(np.mean([r["P"] for r in cnn_list])),
            "P_std": float(np.std([r["P"] for r in cnn_list], ddof=1)),
            "time_mean": float(np.mean([r["time"] for r in cnn_list])),
            "all_test_accs": [r["test_acc"] for r in cnn_list],
            "all_Ps": [r["P"] for r in cnn_list],
        }

    # ======== Random Forest (on PCA features) ========
    print(f"\n--- Random Forest ---")
    from experiments.benchmark_baselines import train_rf
    rf_res = train_rf(train_x_pca, train_y, test_x_pca, test_y, n_estimators=200)
    print("  test={:.4f} P~{}".format(rf_res['test_acc'], rf_res['P']))
    results["methods"]["RandomForest"] = {
        "test_acc_mean": rf_res["test_acc"],
        "test_acc_std": 0.0,
        "P_mean": rf_res["P"],
        "P_std": 0.0,
        "time_mean": rf_res["time"],
    }

    return results


def summarize(results_list):
    print(f"\n{'='*70}")
    print("Summary: Test Accuracy (mean ± std)  |  Parameters (mean ± std)")
    print(f"{'='*70}")

    for res in results_list:
        ds = res["dataset"]
        print(f"\n{ds} (PCA={res.get('pca_dim', '')}):")
        print(f"  {'Method':<16} {'Test Acc':<20} {'Params':<16} {'Time (s)':<10} {'Efficiency':<16}")
        print(f"  {'-'*16} {'-'*20} {'-'*16} {'-'*10} {'-'*16}")
        for method, metrics in res["methods"].items():
            ta = metrics["test_acc_mean"]
            ts = metrics.get("test_acc_std", 0)
            pm = metrics["P_mean"]
            ps = metrics.get("P_std", 0)
            tm = metrics["time_mean"]
            # Efficiency: accuracy / log(params)
            eff = ta / (np.log10(pm + 1))
            print(f"  {method:<16} {ta:.4f}±{ts:.4f}   {pm:>8.0f}±{ps:.0f}   {tm:.1f}     {eff:.4f}")

    print(f"\n{'='*70}")
    print("Welch's t-test (p<0.05, vs MGN)")
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
            t_stat, p_val = stats.ttest_ind(mgn_accs, other_accs, equal_var=False)
            sig = "SIG" if p_val < 0.05 else "n.s."
            delta = metrics["test_acc_mean"] - res["methods"]["MGN"]["test_acc_mean"]
            print(f"  MGN vs {method:<12}: {sig:>4} (Δ={delta:+.4f}, p={p_val:.4f})")


def plot_results(results_list, save_dir="experiments"):
    markers = {"MGN": "o", "BestMLP": "s", "MoE": "^", "SmallCNN": "D", "RandomForest": "v"}
    colors = {"MGN": "#E74C3C", "BestMLP": "#3498DB", "MoE": "#2ECC71",
              "SmallCNN": "#F39C12", "RandomForest": "#9B59B6"}

    datasets = [r["dataset"] for r in results_list]
    methods = [m for m in ["MGN", "BestMLP", "MoE", "SmallCNN", "RandomForest"] if m in results_list[0]["methods"]]

    # Accuracy bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)
    for idx, method in enumerate(methods):
        means = [r["methods"].get(method, {}).get("test_acc_mean", 0) for r in results_list]
        stds = [r["methods"].get(method, {}).get("test_acc_std", 0) for r in results_list]
        offset = (idx - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=method, color=colors.get(method, f"C{idx}"),
               alpha=0.85, edgecolor="k", linewidth=0.5)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none", capsize=3, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Phase 2: Real Dataset Benchmark", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Parameter efficiency
    ax = axes[1]
    for idx, method in enumerate(methods):
        means = [r["methods"].get(method, {}).get("P_mean", 0) for r in results_list]
        offset = (idx - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=method, color=colors.get(method, f"C{idx}"),
               alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Parameters (P)", fontsize=12)
    ax.set_title("Model Size Comparison", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "benchmark_accuracy_params.png"), dpi=120)
    plt.close()
    print(f"  [Saved benchmark_accuracy_params.png]")

    # Efficiency scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    for r_idx, res in enumerate(results_list):
        ds_label = res["dataset"]
        for method in methods:
            if method not in res["methods"]:
                continue
            ta = res["methods"][method]["test_acc_mean"]
            pm = res["methods"][method]["P_mean"]
            ax.scatter(pm, ta, marker=markers.get(method, "o"), color=colors.get(method, f"C{methods.index(method)}"),
                       s=120, alpha=0.8, edgecolor="k", linewidth=0.5,
                       label=f"{method}" if r_idx == 0 else "")
            ax.annotate(f"{ds_label[:4]}", (pm, ta), xytext=(3, 3),
                        textcoords="offset points", fontsize=7, alpha=0.8)
    ax.set_xlabel("Parameters (P, log scale)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Efficiency: Accuracy vs Model Size", fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "benchmark_efficiency.png"), dpi=120)
    plt.close()
    print(f"  [Saved benchmark_efficiency.png]")

    # Parameter efficiency ratio bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    mgn_ps = [r["methods"]["MGN"]["P_mean"] for r in results_list]
    for idx, method in enumerate(methods):
        if method == "MGN":
            continue
        ratios = []
        for r_idx, res in enumerate(results_list):
            if method not in res["methods"]:
                continue
            ratio = res["methods"][method]["P_mean"] / mgn_ps[r_idx]
            ratios.append(ratio)
        if ratios:
            offset = (idx - len(methods) / 2 + 0.5) * width
            ax.bar(x + offset, ratios, width, label=method, color=colors.get(method, f"C{idx}"),
                   alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("P / P_MGN (ratio)", fontsize=12)
    ax.set_title("Parameter Inefficiency vs MGN", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "benchmark_param_ratio.png"), dpi=120)
    plt.close()
    print(f"  [Saved benchmark_param_ratio.png]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="mnist,fashion_mnist")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    ds_names = [d.strip() for d in args.datasets.split(",")]
    all_results = []
    for ds in ds_names:
        res = run_benchmark(ds, seeds=args.seeds, pca_dim=args.pca_dim, quick=args.quick)
        all_results.append(res)
        with open(f"experiments/benchmark_{ds}.json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"  [Saved experiments/benchmark_{ds}.json]")

    with open("experiments/benchmark_all.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("  [Saved experiments/benchmark_all.json]")

    summarize(all_results)
    plot_results(all_results)
