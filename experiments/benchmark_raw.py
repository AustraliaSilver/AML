"""
Phase 2 Benchmark: AML on raw image data (no PCA).
Usage: python experiments/benchmark_raw.py --datasets cifar10 --seeds 3
"""
import sys, os, json, time, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '/kaggle/working/ALM-V1')

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from mgn import ModelGenerator, TaskNetwork, MetaLoss, MetaTrainer
from experiments.benchmark_datasets import get_dataset
from experiments.benchmark_baselines import FixedMLP, FixedMoE, SmallCNN, train_torch, search_best_mlp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_benchmark(dataset_name, seeds=3, quick=False):
    print(f"\n{'='*70}")
    print(f"Benchmark: {dataset_name} (RAW, seeds={seeds})")
    print(f"{'='*70}")

    max_train = 3000 if quick else None
    max_test = 500 if quick else 2000
    epochs = 80 if quick else 150

    data = get_dataset(dataset_name, max_train=max_train, max_test=max_test)
    
    train_x = data["train_x"].float() / 255.0
    test_x = data["test_x"].float() / 255.0
    
    # Normalize
    train_x = (train_x - train_x.mean()) / (train_x.std() + 1e-8)
    test_x = (test_x - test_x.mean()) / (test_x.std() + 1e-8)
    train_y, test_y = data["train_y"], data["test_y"]

    print(f"  Input={data['input_dim']} Output={data['output_dim']}")
    print(f"  Train: {data['n_train']} Test: {data['n_test']}")

    results = {"dataset": dataset_name, "seeds": seeds, "methods": {}}

    # AML on raw flattened features
    in_channels = 1 if data["input_dim"] in (784,) else 3
    
    print(f"\n--- AML-Raw (input_dim={data['input_dim']}, {'GPU' if DEVICE=='cuda' else 'CPU'}, {seeds} seeds) ---")
    aml_list = []
    for seed in range(seeds):
        set_seed(42 + seed)
        gen = ModelGenerator(
            input_dim=data["input_dim"], output_dim=data["output_dim"],
            max_layers=2, max_dim=64, dim_choices=(32, 64),
            latent_dim=32, noise_dim=32, num_prototypes=8,
        )
        exec_ = TaskNetwork(dropout=0.2)
        loss = MetaLoss(lambda_p=1e-4, lambda_c=1e-7, lambda_m=1e-8,
                      lambda_balance=0.5, lambda_stability=0.1, lambda_design=1e-4)
        
        trainer = MetaTrainer(gen, exec_, loss, lr=1e-3, device=DEVICE)
        
        # Add L2 regularization
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=256, shuffle=True)
        
        t0 = time.time()
        for epoch in range(epochs):
            gen.train()
            for xb, yb in loader:
                trainer.optimizer.zero_grad()
                model = gen(xb, tau=0.4)
                logits = exec_(xb, model)
                l2 = sum(p.pow(2).sum() for p in model.weights.values()) * 1e-5
                task_loss = torch.nn.functional.cross_entropy(logits, yb, label_smoothing=0.1)
                reg = loss(xb, yb, model)['resource_cost']
                (task_loss + reg + l2).backward()
                trainer.optimizer.step()
            
            if epoch % 40 == 0:
                gen.eval()
                with torch.no_grad():
                    model = gen(test_x)
                    acc = (exec_(test_x, model).argmax(-1) == test_y).float().mean().item()
                    P = model.structure.effective_params_formula().mean().item()
                print(f"  Seed {seed} Epoch {epoch}: acc={acc:.4f} P={P:.0f}")
        
        elapsed = time.time() - t0
        gen.eval()
        with torch.no_grad():
            model = gen(test_x.to(DEVICE))
            test_acc = (exec_(test_x.to(DEVICE), model).argmax(-1) == test_y.to(DEVICE)).float().mean().item()
            train_model = gen(train_x.to(DEVICE))
            train_acc = (exec_(train_x.to(DEVICE), train_model).argmax(-1) == train_y.to(DEVICE)).float().mean().item()
            P = model.structure.effective_params_formula().mean().item()
        
        res = {"test_acc": test_acc, "train_acc": train_acc, "P": P, "time": elapsed}
        aml_list.append(res)
        print(f"  Seed {seed}: test={res['test_acc']:.4f} train={res['train_acc']:.4f} P={res['P']:.0f}")
    
    results["methods"]["AML-Raw"] = {
        "test_acc_mean": float(np.mean([r["test_acc"] for r in aml_list])),
        "test_acc_std": float(np.std([r["test_acc"] for r in aml_list], ddof=1)),
        "P_mean": float(np.mean([r["P"] for r in aml_list])),
        "time_mean": float(np.mean([r["time"] for r in aml_list])),
        "overfitting_gap": float(np.mean([r["train_acc"] - r["test_acc"] for r in aml_list])),
    }

    # SmallCNN baseline (raw images)
    print(f"\n--- SmallCNN (raw images, {seeds} seeds) ---")
    cnn_list = []
    for seed in range(seeds):
        set_seed(42 + seed)
        cnn = SmallCNN(in_channels, data["input_dim"], data["output_dim"])
        res = train_torch(cnn, train_x, train_y, test_x, test_y, epochs=epochs, device=DEVICE)
        cnn_list.append(res)
        print(f"  Seed {seed}: test={res['test_acc']:.4f} P={res['P']}")

    results["methods"]["SmallCNN"] = {
        "test_acc_mean": float(np.mean([r["test_acc"] for r in cnn_list])),
        "P_mean": float(np.mean([r["P"] for r in cnn_list])),
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="cifar10")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    ds_names = [d.strip() for d in args.datasets.split(",")]
    all_results = []
    for ds in ds_names:
        res = run_benchmark(ds, seeds=args.seeds, quick=args.quick)
        all_results.append(res)
        with open(f"experiments/benchmark_raw_{ds}.json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"  [Saved experiments/benchmark_raw_{ds}.json]")

    # Summary
    print("\n" + "="*70)
    print("Summary: Overfitting Analysis")
    print("="*70)
    for res in all_results:
        ds = res["dataset"]
        aml = res["methods"].get("AML-Raw", {})
        cnn = res["methods"].get("SmallCNN", {})
        gap = aml.get("overfitting_gap", 0)
        print(f"\n{ds}:")
        print(f"  AML-Raw: {aml.get('test_acc_mean', 0):.4f} (gap={gap:.3f}) P={aml.get('P_mean', 0):.0f}")
        print(f"  SmallCNN: {cnn.get('test_acc_mean', 0):.4f} P={cnn.get('P_mean', 0):.0f}")