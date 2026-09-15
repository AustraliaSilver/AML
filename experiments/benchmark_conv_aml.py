"""
Phase 3 Benchmark: ConvAML (Convolutional Architecture Mode Learning) vs Baselines on Raw Image Datasets
Directly evaluates ConvAML on spatial image datasets (MNIST, Fashion-MNIST, CIFAR-10) without PCA feature reduction.
"""

import sys, os, json, time, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")

from mgn import ModelGenerator, TaskNetwork, MetaLoss, MetaTrainer, compute_aml_efficiency
from experiments.benchmark_datasets import get_dataset, cifar_hflip_flat
from experiments.benchmark_baselines import SmallCNN, FixedMLP, ResNetBlockBaseline, train_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

def set_seed(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_conv_aml(input_dim, output_dim, in_channels, fused=False):
    lp = 0.0
    # Huge architecture: L=4, D=128 => p_allow inflated to accommodate very large nets
    p_allow = 562186.0 if in_channels == 3 else 506762.0
    gen = ModelGenerator(
        input_dim=input_dim, output_dim=output_dim,
        max_layers=4, max_dim=128, dim_choices=(32, 64, 96, 128),
        latent_dim=64, noise_dim=64, num_prototypes=4,
        use_conv=True, in_channels=in_channels
    )
    exec_ = TaskNetwork(fused=fused)
    # lambda_balance=0: the -entropy(pi_avg) routing-balance term (weight 1.0)
    # fought the task objective and cost ~3 points in controlled ablations
    # (probes/probe_gate_collapse.py, round 2); diversity emerges without it.
    p_allow = 562186.0 if in_channels == 3 else 506762.0
    loss = MetaLoss(lambda_p=lp, lambda_c=1e-9, lambda_m=0,
                    lambda_balance=0.0, lambda_stability=1e-6, lambda_design=0,
                    lambda_eff=0.2, p_allowance=p_allow)
    return gen, exec_, loss

def train_conv_aml(gen, exec_, loss, train_x, train_y, val_x, val_y, test_x, test_y, epochs=100, batch_size=64, device=DEVICE, compile=False, compile_mode="default", gate_floor=0.0, gate_floor_end=0.0, freeze_arch_at=None):
    t0 = time.time()
    # lr=1e-3: matches the plain baselines' optimizer scale; +2 points over
    # the old 5e-4 in controlled ablation (probes/probe_gate_collapse.py,
    # rounds 4-5). lambda_balance=0 likewise per round 2 (+3.1).
    # weight_decay=0.01: non-monotonic sweep found 0.05 too strong and 0
    # too weak (+0.9 vs both, round 6).
    trainer = MetaTrainer(gen, exec_, loss, lr=1e-3, device=device,
                          weight_decay=0.01, preprocess_fn=cifar_hflip_flat,
                          compile=compile, compile_mode=compile_mode)
    trainer.train(train_x, train_y, val_x=val_x, val_y=val_y, epochs=epochs, batch_size=batch_size,
                  gate_floor_start=gate_floor, gate_floor_end=gate_floor_end,
                  freeze_arch_at=freeze_arch_at)
    elapsed = time.time() - t0
    gen.eval()
    exec_.eval()  # executor carries dropout; leaving it in train mode
    with torch.no_grad():  # depresses every accuracy measured here.
        def eval_conv_acc(x_tensor, y_tensor):
            correct, total = 0, 0
            for i in range(0, len(x_tensor), 256):
                xb = x_tensor[i:i+256].to(device)
                yb = y_tensor[i:i+256].to(device)
                m = gen(xb)
                correct += (exec_(xb, m).argmax(-1) == yb).sum().item()
                total += len(yb)
            return correct / total if total > 0 else 0.0

        test_acc = eval_conv_acc(test_x, test_y)
        train_acc = eval_conv_acc(train_x, train_y)
        
        sample_x = test_x[:512].to(device)
        sample_m = gen(sample_x)
        P = sample_m.structure.effective_params_formula().mean().item()
        L = sample_m.structure.num_layers.mean().item()
        D = sample_m.structure.hidden_dim.mean().item()
        S = gen(train_x[:10].to(DEVICE), tau=0.1, noise_scale=0.0)
        pi = F.softmax(gen.prototype_selector(gen.encoder(train_x[:10].view(10, -1).to(DEVICE))), dim=-1)
        entropy = -(pi.mean(0) * torch.log(pi.mean(0) + 1e-8)).sum().item()
        n_used = (pi.mean(0) > 0.01).sum().item()
        
        # Calculate ResNet block mode probability
        resnet_pct = 0.0
        vit_pct = 0.0
        if getattr(S.structure, "block_mode", None) is not None:
            resnet_pct = S.structure.block_mode[:, :, 1].mean().item() * 100.0
            if S.structure.block_mode.shape[-1] > 2:
                vit_pct = S.structure.block_mode[:, :, 2].mean().item() * 100.0
        
    return {
        "test_acc": test_acc, "train_acc": train_acc, "P": P, "L": L, "D": D,
        "entropy": entropy, "prototypes_used": n_used, "resnet_pct": resnet_pct, "vit_pct": vit_pct, "time": elapsed
    }

def run_conv_aml_benchmark(dataset_name="fashion_mnist", seeds=3, quick=False, fast=False, full=False, fixed_arch=False, only_baselines=False, compile=False, compile_mode="default", batch_size=64, gate_floor=0.0, gate_floor_end=0.0, freeze_arch_at=None, fused_executor=False):
    print(f"\n{'='*70}")
    print(f"ConvAML Benchmark: {dataset_name} (seeds={seeds}, full={full}, fixed_arch={fixed_arch})")
    print(f"{'='*70}")
    sys.stdout.flush()

    if full:
        max_train = None
        max_test = None
        epochs = 12
    else:
        max_train = 1200 if fast else (3000 if quick else 5000)
        max_test = 400 if fast else (1000 if quick else 2000)
        epochs = 25 if fast else (60 if quick else 120)

    # Check if a custom epochs value was passed globally
    if hasattr(args, "epochs") and args.epochs is not None:
        epochs = args.epochs

    data = get_dataset(dataset_name, max_train=max_train, max_test=max_test)
    in_channels = 3 if dataset_name == "cifar10" else 1
    print(f"  Dataset: {dataset_name} | Input Dim: {data['input_dim']} ({in_channels} channels) | Output: {data['output_dim']}")
    print(f"  Train samples: {data['n_train']} | Test samples: {data['n_test']}")
    sys.stdout.flush()

    results = {"dataset": dataset_name, "seeds": seeds, "methods": {}}
    ctag = ((" [compiled]" if compile else "")
            + (" [fused-exec]" if fused_executor else "")
            + (f" [gf={gate_floor}->{gate_floor_end}]" if gate_floor or gate_floor_end else "")
            + (f" [frz@{freeze_arch_at}]" if freeze_arch_at else ""))

    if not only_baselines:
        # 1. ConvAML
        print(f"\n--- ConvAML ({'GPU' if DEVICE=='cuda' else 'CPU'}, {seeds} seeds) ---")
        sys.stdout.flush()
        conv_list = []
        for seed in range(seeds):
            set_seed(42 + seed)
            gen, exec_, loss = make_conv_aml(data["input_dim"], data["output_dim"], in_channels, fused=fused_executor)
            
            if fixed_arch:
                for p in gen.prototype_selector.parameters(): p.requires_grad = False
                for p in gen.structure_delta.parameters(): p.requires_grad = False
                for p in gen.block_mode_head.parameters(): p.requires_grad = False
                
            n_train = len(data["train_x"])
            n_val = int(n_train * 0.2)
            indices = torch.randperm(n_train)
            train_idx = indices[:-n_val]
            val_idx = indices[-n_val:]
            
            trx = data["train_x"][train_idx]
            try_ = data["train_y"][train_idx]
            vx = data["train_x"][val_idx]
            vy = data["train_y"][val_idx]
                
            res = train_conv_aml(gen, exec_, loss, trx, try_, vx, vy, data["test_x"], data["test_y"], epochs=epochs, batch_size=batch_size, compile=compile, compile_mode=compile_mode, gate_floor=gate_floor, gate_floor_end=gate_floor_end, freeze_arch_at=freeze_arch_at)
            conv_list.append(res)
            print(f"  [ConvAML Seed {seed}]{ctag} Test Acc={res['test_acc']:.4f} (train {res['train_acc']:.4f}) | P_eff={int(res['P'])} | L={res['L']:.2f} | D={res['D']:.1f} | ResNet={res.get('resnet_pct', 0.0):.1f}% | ViT={res.get('vit_pct', 0.0):.1f}% | H={res.get('entropy', 0.0):.3f} | t={int(res['time'])}s")
            sys.stdout.flush()

        results["methods"]["ConvAML"] = {
            "test_acc_mean": float(np.mean([r["test_acc"] for r in conv_list])),
            "test_acc_std": float(np.std([r["test_acc"] for r in conv_list], ddof=1)) if seeds > 1 else 0.0,
            "P_mean": float(np.mean([r["P"] for r in conv_list])),
            "P_std": float(np.std([r["P"] for r in conv_list], ddof=1)) if seeds > 1 else 0.0,
            "time_mean": float(np.mean([r["time"] for r in conv_list])),
            "all_test_accs": [r["test_acc"] for r in conv_list],
            "all_train_accs": [r["train_acc"] for r in conv_list],
            "all_Ps": [r["P"] for r in conv_list],
            "avg_L": float(np.mean([r.get("L", 0.0) for r in conv_list])),
            "avg_entropy": float(np.mean([r.get("entropy", 0.0) for r in conv_list])),
            "avg_prototypes_used": float(np.mean([r.get("prototypes_used", 0) for r in conv_list])),
        }

        # ConvAML No O
        print(f"\n--- ConvAML (No O) (CPU, {seeds} seeds) ---")
        sys.stdout.flush()
        conv_no_o_list = []
        for seed in range(seeds):
            set_seed(42 + seed)
            gen, exec_, loss = make_conv_aml(data["input_dim"], data["output_dim"], in_channels, fused=fused_executor)
            loss.lambda_eff = 0.0
            
            if fixed_arch:
                for p in gen.prototype_selector.parameters(): p.requires_grad = False
                for p in gen.structure_delta.parameters(): p.requires_grad = False
                for p in gen.block_mode_head.parameters(): p.requires_grad = False
            
            n_train = len(data["train_x"])
            n_val = int(n_train * 0.2)
            indices = torch.randperm(n_train)
            train_idx = indices[:-n_val]
            val_idx = indices[-n_val:]
            trx = data["train_x"][train_idx]
            try_ = data["train_y"][train_idx]
            vx = data["train_x"][val_idx]
            vy = data["train_y"][val_idx]
                
            res = train_conv_aml(gen, exec_, loss, trx, try_, vx, vy, data["test_x"], data["test_y"], epochs=epochs, batch_size=batch_size, compile=compile, compile_mode=compile_mode, gate_floor=gate_floor, gate_floor_end=gate_floor_end, freeze_arch_at=freeze_arch_at)
            conv_no_o_list.append(res)
            print(f"  [ConvAML No O Seed {seed}] Test Acc={res['test_acc']:.4f} (train {res['train_acc']:.4f}) | P_eff={res['P']:.0f} | L={res['L']:.2f} | D={res['D']:.1f} | ResNet={res.get('resnet_pct', 0.0):.1f}% | H={res.get('entropy', 0.0):.3f} | t={res['time']:.0f}s")
            sys.stdout.flush()

        results["methods"]["ConvAML_No_O"] = {
            "test_acc_mean": float(np.mean([r["test_acc"] for r in conv_no_o_list])),
            "test_acc_std": float(np.std([r["test_acc"] for r in conv_no_o_list], ddof=1)) if seeds > 1 else 0.0,
            "P_mean": float(np.mean([r["P"] for r in conv_no_o_list])),
            "P_std": float(np.std([r["P"] for r in conv_no_o_list], ddof=1)) if seeds > 1 else 0.0,
            "time_mean": float(np.mean([r["time"] for r in conv_no_o_list])),
            "all_test_accs": [r["test_acc"] for r in conv_no_o_list],
            "all_train_accs": [r["train_acc"] for r in conv_no_o_list],
        }

    # 2. SmallCNN Baseline
    print(f"\n--- SmallCNN Baseline (Raw Image CNN, {seeds} seeds) ---")
    sys.stdout.flush()
    cnn_list = []
    for seed in range(seeds):
        set_seed(42 + seed)
        cnn = SmallCNN(in_channels, data["input_dim"], data["output_dim"])
        res = train_torch(cnn, data["train_x"], data["train_y"], data["test_x"], data["test_y"], epochs=epochs, device=DEVICE)
        cnn_list.append(res)
        print(f"  [SmallCNN Seed {seed}] Test Acc={res['test_acc']:.4f} (train {res['train_acc']:.4f}) | P={res['P']} | t={res['time']:.0f}s")
        sys.stdout.flush()

    results["methods"]["SmallCNN"] = {
        "test_acc_mean": float(np.mean([r["test_acc"] for r in cnn_list])),
        "test_acc_std": float(np.std([r["test_acc"] for r in cnn_list], ddof=1)) if seeds > 1 else 0.0,
        "P_mean": float(np.mean([r["P"] for r in cnn_list])),
        "P_std": float(np.std([r["P"] for r in cnn_list], ddof=1)) if seeds > 1 else 0.0,
        "time_mean": float(np.mean([r["time"] for r in cnn_list])),
        "all_test_accs": [r["test_acc"] for r in cnn_list],
        "all_train_accs": [r["train_acc"] for r in cnn_list],
        "all_Ps": [r["P"] for r in cnn_list],
    }

    # 3. Dense MLP Baseline
    print(f"\n--- FixedMLP Baseline (L=2, D=64, {seeds} seeds) ---")
    sys.stdout.flush()
    mlp_list = []
    for seed in range(seeds):
        set_seed(42 + seed)
        mlp = FixedMLP(data["input_dim"], 64, 2, data["output_dim"])
        res = train_torch(mlp, data["train_x"], data["train_y"], data["test_x"], data["test_y"], epochs=epochs, device=DEVICE)
        mlp_list.append(res)
        print(f"  [FixedMLP Seed {seed}] Test Acc={res['test_acc']:.4f} (train {res['train_acc']:.4f}) | P={res['P']} | t={res['time']:.0f}s")
        sys.stdout.flush()

    results["methods"]["FixedMLP"] = {
        "test_acc_mean": float(np.mean([r["test_acc"] for r in mlp_list])),
        "test_acc_std": float(np.std([r["test_acc"] for r in mlp_list], ddof=1)) if seeds > 1 else 0.0,
        "P_mean": float(np.mean([r["P"] for r in mlp_list])),
        "P_std": float(np.std([r["P"] for r in mlp_list], ddof=1)) if seeds > 1 else 0.0,
        "time_mean": float(np.mean([r["time"] for r in mlp_list])),
        "all_test_accs": [r["test_acc"] for r in mlp_list],
        "all_Ps": [r["P"] for r in mlp_list],
    }

    # 4. ResNet Block Baseline (ResNet-8)
    print(f"\n--- ResNetBlock Baseline (ResNet-8, {seeds} seeds) ---")
    sys.stdout.flush()
    resnet_list = []
    for seed in range(seeds):
        set_seed(42 + seed)
        resnet = ResNetBlockBaseline(in_channels, data["input_dim"], data["output_dim"], base_channels=32)
        res = train_torch(resnet, data["train_x"], data["train_y"], data["test_x"], data["test_y"], epochs=epochs, device=DEVICE)
        resnet_list.append(res)
        print(f"  [ResNetBlock Seed {seed}] Test Acc={res['test_acc']:.4f} (train {res['train_acc']:.4f}) | P={res['P']} | t={res['time']:.0f}s")
        sys.stdout.flush()

    results["methods"]["ResNetBlock"] = {
        "test_acc_mean": float(np.mean([r["test_acc"] for r in resnet_list])),
        "test_acc_std": float(np.std([r["test_acc"] for r in resnet_list], ddof=1)) if seeds > 1 else 0.0,
        "P_mean": float(np.mean([r["P"] for r in resnet_list])),
        "P_std": float(np.std([r["P"] for r in resnet_list], ddof=1)) if seeds > 1 else 0.0,
        "time_mean": float(np.mean([r["time"] for r in resnet_list])),
        "all_test_accs": [r["test_acc"] for r in resnet_list],
        "all_Ps": [r["P"] for r in resnet_list],
    }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="fashion_mnist")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--fixed-arch", action="store_true")
    parser.add_argument("--only-baselines", action="store_true")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile generator+executor (fuses micro-kernels; "
                             "same math, faster; compile cost paid once). "
                             "Requires working inductor (Linux). On Windows "
                             "torch 2.5 inductor is broken (WinError 183) and "
                             "the run falls back to eager with a warning.")
    parser.add_argument("--compile-mode", type=str, default="default",
                        help="dynamo mode, e.g. default | reduce-overhead | max-autotune")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="ConvAML batch size (baselines keep their own). "
                             "Larger batches raise arithmetic intensity: fewer "
                             "kernel launches per epoch on launch-bound GPUs.")
    parser.add_argument("--gate-floor", type=float, default=0.0,
                        help="Depth keep-alive: starting floor for layer gates, "
                             "linearly annealed to --gate-floor-end over the run "
                             "(0.0 = off, original behaviour). Keeps deep paths "
                             "gradient-alive while the efficiency penalty ramps.")
    parser.add_argument("--gate-floor-end", type=float, default=0.0,
                        help="Final gate floor (default 0 = full prune allowed). "
                             "Set equal to --gate-floor for constant keep-alive.")
    parser.add_argument("--freeze-arch-at", type=float, default=None,
                        help="Two-phase training: freeze structure params "
                             "(prototype_selector, structure_delta, "
                             "block_mode_head) at this fraction of epochs, "
                             "then finetune weights only. E.g. 0.5. "
                             "Single-phase search overfits the arch to train "
                              "noise; freezing recovers generalization.")
    parser.add_argument("--fused-executor", action="store_true",
                        help="Route the TaskNetwork conv branch through the "
                             "fused CUDA v2 pipeline (mgn.accel._C). ~1.7-2x "
                             "faster fwd+bwd on the benchmark configs; eager "
                             "fallback on any unmet precondition (CPU, 3-mode "
                             "blocks, >5 layers, layout mismatch).")
    args = parser.parse_args()

    ds_names = [d.strip() for d in args.datasets.split(",")]
    all_res = []
    for ds in ds_names:
        r = run_conv_aml_benchmark(ds, seeds=args.seeds, quick=args.quick, fast=args.fast, full=args.full, fixed_arch=args.fixed_arch, only_baselines=args.only_baselines, compile=args.compile, compile_mode=args.compile_mode, batch_size=args.batch_size, gate_floor=args.gate_floor, gate_floor_end=args.gate_floor_end, freeze_arch_at=args.freeze_arch_at, fused_executor=args.fused_executor)
        all_res.append(r)
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"conv_aml_{ds}.json")
        with open(out_path, "w") as f:
            json.dump(r, f, indent=2)

    # Upsert into the aggregate file by dataset so that running a single
    # dataset does not silently discard previously recorded results for
    # the others.
    all_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conv_aml_all.json")
    try:
        with open(all_path) as f:
            merged = json.load(f)
        if not isinstance(merged, list):
            merged = []
    except (FileNotFoundError, json.JSONDecodeError):
        merged = []
    by_ds = {e.get("dataset"): e for e in merged}
    for r in all_res:
        by_ds[r["dataset"]] = r
    merged = list(by_ds.values())

    with open(all_path, "w") as f:
        json.dump(merged, f, indent=2)

    print("\n" + "=" * 70)
    print("ConvAML Benchmark Summary Table")
    print("=" * 70)
    for r in all_res:
        ds = r["dataset"]
        print(f"\nDataset: {ds}")
        print(f"  {'Method':<14} {'Test Acc (mean±std)':<24} {'Params (P_eff)':<16} {'Efficiency*':<14}")
        print(f"  {'-'*14} {'-'*24} {'-'*16} {'-'*14}")
        for method, m in r["methods"].items():
            ta = m["test_acc_mean"]
            ts = m.get("test_acc_std", 0.0)
            pm = m["P_mean"]
            p_allow = 110986.0 if ds in ("mnist", "fashion_mnist") else 148426.0
            eff = compute_aml_efficiency(ta, p_allow, pm)
            print(f"  {method:<14} {ta:.4f}±{ts:.4f}          {pm:>8.0f}        {eff:.5f}")
    print("\n* Balanced Efficiency = [a / log10(p^2/q)] * (p/q)^(a * |a log10 a|) (where p is baseline parameter allowance)")
