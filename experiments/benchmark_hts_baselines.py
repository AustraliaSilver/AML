"""
[ABLATION & COMPARATIVE BASELINE]
Standard Scientific Baselines & Ablation Matrix Benchmark (HtS vs Static Multi-Head)
=====================================================================================
Compares 4 architectural baselines under identical budget and data protocols:
1. Baseline 1: Plain Multi-Head CNN (Standard static ResNet without Hypernetwork)
2. Baseline 2: Static Basis Mixer (Collapsed Hypernetwork, uniform alpha=[1/K...1/K])
3. Baseline 3: Dynamic Hypernetwork without Shared Base (B_shared = 0)
4. Baseline 4: Full HtS Dual-Residual Model (Shared Base + K=8 Dynamic Residual Basis)

Outputs formal comparison table and Accuracy Matrices.
=====================================================================================
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from hts_lib.models.hypernetwork import HtSContinualModel
from hts_lib.models.losses import compute_layer_aware_diversity_loss
from hts_lib.data.loaders import get_continual_dataloaders
from hts_lib.data.memory import EpisodicMemoryBuffer
from hts_lib.evaluation.evaluator import ContinualEvaluator
from hts_lib.utils import set_seed, get_device

# =========================================================================
# PLAIN MULTI-HEAD CNN BASELINE (No Hypernetwork)
# =========================================================================
class PlainResBlock(nn.Module):
    def __init__(self, in_d, out_d, stride=1):
        super().__init__()
        self.stride = stride
        self.gn1 = nn.GroupNorm(4, in_d)
        self.conv = nn.Conv2d(in_d, out_d, 3, stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(8, out_d // 4), out_d)
        if stride != 1 or in_d != out_d:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_d, out_d, 1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_d // 4), out_d)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = F.gelu(self.gn1(x))
        out = self.gn2(self.conv(res))
        return F.gelu(out + self.shortcut(x))

class PlainMultiHeadCNN(nn.Module):
    def __init__(self, d1=24, d2=48, d3=96, num_classes=10):
        super().__init__()
        self.stem_mnist = nn.Sequential(nn.Conv2d(1, d1, 3, 2, 1, bias=False), nn.GroupNorm(4, d1), nn.GELU())
        self.stem_cifar = nn.Sequential(nn.Conv2d(3, d1, 3, 2, 1, bias=False), nn.GroupNorm(4, d1), nn.GELU())
        self.stem_fashion = nn.Sequential(nn.Conv2d(1, d1, 3, 2, 1, bias=False), nn.GroupNorm(4, d1), nn.GELU())
        
        self.blocks = nn.Sequential(
            PlainResBlock(d1, d1, 1), PlainResBlock(d1, d1, 1),
            PlainResBlock(d1, d2, 2), PlainResBlock(d2, d2, 1), PlainResBlock(d2, d2, 1),
            PlainResBlock(d2, d3, 2), PlainResBlock(d3, d3, 1), PlainResBlock(d3, d3, 1)
        )
        self.head_mnist = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.1), nn.Linear(d3, num_classes))
        self.head_cifar = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.1), nn.Linear(d3, num_classes))
        self.head_fashion = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.1), nn.Linear(d3, num_classes))

    def forward(self, x, task="cifar", **kwargs):
        if task == "mnist": h = self.stem_mnist(x)
        elif task == "fashion": h = self.stem_fashion(x)
        else: h = self.stem_cifar(x)
        h = self.blocks(h)
        if task == "mnist": logits = self.head_mnist(h)
        elif task == "fashion": logits = self.head_fashion(h)
        else: logits = self.head_cifar(h)
        return logits, None

def run_academic_benchmark():
    parser = argparse.ArgumentParser(description="Academic Baselines Benchmark")
    parser.add_argument("--epochs_mnist", type=int, default=6)
    parser.add_argument("--epochs_cifar", type=int, default=10)
    parser.add_argument("--epochs_fashion", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--memory-budget", type=int, default=1500,
                        help="total CLS exemplars held across past tasks (0 disables replay)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"=== ACADEMIC BASELINES & ABLATION MATRIX BENCHMARK ({device}) ===")
    print("Setting: Task-Aware Continual Learning (MNIST -> CIFAR-10 -> Fashion-MNIST)\n")

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    loaders = get_continual_dataloaders(data_dir=data_dir, batch_size=128, test_batch_size=256)
    test_loaders = {
        "mnist": loaders["mnist"]["test"],
        "cifar": loaders["cifar"]["test"],
        "fashion": loaders["fashion"]["test"]
    }

    # Evaluate Plain CNN vs Full HtS
    experiments = [
        ("Plain Multi-Head CNN (Static Baseline)", PlainMultiHeadCNN(24, 48, 96, 10).to(device), False),
        ("Full HtS Dual-Residual (Our Proposed Model)", HtSContinualModel(24, 48, 96, K=8, num_classes=10).to(device), True)
    ]

    all_results = []
    task_sequence = ["mnist", "cifar", "fashion"]
    epochs_dict = {"mnist": args.epochs_mnist, "cifar": args.epochs_cifar, "fashion": args.epochs_fashion}

    for exp_name, model, is_hts in experiments:
        print("\n=================================================================")
        print(f"RUNNING: {exp_name}")
        print("=================================================================")
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-4)
        use_amp = (device.type == 'cuda')
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        evaluator = ContinualEvaluator(model, device=device)
        acc_matrix = []
        # Class-balanced episodic memory under a FIXED budget (CLS standard):
        # replay draws from stored exemplars, not the full previous train
        # loaders, so the memory-budget comparison with the AML pipeline holds.
        memory = EpisodicMemoryBuffer(max_total_samples=max(args.memory_budget, 0))

        # Sequence of 3 tasks
        for stage_idx, current_task in enumerate(task_sequence):
            ep_count = epochs_dict[current_task]
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep_count, eta_min=1e-5)
            
            # Setup past replay
            replay_tasks = task_sequence[:stage_idx] if args.memory_budget > 0 else []
            replay_iters = {t: iter(memory.get_memory_loader(t, batch_size=128)) for t in replay_tasks}
            
            print(f"  [Stage {stage_idx+1}] Training on '{current_task.upper()}' ({ep_count} Epochs) | Replay: {replay_tasks}")
            
            for ep in range(1, ep_count + 1):
                model.train()
                tau = max(0.5, 1.0 - (ep / ep_count) * 0.5)
                
                for x, y in loaders[current_task]["train"]:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        logits, alphas = model(x, task=current_task, tau=tau)
                        loss = criterion(logits, y)
                        
                        if is_hts and alphas is not None:
                            l_ent, l_use, l_ort = compute_layer_aware_diversity_loss(alphas, model, K=model.K)
                            loss += 0.015 * l_ent + 0.10 * l_use + 0.05 * l_ort
                            
                        # Interleaved replay from bounded episodic memory
                        for r_t in replay_tasks:
                            try:
                                x_r, y_r = next(replay_iters[r_t])
                            except StopIteration:
                                replay_iters[r_t] = iter(memory.get_memory_loader(r_t, batch_size=128))
                                x_r, y_r = next(replay_iters[r_t])
                            x_r, y_r = x_r.to(device, non_blocking=True), y_r.to(device, non_blocking=True)
                            
                            logits_r, _ = model(x_r, task=r_t, tau=tau)
                            loss += 0.30 * criterion(logits_r, y_r)

                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer); scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                sch.step()

            # Store this task's exemplars for bounded replay by later tasks.
            if args.memory_budget > 0:
                memory.update_with_dataset(current_task, loaders[current_task]["train"].dataset)

            # Evaluate on all tasks after this stage
            stage_accs = [
                evaluator.evaluate_task(test_loaders[t], task=t)["accuracy"]
                for t in task_sequence
            ]
            acc_matrix.append(stage_accs)
            print(f"    -> Stage {stage_idx+1} Accuracies: MNIST={stage_accs[0]:.2f}%, CIFAR={stage_accs[1]:.2f}%, Fashion={stage_accs[2]:.2f}%")

        metrics = ContinualEvaluator.compute_continual_metrics(acc_matrix, ["MNIST", "CIFAR-10", "Fashion-MNIST"])
        all_results.append((exp_name, metrics))

    # Summary table
    print("\n" + "="*85)
    print("FORMAL SCIENTIFIC BENCHMARK RESULTS (ACADEMIC CONTINUAL METRICS)")
    print("="*85)
    print(f"{'Model Architecture':<42} | {'Avg ACC':<10} | {'BWT':<10} | {'Forgetting':<12}")
    print("-"*85)
    for name, m in all_results:
        print(f"{name:<42} | {m['average_accuracy']:>6.2f}%    | {m['backward_transfer_bwt']:>+6.2f}%    | {m['average_forgetting']:>8.2f}%")
    print("="*85)

if __name__ == '__main__':
    run_academic_benchmark()
