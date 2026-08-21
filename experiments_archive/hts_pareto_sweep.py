"""
HtS PARETO FRONTIER SWEEP: Accuracy <-> Dynamic Diversity <-> Forgetting
========================================================================
Quét 4 cấu hình lambda đại diện cho 4 mức độ phân hóa (Regimes):
  1. Config A: Lambda_ortho = 0.00, Lambda_ent = 0.00  (Dense Collapse Baseline)
  2. Config B: Lambda_ortho = 0.02, Lambda_ent = 0.005 (Mild Diversity - Sweet Spot candidate)
  3. Config C: Lambda_ortho = 0.05, Lambda_ent = 0.01  (Balanced Specialization)
  4. Config D: Lambda_ortho = 0.10, Lambda_ent = 0.02  (High Specialization / Sparse)

Đo đạc chính xác trên từng cấu hình:
  - CIFAR-10 Accuracy (%)
  - MNIST Initial & Retest Accuracy (%)
  - Forgetting (Delta %)
  - Cosine Diversity: cos(W_i, W_j) tại Stage 1 và Stage 3
  - Effective Entropy & Usage KL
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from hts_diverse_regularization import (
    HtSDiverseModel, compute_diversity_losses, evaluate_fast, measure_diversity_metrics, set_seed
)

def run_single_config(config_name, lambda_ortho, lambda_ent, lambda_use=0.10, num_ep_mnist=6, num_ep_cifar=10):
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=================================================================")
    print(f"RUNNING CONFIG: {config_name}")
    print(f"  * Lambda_ortho: {lambda_ortho:<6} | Lambda_entropy: {lambda_ent:<6} | Lambda_usage: {lambda_use:<6}")
    print(f"=================================================================")

    # Transforms & Loaders
    tf_c_tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    tf_c_te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    tf_m_tr = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tf_m_te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    c_tr = DataLoader(datasets.CIFAR10('./data', True, download=True, transform=tf_c_tr), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    c_te = DataLoader(datasets.CIFAR10('./data', False, download=True, transform=tf_c_te), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    m_tr = DataLoader(datasets.MNIST('./data', True, download=True, transform=tf_m_tr), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    m_te = DataLoader(datasets.MNIST('./data', False, download=True, transform=tf_m_te), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = HtSDiverseModel(d1=24, d2=48, d3=96, K=4, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    scaler = torch.amp.GradScaler('cuda')

    # Phase 1: Train MNIST
    opt1 = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=num_ep_mnist, eta_min=1e-5)

    for ep in range(1, num_ep_mnist + 1):
        model.train()
        tau = max(0.5, 1.0 - (ep / num_ep_mnist) * 0.5)
        for x, y in m_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt1.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, alphas = model(x, task="mnist", tau=tau)
                loss_ce = criterion(logits, y)
                l_ent, l_use, l_ort = compute_diversity_losses(alphas, model, K=4)
                loss = loss_ce + lambda_ent * l_ent + lambda_use * l_use + lambda_ortho * l_ort
            scaler.scale(loss).backward()
            scaler.step(opt1); scaler.update()
        sch1.step()

    mnist_init_acc, _, _ = evaluate_fast(model, m_te, device, task="mnist")
    print(f"  -> Phase 1 MNIST Initial Acc: {mnist_init_acc:.2f}%")

    # Phase 2: Train CIFAR-10 with Streaming Replay
    opt2 = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=5e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=num_ep_cifar, eta_min=1e-5)
    mnist_stream = iter(m_tr)

    for ep in range(1, num_ep_cifar + 1):
        model.train()
        tau = max(0.5, 1.0 - (ep / num_ep_cifar) * 0.5)
        for x_c, y_c in c_tr:
            x_c, y_c = x_c.to(device, non_blocking=True), y_c.to(device, non_blocking=True)
            try:
                x_m, y_m = next(mnist_stream)
            except StopIteration:
                mnist_stream = iter(m_tr)
                x_m, y_m = next(mnist_stream)
            x_m, y_m = x_m.to(device, non_blocking=True), y_m.to(device, non_blocking=True)

            opt2.zero_grad()
            with torch.amp.autocast('cuda'):
                logits_c, alphas_c = model(x_c, task="cifar", tau=tau)
                loss_c = criterion(logits_c, y_c)
                
                logits_m, alphas_m = model(x_m, task="mnist", tau=tau)
                loss_m = criterion(logits_m, y_m)
                
                l_ent_c, l_use_c, l_ort = compute_diversity_losses(alphas_c, model, K=4)
                l_ent_m, l_use_m, _ = compute_diversity_losses(alphas_m, model, K=4)
                
                loss_div = lambda_ent * (l_ent_c + l_ent_m)*0.5 + lambda_use * (l_use_c + l_use_m)*0.5 + lambda_ortho * l_ort
                loss = loss_c + 0.35 * loss_m + loss_div

            scaler.scale(loss).backward()
            scaler.step(opt2); scaler.update()
        sch2.step()

    cifar_acc, _, _ = evaluate_fast(model, c_te, device, task="cifar")
    mnist_retest_acc, _, _ = evaluate_fast(model, m_te, device, task="mnist")
    
    # Đo đạc Diversity Metrics
    div = measure_diversity_metrics(model, m_te, c_te, device)
    
    result = {
        "Config": config_name,
        "Lambda_ortho": lambda_ortho,
        "Lambda_entropy": lambda_ent,
        "CIFAR-10 Acc": round(cifar_acc, 2),
        "MNIST Init Acc": round(mnist_init_acc, 2),
        "MNIST Retest Acc": round(mnist_retest_acc, 2),
        "Delta Forgetting": round(mnist_init_acc - mnist_retest_acc, 2),
        "Cos Stage 1 (B1)": round(div["Stage 1 (Block 1) Intra-CIFAR Cosine"], 4),
        "Cos Stage 3 (B8)": round(div["Stage 3 (Block 8) Intra-CIFAR Cosine"], 4),
        "Inter-Task Cos": round(div["Stage 3 (Block 8) Inter-Task Cosine"], 4),
    }
    
    print(f"  -> Results: CIFAR={cifar_acc:.2f}% | MNIST_Retest={mnist_retest_acc:.2f}% | Cos_B8={div['Stage 3 (Block 8) Intra-CIFAR Cosine']:.4f}")
    return result

def run_sweep():
    print("=== BAT DAU HtS PARETO FRONTIER SWEEP ===")
    
    configs = [
        ("Config A (Dense Collapse)",    0.00, 0.000, 0.00),
        ("Config B (Mild Diversity)",     0.02, 0.005, 0.10),
        ("Config C (Balanced Diversity)", 0.05, 0.010, 0.10),
        ("Config D (High Specialization)",0.10, 0.020, 0.15)
    ]
    
    results = []
    t_start = time.time()
    
    for name, l_ortho, l_ent, l_use in configs:
        res = run_single_config(name, l_ortho, l_ent, l_use, num_ep_mnist=6, num_ep_cifar=10)
        results.append(res)
        
    print("\n" + "="*85)
    print("TONG KET HtS PARETO FRONTIER: ACCURACY <-> DYNAMIC DIVERSITY <-> FORGETTING")
    print("="*85)
    print(f"{'Config Name':<28} | {'CIFAR Acc':<10} | {'MNIST Ret':<10} | {'Delta':<7} | {'Cos Stage 1':<12} | {'Cos Stage 3':<12}")
    print("-"*85)
    for r in results:
        print(f"{r['Config']:<28} | {r['CIFAR-10 Acc']:>6.2f}%   | {r['MNIST Retest Acc']:>6.2f}%   | {r['Delta Forgetting']:>+5.2f}% | {r['Cos Stage 1 (B1)']:>10.4f}   | {r['Cos Stage 3 (B8)']:>10.4f}")
    print("="*85)
    print(f"Tong thoi gian chay Sweep: {(time.time() - t_start)/60:.1f} phut\n")

if __name__ == '__main__':
    run_sweep()
