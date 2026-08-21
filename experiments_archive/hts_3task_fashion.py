"""
HtS 3-TASK CONTINUAL LEARNING BENCHMARK: MNIST -> CIFAR-10 -> FASHION-MNIST
===========================================================================
Kiểm tra khả năng Continual Learning trên 3 miền dữ liệu hoàn toàn khác biệt:
  Task 1: MNIST (Chữ số viết tay đen trắng, 28x28)
  Task 2: CIFAR-10 (Ảnh vật thể tự nhiên đa màu sắc phức tạp, 32x32)
  Task 3: Fashion-MNIST (Quần áo / Giày dép thời trang phức tạp, 28x28)

Mục tiêu cốt lõi:
- Đạt Fashion-MNIST Acc > 91-93%
- Kiểm tra xem MNIST và CIFAR-10 có bị giảm Acc sau khi học Fashion-MNIST hay không.
- Phân tích sự phân bổ của K=8 Router Experts trên cả 3 Tasks.
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from hts_dual_residual import (
    HtSDualResidualModel, compute_layer_aware_diversity_loss, evaluate_fast, set_seed
)

# Mở rộng Model để hỗ trợ Stem và Head cho Fashion-MNIST
class HtSThreeTaskModel(HtSDualResidualModel):
    def __init__(self, d1=24, d2=48, d3=96, K=8, num_classes=10):
        super().__init__(d1=d1, d2=d2, d3=d3, K=K, num_classes=num_classes)
        
        # Native Stem cho Fashion-MNIST (28x28 grayscale -> 14x14)
        self.stem_fashion = nn.Sequential(
            nn.Conv2d(1, d1, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, d1),
            nn.GELU()
        )
        
        # Classification Head cho Fashion-MNIST
        self.head_fashion = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(d3, num_classes)
        )

    def forward(self, x, task="cifar", tau=0.8):
        B = x.shape[0]
        if task == "mnist":
            h = self.stem_mnist(x)
        elif task == "fashion":
            h = self.stem_fashion(x)
        else:
            h = self.stem_cifar(x)
        
        logits_alpha = self.task_ctrl(h).view(B, 8, self.K)
        
        if self.training:
            alphas = F.gumbel_softmax(logits_alpha, tau=tau, hard=False, dim=-1)
        else:
            alphas = F.softmax(logits_alpha / tau, dim=-1)
            
        # Stage 1 (D=24)
        h = self.b1(h, self.basis_sys.get_kernel('b1', alphas[:, 0]))
        h = self.b2(h, self.basis_sys.get_kernel('b1', alphas[:, 1]))
        
        # Stage 2 (D=48)
        h = self.b3(h, self.basis_sys.get_kernel('b1_2', alphas[:, 2]))
        h = self.b4(h, self.basis_sys.get_kernel('b2', alphas[:, 3]))
        h = self.b5(h, self.basis_sys.get_kernel('b2', alphas[:, 4]))
        
        # Stage 3 (D=96)
        h = self.b6(h, self.basis_sys.get_kernel('b2_3', alphas[:, 5]))
        h = self.b7(h, self.basis_sys.get_kernel('b3', alphas[:, 6]))
        h = self.b8(h, self.basis_sys.get_kernel('b3', alphas[:, 7]))
        
        if task == "mnist":
            logits = self.head_mnist(h)
        elif task == "fashion":
            logits = self.head_fashion(h)
        else:
            logits = self.head_cifar(h)
            
        return logits, alphas

def measure_3task_alpha_distribution(model, m_loader, c_loader, f_loader, device):
    model.eval()
    with torch.no_grad():
        x_m, _ = next(iter(m_loader))
        x_c, _ = next(iter(c_loader))
        x_f, _ = next(iter(f_loader))
        
        x_m, x_c, x_f = x_m.to(device), x_c.to(device), x_f.to(device)
        
        _, a_m = model(x_m, task="mnist", tau=0.5)
        _, a_c = model(x_c, task="cifar", tau=0.5)
        _, a_f = model(x_f, task="fashion", tau=0.5)
        
        # Stage 3 Block 8 alpha mean
        mean_m = a_m[:, 7].mean(dim=0).cpu().numpy().round(2).tolist()
        mean_c = a_c[:, 7].mean(dim=0).cpu().numpy().round(2).tolist()
        mean_f = a_f[:, 7].mean(dim=0).cpu().numpy().round(2).tolist()
        
        return mean_m, mean_c, mean_f

def run_3task_continual():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== HtS 3-TASK CONTINUAL LEARNING BENCHMARK ({device}) ===")
    print("Thu tu hoc: Task 1 (MNIST) -> Task 2 (CIFAR-10) -> Task 3 (Fashion-MNIST)\n")

    model = HtSThreeTaskModel(d1=24, d2=48, d3=96, K=8, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    scaler = torch.amp.GradScaler('cuda')

    # Data Transforms
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
    tf_f_tr = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    tf_f_te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    m_tr = DataLoader(datasets.MNIST('./data', True, download=True, transform=tf_m_tr), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    m_te = DataLoader(datasets.MNIST('./data', False, download=True, transform=tf_m_te), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    
    c_tr = DataLoader(datasets.CIFAR10('./data', True, download=True, transform=tf_c_tr), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    c_te = DataLoader(datasets.CIFAR10('./data', False, download=True, transform=tf_c_te), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    
    f_tr = DataLoader(datasets.FashionMNIST('./data', True, download=True, transform=tf_f_tr), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    f_te = DataLoader(datasets.FashionMNIST('./data', False, download=True, transform=tf_f_te), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    lambda_ent = 0.015
    lambda_use = 0.10
    lambda_ort = 0.05

    # =========================================================================
    # PHASE 1: TRAIN MNIST (8 Epochs)
    # =========================================================================
    print("------------------------------------------------------------")
    print("PHASE 1: TRAIN TASK 1 (MNIST) (8 Epochs)")
    print("------------------------------------------------------------")
    opt1 = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=8, eta_min=1e-5)

    for ep in range(1, 9):
        model.train()
        t0 = time.time()
        tau = max(0.5, 1.0 - (ep / 8) * 0.5)
        for x, y in m_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt1.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, alphas = model(x, task="mnist", tau=tau)
                loss_ce = criterion(logits, y)
                l_ent, l_use, l_ort = compute_layer_aware_diversity_loss(alphas, model, K=8)
                loss = loss_ce + lambda_ent * l_ent + lambda_use * l_use + lambda_ort * l_ort
            scaler.scale(loss).backward()
            scaler.step(opt1); scaler.update()
        sch1.step()
        
        acc, errors, total_m = evaluate_fast(model, m_te, device, task="mnist")
        print(f"Epoch {ep:2d}/8 ({time.time()-t0:4.1f}s) -> MNIST Acc: {acc:.2f}% | So anh sai: {errors}/{total_m}")

    mnist_p1, _, _ = evaluate_fast(model, m_te, device, task="mnist")
    print(f"\n[PHASE 1 HOAN TAT] MNIST Dat: {mnist_p1:.2f}%\n")

    # =========================================================================
    # PHASE 2: TRAIN CIFAR-10 (12 Epochs) + MNIST Streaming Replay
    # =========================================================================
    print("------------------------------------------------------------")
    print("PHASE 2: TRAIN TASK 2 (CIFAR-10) + MNIST REPLAY (12 Epochs)")
    print("------------------------------------------------------------")
    opt2 = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=5e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=12, eta_min=1e-5)
    mnist_stream = iter(m_tr)

    for ep in range(1, 13):
        model.train()
        t0 = time.time()
        tau = max(0.5, 1.0 - (ep / 12) * 0.5)
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
                
                l_ent_c, l_use_c, l_ort = compute_layer_aware_diversity_loss(alphas_c, model, K=8)
                l_ent_m, l_use_m, _ = compute_layer_aware_diversity_loss(alphas_m, model, K=8)
                
                loss_div = lambda_ent * (l_ent_c + l_ent_m)*0.5 + lambda_use * (l_use_c + l_use_m)*0.5 + lambda_ort * l_ort
                loss = loss_c + 0.30 * loss_m + loss_div

            scaler.scale(loss).backward()
            scaler.step(opt2); scaler.update()
        sch2.step()
        
        c_acc, _, _ = evaluate_fast(model, c_te, device, task="cifar")
        m_acc, m_err, total_m = evaluate_fast(model, m_te, device, task="mnist")
        print(f"Epoch {ep:2d}/12 ({time.time()-t0:4.1f}s) -> CIFAR-10: {c_acc:.2f}% | MNIST Retest: {m_acc:.2f}% (Sai: {m_err}/{total_m})")

    mnist_p2, _, _ = evaluate_fast(model, m_te, device, task="mnist")
    cifar_p2, _, _ = evaluate_fast(model, c_te, device, task="cifar")
    print(f"\n[PHASE 2 HOAN TAT] CIFAR-10 Dat: {cifar_p2:.2f}% | MNIST Duy Tri: {mnist_p2:.2f}%\n")

    # =========================================================================
    # PHASE 3: TRAIN FASHION-MNIST (10 Epochs) + MNIST/CIFAR Dual Replay
    # =========================================================================
    print("------------------------------------------------------------")
    print("PHASE 3: TRAIN TASK 3 (FASHION-MNIST) + DUAL REPLAY (10 Epochs)")
    print("------------------------------------------------------------")
    opt3 = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-4)
    sch3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=10, eta_min=1e-5)
    
    mnist_stream = iter(m_tr)
    cifar_stream = iter(c_tr)

    for ep in range(1, 11):
        model.train()
        t0 = time.time()
        tau = max(0.5, 1.0 - (ep / 10) * 0.5)
        for x_f, y_f in f_tr:
            x_f, y_f = x_f.to(device, non_blocking=True), y_f.to(device, non_blocking=True)
            
            # Replay Task 1 (MNIST)
            try:
                x_m, y_m = next(mnist_stream)
            except StopIteration:
                mnist_stream = iter(m_tr)
                x_m, y_m = next(mnist_stream)
            x_m, y_m = x_m.to(device, non_blocking=True), y_m.to(device, non_blocking=True)

            # Replay Task 2 (CIFAR-10)
            try:
                x_c, y_c = next(cifar_stream)
            except StopIteration:
                cifar_stream = iter(c_tr)
                x_c, y_c = next(cifar_stream)
            x_c, y_c = x_c.to(device, non_blocking=True), y_c.to(device, non_blocking=True)

            opt3.zero_grad()
            with torch.amp.autocast('cuda'):
                logits_f, alphas_f = model(x_f, task="fashion", tau=tau)
                loss_f = criterion(logits_f, y_f)
                
                logits_m, alphas_m = model(x_m, task="mnist", tau=tau)
                loss_m = criterion(logits_m, y_m)
                
                logits_c, alphas_c = model(x_c, task="cifar", tau=tau)
                loss_c = criterion(logits_c, y_c)
                
                l_ent_f, l_use_f, l_ort = compute_layer_aware_diversity_loss(alphas_f, model, K=8)
                l_ent_m, l_use_m, _ = compute_layer_aware_diversity_loss(alphas_m, model, K=8)
                l_ent_c, l_use_c, _ = compute_layer_aware_diversity_loss(alphas_c, model, K=8)
                
                loss_div = lambda_ent * (l_ent_f + l_ent_m + l_ent_c)/3.0 + lambda_use * (l_use_f + l_use_m + l_use_c)/3.0 + lambda_ort * l_ort
                loss = loss_f + 0.25 * loss_m + 0.35 * loss_c + loss_div

            scaler.scale(loss).backward()
            scaler.step(opt3); scaler.update()
        sch3.step()
        
        f_acc, _, _ = evaluate_fast(model, f_te, device, task="fashion")
        m_acc, m_err, total_m = evaluate_fast(model, m_te, device, task="mnist")
        c_acc, _, _ = evaluate_fast(model, c_te, device, task="cifar")
        print(f"Epoch {ep:2d}/10 ({time.time()-t0:4.1f}s) -> Fashion: {f_acc:.2f}% | MNIST: {m_acc:.2f}% (Sai: {m_err}/{total_m}) | CIFAR: {c_acc:.2f}%")

    # Đánh giá chung cuộc toàn bộ 3 Tasks
    mnist_final, mnist_err_final, _ = evaluate_fast(model, m_te, device, task="mnist")
    cifar_final, _, _ = evaluate_fast(model, c_te, device, task="cifar")
    fashion_final, _, _ = evaluate_fast(model, f_te, device, task="fashion")
    
    alpha_m, alpha_c, alpha_f = measure_3task_alpha_distribution(model, m_te, c_te, f_te, device)

    print("\n" + "="*75)
    print("TONG KET HtS 3-TASK CONTINUAL LEARNING (MNIST -> CIFAR-10 -> FASHION-MNIST)")
    print("="*75)
    print(f"{'Task Name':<20} | {'Acc Ban Dau':<14} | {'Acc Sau CIFAR':<14} | {'Acc Sau Fashion':<14} | {'Forgetting':<10}")
    print("-"*75)
    print(f"{'1. MNIST':<20} | {mnist_p1:>11.2f}% | {mnist_p2:>11.2f}% | {mnist_final:>11.2f}% | {mnist_final - mnist_p1:>+8.2f}%")
    print(f"{'2. CIFAR-10':<20} | {'—':>12} | {cifar_p2:>11.2f}% | {cifar_final:>11.2f}% | {cifar_final - cifar_p2:>+8.2f}%")
    print(f"{'3. Fashion-MNIST':<20} | {'—':>12} | {'—':>12} | {fashion_final:>11.2f}% | {'New Task':>10}")
    print("="*75)
    
    print("\nPHAN PHOI ROUTER EXPERTS CHO 3 TASKS (STAGE 3 BLOCK 8):")
    print(f"  * MNIST Alpha Distr        : {alpha_m}")
    print(f"  * CIFAR-10 Alpha Distr     : {alpha_c}")
    print(f"  * Fashion-MNIST Alpha Distr: {alpha_f}")
    print("="*75)

if __name__ == '__main__':
    run_3task_continual()
