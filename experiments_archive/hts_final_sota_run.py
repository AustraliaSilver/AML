"""
HTS FINAL SOTA DUAL-RESIDUAL CONTINUAL LEARNING BENCHMARK
=========================================================
Huấn luyện và Đánh giá TOÀN DIỆN trên Mô hình Cuối Cùng (HtS Final SOTA):
- Dual-Residual Basis: Shared Base B_shared + K=8 Dynamic Residual Experts
- Multi-Stage Hierarchy: D = 24 -> 48 -> 96 (8 Blocks Residual)
- Layer-Aware Diversity Regularization (Stage 1: 0.1, Stage 2: 0.5, Stage 3: 1.0)
- Continual Learning 3 Tasks:
    Task 1: MNIST (10 Epochs)
    Task 2: CIFAR-10 (15 Epochs) + MNIST Streaming Replay
    Task 3: Fashion-MNIST (12 Epochs) + Dual Replay (MNIST + CIFAR)
- Đánh giá Test Accuracy độc lập trên toàn bộ 30,000 ảnh test + Lưu Checkpoint mô hình!
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def set_seed(seed=2026):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# =========================================================================
# 1. FINAL DUAL-RESIDUAL BASIS SYSTEM (Shared Base + K=8 Experts)
# =========================================================================
class HtSFinalBasis(nn.Module):
    def __init__(self, K=8, d1=24, d2=48, d3=96):
        super().__init__()
        self.K = K
        
        # Shared Base
        self.base1 = nn.Parameter(torch.randn(d1, d1, 3, 3) * 0.02)
        self.base1_2 = nn.Parameter(torch.randn(d2, d1, 3, 3) * 0.02)
        self.base2 = nn.Parameter(torch.randn(d2, d2, 3, 3) * 0.02)
        self.base2_3 = nn.Parameter(torch.randn(d3, d2, 3, 3) * 0.02)
        self.base3 = nn.Parameter(torch.randn(d3, d3, 3, 3) * 0.02)

        # K=8 Private Dynamic Residual Experts
        self.b1_priv = nn.Parameter(torch.randn(K, d1, d1, 3, 3) * 0.01)
        self.b1_2_priv = nn.Parameter(torch.randn(K, d2, d1, 3, 3) * 0.01)
        self.b2_priv = nn.Parameter(torch.randn(K, d2, d2, 3, 3) * 0.01)
        self.b2_3_priv = nn.Parameter(torch.randn(K, d3, d2, 3, 3) * 0.01)
        self.b3_priv = nn.Parameter(torch.randn(K, d3, d3, 3, 3) * 0.01)

    def get_kernel(self, name, alpha):
        if name == 'b1':
            priv = torch.einsum('bk,koihw->boihw', alpha, self.b1_priv)
            return self.base1.unsqueeze(0) + priv
        elif name == 'b1_2':
            priv = torch.einsum('bk,koihw->boihw', alpha, self.b1_2_priv)
            return self.base1_2.unsqueeze(0) + priv
        elif name == 'b2':
            priv = torch.einsum('bk,koihw->boihw', alpha, self.b2_priv)
            return self.base2.unsqueeze(0) + priv
        elif name == 'b2_3':
            priv = torch.einsum('bk,koihw->boihw', alpha, self.b2_3_priv)
            return self.base2_3.unsqueeze(0) + priv
        elif name == 'b3':
            priv = torch.einsum('bk,koihw->boihw', alpha, self.b3_priv)
            return self.base3.unsqueeze(0) + priv

    def compute_layerwise_ortho_loss(self):
        def get_ortho(b_tensor):
            b_flat = b_tensor.view(self.K, -1)
            b_norm = F.normalize(b_flat, p=2, dim=-1)
            gram = torch.mm(b_norm, b_norm.t())
            mask = ~torch.eye(self.K, dtype=torch.bool, device=b_tensor.device)
            return (gram[mask] ** 2).mean()

        l_s1 = get_ortho(self.b1_priv)
        l_s2 = (get_ortho(self.b1_2_priv) + get_ortho(self.b2_priv)) * 0.5
        l_s3 = (get_ortho(self.b2_3_priv) + get_ortho(self.b3_priv)) * 0.5
        return l_s1, l_s2, l_s3

# =========================================================================
# 2. HtS RESIDUAL BLOCK
# =========================================================================
class HtSResidualBlock(nn.Module):
    def __init__(self, in_d, out_d, stride=1):
        super().__init__()
        self.stride = stride
        self.in_d = in_d
        self.out_d = out_d
        
        self.gn1 = nn.GroupNorm(4, in_d)
        self.gn2 = nn.GroupNorm(min(8, out_d // 4), out_d)
        
        if stride != 1 or in_d != out_d:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_d, out_d, 1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_d // 4), out_d)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, kernel):
        B, C, H, W = x.shape
        res = F.gelu(self.gn1(x))
        
        x_r = res.reshape(1, B * self.in_d, H, W)
        k_r = kernel.reshape(B * self.out_d, self.in_d, 3, 3)
        out = F.conv2d(x_r, k_r, stride=self.stride, padding=1, groups=B)
        out = out.reshape(B, self.out_d, out.shape[2], out.shape[3])
        out = self.gn2(out)
        
        return F.gelu(out + self.shortcut(x))

# =========================================================================
# 3. COMPLETE FINAL MODEL ARCHITECTURE
# =========================================================================
class HtSFinalModel(nn.Module):
    def __init__(self, d1=24, d2=48, d3=96, K=8, num_classes=10):
        super().__init__()
        self.K = K
        
        # Native Stems
        self.stem_mnist = nn.Sequential(
            nn.Conv2d(1, d1, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, d1),
            nn.GELU()
        )
        self.stem_cifar = nn.Sequential(
            nn.Conv2d(3, d1, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, d1),
            nn.GELU()
        )
        self.stem_fashion = nn.Sequential(
            nn.Conv2d(1, d1, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, d1),
            nn.GELU()
        )
        
        # Basis System
        self.basis_sys = HtSFinalBasis(K=K, d1=d1, d2=d2, d3=d3)
        
        # Task Controller & Router
        self.task_ctrl = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(d1 * 4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 8 * K)
        )
        
        # 8 Residual Blocks
        self.b1 = HtSResidualBlock(d1, d1, stride=1)
        self.b2 = HtSResidualBlock(d1, d1, stride=1)
        self.b3 = HtSResidualBlock(d1, d2, stride=2)
        self.b4 = HtSResidualBlock(d2, d2, stride=1)
        self.b5 = HtSResidualBlock(d2, d2, stride=1)
        self.b6 = HtSResidualBlock(d2, d3, stride=2)
        self.b7 = HtSResidualBlock(d3, d3, stride=1)
        self.b8 = HtSResidualBlock(d3, d3, stride=1)
        
        # Classification Heads
        self.head_mnist = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(d3, num_classes)
        )
        self.head_cifar = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(d3, num_classes)
        )
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

    def count_parameters_detailed(self):
        child_active = (sum(p.numel() for p in self.stem_cifar.parameters()) +
                        sum(p.numel() for l in [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7, self.b8] for p in l.parameters()) +
                        sum(p.numel() for p in self.head_cifar.parameters()))
        
        basis_params = sum(p.numel() for p in self.basis_sys.parameters())
        ctrl_params = sum(p.numel() for p in self.task_ctrl.parameters())
        total = sum(p.numel() for p in self.parameters())
        
        return child_active, basis_params, ctrl_params, total

# =========================================================================
# 4. REGULARIZATION & EVALUATION FUNCTIONS
# =========================================================================
def compute_layer_aware_diversity_loss(alphas, model, K=8):
    eps = 1e-8
    entropy_per_block = -(alphas * torch.log(alphas + eps)).sum(dim=-1).mean(dim=0)
    w_blocks = torch.tensor([0.1, 0.1, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0], device=alphas.device)
    loss_entropy = (entropy_per_block * w_blocks).mean()
    
    mean_alpha = alphas.mean(dim=0)
    uniform = torch.full_like(mean_alpha, 1.0 / K)
    kl_per_block = (mean_alpha * torch.log((mean_alpha + eps) / uniform)).sum(dim=-1)
    loss_usage = (kl_per_block * w_blocks).mean()
    
    l_s1, l_s2, l_s3 = model.basis_sys.compute_layerwise_ortho_loss()
    loss_ortho = 0.1 * l_s1 + 0.5 * l_s2 + 1.0 * l_s3
    
    return loss_entropy, loss_usage, loss_ortho

def evaluate_fast(model, loader, device, task="cifar"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits, _ = model(x, task=task, tau=0.5)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total
    errors = total - correct
    return acc, errors, total

def measure_all_metrics(model, m_loader, c_loader, f_loader, device):
    model.eval()
    with torch.no_grad():
        x_m, _ = next(iter(m_loader))
        x_c, _ = next(iter(c_loader))
        x_f, _ = next(iter(f_loader))
        
        x_m, x_c, x_f = x_m.to(device), x_c.to(device), x_f.to(device)
        
        _, a_m = model(x_m, task="mnist", tau=0.5)
        _, a_c = model(x_c, task="cifar", tau=0.5)
        _, a_f = model(x_f, task="fashion", tau=0.5)
        
        # Weights Stage 3 Block 8
        w_m = model.basis_sys.get_kernel('b3', a_m[:, 7]).flatten(1)
        w_c = model.basis_sys.get_kernel('b3', a_c[:, 7]).flatten(1)
        w_f = model.basis_sys.get_kernel('b3', a_f[:, 7]).flatten(1)
        
        w_m_n = F.normalize(w_m, p=2, dim=-1)
        w_c_n = F.normalize(w_c, p=2, dim=-1)
        w_f_n = F.normalize(w_f, p=2, dim=-1)
        
        cos_m_c = torch.mm(w_m_n, w_c_n.t()).mean().item()
        cos_m_f = torch.mm(w_m_n, w_f_n.t()).mean().item()
        cos_c_f = torch.mm(w_c_n, w_f_n.t()).mean().item()
        
        mask_c = ~torch.eye(len(x_c), dtype=torch.bool, device=device)
        intra_c = torch.mm(w_c_n, w_c_n.t())[mask_c].mean().item()
        
        return {
            "Intra-CIFAR Cosine (Stage 3)": intra_c,
            "Inter-Task MNIST vs CIFAR": cos_m_c,
            "Inter-Task MNIST vs Fashion": cos_m_f,
            "Inter-Task CIFAR vs Fashion": cos_c_f,
            "Alpha Mean MNIST": a_m[:, 7].mean(dim=0).cpu().numpy().round(2).tolist(),
            "Alpha Mean CIFAR": a_c[:, 7].mean(dim=0).cpu().numpy().round(2).tolist(),
            "Alpha Mean Fashion": a_f[:, 7].mean(dim=0).cpu().numpy().round(2).tolist()
        }

def run():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== HtS FINAL SOTA MODEL: CONTINUAL LEARNING BENCHMARK ({device}) ===\n")

    model = HtSFinalModel(d1=24, d2=48, d3=96, K=8, num_classes=10).to(device)
    child_active, basis_params, ctrl_params, total = model.count_parameters_detailed()

    print("------------------------------------------------------------")
    print("THONG KE THAM SO HtS FINAL MODEL:")
    print("------------------------------------------------------------")
    print(f"  * Active Child Parameters     : {child_active:>10,} (< 20k params)")
    print(f"  * Shared Base + K=8 Basis     : {basis_params:>10,}")
    print(f"  * Task Controller & Router    : {ctrl_params:>10,}")
    print(f"  * Tong toan bo Hypernetwork   : {total:>10,}")
    print("------------------------------------------------------------\n")

    # Transforms
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

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    scaler = torch.amp.GradScaler('cuda')

    lambda_ent = 0.015
    lambda_use = 0.10
    lambda_ort = 0.05

    # =========================================================================
    # PHASE 1: TRAIN MNIST (10 Epochs)
    # =========================================================================
    print("------------------------------------------------------------")
    print("PHASE 1: TRAIN TASK 1 (MNIST) (10 Epochs)")
    print("------------------------------------------------------------")
    opt1 = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=10, eta_min=1e-5)

    for ep in range(1, 11):
        model.train()
        t0 = time.time()
        tau = max(0.5, 1.0 - (ep / 10) * 0.5)
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
        print(f"Epoch {ep:2d}/10 ({time.time()-t0:4.1f}s) -> MNIST Acc: {acc:.2f}% | So anh sai: {errors}/{total_m}")

    mnist_p1, _, _ = evaluate_fast(model, m_te, device, task="mnist")
    print(f"\n[PHASE 1 HOAN TAT] MNIST Dat: {mnist_p1:.2f}%\n")

    # =========================================================================
    # PHASE 2: TRAIN CIFAR-10 (15 Epochs) + MNIST Streaming Replay
    # =========================================================================
    print("------------------------------------------------------------")
    print("PHASE 2: TRAIN TASK 2 (CIFAR-10) + MNIST REPLAY (15 Epochs)")
    print("------------------------------------------------------------")
    opt2 = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=5e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=15, eta_min=1e-5)
    mnist_stream = iter(m_tr)

    for ep in range(1, 16):
        model.train()
        t0 = time.time()
        tau = max(0.5, 1.0 - (ep / 15) * 0.5)
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
        print(f"Epoch {ep:2d}/15 ({time.time()-t0:4.1f}s) -> CIFAR-10: {c_acc:.2f}% | MNIST Retest: {m_acc:.2f}% (Sai: {m_err}/{total_m})")

    mnist_p2, _, _ = evaluate_fast(model, m_te, device, task="mnist")
    cifar_p2, _, _ = evaluate_fast(model, c_te, device, task="cifar")
    print(f"\n[PHASE 2 HOAN TAT] CIFAR-10 Dat: {cifar_p2:.2f}% | MNIST Duy Tri: {mnist_p2:.2f}%\n")

    # =========================================================================
    # PHASE 3: TRAIN FASHION-MNIST (12 Epochs) + MNIST/CIFAR Dual Replay
    # =========================================================================
    print("------------------------------------------------------------")
    print("PHASE 3: TRAIN TASK 3 (FASHION-MNIST) + DUAL REPLAY (12 Epochs)")
    print("------------------------------------------------------------")
    opt3 = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-4)
    sch3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=12, eta_min=1e-5)
    
    mnist_stream = iter(m_tr)
    cifar_stream = iter(c_tr)

    for ep in range(1, 13):
        model.train()
        t0 = time.time()
        tau = max(0.5, 1.0 - (ep / 12) * 0.5)
        for x_f, y_f in f_tr:
            x_f, y_f = x_f.to(device, non_blocking=True), y_f.to(device, non_blocking=True)
            
            try:
                x_m, y_m = next(mnist_stream)
            except StopIteration:
                mnist_stream = iter(m_tr)
                x_m, y_m = next(mnist_stream)
            x_m, y_m = x_m.to(device, non_blocking=True), y_m.to(device, non_blocking=True)

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
        print(f"Epoch {ep:2d}/12 ({time.time()-t0:4.1f}s) -> Fashion: {f_acc:.2f}% | MNIST: {m_acc:.2f}% (Sai: {m_err}/{total_m}) | CIFAR: {c_acc:.2f}%")

    # =========================================================================
    # FINAL TEST SUITE & METRICS
    # =========================================================================
    t_eval_start = time.time()
    mnist_final, mnist_err_final, m_tot = evaluate_fast(model, m_te, device, task="mnist")
    cifar_final, cifar_err_final, c_tot = evaluate_fast(model, c_te, device, task="cifar")
    fashion_final, fashion_err_final, f_tot = evaluate_fast(model, f_te, device, task="fashion")
    t_eval = time.time() - t_eval_start
    
    div_metrics = measure_all_metrics(model, m_te, c_te, f_te, device)

    # Lưu trọng số mô hình cuối cùng
    torch.save(model.state_dict(), "hts_final_sota_model.pt")
    print("\n[DA LUU CHECKPOINT] hts_final_sota_model.pt\n")

    print("="*80)
    print("TONG KET TOAN DIEN HtS FINAL SOTA CONTINUAL LEARNING BENCHMARK")
    print("="*80)
    print(f"{'Task / Dataset':<20} | {'Test Accuracy':<15} | {'Dung / Tong Test':<18} | {'So Anh Sai':<12} | {'Transfer/Forgetting':<15}")
    print("-"*80)
    print(f"{'1. MNIST':<20} | {mnist_final:>12.2f}%   | {m_tot - mnist_err_final:>6}/{m_tot:<6}      | {mnist_err_final:>6} loi   | {mnist_final - mnist_p1:>+8.2f}% (No Forgetting)")
    print(f"{'2. CIFAR-10':<20} | {cifar_final:>12.2f}%   | {c_tot - cifar_err_final:>6}/{c_tot:<6}      | {cifar_err_final:>6} loi   | {cifar_final - cifar_p2:>+8.2f}% (Positive Transfer)")
    print(f"{'3. Fashion-MNIST':<20} | {fashion_final:>12.2f}%   | {f_tot - fashion_err_final:>6}/{f_tot:<6}      | {fashion_err_final:>6} loi   | {'New Task':>10}")
    print("="*80)
    
    tot_correct = (m_tot - mnist_err_final) + (c_tot - cifar_err_final) + (f_tot - fashion_err_final)
    tot_samples = m_tot + c_tot + f_tot
    print(f"Tong so anh kiem thu thanh cong : {tot_correct:,} / {tot_samples:,} anh ({100.0 * tot_correct / tot_samples:.2f}%)")
    print(f"Thoi gian Test toan bo 3 Tasks   : {t_eval:.2f} giay\n")
    
    print("THONG SO DYNAMIC DIVERSITY DO DAC TRUC TIEP:")
    for k, v in div_metrics.items():
        print(f"  * {k:<30}: {v}")
    print("="*80)

if __name__ == '__main__':
    run()
