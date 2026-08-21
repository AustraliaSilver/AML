"""
HtS MULTI-STAGE RESIDUAL HYPERNETWORK (CIFAR-10 > 80-85% & MNIST > 99.6%)
========================================================================
Thiết kế tối ưu:
- Multi-Stage Channel Width: D = 24 -> 48 -> 96
- HtS Basis Count K = 4 cho từng Stage
- Tổng tham số toàn bộ Hypernetwork: ~650k parameters!
- Active parameters mạng con: < 40k parameters!
- True Residual Connections: H(x) = x + Conv_Basis(x)
- Mixed Precision (AMP) + CIFAR-10 AutoAugment + Streaming Replay
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
# 1. MULTI-STAGE HtS BASIS SYSTEM (K=4, D=24->48->96, Total ~640k params)
# =========================================================================
class HtSMultiStageBasis(nn.Module):
    def __init__(self, K=4, d1=24, d2=48, d3=96):
        super().__init__()
        self.K = K
        # Stage 1: [4, 24, 24, 3, 3] -> 20.7k
        self.b1 = nn.Parameter(torch.randn(K, d1, d1, 3, 3) * 0.02)
        # Stage 1->2 Transition: [4, 48, 24, 3, 3] -> 41.4k
        self.b1_2 = nn.Parameter(torch.randn(K, d2, d1, 3, 3) * 0.02)
        # Stage 2: [4, 48, 48, 3, 3] -> 82.9k
        self.b2 = nn.Parameter(torch.randn(K, d2, d2, 3, 3) * 0.02)
        # Stage 2->3 Transition: [4, 96, 48, 3, 3] -> 165.8k
        self.b2_3 = nn.Parameter(torch.randn(K, d3, d2, 3, 3) * 0.02)
        # Stage 3: [4, 96, 96, 3, 3] -> 331.7k
        self.b3 = nn.Parameter(torch.randn(K, d3, d3, 3, 3) * 0.02)
        # Total Basis Params = 642.5k params!

    def get_kernel(self, name, alpha):
        # alpha: [B, K]
        if name == 'b1':
            return torch.einsum('bk,koihw->boihw', alpha, self.b1)
        elif name == 'b1_2':
            return torch.einsum('bk,koihw->boihw', alpha, self.b1_2)
        elif name == 'b2':
            return torch.einsum('bk,koihw->boihw', alpha, self.b2)
        elif name == 'b2_3':
            return torch.einsum('bk,koihw->boihw', alpha, self.b2_3)
        elif name == 'b3':
            return torch.einsum('bk,koihw->boihw', alpha, self.b3)

# =========================================================================
# 2. HtS RESIDUAL BLOCK VỚI ĐƯỜNG TRUYỀN THẲNG SKIP CONNECTION
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
# 3. COMPLETE MULTI-STAGE HtS MODEL
# =========================================================================
class HtSMultiStageModel(nn.Module):
    def __init__(self, d1=24, d2=48, d3=96, K=4, num_classes=10):
        super().__init__()
        self.K = K
        
        # Native Stems
        self.stem_mnist = nn.Sequential(
            nn.Conv2d(1, d1, 3, stride=2, padding=1, bias=False), # 28x28 -> 14x14
            nn.GroupNorm(4, d1),
            nn.GELU()
        )
        self.stem_cifar = nn.Sequential(
            nn.Conv2d(3, d1, 3, stride=2, padding=1, bias=False), # 32x32 -> 16x16
            nn.GroupNorm(4, d1),
            nn.GELU()
        )
        
        # Basis System (~642k params)
        self.basis_sys = HtSMultiStageBasis(K=K, d1=d1, d2=d2, d3=d3)
        
        # Task Controller & Alpha Generator (~12k params)
        self.task_ctrl = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(d1 * 4, 32),
            nn.SiLU()
        )
        self.alpha_gen = nn.Linear(32, 8 * K) # 8 blocks x K coeffs
        
        # Stage 1: D=24, 2 blocks (16x16 / 14x14)
        self.b1 = HtSResidualBlock(d1, d1, stride=1)
        self.b2 = HtSResidualBlock(d1, d1, stride=1)
        
        # Stage 2: D=48, 3 blocks (downsample 8x8 / 7x7)
        self.b3 = HtSResidualBlock(d1, d2, stride=2)
        self.b4 = HtSResidualBlock(d2, d2, stride=1)
        self.b5 = HtSResidualBlock(d2, d2, stride=1)
        
        # Stage 3: D=96, 3 blocks (downsample 4x4 / 4x4)
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

    def forward(self, x, task="cifar"):
        B = x.shape[0]
        h = self.stem_mnist(x) if task == "mnist" else self.stem_cifar(x)
        
        z = self.task_ctrl(h)
        alphas = F.softmax(self.alpha_gen(z).view(B, 8, self.K), dim=-1)
        
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
        
        logits = self.head_mnist(h) if task == "mnist" else self.head_cifar(h)
        return logits

    def count_parameters_detailed(self):
        child_active = (sum(p.numel() for p in self.stem_cifar.parameters()) +
                        sum(p.numel() for l in [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7, self.b8] for p in l.parameters()) +
                        sum(p.numel() for p in self.head_cifar.parameters()))
        
        basis_params = sum(p.numel() for p in self.basis_sys.parameters())
        ctrl_params = sum(p.numel() for p in self.task_ctrl.parameters()) + sum(p.numel() for p in self.alpha_gen.parameters())
        total = sum(p.numel() for p in self.parameters())
        
        return child_active, basis_params, ctrl_params, total

def evaluate_fast(model, loader, device, task="cifar"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(x, task=task)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total
    errors = total - correct
    return acc, errors, total

def run():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== HtS MULTI-STAGE RESIDUAL HYPERNETWORK ({device}) ===")
    print("Muc tieu: Dot pha CIFAR-10 len >80-85% voi tong tham so ~650k\n")

    model = HtSMultiStageModel(d1=24, d2=48, d3=96, K=4, num_classes=10).to(device)
    child_active, basis_params, ctrl_params, total = model.count_parameters_detailed()
    
    print("------------------------------------------------------------")
    print("THONG KE THAM SO HtS MULTI-STAGE:")
    print("------------------------------------------------------------")
    print(f"  * Active Child Parameters     : {child_active:>10,} (< 20k params)")
    print(f"  * Shared HtS Multi-Scale Basis: {basis_params:>10,} (~642k params)")
    print(f"  * Task Controller & Router    : {ctrl_params:>10,}")
    print(f"  * Tong toan bo Hypernetwork   : {total:>10,} (~655k params)")
    print("------------------------------------------------------------\n")

    # Data Transforms SOTA
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

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    scaler = torch.amp.GradScaler('cuda')

    # =========================================================================
    # PHASE 1: TRAIN MNIST (8 Epochs)
    # =========================================================================
    print("------------------------------------------------------------")
    print("PHASE 1: TRAIN MNIST TREN HtS MULTI-STAGE (8 Epochs)")
    print("------------------------------------------------------------")
    opt1 = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=8, eta_min=1e-5)

    for ep in range(1, 9):
        model.train()
        t0 = time.time()
        for x, y in m_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt1.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = criterion(model(x, task="mnist"), y)
            scaler.scale(loss).backward()
            scaler.step(opt1); scaler.update()
        sch1.step()
        
        acc, errors, total_m = evaluate_fast(model, m_te, device, task="mnist")
        print(f"Epoch {ep:2d}/8 ({time.time()-t0:4.1f}s) -> MNIST Acc: {acc:.2f}% | So anh sai: {errors}/{total_m}")

    mnist_init_acc, mnist_init_err, _ = evaluate_fast(model, m_te, device, task="mnist")
    print(f"\n[PHASE 1 HOAN TAT] MNIST Dat: {mnist_init_acc:.2f}% (Sai: {mnist_init_err} anh)\n")

    # =========================================================================
    # PHASE 2: TRAIN CIFAR-10 (15 Epochs) -> DOT PHA ACCURACY
    # =========================================================================
    print("------------------------------------------------------------")
    print("PHASE 2: TRAIN CIFAR-10 MULTI-STAGE (15 Epochs)")
    print("------------------------------------------------------------")
    opt2 = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=5e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=15, eta_min=1e-5)

    mnist_stream = iter(m_tr)

    for ep in range(1, 16):
        model.train()
        t0 = time.time()
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
                loss_c = criterion(model(x_c, task="cifar"), y_c)
                loss_m = criterion(model(x_m, task="mnist"), y_m)
                loss = loss_c + 0.35 * loss_m

            scaler.scale(loss).backward()
            scaler.step(opt2); scaler.update()
            
        sch2.step()
        
        c_acc, c_err, total_c = evaluate_fast(model, c_te, device, task="cifar")
        m_ret_acc, m_ret_err, total_m = evaluate_fast(model, m_te, device, task="mnist")
        print(f"Epoch {ep:2d}/15 ({time.time()-t0:4.1f}s) -> CIFAR-10: {c_acc:.2f}% | MNIST Retest: {m_ret_acc:.2f}% (Sai: {m_ret_err}/{total_m})")

    cifar_final_acc, cifar_final_err, _ = evaluate_fast(model, c_te, device, task="cifar")
    mnist_final_acc, mnist_final_err, total_m = evaluate_fast(model, m_te, device, task="mnist")

    print("\n" + "="*65)
    print("TONG KET HtS MULTI-STAGE RESIDUAL HYPERNETWORK")
    print("="*65)
    print(f"Tong Parameters Model         : {total:,} (~655k)")
    print(f"Active Child Parameters       : {child_active:,} (< 20k)")
    print(f"MNIST Ban Dau (Phase 1)       : {mnist_init_acc:.2f}% (Sai: {mnist_init_err} anh)")
    print(f"CIFAR-10 Dat Duoc (Phase 2)   : {cifar_final_acc:.2f}% (Dot pha!)")
    print(f"MNIST Sau Khi Hoc CIFAR-10    : {mnist_final_acc:.2f}% (Sai: {mnist_final_err} anh)")
    print(f"Muc Do Chuyen Giao (Transfer) : {mnist_final_acc - mnist_init_acc:+.2f}%")
    print("="*65)

if __name__ == '__main__':
    run()
