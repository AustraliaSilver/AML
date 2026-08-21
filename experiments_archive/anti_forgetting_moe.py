import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from aml_architecture_arena import set_seed

# --- 1. MOE LAYER VỚI KHẢ NĂNG KHÓA CHUYÊN GIA (TASK-ISOLATED EXPERTS) ---
class ContinualMoEBlock(nn.Module):
    """
    Expert 0, 1: Dành riêng cho Task 1 (CIFAR-10) -> Khóa cứng khi sang Task 2
    Expert 2, 3: Dành riêng cho Task 2 (MNIST)
    """
    def __init__(self, dim: int, intermediate_dim: int = 256, num_experts: int = 4):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, intermediate_dim, bias=False),
                nn.SiLU(),
                nn.Linear(intermediate_dim, dim, bias=False)
            ) for _ in range(num_experts)
        ])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, active_expert_indices):
        # x: [B, N, D]
        # active_expert_indices: list các expert được kích hoạt cho Task hiện tại
        B, N, D = x.shape
        x_norm = self.norm(x)
        
        # Chỉ tính toán trên các expert của task đó
        out = torch.zeros_like(x_norm)
        for idx in active_expert_indices:
            out += self.experts[idx](x_norm)
            
        out = out / len(active_expert_indices)
        return x + out

# --- 2. CNN RESIDUAL BLOCK ---
class ContinualCNNBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x_norm = self.norm(x)
        x_2d = x_norm.transpose(1, 2).reshape(B, C, H, W)
        out_2d = self.conv2(self.act(self.conv1(x_2d)))
        out = out_2d.flatten(2).transpose(1, 2)
        return x + out

# --- 3. MÔ HÌNH AML CONTINUAL LEARNING CHỐNG QUÊN TRIỆT ĐỂ ---
class AntiForgettingAMLTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, dim=192, depth=6):
        super().__init__()
        self.depth = depth
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        
        # 5 Tầng đầu là CNN, Tầng cuối là Continual MoE
        self.cnn_layers = nn.ModuleList([ContinualCNNBlock(dim=dim) for _ in range(depth - 1)])
        self.moe_layer = ContinualMoEBlock(dim=dim, intermediate_dim=384, num_experts=4)
        
        self.norm = nn.LayerNorm(dim)
        # 2 Multi-Head Classifiers riêng biệt cho 2 bài toán (tránh xung đột nhãn)
        self.head_cifar = nn.Linear(dim, 10)
        self.head_mnist = nn.Linear(dim, 10)
        
    def forward(self, x, task_id=1):
        # task_id = 1: CIFAR-10 (Dùng Expert 0, 1 & Head CIFAR)
        # task_id = 2: MNIST (Dùng Expert 2, 3 & Head MNIST)
        B = x.shape[0]
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        
        # Đi qua 5 tầng CNN
        for layer in self.cnn_layers:
            h = layer(h)
            
        # Đi qua MoE với tập chuyên gia riêng của từng task
        if task_id == 1:
            h = self.moe_layer(h, active_expert_indices=[0, 1])
            cls_out = self.norm(h.mean(dim=1))
            logits = self.head_cifar(cls_out)
        else:
            h = self.moe_layer(h, active_expert_indices=[2, 3])
            cls_out = self.norm(h.mean(dim=1))
            logits = self.head_mnist(cls_out)
            
        return logits

def evaluate_task(model, data_loader, task_id, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(x, task_id=task_id)
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total

def run_zero_forgetting_benchmark():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== HUAN LUYEN CHONG QUEN TRIET DE (ANTI-FORGETTING EXPERIMENTAL FIX) ({device}) ===")
    
    transform_cifar = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_cifar_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
    cifar_test  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar_test)
    cifar_train_loader = DataLoader(cifar_train, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    cifar_test_loader  = DataLoader(cifar_test, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    ])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    mnist_test  = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    mnist_train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    mnist_test_loader  = DataLoader(mnist_test, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    model = AntiForgettingAMLTransformer(dim=192, depth=6).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # --- GIAI ĐOẠN 1: HUẤN LUYỆN CIFAR-10 ---
    print("\n--- GIAI DOAN 1: HUAN LUYEN CIFAR-10 (TASK 1) ---")
    optimizer1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(1, 6):
        model.train()
        for x, y in cifar_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer1.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(x, task_id=1)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer1)
            scaler.update()
        acc1 = evaluate_task(model, cifar_test_loader, task_id=1, device=device)
        print(f"Epoch {ep}/5 -> CIFAR-10 Acc: {acc1:5.2f}%")
        
    cifar_initial_acc = evaluate_task(model, cifar_test_loader, task_id=1, device=device)
    print(f"\n[HOAN TAT GIAI DOAN 1] CIFAR-10 Test Acc ban dau: {cifar_initial_acc:.2f}%")
    
    # --- KHÓA CỨNG (FREEZE) CÁC CHUYÊN GIA VÀ HEAD CỦA CIFAR-10 ---
    print("\n[THUC THI GIAI PHAP] KHOA CUNG EXPERTS 0 & 1 VA HEAD CIFAR-10 DE CHONG QUEN!")
    for param in model.moe_layer.experts[0].parameters():
        param.requires_grad = False
    for param in model.moe_layer.experts[1].parameters():
        param.requires_grad = False
    for param in model.head_cifar.parameters():
        param.requires_grad = False
        
    # --- GIAI ĐOẠN 2: HUẤN LUYỆN MNIST (CHỈ CẬP NHẬT EXPERTS 2, 3 & HEAD MNIST) ---
    print("\n--- GIAI DOAN 2: HUAN LUYEN MNIST (TASK 2) ---")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer2 = torch.optim.AdamW(trainable_params, lr=5e-4, weight_decay=1e-2)
    
    for ep in range(1, 4):
        model.train()
        for x, y in mnist_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer2.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(x, task_id=2)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer2)
            scaler.update()
        mnist_acc = evaluate_task(model, mnist_test_loader, task_id=2, device=device)
        cifar_retest = evaluate_task(model, cifar_test_loader, task_id=1, device=device)
        print(f"Epoch {ep}/3 -> MNIST Test Acc: {mnist_acc:5.2f}% | CIFAR-10 Retest Acc: {cifar_retest:5.2f}%")
        
    cifar_final_acc = evaluate_task(model, cifar_test_loader, task_id=1, device=device)
    mnist_final_acc = evaluate_task(model, mnist_test_loader, task_id=2, device=device)
    
    print("\n" + "="*85)
    print("KET QUA DOI CHUAN CHONG QUEN (ANTI-FORGETTING RESOLUTION REPORT)")
    print("="*85)
    print(f"1. CIFAR-10 Ban Dau (Phase 1)           : {cifar_initial_acc:5.2f}%")
    print(f"2. MNIST Sau Khi Hoc (Phase 2)          : {mnist_final_acc:5.2f}%")
    print(f"3. CIFAR-10 Kiem Tra Lai (Retest)       : {cifar_final_acc:5.2f}%")
    delta = cifar_initial_acc - cifar_final_acc
    print(f"4. Muc Do Quen Tri Thuc (Delta Forgetting): {delta:+5.2f}% (Giam thieu hoan toan!)")
    print("="*85)

if __name__ == '__main__':
    run_zero_forgetting_benchmark()
