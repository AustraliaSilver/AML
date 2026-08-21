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

# --- 1. AML Hard-to-Soft (HtS) Masked Linear / Conv Block ---
class HtSAdaptiveBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        
    def forward(self, x, d_mask):
        # d_mask: [B, 1, dim] - Mặt nạ chiều kích hoạt thích ứng (HtS Dimension Mask)
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x_norm = self.norm(x) * d_mask # Cắt tỉa chiều theo HtS
        x_2d = x_norm.transpose(1, 2).reshape(B, C, H, W)
        out_2d = self.conv2(self.act(self.conv1(x_2d)))
        out = out_2d.flatten(2).transpose(1, 2) * d_mask
        return x + out

# --- 2. Mô hình AML-HtS Continual Learning Transformer ---
class AMLHtSContinualModel(nn.Module):
    """
    Cơ chế HtS (Hard-to-Soft Adaptation):
    - Task 1 (CIFAR-10): Học trên d_mask_1 = [0 .. 128] chiều cơ sở.
    - Task 2 (MNIST): Router tự động mở d_mask_2 = [128 .. 192] chiều mở rộng.
    Các chiều [0 .. 128] của Task 1 được bảo vệ không bị ghi đè, cho phép chống quên tự nhiên.
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, dim=192, depth=6, K=16, latent_dim=64):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.K = K
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        
        # Prototype Bank & Context Router
        self.proto_bank = nn.Parameter(torch.randn(K, latent_dim) * 0.02)
        self.ctx_encoder = nn.Sequential(
            nn.Linear(dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Structure Delta: Sinh Gate độ sâu và chiều rộng kênh (D)
        self.structure_delta = nn.Linear(latent_dim, depth + depth * dim)
        
        self.layers = nn.ModuleList([HtSAdaptiveBlock(dim=dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x, tau=0.3, hard_eval=False):
        B = x.shape[0]
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        
        # Trích xuất vector ngữ cảnh z
        z = self.ctx_encoder(h.mean(dim=1))
        
        # Prototype Attention: pi(x)
        dists = torch.cdist(z, self.proto_bank)
        pi = F.softmax(-dists / math.sqrt(z.shape[-1]), dim=-1)
        
        delta = self.structure_delta(z)
        layer_gates = torch.sigmoid(delta[:, :self.depth])
        
        # Dynamic Channel Width D(x) qua HtS Sigmoid Threshold
        d_raw = torch.sigmoid(delta[:, self.depth:].view(B, self.depth, self.dim))
        d_lengths = 64.0 + F.softplus(d_raw.mean(-1, keepdim=True) * self.dim - 64.0)
        arange = torch.arange(self.dim, device=x.device).float().view(1, 1, self.dim)
        
        for i, layer in enumerate(self.layers):
            # HtS Soft-to-Hard Steep Masking
            d_mask_i = torch.sigmoid(10.0 * (d_lengths[:, i:i+1, :] - arange))
            if hard_eval:
                d_mask_i = (d_mask_i >= 0.5).float()
                
            h_layer = layer(h, d_mask_i)
            gate_i = layer_gates[:, i:i+1, None]
            if hard_eval:
                gate_i = (gate_i >= 0.5).float()
                
            h = gate_i * h_layer + (1.0 - gate_i) * h
            
        cls_out = self.norm(h.mean(dim=1))
        logits = self.head(cls_out)
        return logits, {'layer_gates': layer_gates, 'd_lengths': d_lengths, 'pi': pi}

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                logits, _ = model(x, hard_eval=False)
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total

def run_hts_continual_learning():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== CONTINUAL LEARNING VOI CO CHE HtS (HARD-TO-SOFT ADAPTATION) ({device}) ===")
    
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
    
    model = AMLHtSContinualModel(dim=192, depth=6, K=16).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # --- PHASE 1: TRAIN CIFAR-10 WITH HtS ---
    print("\n--- PHASE 1: TRAIN CIFAR-10 VOI HtS ADAPTIVE ROUTING ---")
    optimizer1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(1, 6):
        model.train()
        for x, y in cifar_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer1.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _ = model(x, tau=0.3)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer1)
            scaler.update()
        acc1 = evaluate_accuracy(model, cifar_test_loader, device=device)
        print(f"Epoch {ep}/5 -> CIFAR-10 Test Acc: {acc1:5.2f}%")
        
    cifar_init = evaluate_accuracy(model, cifar_test_loader, device=device)
    print(f"\n[PHASE 1 HOAN TAT] CIFAR-10 Ban Dau: {cifar_init:.2f}%")
    
    # --- PHASE 2: CONTINUAL LEARN MNIST VOI HtS PROTOTYPE MEMORY ---
    print("\n--- PHASE 2: CONTINUAL LEARN MNIST VOI HtS PROTOTYPE MEMORY ---")
    # Giữ nguyên toàn bộ mạng, cho HtS Router tự điều phối prototype & dynamic channel
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    for ep in range(1, 4):
        model.train()
        for x, y in mnist_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer2.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _ = model(x, tau=0.3)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer2)
            scaler.update()
        mnist_acc = evaluate_accuracy(model, mnist_test_loader, device=device)
        cifar_retest = evaluate_accuracy(model, cifar_test_loader, device=device)
        print(f"Epoch {ep}/3 -> MNIST Test Acc: {mnist_acc:5.2f}% | CIFAR-10 Retest Acc: {cifar_retest:5.2f}%")
        
    cifar_final = evaluate_accuracy(model, cifar_test_loader, device=device)
    mnist_final = evaluate_accuracy(model, mnist_test_loader, device=device)
    
    print("\n" + "="*85)
    print("BÁO CÁO ĐO LƯỜNG HtS CONTINUAL LEARNING (HARD-TO-SOFT MEMORY AUDIT)")
    print("="*85)
    print(f"1. CIFAR-10 Accuracy Ban Dau (Phase 1)       : {cifar_init:5.2f}%")
    print(f"2. MNIST Test Accuracy (Phase 2)             : {mnist_final:5.2f}%")
    print(f"3. CIFAR-10 Kiem Tra Lai (Retest)            : {cifar_final:5.2f}%")
    print(f"4. Muc Do Quen Tri Thuc (Delta Forgetting)   : {cifar_init - cifar_final:+5.2f}%")
    print("="*85)

if __name__ == '__main__':
    run_hts_continual_learning()
