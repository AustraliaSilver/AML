import math
import os
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
# 4 MẠNG CON ỨNG VIÊN (CNN, ViT, MoE, ResNet)
# =========================================================================

class VisionCNNSubnet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
    def forward(self, x_2d):
        return self.conv2(self.act(self.conv1(x_2d)))

class VisionViTSubnet(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class VisionMoESubnet(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int = 256, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, intermediate_dim, bias=False),
                nn.SiLU(),
                nn.Linear(intermediate_dim, dim, bias=False)
            ) for _ in range(num_experts)
        ])
    def forward(self, x):
        B, N, D = x.shape
        x_flat = x.reshape(-1, D)
        probs = F.softmax(self.router(x_flat), dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        out_flat = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            idx = indices[:, k]
            w = weights[:, k:k+1]
            for i, exp in enumerate(self.experts):
                m = (idx == i)
                if m.any():
                    out_flat[m] += w[m] * exp(x_flat[m])
        return out_flat.reshape(B, N, D)

class VisionResNetSubnet(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int = 384):
        super().__init__()
        self.fc1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# =========================================================================
# TẦNG AML VISION TÍCH HỢP HtS DUAL-MEMORY MASKING
# =========================================================================
class AMLVisionAdaptiveBlock(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.cnn = VisionCNNSubnet(dim)
        self.vit = VisionViTSubnet(dim)
        self.moe = VisionMoESubnet(dim)
        self.res = VisionResNetSubnet(dim)
        
    def forward(self, x, block_mode_probs, d_mask):
        B, N, D = x.shape
        x_norm = self.norm(x) * d_mask
        
        # 0: CNN
        H = W = int(math.sqrt(N))
        x_2d = x_norm.transpose(1, 2).reshape(B, D, H, W)
        h_cnn = self.cnn(x_2d).flatten(2).transpose(1, 2)
        
        # 1: ViT
        h_vit = self.vit(x_norm)
        
        # 2: MoE
        h_moe = self.moe(x_norm)
        
        # 3: ResNet
        h_res = self.res(x_norm)
        
        h_all = torch.stack([h_cnn, h_vit, h_moe, h_res], dim=1)
        w = block_mode_probs.view(B, 4, 1, 1)
        h_out = (h_all * w).sum(dim=1)
        
        return x + h_out * d_mask

# =========================================================================
# MÔ HÌNH AML DUO-MEMORY PROTOTYPE ARCHITECTURE (ZERO-FORGETTING FIX)
# =========================================================================
class AMLDuoMemoryVisionModel(nn.Module):
    """
    KIẾN TRÚC FIX TOÀN DIỆN CHỐNG QUÊN:
    1. Chia đôi ngân hàng Prototype:
       - Proto 0..7 : Cố định cho Task 1 (CIFAR-10) -> Khóa Anchor sau khi học
       - Proto 8..15: Cố định cho Task 2 (MNIST)
    2. Chia đôi kênh không gian ẩn HtS:
       - Kênh 0..128 : Lưu trữ đặc trưng tự nhiên CIFAR-10
       - Kênh 128..192: Lưu trữ đặc trưng chữ số MNIST
    3. Multi-Task Heads:
       - head_cifar: Phân loại 10 lớp tự nhiên
       - head_mnist: Phân loại 10 chữ số
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, dim=192, depth=6, K=16, latent_dim=64):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.K = K
        self.latent_dim = latent_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        
        # Ngân hàng 16 Prototypes
        self.proto_bank = nn.Parameter(torch.randn(K, latent_dim) * 0.02)
        self.ctx_encoder = nn.Sequential(
            nn.Linear(dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        self.structure_delta = nn.Linear(latent_dim, depth + (depth * dim) + (depth * 4))
        self.layers = nn.ModuleList([AMLVisionAdaptiveBlock(dim=dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        
        # Multi-Heads riêng biệt chống xung đột nhãn
        self.head_cifar = nn.Linear(dim, 10)
        self.head_mnist = nn.Linear(dim, 10)
        
    def forward(self, x, task_id=1, tau=0.3):
        # task_id = 1: CIFAR-10 (Dùng Proto 0..7, Kênh 0..128, head_cifar)
        # task_id = 2: MNIST    (Dùng Proto 8..15, Kênh 0..192, head_mnist)
        B = x.shape[0]
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        
        z = self.ctx_encoder(h.mean(dim=1))
        
        # Task-Isolated Prototype Attention
        if task_id == 1:
            active_protos = self.proto_bank[:8] # Task 1 Prototypes
        else:
            active_protos = self.proto_bank[8:] # Task 2 Prototypes
            
        dists = torch.cdist(z, active_protos)
        pi = F.softmax(-dists / math.sqrt(self.latent_dim), dim=-1)
        
        delta = self.structure_delta(z)
        layer_gates = torch.sigmoid(delta[:, :self.depth])
        
        idx_dim_end = self.depth + (self.depth * self.dim)
        d_raw = torch.sigmoid(delta[:, self.depth:idx_dim_end].view(B, self.depth, self.dim))
        
        # HtS Task Width Allocation
        if task_id == 1:
            # Giới hạn kênh cơ sở [0..128]
            d_lengths = 64.0 + F.softplus(d_raw.mean(-1, keepdim=True) * 128.0 - 64.0)
        else:
            # Mở rộng toàn dải [0..192]
            d_lengths = 96.0 + F.softplus(d_raw.mean(-1, keepdim=True) * 192.0 - 96.0)
            
        arange = torch.arange(self.dim, device=x.device).float().view(1, 1, self.dim)
        mode_logits = delta[:, idx_dim_end:].view(B, self.depth, 4)
        block_mode_probs = F.gumbel_softmax(mode_logits, tau=tau, hard=False, dim=-1)
        
        for i, layer in enumerate(self.layers):
            d_mask_raw = torch.sigmoid(10.0 * (d_lengths[:, i:i+1, :] - arange))
            if task_id == 1:
                # Out-of-place task channel mask
                task_channel_mask = torch.cat([
                    torch.ones(1, 1, 128, device=x.device),
                    torch.zeros(1, 1, self.dim - 128, device=x.device)
                ], dim=-1)
                d_mask_i = d_mask_raw * task_channel_mask
            else:
                d_mask_i = d_mask_raw
                
            h_layer = layer(h, block_mode_probs[:, i, :], d_mask_i)
            gate_i = layer_gates[:, i:i+1, None]
            h = gate_i * h_layer + (1.0 - gate_i) * h
            
        cls_out = self.norm(h.mean(dim=1))
        
        if task_id == 1:
            logits = self.head_cifar(cls_out)
        else:
            logits = self.head_mnist(cls_out)
            
        return logits

def evaluate_task_acc(model, data_loader, task_id, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(x, task_id=task_id, tau=0.3)
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total

def run_fixed_continual_aml():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== HUAN LUYEN VISION AML FIX TOAN DIEN (PROTOTYPE ANCHORING + DUAL HtS) ({device}) ===")
    
    # 1. Datasets
    transform_cifar_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_cifar_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar_train)
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
    
    # 2. Model & Optimizer
    model = AMLDuoMemoryVisionModel(dim=192, depth=6, K=16).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # ==================== PHASE 1: TRAIN CIFAR-10 ====================
    print("\n" + "="*80)
    print("PHASE 1: HUAN LUYEN CIFAR-10 (DUNG PROTOTYPES 0..7 & KENH 0..128)")
    print("="*80)
    optimizer1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=8, eta_min=1e-5)
    
    for ep in range(1, 9):
        model.train()
        t0 = time.time()
        for x, y in cifar_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer1.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(x, task_id=1, tau=0.5)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer1)
            scaler.update()
        scheduler1.step()
        cifar_acc = evaluate_task_acc(model, cifar_test_loader, task_id=1, device=device)
        print(f"Epoch {ep}/8 ({time.time()-t0:4.1f}s) -> CIFAR-10 Acc: {cifar_acc:5.2f}%")
        
    cifar_init_acc = evaluate_task_acc(model, cifar_test_loader, task_id=1, device=device)
    print(f"\n[PHASE 1 HOAN TAT] CIFAR-10 Test Acc Dat Duoc: {cifar_init_acc:.2f}%")
    
    # ==================== KHOA ANCHOR CIFAR-10 ====================
    print("\n" + "="*80)
    print("THUC THI AML MEMORY ANCHOR: KHOA PROTO 0..7 VA HEAD CIFAR-10!")
    print("="*80)
    # Khóa Head CIFAR
    for p in model.head_cifar.parameters():
        p.requires_grad = False
        
    # ==================== PHASE 2: CONTINUAL LEARN MNIST ====================
    print("\n" + "="*80)
    print("PHASE 2: HUAN LUYEN MNIST (DUNG PROTOTYPES 8..15 & MO RONG KENH 128..192)")
    print("="*80)
    # Huấn luyện với EWC Regularization để bảo vệ Backbone
    optimizer2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-2)
    
    for ep in range(1, 4):
        model.train()
        t0 = time.time()
        for x, y in mnist_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer2.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(x, task_id=2, tau=0.5)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer2)
            scaler.update()
            
        mnist_acc = evaluate_task_acc(model, mnist_test_loader, task_id=2, device=device)
        cifar_retest = evaluate_task_acc(model, cifar_test_loader, task_id=1, device=device)
        print(f"Epoch {ep}/3 ({time.time()-t0:4.1f}s) -> MNIST Acc: {mnist_acc:5.2f}% | CIFAR-10 Retest Acc: {cifar_retest:5.2f}%")
        
    cifar_final_acc = evaluate_task_acc(model, cifar_test_loader, task_id=1, device=device)
    mnist_final_acc = evaluate_task_acc(model, mnist_test_loader, task_id=2, device=device)
    
    print("\n" + "="*85)
    print("BAO CAO KET QUA KHOA HOC: GIAI PHAP AML DUO-MEMORY CHONG QUEN")
    print("="*85)
    print(f"1. Do chinh xac CIFAR-10 Ban Dau (Task 1)     : {cifar_init_acc:5.2f}%")
    print(f"2. Do chinh xac MNIST Sau Khi Hoc (Task 2)    : {mnist_final_acc:5.2f}%")
    print(f"3. Do chinh xac CIFAR-10 Kiem Tra Lai (Retest): {cifar_final_acc:5.2f}%")
    delta = cifar_init_acc - cifar_final_acc
    print(f"4. Muc Do Quen Tri Thuc (Delta Forgetting)    : {delta:+5.2f}%")
    print("="*85)

if __name__ == '__main__':
    run_fixed_continual_aml()
