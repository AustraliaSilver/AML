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

# --- 1. KHỐI CNN (Conv2D Residual) ---
class CNNResidualBlock(nn.Module):
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

# --- 2. KHỐI ViT (Self-Attention) ---
class ViTAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        B, N, C = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x + self.proj(out)

# --- 3. KHỐI MoE (Top-2 SwiGLU Experts) ---
class MoESwiGLUBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int = 256, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm = nn.LayerNorm(dim)
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
        x_norm = self.norm(x)
        x_flat = x_norm.reshape(-1, D)
        
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
                    
        return x + out_flat.reshape(B, N, D)

# --- 4. KHỐI ResNet (Deep MLP Residual) ---
class ResNetMLPBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int = 384):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, dim)
        
    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))

# --- 5. TẦNG HYBRID MULTI-ARCHITECTURE ARENA ---
class MultiArchLayer(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.cnn = CNNResidualBlock(dim)
        self.vit = ViTAttentionBlock(dim)
        self.moe = MoESwiGLUBlock(dim)
        self.res = ResNetMLPBlock(dim)
        
    def forward(self, x, arch_weights):
        # arch_weights: [B, 4] -> Tỷ lệ chọn: [CNN, ViT, MoE, ResNet]
        B, N, D = x.shape
        h_cnn = self.cnn(x)
        h_vit = self.vit(x)
        h_moe = self.moe(x)
        h_res = self.res(x)
        
        h_all = torch.stack([h_cnn, h_vit, h_moe, h_res], dim=1) # [B, 4, N, D]
        w = arch_weights.view(B, 4, 1, 1)
        return (h_all * w).sum(dim=1)

# --- 6. MÔ HÌNH TOÀN DIỆN VỚI ROUTER ĐA KIẾN TRÚC ---
class FastMultiArchTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, dim=192, depth=6, latent_dim=64):
        super().__init__()
        self.depth = depth
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        
        # Router dự đoán trọng số cho 6 tầng x 4 kiến trúc = 24 outputs
        self.router = nn.Sequential(
            nn.Linear(dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, depth * 4)
        )
        
        self.layers = nn.ModuleList([MultiArchLayer(dim=dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        # x: [B, 3, 32, 32] -> patches: [B, 64, dim]
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        
        # Mean pooling để lấy context toàn ảnh cho router
        img_ctx = h.mean(dim=1)
        router_logits = self.router(img_ctx).view(B, self.depth, 4)
        arch_probs = F.softmax(router_logits, dim=-1) # [B, depth, 4]
        
        for i, layer in enumerate(self.layers):
            h = layer(h, arch_probs[:, i, :])
            
        cls_out = self.norm(h.mean(dim=1))
        logits = self.head(cls_out)
        return logits, arch_probs

def run_fast_architecture_arena():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== FAST ARCHITECTURE ARENA: CNN vs ViT vs MoE vs ResNet ({device}) ===")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    model = FastMultiArchTransformer(dim=192, depth=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    epochs = 8
    best_acc = 0.0
    
    print("\n--- TRAINING 8 EPOCHS & TRACKING ARCHITECTURE PREFERENCE ---")
    t0_all = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _ = model(x)
                loss = criterion(logits, y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * x.size(0)
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        scheduler.step()
        train_acc = 100.0 * correct / total
        
        # Test eval
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda'):
                    logits, _ = model(x)
                _, predicted = logits.max(1)
                test_total += y.size(0)
                test_correct += predicted.eq(y).sum().item()
                
        test_acc = 100.0 * test_correct / test_total
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_multi_arch_cifar10.pth")
            
        print(f"Epoch {epoch:2d}/{epochs:2d} ({time.time()-t0:4.1f}s) | Train Acc: {train_acc:5.2f}% | Test Acc: {test_acc:5.2f}% (Best: {best_acc:5.2f}%)")

    # --- ĐO ĐẠC TRỌNG SỐ LỰA CHỌN TRÊN TOÀN BỘ 10.000 ẢNH TEST ---
    model.load_state_dict(torch.load("best_multi_arch_cifar10.pth", map_location=device, weights_only=True))
    model.eval()
    
    arch_weights_all = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            with torch.amp.autocast('cuda'):
                _, arch_probs = model(x)
            arch_weights_all.append(arch_probs.cpu().numpy())
            
    arch_arr = np.concatenate(arch_weights_all, axis=0) # [10000, 6, 4]
    mean_prefs = np.mean(arch_arr, axis=0) * 100 # [6, 4] in %
    
    print("\n" + "="*88)
    print("BANG KET QUA TY LE UU TIEN KIEN TRUC: CNN vs ViT vs MoE vs ResNet (Tren 10,000 anh test)")
    print("="*88)
    print(f"{'Layer':<10} {'CNN (Conv2D)':<16} {'ViT (Attention)':<18} {'MoE (Sparse)':<16} {'ResNet (MLP)':<16} {'Dominant Arch'}")
    print("-" * 88)
    arch_names = ["CNN", "ViT", "MoE", "ResNet"]
    for i in range(6):
        c, v, m, r = mean_prefs[i]
        dom = arch_names[np.argmax(mean_prefs[i])]
        print(f"Layer {i+1:<4} {c:<15.2f}% {v:<17.2f}% {m:<15.2f}% {r:<15.2f}% {dom}")
        
    print("-" * 88)
    total_avg = np.mean(mean_prefs, axis=0)
    print(f"{'TOTAL':<10} {total_avg[0]:<15.2f}% {total_avg[1]:<17.2f}% {total_avg[2]:<15.2f}% {total_avg[3]:<15.2f}% {arch_names[np.argmax(total_avg)]} (OVERALL WINNER)")
    print("="*88)

if __name__ == '__main__':
    run_fast_architecture_arena()
