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
# 4 ỨNG VIÊN KIẾN TRÚC MẠNG CON (4 VISION BLOCK MODES)
# =========================================================================

# 1. Mode 0: Khối CNN Tích Chập Cục Bộ (Local Inductive Bias)
class VisionCNNSubnet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
    def forward(self, x_2d):
        return self.conv2(self.act(self.conv1(x_2d)))

# 2. Mode 1: Khối ViT Multi-Head Self-Attention (Global Attention)
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

# 3. Mode 2: Khối MoE Sparse Multi-Experts (Top-2 SwiGLU Experts)
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

# 4. Mode 3: Khối ResNet Dense Residual MLP
class VisionResNetSubnet(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int = 384):
        super().__init__()
        self.fc1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# =========================================================================
# TẦNG AML VISION CHỌN KIẾN TRÚC MẠNG CON + ĐIỀU TIẾT HtS MASKING
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
        # x: [B, N, D]
        # block_mode_probs: [B, 4] -> Trọng số chọn [CNN, ViT, MoE, ResNet]
        # d_mask: [B, 1, D] -> Mặt nạ chiều HtS
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
        
        # Tổ hợp tuyến tính theo Mode Routing của AML
        h_all = torch.stack([h_cnn, h_vit, h_moe, h_res], dim=1) # [B, 4, N, D]
        w = block_mode_probs.view(B, 4, 1, 1)
        h_out = (h_all * w).sum(dim=1)
        
        return x + h_out * d_mask

# =========================================================================
# MÔ HÌNH AML VISION TOÀN DIỆN (PROTOTYPE BANK + HtS + ARCHITECTURE SELECTION)
# =========================================================================
class AMLVisionArchitectureSearchModel(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, dim=192, depth=6, K=16, latent_dim=64):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.K = K
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        
        # Prototype Memory Bank
        self.proto_bank = nn.Parameter(torch.randn(K, latent_dim) * 0.02)
        self.ctx_encoder = nn.Sequential(
            nn.Linear(dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Structure Delta: Sinh Gate độ sâu, Chiều rộng kênh D(x) và Lựa chọn mạng con (4 Modes x Depth)
        self.structure_delta = nn.Linear(latent_dim, depth + (depth * dim) + (depth * 4))
        
        self.layers = nn.ModuleList([AMLVisionAdaptiveBlock(dim=dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x, tau=0.3):
        B = x.shape[0]
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        
        # 1. Trích xuất vector ngữ cảnh z từ toàn ảnh
        z = self.ctx_encoder(h.mean(dim=1))
        
        # 2. Phân bổ Prototype: pi(x)
        dists = torch.cdist(z, self.proto_bank)
        pi = F.softmax(-dists / math.sqrt(z.shape[-1]), dim=-1)
        
        # 3. Giải mã cấu trúc mạng qua HtS Delta Head
        delta = self.structure_delta(z)
        
        # Độ sâu từng tầng
        layer_gates = torch.sigmoid(delta[:, :self.depth])
        
        # Chiều rộng kênh D(x)
        idx_dim_end = self.depth + (self.depth * self.dim)
        d_raw = torch.sigmoid(delta[:, self.depth:idx_dim_end].view(B, self.depth, self.dim))
        d_lengths = 64.0 + F.softplus(d_raw.mean(-1, keepdim=True) * self.dim - 64.0)
        arange = torch.arange(self.dim, device=x.device).float().view(1, 1, self.dim)
        
        # Lựa chọn 4 chế độ mạng con (CNN, ViT, MoE, ResNet) cho từng tầng
        mode_logits = delta[:, idx_dim_end:].view(B, self.depth, 4)
        block_mode_probs = F.gumbel_softmax(mode_logits, tau=tau, hard=False, dim=-1)
        
        # 4. Thực thi dòng tính toán thích ứng qua các tầng
        for i, layer in enumerate(self.layers):
            d_mask_i = torch.sigmoid(10.0 * (d_lengths[:, i:i+1, :] - arange))
            h_layer = layer(h, block_mode_probs[:, i, :], d_mask_i)
            gate_i = layer_gates[:, i:i+1, None]
            h = gate_i * h_layer + (1.0 - gate_i) * h
            
        cls_out = self.norm(h.mean(dim=1))
        logits = self.head(cls_out)
        return logits, {
            'layer_gates': layer_gates,
            'd_lengths': d_lengths,
            'block_mode_probs': block_mode_probs,
            'pi': pi
        }

# =========================================================================
# CHƯƠNG TRÌNH HUẤN LUYỆN & KIỂM TRA CHUYÊN SÂU THỊ GIÁC
# =========================================================================
def run_vision_aml_pure_benchmark():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== HUAN LUYEN VISION AML (HtS PROTOTYPE + SUBNET ARCHITECTURE SELECTION) ({device}) ===")
    
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
    
    model = AMLVisionArchitectureSearchModel(dim=192, depth=6, K=16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Khoi tao mo hinh Vision AML: {n_params:,} tham so (~{n_params*4/(1024*1024):.2f} MB)")
    print(f"Cac mang con co the chon: [Mode 0: CNN | Mode 1: ViT | Mode 2: MoE | Mode 3: ResNet]")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    epochs = 8
    best_acc = 0.0
    best_ckpt = "best_vision_aml_arch_search.pth"
    
    print("\n--- BAT DAU HUAN LUYEN TRONG 8 EPOCHS TREN CIFAR-10 ---")
    t0_all = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        t0_ep = time.time()
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _ = model(x, tau=0.5)
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
                    logits, _ = model(x, tau=0.3)
                _, predicted = logits.max(1)
                test_total += y.size(0)
                test_correct += predicted.eq(y).sum().item()
                
        test_acc = 100.0 * test_correct / test_total
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_ckpt)
            
        print(f"Epoch {epoch:2d}/{epochs:2d} ({time.time()-t0_ep:4.1f}s) | Train Loss: {train_loss/total:.4f} | Train Acc: {train_acc:5.2f}% | Test Acc: {test_acc:5.2f}% (Best: {best_acc:5.2f}%)")

    print(f"\nHuan luyen hoan tat trong {(time.time() - t0_all)/60:.2f} phut. Best Test Acc: {best_acc:.2f}%")
    
    # --- PHAN TICH TI LE CHON KIEN TRUC TREN 10,000 ANH TEST ---
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    model.eval()
    
    mode_probs_all = []
    d_lengths_all = []
    gates_all = []
    
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            with torch.amp.autocast('cuda'):
                _, diag = model(x, tau=0.3)
            mode_probs_all.append(diag['block_mode_probs'].cpu().numpy())
            d_lengths_all.append(diag['d_lengths'].cpu().numpy())
            gates_all.append(diag['layer_gates'].cpu().numpy())
            
    mode_arr = np.concatenate(mode_probs_all, axis=0) # [10000, 6, 4]
    d_arr = np.concatenate(d_lengths_all, axis=0)     # [10000, 6, 1]
    g_arr = np.concatenate(gates_all, axis=0)         # [10000, 6]
    
    mean_modes = np.mean(mode_arr, axis=0) * 100
    mean_d = np.mean(d_arr, axis=0).squeeze(-1)
    mean_g = np.mean(g_arr, axis=0) * 100
    
    print("\n" + "="*95)
    print("BÁO CÁO TỰ ĐỘNG CHỌN KIẾN TRÚC MẠNG CON CỦA VISION AML (10,000 ẢNH TEST)")
    print("="*95)
    print(f"{'Tầng':<8} {'Cổng Sâu G':<12} {'Chiều D(x)':<12} {'CNN (%)':<12} {'ViT (%)':<12} {'MoE (%)':<12} {'ResNet (%)':<12} {'Kiến Trúc Thắng'}")
    print("-" * 95)
    arch_names = ["CNN", "ViT", "MoE", "ResNet"]
    for i in range(6):
        c, v, m, r = mean_modes[i]
        dom = arch_names[np.argmax(mean_modes[i])]
        print(f"Layer {i+1:<3} {mean_g[i]:<10.1f}% {mean_d[i]:<11.1f} {c:<11.2f}% {v:<11.2f}% {m:<11.2f}% {r:<11.2f}% {dom}")
        
    print("-" * 95)
    tot_mode = np.mean(mean_modes, axis=0)
    print(f"{'TOTAL':<8} {np.mean(mean_g):<10.1f}% {np.mean(mean_d):<11.1f} {tot_mode[0]:<11.2f}% {tot_mode[1]:<11.2f}% {tot_mode[2]:<11.2f}% {tot_mode[3]:<11.2f}% {arch_names[np.argmax(tot_mode)]} (DOMINANT)")
    print("="*95)

if __name__ == '__main__':
    run_vision_aml_pure_benchmark()
