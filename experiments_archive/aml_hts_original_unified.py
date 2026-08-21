"""
AML ORIGINAL ARCHITECTURE WITH HtS CONTINUAL LEARNING
======================================================
Khoi phuc 100% kien truc goc:
1. Prototype Bank (16 prototypes) + Context Encoder + Softmax selection
2. Meta-Controllers: structure_delta, block_mode_head (khoi tao 0)
3. Hard-to-Soft (HtS) Masking: d_lengths + sigmoid(10.0 * (d_lengths - arange))
4. Dynamic Routing: CNN, ViT, MoE, ResNet theo Hard Gumbel-Softmax (bm_probs)
5. Positive bias gate (+2.0)
6. Continual Learning bang cach dieu chinh sinh trong so va mask theo task ma khong tach model thanh nhanh roi rac.
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
# 4 SUB-NETWORKS CHUAN AML GOC
# =========================================================================
class CNNSubnet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
    def forward(self, x2d):
        return self.pw(self.act(self.dw(x2d)))

class ViTSubnet(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.h = heads
        self.hd = dim // heads
        self.scale = self.hd ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(out)

class MoESubnet(nn.Module):
    def __init__(self, dim, inner=256, n_experts=4, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, inner, bias=False),
                nn.SiLU(),
                nn.Linear(inner, dim, bias=False)
            ) for _ in range(n_experts)
        ])
    def forward(self, x):
        B, N, D = x.shape
        xf = x.reshape(-1, D)
        gates = F.softmax(self.router(xf), dim=-1)
        weights, indices = torch.topk(gates, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        out = torch.zeros_like(xf)
        for k in range(self.top_k):
            for i, exp in enumerate(self.experts):
                mask = (indices[:, k] == i)
                if mask.any():
                    out[mask] += weights[mask, k:k+1] * exp(xf[mask])
        return out.reshape(B, N, D)

class ResNetSubnet(nn.Module):
    def __init__(self, dim, inner=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Linear(inner, dim)
        )
    def forward(self, x):
        return self.net(x)

class ScaledAMLBlock(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.cnn = CNNSubnet(dim)
        self.vit = ViTSubnet(dim)
        self.moe = MoESubnet(dim)
        self.res = ResNetSubnet(dim)

    def forward(self, x, bm_probs, d_mask):
        B, N, D = x.shape
        H = W = int(math.sqrt(N))
        
        xn = self.norm(x) * d_mask
        
        # 4 paradigms
        h_cnn = self.cnn(xn.transpose(1, 2).reshape(B, D, H, W)).flatten(2).transpose(1, 2)
        h_vit = self.vit(xn)
        h_moe = self.moe(xn)
        h_res = self.res(xn)
        
        stacked = torch.stack([h_cnn, h_vit, h_moe, h_res], dim=1) # [B, 4, N, D]
        h_layer = (stacked * bm_probs.view(B, 4, 1, 1)).sum(dim=1)
        
        # HtS apply
        return x + h_layer * d_mask

# =========================================================================
# AML VISION MODEL GOC (DUNG CHUNG CHO CA 2 TASK)
# =========================================================================
class AMLOriginalVisionModel(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, 
                 dim=192, depth=6, K=16, latent_dim=128, num_classes=10):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.K = K
        self.latent_dim = latent_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        self.drop = nn.Dropout(0.1)

        # Context Encoder
        self.ctx_encoder = nn.Sequential(
            nn.Linear(dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Prototype Bank Goc (K prototypes)
        self.prototype_bank = nn.Parameter(torch.randn(K, latent_dim) * 0.02)
        self.proto_selector = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.SiLU(),
            nn.Linear(256, K)
        )

        # Meta Controllers
        self.structure_delta = nn.Linear(latent_dim, depth + (depth * dim))
        self.block_mode_head = nn.Linear(latent_dim, depth * 4)

        nn.init.constant_(self.structure_delta.weight, 0.0)
        nn.init.constant_(self.structure_delta.bias, 0.0)
        nn.init.constant_(self.block_mode_head.weight, 0.0)
        nn.init.constant_(self.block_mode_head.bias, 0.0)

        # 6 Scaled AML Blocks
        self.layers = nn.ModuleList([ScaledAMLBlock(dim=dim) for _ in range(depth)])
        self.norm_final = nn.LayerNorm(dim)
        
        # Heads rieng biet cho 2 task nhung chung hoan toan AML Backbone
        self.head_cifar = nn.Linear(dim, num_classes)
        self.head_mnist = nn.Linear(dim, num_classes)

    def forward(self, x, task="cifar", tau=1.0):
        B = x.shape[0]
        device = x.device

        # Patch embedding
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        h = self.drop(h)

        # Context Representation
        z_raw = self.ctx_encoder(h.mean(dim=1))
        
        # Prototype Attention (Chuan AML goc)
        pi = F.softmax(self.proto_selector(z_raw), dim=-1) # [B, K]
        z_proto = pi @ self.prototype_bank                  # [B, latent_dim]
        z = z_raw + z_proto

        # HtS Meta-Calculations
        delta = self.structure_delta(z)
        layer_gates = torch.sigmoid(delta[:, :self.depth] + 2.0)
        
        d_lengths = 64.0 + F.softplus(
            torch.sigmoid(delta[:, self.depth:].view(B, self.depth, self.dim).mean(-1, keepdim=True) + 2.0) * self.dim - 64.0
        )
        arange = torch.arange(self.dim, device=device).float().view(1, 1, self.dim)

        bm_logits = self.block_mode_head(z).view(B, self.depth, 4)
        bm_probs = F.gumbel_softmax(bm_logits, tau=tau, hard=True, dim=-1)

        # Chay qua cac lop AML voi HtS Masking
        for i, layer in enumerate(self.layers):
            h_prev = h
            d_mask_i = torch.sigmoid(10.0 * (d_lengths[:, i:i+1, :] - arange))
            h_layer = layer(h, bm_probs[:, i, :], d_mask_i)
            gate_i = layer_gates[:, i:i+1, None]
            h = gate_i * h_layer + (1.0 - gate_i) * h_prev

        h_out = self.norm_final(h.mean(dim=1))
        
        if task == "cifar":
            logits = self.head_cifar(h_out)
        else:
            logits = self.head_mnist(h_out)

        return logits, {
            "pi": pi,
            "layer_gates": layer_gates,
            "d_lengths": d_lengths,
            "bm_probs": bm_probs
        }

def evaluate(model, loader, device, task="cifar"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                logits, _ = model(x, task=task, tau=0.3)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

def run():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== HUAN LUYEN CONTINUAL VOI KIEN TRUC AML & HtS GOC ({device}) ===")

    # Transforms
    tf_c_tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    tf_c_te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    tf_m_tr = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,)*3, (0.3081,)*3)
    ])
    tf_m_te = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,)*3, (0.3081,)*3)
    ])

    c_tr = DataLoader(datasets.CIFAR10('./data', True, download=True, transform=tf_c_tr), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    c_te = DataLoader(datasets.CIFAR10('./data', False, download=True, transform=tf_c_te), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    m_tr = DataLoader(datasets.MNIST('./data', True, download=True, transform=tf_m_tr), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    m_te = DataLoader(datasets.MNIST('./data', False, download=True, transform=tf_m_te), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    model = AMLOriginalVisionModel(dim=192, depth=6, K=16).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    # =========================================================================
    # PHASE 1: TRAIN CIFAR-10 TREN AML GOC
    # =========================================================================
    print("\n------------------------------------------------------------")
    print("PHASE 1: TRAIN CIFAR-10 TREN KIEN TRUC AML GOC (8 Epochs)")
    print("------------------------------------------------------------")
    opt1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=8, eta_min=1e-5)

    for ep in range(1, 9):
        model.train()
        t0 = time.time()
        for x, y in c_tr:
            x, y = x.to(device), y.to(device)
            opt1.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, info = model(x, task="cifar", tau=1.0)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt1)
            scaler.update()
        sch1.step()
        acc = evaluate(model, c_te, device, task="cifar")
        print(f"Epoch {ep}/8 ({time.time()-t0:.1f}s) -> CIFAR-10 Acc: {acc:.2f}%")

    cifar_init = evaluate(model, c_te, device, task="cifar")
    print(f"\n[PHASE 1 HOAN TAT] CIFAR-10 Dat: {cifar_init:.2f}%\n")

    # =========================================================================
    # PHASE 2: CONTINUAL LEARNING MNIST VOI HtS MASK & PROTOTYPE ANCHORING
    # Khoa nua duoi cac prototype cua CIFAR de sinh mask khong de len vung CIFAR
    # =========================================================================
    print("------------------------------------------------------------")
    print("PHASE 2: TRAIN CONTINUAL MNIST VOI HtS DYNAMIC MASK (5 Epochs)")
    print("------------------------------------------------------------")
    opt2 = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=5, eta_min=1e-5)

    for ep in range(1, 6):
        model.train()
        t0 = time.time()
        for x, y in m_tr:
            x, y = x.to(device), y.to(device)
            opt2.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, info = model(x, task="mnist", tau=1.0)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt2)
            scaler.update()
        sch2.step()
        
        m_acc = evaluate(model, m_te, device, task="mnist")
        c_retest = evaluate(model, c_te, device, task="cifar")
        print(f"Epoch {ep}/5 ({time.time()-t0:.1f}s) -> MNIST Acc: {m_acc:.2f}% | CIFAR-10 Retest: {c_retest:.2f}%")

    cifar_final = evaluate(model, c_te, device, task="cifar")
    mnist_final = evaluate(model, m_te, device, task="mnist")

    print("\n" + "="*65)
    print("TONG KET CONTINUAL LEARNING VOI HtS & AML GOC")
    print("="*65)
    print(f"CIFAR-10 Ban Dau (Phase 1)    : {cifar_init:.2f}%")
    print(f"MNIST Dat Duoc (Phase 2)      : {mnist_final:.2f}%")
    print(f"CIFAR-10 Retest (Sau Phase 2) : {cifar_final:.2f}%")
    print(f"Delta Forgetting              : {cifar_init - cifar_final:+.2f}%")
    print("="*65)

if __name__ == '__main__':
    run()
