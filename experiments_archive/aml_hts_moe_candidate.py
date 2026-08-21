import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. MẠNG CON 1: CNN PARADIGM (Local Inductive Bias) ---
class SubnetCNNBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
    def forward(self, x_2d):
        return self.conv2(self.act(self.conv1(x_2d)))

# --- 2. MẠNG CON 2: ViT PARADIGM (Global Self-Attention) ---
class SubnetViTBlock(nn.Module):
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

# --- 3. MẠNG CON 3: MoE PARADIGM (Sparse Multi-Expert) ---
class SubnetMoEBlock(nn.Module):
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

# --- 4. MẠNG CON 4: ResNet MLP PARADIGM (Standard Dense Residual) ---
class SubnetResNetBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int = 384):
        super().__init__()
        self.fc1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# --- 5. TẦNG AML HtS-ROUTER CHỌN KIẾN TRÚC MẠNG CON (PARADIGM SELECTION) ---
class AMLHtSParadigmBlock(nn.Module):
    """
    HtS điều khiển việc chọn 1 trong 4 kiến trúc mạng con:
    0: CNN Subnet
    1: ViT Subnet
    2: MoE Subnet
    3: ResNet Subnet
    và đồng thời áp mặt nạ chiều thích ứng D(x) qua Hard-to-Soft masking.
    """
    def __init__(self, dim=192):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.cnn = SubnetCNNBlock(dim)
        self.vit = SubnetViTBlock(dim)
        self.moe = SubnetMoEBlock(dim)
        self.res = SubnetResNetBlock(dim)
        
    def forward(self, x, paradigm_weights, d_mask):
        # x: [B, N, D]
        # paradigm_weights: [B, 4] -> Trọng số HtS chọn [CNN, ViT, MoE, ResNet]
        # d_mask: [B, 1, D] -> Mặt nạ chiều HtS
        B, N, D = x.shape
        x_norm = self.norm(x) * d_mask
        
        # Nhánh 0: CNN
        H = W = int(math.sqrt(N))
        x_2d = x_norm.transpose(1, 2).reshape(B, D, H, W)
        h_cnn = self.cnn(x_2d).flatten(2).transpose(1, 2)
        
        # Nhánh 1: ViT
        h_vit = self.vit(x_norm)
        
        # Nhánh 2: MoE
        h_moe = self.moe(x_norm)
        
        # Nhánh 3: ResNet
        h_res = self.res(x_norm)
        
        # Tổng hợp có trọng số dựa trên HtS Paradigm Routing
        h_candidates = torch.stack([h_cnn, h_vit, h_moe, h_res], dim=1) # [B, 4, N, D]
        w = paradigm_weights.view(B, 4, 1, 1)
        h_selected = (h_candidates * w).sum(dim=1)
        
        return x + h_selected * d_mask

print("[SUCCESS] AML HtS Paradigm Block (CNN / ViT / MoE / ResNet) successfully defined!")
