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

# --- 1. SwiGLU Expert Cho Vision Patches ---
class VisionSwiGLUExpert(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_up = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, dim, bias=False)
        
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

# --- 2. Sparse MoE Vision Block ---
class SparseMoEVisionBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            VisionSwiGLUExpert(dim, intermediate_dim) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        B, N, D = x.shape
        x_flat = x.view(-1, D)
        
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        out_flat = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            exp_idx = topk_indices[:, k]
            exp_weight = topk_weights[:, k:k+1]
            for i, expert in enumerate(self.experts):
                mask = (exp_idx == i)
                if mask.any():
                    out_flat[mask] += exp_weight[mask] * expert(x_flat[mask])
                    
        # Load Balancing Loss (Mixtral Style)
        # density: xác suất trung bình của từng chuyên gia
        # count: tỷ lệ token thực tế được gán cho từng chuyên gia
        density = router_probs.mean(dim=0)
        
        # Tạo one-hot vector cho top-k indices
        one_hot_indices = F.one_hot(topk_indices, num_classes=self.num_experts).float() # [B*N, top_k, num_experts]
        count = one_hot_indices.sum(dim=1).mean(dim=0) / self.top_k # [num_experts]
        
        aux_loss = self.num_experts * torch.sum(density * count)
        
        return out_flat.view(B, N, D), aux_loss, router_probs

# --- 3. Multi-Head Self-Attention ---
class VisionAttention(nn.Module):
    def __init__(self, dim=192, num_heads=6, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

# --- 4. Vision AML-MoE Transformer Layer ---
class AMLMoEVisionLayer(nn.Module):
    def __init__(self, dim=192, num_heads=6, intermediate_dim=384, num_experts=4, top_k=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = VisionAttention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = SparseMoEVisionBlock(dim=dim, intermediate_dim=intermediate_dim, num_experts=num_experts, top_k=top_k)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        moe_out, aux_loss, router_probs = self.moe(self.norm2(x))
        x = x + moe_out
        return x, aux_loss, router_probs

# --- 5. Full Vision MoE Transformer ---
class AMLVisionMoETransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, 
                 dim=192, depth=6, num_heads=6, intermediate_dim=384, num_experts=4, top_k=2, latent_dim=64):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_classes = num_classes
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim) * 0.02)
        
        self.macro_router = nn.Sequential(
            nn.Linear(dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, depth)
        )
        
        self.layers = nn.ModuleList([
            AMLMoEVisionLayer(dim=dim, num_heads=num_heads, intermediate_dim=intermediate_dim, num_experts=num_experts, top_k=top_k)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, patches), dim=1) + self.pos_embed
        
        macro_z = h[:, 0, :]
        macro_gates = torch.sigmoid(self.macro_router(macro_z))
        
        total_aux_loss = 0.0
        expert_distributions = []
        
        for i, layer in enumerate(self.layers):
            h_prev = h
            h_layer, aux_l, r_probs = layer(h)
            gate_i = macro_gates[:, i:i+1, None]
            h = gate_i * h_layer + (1.0 - gate_i) * h_prev
            
            total_aux_loss += aux_l
            expert_distributions.append(r_probs.mean(dim=0).detach().cpu().numpy())
            
        cls_out = self.norm(h[:, 0])
        logits = self.head(cls_out)
        
        return logits, {
            'macro_gates': macro_gates,
            'aux_loss': total_aux_loss / self.depth,
            'expert_dist': expert_distributions
        }

# --- 6. Fast Training Pipeline on CIFAR-10 ---
def train_and_eval_vision_moe():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== TRAINING FAST VISION AML-MoE TRANSFORMER ON CIFAR-10 ({device}) ===")
    
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
    
    model = AMLVisionMoETransformer(
        img_size=32, patch_size=4, in_chans=3, num_classes=10,
        dim=192, depth=6, num_heads=6, intermediate_dim=384,
        num_experts=4, top_k=2
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Architecture : 6 Layers, 4 SwiGLU Experts per layer, Top-2 Sparse Routing")
    print(f"Total Parameters   : {n_params:,} (~{n_params*4/(1024*1024):.2f} MB)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    epochs = 10
    best_acc = 0.0
    
    print("\n--- STAGE 1: TRAINING FAST VISION MOE (10 EPOCHS) ---")
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
                logits, diag = model(x)
                ce_loss = criterion(logits, y)
                loss = ce_loss + 0.01 * diag['aux_loss']
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * x.size(0)
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        scheduler.step()
        train_acc = 100.0 * correct / total
        
        # Validation on Test Set
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
            torch.save(model.state_dict(), "best_vision_moe_cifar10.pth")
            
        ep_time = time.time() - t0_ep
        print(f"Epoch {epoch:2d}/{epochs:2d} ({ep_time:4.1f}s) | Train Loss: {train_loss/total:.4f} | Train Acc: {train_acc:5.2f}% | Test Acc: {test_acc:5.2f}% (Best: {best_acc:5.2f}%)")

    print(f"\nTraining Complete in {(time.time() - t0_all)/60:.2f} mins. Peak Test Accuracy: {best_acc:.2f}%")
    
    # Analyze Expert Utilization
    model.load_state_dict(torch.load("best_vision_moe_cifar10.pth", weights_only=True))
    model.eval()
    with torch.no_grad():
        sample_x, _ = next(iter(test_loader))
        _, diag = model(sample_x.to(device))
        print("\n--- EXPERT UTILIZATION ACROSS ALL 6 LAYERS ---")
        for layer_idx, dist in enumerate(diag['expert_dist']):
            dist_str = ", ".join([f"E{e+1}: {p*100:4.1f}%" for e, p in enumerate(dist)])
            print(f"  * Layer {layer_idx+1} -> [{dist_str}]")

if __name__ == '__main__':
    train_and_eval_vision_moe()
