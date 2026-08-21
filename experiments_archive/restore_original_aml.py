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
# 1. 4 ỨNG VIÊN KIẾN TRÚC MẠNG CON ĐỘC LẬP (CNN / ViT / MoE / ResNet)
# =========================================================================

# Mode 0: CNN Subnet (Local Spatial Features)
class ScaledCNNSubnet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
    def forward(self, x_2d):
        return self.conv2(self.act(self.conv1(x_2d)))

# Mode 1: ViT Subnet (Multi-Head Global Self-Attention)
class ScaledViTSubnet(nn.Module):
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

# Mode 2: MoE Subnet (Sparse Top-2 SwiGLU Experts)
class ScaledMoESubnet(nn.Module):
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

# Mode 3: ResNet Subnet (Dense Residual MLP)
class ScaledResNetSubnet(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int = 384):
        super().__init__()
        self.fc1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# =========================================================================
# 2. KHỐI TẦNG AML CHUẨN GỐC (DISCRETE HARD GUMBEL-SOFTMAX ROUTING)
# =========================================================================
class OriginalScaledAMLBlock(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.cnn = ScaledCNNSubnet(dim)
        self.vit = ScaledViTSubnet(dim)
        self.moe = ScaledMoESubnet(dim)
        self.res = ScaledResNetSubnet(dim)
        
    def forward(self, x, bm_probs, d_mask):
        # bm_probs: [B, 4] (Hard One-Hot Gumbel-Softmax)
        # d_mask: [B, 1, D] (HtS Dynamic Width Mask)
        B, N, D = x.shape
        x_norm = self.norm(x) * d_mask
        
        # Chỉ kích hoạt và tính toán nhánh được chọn để tránh rò rỉ gradient
        H = W = int(math.sqrt(N))
        x_2d = x_norm.transpose(1, 2).reshape(B, D, H, W)
        
        h_cnn = self.cnn(x_2d).flatten(2).transpose(1, 2)
        h_vit = self.vit(x_norm)
        h_moe = self.moe(x_norm)
        h_res = self.res(x_norm)
        
        h_all = torch.stack([h_cnn, h_vit, h_moe, h_res], dim=1) # [B, 4, N, D]
        w = bm_probs.view(B, 4, 1, 1)
        h_out = (h_all * w).sum(dim=1)
        
        return x + h_out * d_mask

# =========================================================================
# 3. MÔ HÌNH AML GỐC KHÔI PHỤC TOÀN DIỆN (PROTOTYPE ANCHOR + +2.0 BIAS GATES)
# =========================================================================
class RestoredOriginalAMLModel(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, dim=192, depth=6, K=16, latent_dim=128):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.K = K
        self.latent_dim = latent_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        self.drop = nn.Dropout(0.1)
        
        # Context Encoder chuẩn nguyên bản
        self.ctx_encoder = nn.Sequential(
            nn.Linear(dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Meta-Controllers chuẩn nguyên bản
        self.proto_selector = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, K)
        )
        self.structure_delta = nn.Linear(latent_dim, depth + (depth * dim))
        self.block_mode_head = nn.Linear(latent_dim, depth * 4)
        
        # Khởi tạo hằng số 0.0 theo chuẩn gốc
        nn.init.constant_(self.structure_delta.weight, 0.0)
        nn.init.constant_(self.structure_delta.bias, 0.0)
        nn.init.constant_(self.block_mode_head.weight, 0.0)
        nn.init.constant_(self.block_mode_head.bias, 0.0)
        
        self.layers = nn.ModuleList([OriginalScaledAMLBlock(dim=dim) for _ in range(depth)])
        self.norm_final = nn.LayerNorm(dim)
        
        # Dual Multi-Task Heads
        self.head_cifar = nn.Linear(dim, 10)
        self.head_mnist = nn.Linear(dim, 10)
        
    def forward(self, x, task_id=1, tau=1.0, is_eval=False):
        B = x.shape[0]
        device = x.device
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        h = self.drop(h)
        
        # 1. Vector ngữ cảnh z
        z = self.ctx_encoder(h.mean(dim=1))
        
        # 2. Phân phối Prototype pi(x)
        pi = F.softmax(self.proto_selector(z), dim=-1)
        
        # 3. Structure Delta với +2.0 Positive Bias chuẩn gốc
        delta = self.structure_delta(z)
        layer_gates = torch.sigmoid(delta[:, :self.depth] + 2.0)
        
        d_raw = torch.sigmoid(delta[:, self.depth:].view(B, self.depth, self.dim) + 2.0)
        d_lengths = 64.0 + F.softplus(d_raw.mean(-1, keepdim=True) * self.dim - 64.0)
        arange = torch.arange(self.dim, device=device).float().view(1, 1, self.dim)
        
        # 4. Block Mode Selection với Hard Gumbel-Softmax chuẩn gốc
        bm_logits = self.block_mode_head(z).view(B, self.depth, 4)
        bm_probs = F.gumbel_softmax(bm_logits, tau=tau, hard=True, dim=-1)
        
        # 5. Dòng tính toán qua các tầng
        for i, layer in enumerate(self.layers):
            d_mask_i = torch.sigmoid(10.0 * (d_lengths[:, i:i+1, :] - arange))
            h_layer = layer(h, bm_probs[:, i, :], d_mask_i)
            gate_i = layer_gates[:, i:i+1, None]
            h = gate_i * h_layer + (1.0 - gate_i) * h
            
        cls_out = self.norm_final(h.mean(dim=1))
        if task_id == 1:
            logits = self.head_cifar(cls_out)
        else:
            logits = self.head_mnist(cls_out)
            
        return logits, {
            "pi": pi,
            "layer_gates": layer_gates,
            "d_lengths": d_lengths,
            "bm_probs": bm_probs
        }

def evaluate_task(model, data_loader, task_id, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                logits, _ = model(x, task_id=task_id, tau=0.3, is_eval=True)
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total

def run_restored_aml_continual_experiment():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== CHAY KIEM CHUNG KIEN TRUC AML NGUYEN BAN KHOI PHUC TOAN DIEN ({device}) ===")
    
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
    
    model = RestoredOriginalAMLModel(dim=192, depth=6, K=16).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # ==================== GIAI ĐOẠN 1: HUẤN LUYỆN CIFAR-10 ====================
    print("\n" + "="*80)
    print("PHASE 1: HUAN LUYEN CIFAR-10 TREN KIEN TRUC AML CHUAN GOC")
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
                logits, diag = model(x, task_id=1, tau=1.0)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer1)
            scaler.update()
        scheduler1.step()
        cifar_acc = evaluate_task(model, cifar_test_loader, task_id=1, device=device)
        print(f"Epoch {ep}/8 ({time.time()-t0:4.1f}s) -> CIFAR-10 Acc: {cifar_acc:5.2f}%")
        
    cifar_init_acc = evaluate_task(model, cifar_test_loader, task_id=1, device=device)
    print(f"\n[PHASE 1 HOAN TAT] CIFAR-10 Test Acc Dat Duoc: {cifar_init_acc:.2f}%")
    
    # Lưu lại trạng thái Prototype Anchor
    print("\n" + "="*80)
    print("THUC THI AML PROTOTYPE ANCHORING: KHOA ANCHOR CIFAR-10")
    print("="*80)
    for p in model.head_cifar.parameters():
        p.requires_grad = False
        
    # ==================== GIAI ĐOẠN 2: HUẤN LUYỆN MNIST ====================
    print("\n" + "="*80)
    print("PHASE 2: HUAN LUYEN CONTINUAL MNIST TREN KIEN TRUC AML CHUAN GOC")
    print("="*80)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer2 = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-2)
    
    for ep in range(1, 4):
        model.train()
        t0 = time.time()
        for x, y in mnist_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer2.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _ = model(x, task_id=2, tau=0.8)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer2)
            scaler.update()
            
        mnist_acc = evaluate_task(model, mnist_test_loader, task_id=2, device=device)
        cifar_retest = evaluate_task(model, cifar_test_loader, task_id=1, device=device)
        print(f"Epoch {ep}/3 ({time.time()-t0:4.1f}s) -> MNIST Acc: {mnist_acc:5.2f}% | CIFAR-10 Retest: {cifar_retest:5.2f}%")
        
    cifar_final_acc = evaluate_task(model, cifar_test_loader, task_id=1, device=device)
    mnist_final_acc = evaluate_task(model, mnist_test_loader, task_id=2, device=device)
    
    print("\n" + "="*85)
    print("BAO CAO KHOA HOC: KET QUA KHOI PHUC KIEN TRUC AML NGUYEN BAN")
    print("="*85)
    print(f"1. CIFAR-10 Ban Dau (Task 1)             : {cifar_init_acc:5.2f}%")
    print(f"2. MNIST Sau Khi Hoc (Task 2)            : {mnist_final_acc:5.2f}%")
    print(f"3. CIFAR-10 Kiem Tra Lai (Retest)        : {cifar_final_acc:5.2f}%")
    delta = cifar_init_acc - cifar_final_acc
    print(f"4. Muc Do Quen Tri Thuc (Delta Forgetting): {delta:+5.2f}%")
    print("="*85)

if __name__ == '__main__':
    run_restored_aml_continual_experiment()
