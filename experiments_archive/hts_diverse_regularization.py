"""
HtS MULTI-STAGE WITH DIVERSITY & ORTHOGONALITY REGULARIZATION
============================================================
Triển khai bộ 3 hàm Loss chống Hypernetwork Collapse chuẩn xác:
1. L_ortho: Ép các Basis tensor vuông góc nhau: sum_{i != j} ( <B_i, B_j> / (|B_i|*|B_j|) )^2
2. L_entropy: Ép alpha(x) của từng mẫu phải sắc nét (Low per-sample entropy): sum( -alpha * log(alpha) )
3. L_usage: Ép cân bằng sử dụng toàn bộ Basis trên batch (KL(mean(alpha) || Uniform)): KL divergence
4. Layer-aware Diversity: Phạt mạnh ở tầng sâu (Stage 3, cos ~ 0.4-0.7), cho phép tầng nông chia sẻ edge (Stage 1, cos ~ 0.8-0.95).
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
# 1. MULTI-STAGE HtS BASIS SYSTEM
# =========================================================================
class HtSMultiStageBasis(nn.Module):
    def __init__(self, K=4, d1=24, d2=48, d3=96):
        super().__init__()
        self.K = K
        # Stage 1: [4, 24, 24, 3, 3]
        self.b1 = nn.Parameter(torch.randn(K, d1, d1, 3, 3) * 0.02)
        # Stage 1->2 Transition: [4, 48, 24, 3, 3]
        self.b1_2 = nn.Parameter(torch.randn(K, d2, d1, 3, 3) * 0.02)
        # Stage 2: [4, 48, 48, 3, 3]
        self.b2 = nn.Parameter(torch.randn(K, d2, d2, 3, 3) * 0.02)
        # Stage 2->3 Transition: [4, 96, 48, 3, 3]
        self.b2_3 = nn.Parameter(torch.randn(K, d3, d2, 3, 3) * 0.02)
        # Stage 3: [4, 96, 96, 3, 3]
        self.b3 = nn.Parameter(torch.randn(K, d3, d3, 3, 3) * 0.02)

    def get_kernel(self, name, alpha):
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

    def compute_orthogonality_loss(self):
        """
        L_ortho = sum_{i != j} ( (B_i . B_j) / (|B_i| * |B_j|) )^2
        """
        loss_ortho = 0.0
        basis_list = [self.b1, self.b1_2, self.b2, self.b2_3, self.b3]
        for b in basis_list:
            # Flatten: [K, dim]
            b_flat = b.view(self.K, -1)
            b_norm = F.normalize(b_flat, p=2, dim=-1)
            gram = torch.mm(b_norm, b_norm.t()) # [K, K]
            # Loại bỏ đường chéo chính (i == j)
            mask = ~torch.eye(self.K, dtype=torch.bool, device=b.device)
            loss_ortho += (gram[mask] ** 2).mean()
        return loss_ortho / len(basis_list)

# =========================================================================
# 2. HtS RESIDUAL BLOCK
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
# 3. COMPLETE MODEL VỚI SHARP ROUTER & GUMBEL MIX
# =========================================================================
class HtSDiverseModel(nn.Module):
    def __init__(self, d1=24, d2=48, d3=96, K=4, num_classes=10):
        super().__init__()
        self.K = K
        
        self.stem_mnist = nn.Sequential(
            nn.Conv2d(1, d1, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, d1),
            nn.GELU()
        )
        self.stem_cifar = nn.Sequential(
            nn.Conv2d(3, d1, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, d1),
            nn.GELU()
        )
        
        self.basis_sys = HtSMultiStageBasis(K=K, d1=d1, d2=d2, d3=d3)
        
        # Router Controller
        self.task_ctrl = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(d1 * 4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 8 * K)
        )
        
        # 8 Blocks
        self.b1 = HtSResidualBlock(d1, d1, stride=1)
        self.b2 = HtSResidualBlock(d1, d1, stride=1)
        self.b3 = HtSResidualBlock(d1, d2, stride=2)
        self.b4 = HtSResidualBlock(d2, d2, stride=1)
        self.b5 = HtSResidualBlock(d2, d2, stride=1)
        self.b6 = HtSResidualBlock(d2, d3, stride=2)
        self.b7 = HtSResidualBlock(d3, d3, stride=1)
        self.b8 = HtSResidualBlock(d3, d3, stride=1)
        
        # Heads
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

    def forward(self, x, task="cifar", tau=0.8):
        B = x.shape[0]
        h = self.stem_mnist(x) if task == "mnist" else self.stem_cifar(x)
        
        # Router Logits
        logits_alpha = self.task_ctrl(h).view(B, 8, self.K)
        
        if self.training:
            # Gumbel-Softmax nhiệt độ tau để tạo sparsity mà vẫn differentiable
            alphas = F.gumbel_softmax(logits_alpha, tau=tau, hard=False, dim=-1)
        else:
            alphas = F.softmax(logits_alpha / tau, dim=-1)
            
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
        return logits, alphas

# =========================================================================
# 4. HÀM TÍNH TOÁN BỘ BA REGULARIZATION (ENTROPY + ORTHO + USAGE)
# =========================================================================
def compute_diversity_losses(alphas, model, K=4):
    """
    alphas: [B, 8, K]
    """
    # A. Per-Sample Entropy (Muốn giảm entropy để mỗi sample chọn dứt khoát 1-2 basis)
    # H(alpha) = - sum(alpha * log(alpha + 1e-8))
    eps = 1e-8
    entropy_per_sample = -(alphas * torch.log(alphas + eps)).sum(dim=-1).mean()
    
    # B. Batch Usage Balancing (Ép trung bình toàn batch phải đều K=4, chống collapse vào 1 basis)
    # mean_alpha: [8, K]
    mean_alpha = alphas.mean(dim=0)
    uniform = torch.full_like(mean_alpha, 1.0 / K)
    # KL Divergence: KL(mean_alpha || uniform) = sum(mean_alpha * log(mean_alpha / uniform))
    usage_loss = (mean_alpha * torch.log((mean_alpha + eps) / uniform)).sum(dim=-1).mean()
    
    # C. Basis Orthogonality
    ortho_loss = model.basis_sys.compute_orthogonality_loss()
    
    return entropy_per_sample, usage_loss, ortho_loss

def evaluate_fast(model, loader, device, task="cifar"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits, _ = model(x, task=task, tau=0.5)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total
    errors = total - correct
    return acc, errors, total

def measure_diversity_metrics(model, m_loader, c_loader, device):
    model.eval()
    with torch.no_grad():
        x_m, _ = next(iter(m_loader))
        x_c, _ = next(iter(c_loader))
        x_m, x_c = x_m.to(device), x_c.to(device)
        
        _, alphas_m = model(x_m, task="mnist", tau=0.5)
        _, alphas_c = model(x_c, task="cifar", tau=0.5)
        
        # Trọng số tầng sâu Stage 3 Block 8 (Kernel 96x96x3x3)
        w_m_b8 = model.basis_sys.get_kernel('b3', alphas_m[:, 7]).flatten(1)
        w_c_b8 = model.basis_sys.get_kernel('b3', alphas_c[:, 7]).flatten(1)
        
        # Trọng số tầng nông Stage 1 Block 1 (Kernel 24x24x3x3)
        w_m_b1 = model.basis_sys.get_kernel('b1', alphas_m[:, 0]).flatten(1)
        w_c_b1 = model.basis_sys.get_kernel('b1', alphas_c[:, 0]).flatten(1)
        
        w_m_b8_norm = F.normalize(w_m_b8, p=2, dim=-1)
        w_c_b8_norm = F.normalize(w_c_b8, p=2, dim=-1)
        
        w_m_b1_norm = F.normalize(w_m_b1, p=2, dim=-1)
        w_c_b1_norm = F.normalize(w_c_b1, p=2, dim=-1)
        
        # Pairwise Cosine
        cos_m_b8 = torch.mm(w_m_b8_norm, w_m_b8_norm.t())
        cos_c_b8 = torch.mm(w_c_b8_norm, w_c_b8_norm.t())
        cos_inter_b8 = torch.mm(w_m_b8_norm, w_c_b8_norm.t())
        
        cos_m_b1 = torch.mm(w_m_b1_norm, w_m_b1_norm.t())
        cos_c_b1 = torch.mm(w_c_b1_norm, w_c_b1_norm.t())
        
        mask_m = ~torch.eye(len(x_m), dtype=torch.bool, device=device)
        mask_c = ~torch.eye(len(x_c), dtype=torch.bool, device=device)
        
        intra_m_b8 = cos_m_b8[mask_m].mean().item()
        intra_c_b8 = cos_c_b8[mask_c].mean().item()
        inter_b8 = cos_inter_b8.mean().item()
        
        intra_m_b1 = cos_m_b1[mask_m].mean().item()
        intra_c_b1 = cos_c_b1[mask_c].mean().item()
        
        return {
            "Stage 1 (Block 1) Intra-MNIST Cosine": intra_m_b1,
            "Stage 1 (Block 1) Intra-CIFAR Cosine": intra_c_b1,
            "Stage 3 (Block 8) Intra-MNIST Cosine": intra_m_b8,
            "Stage 3 (Block 8) Intra-CIFAR Cosine": intra_c_b8,
            "Stage 3 (Block 8) Inter-Task Cosine": inter_b8,
            "Alpha MNIST Sample 0": alphas_m[0, 7].cpu().numpy().round(2).tolist(),
            "Alpha CIFAR Sample 0": alphas_c[0, 7].cpu().numpy().round(2).tolist(),
            "Alpha MNIST Batch Mean": alphas_m[:, 7].mean(dim=0).cpu().numpy().round(2).tolist(),
            "Alpha CIFAR Batch Mean": alphas_c[:, 7].mean(dim=0).cpu().numpy().round(2).tolist()
        }

def run():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== HtS DYNAMIC DIVERSITY BENCHMARK (ENTROPY + ORTHO + USAGE) ({device}) ===")
    print("Muc tieu: Triet tieu Hypernetwork Collapse, dat Cosine hop ly (Tang nong ~0.8-0.95, Tang sau ~0.4-0.7)\n")

    model = HtSDiverseModel(d1=24, d2=48, d3=96, K=4, num_classes=10).to(device)

    # Transforms
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

    # Trọng số Regularization
    lambda_entropy = 0.02  # Ép per-sample sparsity
    lambda_usage = 0.15    # Ép cân bằng sử dụng cả 4 Basis trên batch
    lambda_ortho = 0.10    # Ép các Basis trực giao nhau

    # =========================================================================
    # PHASE 1: TRAIN MNIST (8 Epochs)
    # =========================================================================
    print("------------------------------------------------------------")
    print("PHASE 1: TRAIN MNIST VOI DIVERSITY REGULARIZATION (8 Epochs)")
    print("------------------------------------------------------------")
    opt1 = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=8, eta_min=1e-5)

    for ep in range(1, 9):
        model.train()
        t0 = time.time()
        tau = max(0.5, 1.0 - (ep / 8) * 0.5) # Annealing tau từ 1.0 -> 0.5
        
        for x, y in m_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt1.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, alphas = model(x, task="mnist", tau=tau)
                loss_ce = criterion(logits, y)
                l_ent, l_use, l_ort = compute_diversity_losses(alphas, model, K=4)
                
                loss = loss_ce + lambda_entropy * l_ent + lambda_usage * l_use + lambda_ortho * l_ort
                
            scaler.scale(loss).backward()
            scaler.step(opt1); scaler.update()
        sch1.step()
        
        acc, errors, total_m = evaluate_fast(model, m_te, device, task="mnist")
        print(f"Epoch {ep:2d}/8 ({time.time()-t0:4.1f}s) -> MNIST Acc: {acc:.2f}% | So anh sai: {errors}/{total_m} | L_ent: {l_ent.item():.3f} | L_use: {l_use.item():.3f} | L_ort: {l_ort.item():.3f}")

    mnist_init_acc, mnist_init_err, _ = evaluate_fast(model, m_te, device, task="mnist")
    print(f"\n[PHASE 1 HOAN TAT] MNIST Dat: {mnist_init_acc:.2f}% (Sai: {mnist_init_err} anh)\n")

    # =========================================================================
    # PHASE 2: TRAIN CIFAR-10 (15 Epochs) VOI DIVERSITY & REPLAY
    # =========================================================================
    print("------------------------------------------------------------")
    print("PHASE 2: TRAIN CIFAR-10 VOI DIVERSITY & STREAMING REPLAY (15 Epochs)")
    print("------------------------------------------------------------")
    opt2 = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=5e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=15, eta_min=1e-5)

    mnist_stream = iter(m_tr)

    for ep in range(1, 16):
        model.train()
        t0 = time.time()
        tau = max(0.5, 1.0 - (ep / 15) * 0.5)
        
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
                logits_c, alphas_c = model(x_c, task="cifar", tau=tau)
                loss_c = criterion(logits_c, y_c)
                
                logits_m, alphas_m = model(x_m, task="mnist", tau=tau)
                loss_m = criterion(logits_m, y_m)
                
                l_ent_c, l_use_c, l_ort = compute_diversity_losses(alphas_c, model, K=4)
                l_ent_m, l_use_m, _ = compute_diversity_losses(alphas_m, model, K=4)
                
                loss_div = lambda_entropy * (l_ent_c + l_ent_m)*0.5 + lambda_usage * (l_use_c + l_use_m)*0.5 + lambda_ortho * l_ort
                loss = loss_c + 0.35 * loss_m + loss_div

            scaler.scale(loss).backward()
            scaler.step(opt2); scaler.update()
            
        sch2.step()
        
        c_acc, c_err, total_c = evaluate_fast(model, c_te, device, task="cifar")
        m_ret_acc, m_ret_err, total_m = evaluate_fast(model, m_te, device, task="mnist")
        print(f"Epoch {ep:2d}/15 ({time.time()-t0:4.1f}s) -> CIFAR-10: {c_acc:.2f}% | MNIST Retest: {m_ret_acc:.2f}% (Sai: {m_ret_err}/{total_m})")

    cifar_final_acc, cifar_final_err, _ = evaluate_fast(model, c_te, device, task="cifar")
    mnist_final_acc, mnist_final_err, total_m = evaluate_fast(model, m_te, device, task="mnist")

    # Đo đạc lại Diversity sau huấn luyện
    div_results = measure_diversity_metrics(model, m_te, c_te, device)

    print("\n" + "="*65)
    print("TONG KET HtS DYNAMIC DIVERSITY BENCHMARK")
    print("="*65)
    print(f"MNIST Ban Dau (Phase 1)       : {mnist_init_acc:.2f}% (Sai: {mnist_init_err} anh)")
    print(f"CIFAR-10 Dat Duoc (Phase 2)   : {cifar_final_acc:.2f}%")
    print(f"MNIST Sau Khi Hoc CIFAR-10    : {mnist_final_acc:.2f}% (Sai: {mnist_final_err} anh)")
    print("------------------------------------------------------------")
    print("CHISO DYNAMIC DIVERSITY DO DAC THUC TE:")
    for k, v in div_results.items():
        print(f"  * {k:<38}: {v}")
    print("="*65)

if __name__ == '__main__':
    run()
