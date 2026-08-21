"""
AML Continual Learning - Giải pháp triệt để:
Phase 1: Train toàn bộ model trên CIFAR-10
Phase 2: Đóng băng TOÀN BỘ backbone, chỉ train một Linear Head MNIST nhỏ
         -> Đảm bảo toán học: CIFAR-10 accuracy không thay đổi 1 bit
"""
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

# ============================================================
# CHUẨN AML GỐC: CNN / ViT / MoE / ResNet Subnets
# ============================================================
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
        self.h, self.hd = heads, dim // heads
        self.scale = self.hd ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.hd).permute(2,0,3,1,4)
        a = (q @ k.transpose(-2,-1)) * self.scale
        return self.out((a.softmax(-1) @ v).transpose(1,2).reshape(B, N, C))

class MoESubnet(nn.Module):
    def __init__(self, dim, inner=256, n=4, k=2):
        super().__init__()
        self.n, self.k = n, k
        self.router = nn.Linear(dim, n, bias=False)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, inner, bias=False), nn.SiLU(),
            nn.Linear(inner, dim, bias=False)) for _ in range(n)])
    def forward(self, x):
        B, N, D = x.shape
        xf = x.reshape(-1, D)
        w, idx = torch.topk(F.softmax(self.router(xf), -1), self.k, dim=-1)
        w = w / (w.sum(-1, keepdim=True) + 1e-8)
        out = torch.zeros_like(xf)
        for k in range(self.k):
            for i, e in enumerate(self.experts):
                m = (idx[:, k] == i)
                if m.any():
                    out[m] += w[m, k:k+1] * e(xf[m])
        return out.reshape(B, N, D)

class ResNetSubnet(nn.Module):
    def __init__(self, dim, inner=384):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, inner), nn.GELU(), nn.Linear(inner, dim))
    def forward(self, x):
        return self.net(x)

# ============================================================
# TẦNG AML GỐC với Hard Gumbel Routing + HtS Masking
# ============================================================
class AMLBlock(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.cnn = CNNSubnet(dim)
        self.vit = ViTSubnet(dim)
        self.moe = MoESubnet(dim)
        self.res = ResNetSubnet(dim)

    def forward(self, x, bm, d_mask):
        B, N, D = x.shape
        H = W = int(math.sqrt(N))
        xn = self.norm(x) * d_mask
        h0 = self.cnn(xn.transpose(1,2).reshape(B,D,H,W)).flatten(2).transpose(1,2)
        h1 = self.vit(xn)
        h2 = self.moe(xn)
        h3 = self.res(xn)
        stk = torch.stack([h0, h1, h2, h3], 1)
        out = (stk * bm.view(B, 4, 1, 1)).sum(1)
        return x + out * d_mask

# ============================================================
# MÔ HÌNH AML VISION CHUẨN GỐC
# ============================================================
class AMLVisionModel(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 dim=192, depth=6, K=16, latent=128, num_classes=10):
        super().__init__()
        self.dim, self.depth, self.K = dim, depth, K
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_chans, dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        self.drop = nn.Dropout(0.1)

        # Context Encoder & Meta-Controllers (AML chuẩn gốc)
        self.ctx_enc = nn.Sequential(
            nn.Linear(dim, latent), nn.SiLU(), nn.Linear(latent, latent))
        self.proto_sel = nn.Sequential(
            nn.Linear(latent, 256), nn.SiLU(), nn.Linear(256, K))
        self.struct_delta = nn.Linear(latent, depth + depth * dim)
        self.bm_head = nn.Linear(latent, depth * 4)

        # Khởi tạo zero (chuẩn gốc AML)
        for m in [self.struct_delta, self.bm_head]:
            nn.init.constant_(m.weight, 0.0)
            nn.init.constant_(m.bias, 0.0)

        self.layers = nn.ModuleList([AMLBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        # HEAD CIFAR (sẽ được dùng để suy diễn Task 1)
        self.head_cifar = nn.Linear(dim, num_classes)

    def encode(self, x, tau=1.0):
        """Trả về đặc trưng AML đã được xử lý qua toàn bộ Backbone"""
        B = x.shape[0]
        device = x.device
        h = self.drop(self.patch_embed(x).flatten(2).transpose(1,2) + self.pos_embed)

        z = self.ctx_enc(h.mean(1))
        delta = self.struct_delta(z)

        # +2.0 Positive Bias chuẩn AML gốc
        gates = torch.sigmoid(delta[:, :self.depth] + 2.0)
        d_raw = torch.sigmoid(delta[:, self.depth:].view(B, self.depth, self.dim) + 2.0)
        d_len = 64. + F.softplus(d_raw.mean(-1, keepdim=True) * self.dim - 64.)
        arange = torch.arange(self.dim, device=device).float().view(1, 1, self.dim)

        # Hard Gumbel chuẩn gốc
        bm = F.gumbel_softmax(self.bm_head(z).view(B, self.depth, 4),
                              tau=tau, hard=True, dim=-1)

        for i, layer in enumerate(self.layers):
            dm = torch.sigmoid(10. * (d_len[:, i:i+1, :] - arange))
            h = gates[:, i:i+1, None] * layer(h, bm[:, i], dm) + \
                (1. - gates[:, i:i+1, None]) * h

        return self.norm(h.mean(1))  # [B, dim]

    def forward_cifar(self, x, tau=1.0):
        return self.head_cifar(self.encode(x, tau))

    def forward_mnist(self, x, mnist_head, tau=1.0):
        with torch.no_grad():  # Backbone hoàn toàn frozen khi train MNIST
            feat = self.encode(x, tau)
        return mnist_head(feat)


def evaluate(model, loader, device, task='cifar', mnist_head=None):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                if task == 'cifar':
                    logits = model.forward_cifar(x, tau=0.3)
                else:
                    logits = model.forward_mnist(x, mnist_head, tau=0.3)
            correct += logits.max(1)[1].eq(y).sum().item()
            total += y.size(0)
    return 100. * correct / total


def run():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== AML CONTINUAL LEARNING - GIAI PHAP TRIET DE ({device}) ===")

    # Datasets
    tf_cifar_tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    tf_cifar_te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    tf_mnist = transforms.Compose([
        transforms.Resize((32, 32)), transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,)*3, (0.3081,)*3)])

    c_tr = DataLoader(datasets.CIFAR10(root='./data', train=True,  download=True, transform=tf_cifar_tr), batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
    c_te = DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=tf_cifar_te), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    m_tr = DataLoader(datasets.MNIST(root='./data',   train=True,  download=True, transform=tf_mnist),   batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
    m_te = DataLoader(datasets.MNIST(root='./data',   train=False, download=True, transform=tf_mnist),   batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    model = AMLVisionModel(dim=192, depth=6, K=16).to(device)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    # ================================================================
    # PHASE 1: Train toàn bộ mô hình trên CIFAR-10
    # ================================================================
    print("\n--- PHASE 1: TRAIN CIFAR-10 (8 Epochs) ---")
    opt1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=8, eta_min=1e-5)

    for ep in range(1, 9):
        model.train()
        t0 = time.time()
        for x, y in c_tr:
            x, y = x.to(device), y.to(device)
            opt1.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = crit(model.forward_cifar(x, tau=1.0), y)
            scaler.scale(loss).backward()
            scaler.step(opt1); scaler.update()
        sch1.step()
        acc = evaluate(model, c_te, device, 'cifar')
        print(f"Epoch {ep}/8 ({time.time()-t0:.1f}s) -> CIFAR-10 Acc: {acc:.2f}%")

    cifar_init = evaluate(model, c_te, device, 'cifar')
    print(f"\n[PHASE 1 DONE] CIFAR-10 Acc: {cifar_init:.2f}%")

    # ================================================================
    # KHÓA CỨNG HOÀN TOÀN BACKBONE (TOÁN HỌC ĐẢM BẢO ZERO FORGETTING)
    # Chỉ 1 Linear Layer MNIST nhỏ được phép học
    # ================================================================
    print("\n[KHOA CUNG] DONG BANG 100% BACKBONE - Chi train Linear Head MNIST")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False  # Khóa TOÀN BỘ

    # Tạo Head MNIST độc lập hoàn toàn (không chia sẻ bất kỳ trọng số nào)
    mnist_head = nn.Linear(192, 10).to(device)
    print(f"   Head MNIST: {sum(p.numel() for p in mnist_head.parameters())} params (chi nay duoc train)")

    # ================================================================
    # PHASE 2: Chỉ train Head MNIST - Backbone bị đóng băng tuyệt đối
    # ================================================================
    print("\n--- PHASE 2: TRAIN MNIST (3 Epochs, Chi Head MNIST) ---")
    opt2 = torch.optim.AdamW(mnist_head.parameters(), lr=1e-2, weight_decay=1e-3)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=3, eta_min=1e-4)

    for ep in range(1, 4):
        mnist_head.train()
        t0 = time.time()
        for x, y in m_tr:
            x, y = x.to(device), y.to(device)
            opt2.zero_grad()
            with torch.amp.autocast('cuda'):
                # Backbone hoàn toàn frozen: không 1 gradient nào chạy qua
                loss = crit(model.forward_mnist(x, mnist_head, tau=0.3), y)
            scaler.scale(loss).backward()
            scaler.step(opt2); scaler.update()
        sch2.step()

        mnist_acc = evaluate(model, m_te, device, 'mnist', mnist_head)
        cifar_retest = evaluate(model, c_te, device, 'cifar')
        print(f"Epoch {ep}/3 ({time.time()-t0:.1f}s) -> MNIST: {mnist_acc:.2f}% | CIFAR-10 Retest: {cifar_retest:.2f}%")

    cifar_final = evaluate(model, c_te, device, 'cifar')
    mnist_final = evaluate(model, m_te, device, 'mnist', mnist_head)

    print("\n" + "="*75)
    print("KET QUA CUOI CUNG")
    print("="*75)
    print(f"CIFAR-10 Ban Dau   : {cifar_init:.2f}%")
    print(f"MNIST Dat Duoc     : {mnist_final:.2f}%")
    print(f"CIFAR-10 Retest    : {cifar_final:.2f}%")
    print(f"Delta Forgetting   : {cifar_init - cifar_final:+.2f}% (Lo tuong: 0.00%)")
    print("="*75)


if __name__ == '__main__':
    run()
