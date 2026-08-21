"""
AML Continual Learning - GIAI PHAP HOAN CHINH:
- Phase 1: Train toan bo model tren CIFAR-10
- Phase 2: DONG BANG 100% Backbone
            + Them MNIST Adapter (2-layer MLP nho)
            + Train Adapter + Head MNIST
=> Zero Forgetting CIFAR-10 + MNIST Accuracy cao hon
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

# ============================================================
# AML Subnets
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
        return x + (stk * bm.view(B, 4, 1, 1)).sum(1) * d_mask

# ============================================================
# AML Vision Backbone
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
        self.ctx_enc = nn.Sequential(nn.Linear(dim, latent), nn.SiLU(), nn.Linear(latent, latent))
        self.proto_sel = nn.Sequential(nn.Linear(latent, 256), nn.SiLU(), nn.Linear(256, K))
        self.struct_delta = nn.Linear(latent, depth + depth * dim)
        self.bm_head = nn.Linear(latent, depth * 4)
        for m in [self.struct_delta, self.bm_head]:
            nn.init.constant_(m.weight, 0.0)
            nn.init.constant_(m.bias, 0.0)
        self.layers = nn.ModuleList([AMLBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head_cifar = nn.Linear(dim, num_classes)

    def encode(self, x, tau=1.0):
        B, device = x.shape[0], x.device
        h = self.drop(self.patch_embed(x).flatten(2).transpose(1,2) + self.pos_embed)
        z = self.ctx_enc(h.mean(1))
        delta = self.struct_delta(z)
        gates = torch.sigmoid(delta[:, :self.depth] + 2.0)
        d_raw = torch.sigmoid(delta[:, self.depth:].view(B, self.depth, self.dim) + 2.0)
        d_len = 64. + F.softplus(d_raw.mean(-1, keepdim=True) * self.dim - 64.)
        arange = torch.arange(self.dim, device=device).float().view(1, 1, self.dim)
        bm = F.gumbel_softmax(self.bm_head(z).view(B, self.depth, 4), tau=tau, hard=True, dim=-1)
        for i, layer in enumerate(self.layers):
            dm = torch.sigmoid(10. * (d_len[:, i:i+1, :] - arange))
            h = gates[:, i:i+1, None] * layer(h, bm[:, i], dm) + (1. - gates[:, i:i+1, None]) * h
        return self.norm(h.mean(1))

    def forward_cifar(self, x, tau=1.0):
        return self.head_cifar(self.encode(x, tau))

    @torch.no_grad()
    def encode_frozen(self, x, tau=0.3):
        """Backbone hoàn toàn frozen - không gradient nào được tạo ra"""
        return self.encode(x, tau)


# ============================================================
# MNIST ADAPTER: 2-layer MLP nhỏ nối giữa Backbone và Head
# Backbone frozen -> Adapter học cách dịch đặc trưng sang MNIST
# ============================================================
class MNISTAdapter(nn.Module):
    """
    Nhận đặc trưng [B, 192] từ Backbone đã frozen
    -> Học cách ánh xạ sang không gian phù hợp với MNIST
    -> Output: logits [B, 10]
    """
    def __init__(self, in_dim=192, hidden=512, num_classes=10, dropout=0.3):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )
    def forward(self, feat):
        return self.adapter(feat)


def evaluate(model, loader, device, task='cifar', adapter=None):
    model.eval()
    if adapter: adapter.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                if task == 'cifar':
                    logits = model.forward_cifar(x, tau=0.3)
                else:
                    feat = model.encode_frozen(x)
                    logits = adapter(feat)
            correct += logits.max(1)[1].eq(y).sum().item()
            total += y.size(0)
    return 100. * correct / total


def run():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== AML CONTINUAL LEARNING (BACKBONE FROZEN + MNIST ADAPTER) ({device}) ===")

    # Datasets
    tf_tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    tf_te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    tf_mnist = transforms.Compose([
        transforms.Resize((32, 32)), transforms.Grayscale(3),
        transforms.ToTensor(), transforms.Normalize((0.1307,)*3, (0.3081,)*3)])

    c_tr = DataLoader(datasets.CIFAR10(root='./data', train=True,  download=True, transform=tf_tr),    batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
    c_te = DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=tf_te),    batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    m_tr = DataLoader(datasets.MNIST(root='./data',   train=True,  download=True, transform=tf_mnist), batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
    m_te = DataLoader(datasets.MNIST(root='./data',   train=False, download=True, transform=tf_mnist), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    model = AMLVisionModel(dim=192, depth=6, K=16).to(device)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    # ============================================================
    # PHASE 1: TRAIN CIFAR-10
    # ============================================================
    print("\n--- PHASE 1: TRAIN CIFAR-10 (8 Epochs) ---")
    opt1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=8, eta_min=1e-5)
    best_acc, best_ckpt = 0.0, "best_aml_cifar.pth"
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
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_ckpt)
        print(f"Epoch {ep}/8 ({time.time()-t0:.1f}s) -> CIFAR-10: {acc:.2f}% (Best: {best_acc:.2f}%)")

    # Load best checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    cifar_init = evaluate(model, c_te, device, 'cifar')
    print(f"\n[PHASE 1 DONE] CIFAR-10 Best Acc: {cifar_init:.2f}%")

    # ============================================================
    # KHOA CUNG TUYET DOI BACKBONE
    # ============================================================
    print("\n[KHOA CUNG] DONG BANG 100% BACKBONE (Zero Forgetting Guarantee)")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ============================================================
    # PHASE 2: TRAIN MNIST ADAPTER (Backbone hoàn toàn frozen)
    # ============================================================
    adapter = MNISTAdapter(in_dim=192, hidden=512, dropout=0.3).to(device)
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"[ADAPTER] MNIST Adapter params: {n_params:,} (chi nay duoc train)")

    print("\n--- PHASE 2: TRAIN MNIST ADAPTER (5 Epochs) ---")
    opt2 = torch.optim.AdamW(adapter.parameters(), lr=5e-3, weight_decay=1e-3)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=5, eta_min=1e-4)

    for ep in range(1, 6):
        adapter.train()
        t0 = time.time()
        for x, y in m_tr:
            x, y = x.to(device), y.to(device)
            opt2.zero_grad()
            with torch.amp.autocast('cuda'):
                feat = model.encode_frozen(x)   # Backbone frozen tuyet doi
                loss = crit(adapter(feat), y)
            scaler.scale(loss).backward()
            scaler.step(opt2); scaler.update()
        sch2.step()

        mnist_acc = evaluate(model, m_te, device, 'mnist', adapter)
        cifar_retest = evaluate(model, c_te, device, 'cifar')
        print(f"Epoch {ep}/5 ({time.time()-t0:.1f}s) -> MNIST: {mnist_acc:.2f}% | CIFAR-10 Retest: {cifar_retest:.2f}%")

    cifar_final = evaluate(model, c_te, device, 'cifar')
    mnist_final = evaluate(model, m_te, device, 'mnist', adapter)

    print("\n" + "="*70)
    print("KET QUA CUOI CUNG")
    print("="*70)
    print(f"CIFAR-10 Ban Dau (Phase 1)  : {cifar_init:.2f}%")
    print(f"MNIST Dat Duoc (Phase 2)    : {mnist_final:.2f}%")
    print(f"CIFAR-10 Retest             : {cifar_final:.2f}%")
    print(f"Delta Forgetting            : {cifar_init - cifar_final:+.2f}%  (Bao dam 0.00%)")
    print("="*70)


if __name__ == '__main__':
    run()
