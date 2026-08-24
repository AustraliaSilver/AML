"""
diagnose_trunkv2.py — Test normalization variants on the SAME trunk,
static-oracle form (classic CNN), eager executor, isolated variable = norm.

Arms:
  gn1 : F.group_norm(h, 1)            (current — no affine, volume stats)
  gn8 : GroupNorm(8, C, affine=True)  (per-group channel stats + affine)
  bn  : BatchNorm2d(C, affine=True)   (standard CNN norm)

Relative accuracy between arms identifies the norm bottleneck.
"""

import os, sys, time, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "AML_Core_Code"))

from train_cv_cifar_18 import get_prototype_layout

L, D, K, NUM_CLASSES = 18, 64, 32, 10


class NormWrap(nn.Module):
    def __init__(self, kind, C):
        super().__init__()
        self.kind = kind
        if kind == "gn8":
            self.n = nn.GroupNorm(8, C, affine=True)
        elif kind == "bn":
            self.n = nn.BatchNorm2d(C, affine=True)
        elif kind == "in":
            self.n = nn.InstanceNorm2d(C, affine=False)
        elif kind == "in_af":
            self.n = nn.InstanceNorm2d(C, affine=True)

    def forward(self, h):
        if self.kind == "gn1":
            return F.group_norm(h, 1)
        return self.n(h)


class TrunkNet(nn.Module):
    def __init__(self, norm_kind):
        super().__init__()
        total_dim, _, _, _, _ = get_prototype_layout(L, D, NUM_CLASSES)
        self.protos = nn.Parameter(torch.zeros(K, total_dim))
        import math
        nn.init.kaiming_uniform_(self.protos, a=math.sqrt(5))
        self.norms = nn.ModuleList([NormWrap(norm_kind, D) for _ in range(L)])
        self.fc = nn.Linear(16 * D, NUM_CLASSES)


def build(norm_kind, device):
    net = TrunkNet(norm_kind).to(device)
    total_dim, offsets, li, wo, bo = get_prototype_layout(L, D, NUM_CLASSES)
    return net, offsets, wo, bo


def forward_net(net, offsets, wo, bo, x):
    """Static-oracle form: pi=one-hot row0, gates ON, full dims, ResNet mode."""
    B = x.shape[0]
    protos = net.protos
    pi = torch.zeros(B, K, device=x.device)
    pi[:, 0] = 1.0
    h = x
    Cin = 3
    H, W = 32, 32
    for i in range(L):
        w_start, w_end, b_start, b_end, in_c = offsets[i]
        Wm = torch.mm(pi, protos[:, w_start:w_end]).view(B, D, Cin, 3, 3)
        bx = torch.mm(pi, protos[:, b_start:b_end])
        hc = F.conv2d(h.reshape(1, B * Cin, H, W), Wm.reshape(B * D, Cin, 3, 3),
                      bx.reshape(-1), padding=1, groups=B).view(B, D, H, W)
        hn = net.norms[i](hc)
        hp = F.relu(hn)
        he = hp if i == 0 else h
        hr = F.relu(he + hn)
        h = hr if i > 0 else hp
        if i == 5 or i == 11:
            h = F.max_pool2d(h, 2, 2)
            H //= 2; W //= 2
        Cin = D
    feat = F.adaptive_avg_pool2d(h, (4, 4)).reshape(B, -1)
    feat = F.layer_norm(feat, [feat.shape[-1]])
    feat = F.dropout(feat, p=0.2, training=net.training)
    return net.fc(feat)


def run_arm(kind, epochs, device, tl, el):
    set_seed = torch.manual_seed(42)
    net, offsets, wo, bo = build(kind, device)
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 0.0
    for ep in range(epochs):
        net.train()
        t0, tot = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = forward_net(net, offsets, wo, bo, x)
                loss = F.cross_entropy(out.float(), y, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        sch.step()
        # eval
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in el:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = forward_net(net, offsets, wo, bo, x)
                correct += (out.argmax(-1) == y).sum().item()
                total += y.numel()
        acc = correct / total
        best = max(best, acc)
        print(f"  [{kind}] ep {ep+1}/{epochs} | {time.time()-t0:5.1f}s | "
              f"loss {tot/len(tl):.4f} | acc {acc*100:.2f}% | best {best*100:.2f}%", flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    tf_test = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    root = os.path.join(_ROOT, "data")
    tr = torchvision.datasets.CIFAR10(root, train=True, download=False, transform=tf_train)
    te = torchvision.datasets.CIFAR10(root, train=False, download=False, transform=tf_test)
    tl = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True, num_workers=3,
                                     pin_memory=True, persistent_workers=True, drop_last=True)
    el = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False, num_workers=3,
                                     pin_memory=True, persistent_workers=True)

    results = {}
    for kind in ["in", "in_af"]:
        print(f"\n===== NORM ARM: {kind} =====", flush=True)
        results[kind] = run_arm(kind, args.epochs, device, tl, el)

    print("\n===== TRUNK-NORM VERDICT =====")
    for k, v in results.items():
        print(f"  {k:5s}: best {v*100:.2f}%")
