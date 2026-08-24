"""
diagnose_bottleneck.py — Isolate WHY the AML supernet caps at ~78% on CIFAR.

Three arms, identical executor/training budget:
  B "static-oracle": fixed one-hot pi + fully-open gates + ResNet-only modes
                     -> upper bound of the trunk as a CLASSIC CNN
  A "fixed-w":       fixed one-hot pi, structure still dynamic
                     -> cost of per-sample WEIGHT MIXING alone
  C "dynamic":       everything dynamic (control)

If B >> C: the dynamic machinery itself is the bottleneck.
If B ~ C ~ 78: the trunk design (GN-over-volume, single linear head,
               uniform width) is the bottleneck, independent of dynamics.
"""

import os, sys, time, argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "AML_Core_Code"))

from train import ConvAML_CIFAR_TrueDynamic, set_seed


def make_loaders(workers=3):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    root = os.path.join(_ROOT, "data")
    tr = torchvision.datasets.CIFAR10(root, train=True, download=False, transform=tf_train)
    te = torchvision.datasets.CIFAR10(root, train=False, download=False, transform=tf_test)
    tl = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True, num_workers=workers,
                                     pin_memory=True, persistent_workers=True, drop_last=True)
    el = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False, num_workers=workers,
                                     pin_memory=True, persistent_workers=True)
    return tl, el


def force_static_structure(model):
    """Pin gates ON, dims FULL, modes RESNET — permanently."""
    with torch.no_grad():
        sd = model.structure_delta
        sd.bias[:model.max_layers] += 8.0                       # layer logits ~10 -> soft ~1.0
        sd.bias[model.max_layers:] += 8.0                       # d_logit high -> full masks
        bm = model.block_mode_head
        nn_bm = torch.zeros_like(bm.bias.data.view(model.max_layers, 2))
        nn_bm[:, 1] = 12.0                                      # hard ResNet preference
        bm.bias.data.copy_(nn_bm.view(-1))
    for p in list(model.structure_delta.parameters()) + list(model.block_mode_head.parameters()):
        p.requires_grad_(False)


@torch.no_grad()
def evaluate(model, el, device):
    model.eval()
    correct, total = 0, 0
    for x, y in el:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out, _, _, _ = model(x, tau=0.1)
        correct += (out.argmax(-1) == y).sum().item()
        total += y.numel()
    model.train()
    return correct / total


def run_arm(arm, epochs, device, tl, el):
    set_seed(42)
    model = ConvAML_CIFAR_TrueDynamic(K=32, max_dim=64, max_layers=18).to(device)
    torch.backends.cudnn.benchmark = True

    fixed_pi = None
    if arm in ("static-oracle", "fixed-w"):
        # Fixed one-hot prototype routing: same generated weights for EVERY sample
        k0 = 0
        fixed_pi = torch.zeros(model.K, device=device)
        fixed_pi[k0] = 1.0
        model.caf_attention = lambda feats: fixed_pi.expand(feats.shape[0], -1)
        # Freeze router so nothing pulls it around
        for p in model.prototype_selector.parameters():
            p.requires_grad_(False)
    if arm == "static-oracle":
        force_static_structure(model)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = F.cross_entropy

    best = 0.0
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(x)[0]
                loss = ce(out.float(), y, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += loss.item()
        sch.step()
        acc = evaluate(model, el, device)
        best = max(best, acc)
        print(f"  [{arm}] ep {ep+1}/{epochs} | {time.time()-t0:5.1f}s | "
              f"loss {tot/len(tl):.4f} | acc {acc*100:.2f}% | best {best*100:.2f}%", flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--arms', type=str, default='static-oracle,fixed-w,dynamic')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tl, el = make_loaders()
    results = {}
    for arm in args.arms.split(','):
        print(f"\n===== ARM: {arm} =====", flush=True)
        results[arm] = run_arm(arm.strip(), args.epochs, device, tl, el)

    print("\n===== BOTTLENECK VERDICT =====")
    b = results.get('static-oracle'); a = results.get('fixed-w'); c = results.get('dynamic')
    for k, v in results.items():
        print(f"  {k:14s}: best {v*100:.2f}%")
    if b is not None and c is not None:
        gap_dyn = (b - c) * 100
        print(f"\n  Dynamic-vs-static gap : {gap_dyn:.2f} pts "
              f"({'DYNAMIC MACHINERY is the bottleneck' if gap_dyn > 5 else 'trunk design is the bottleneck' if abs(gap_dyn) <= 5 and b < 85 else 'mixed'})")
        if b < 85:
            print(f"  Static oracle itself < 85% => TRUNK DESIGN (norm/head/width) caps accuracy.")
