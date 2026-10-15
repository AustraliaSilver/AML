"""Train script for Hybrid AML (CNN + ViT dynamic search).
Demonstrates dynamic topology selection across PlainConv, ResNet, and Vision Transformer blocks.
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mgn.blocks import HybridBlock
from mgn.blocks.cnn_block import make_norm
from mgn.generator import DistanceRouter


class HybridAMLNetwork(nn.Module):
    """Hybrid AML Network capable of routing each layer between:
    - Mode 0: PlainConvBlock
    - Mode 1: ResNetBlock
    - Mode 2: VisionTransformerBlock
    """
    def __init__(self, num_classes=10, max_layers=6, hidden_dim=48, K=8, num_heads=4,
                 norm_type='in'):
        super().__init__()
        self.max_layers = max_layers
        self.hidden_dim = hidden_dim
        self.K = K

        # Layers: First conv stem, followed by dynamic HybridBlocks
        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1),
            make_norm(norm_type, hidden_dim),
            nn.ReLU()
        )

        # Ultra-fast routing head: Reuses stem features via adaptive pooling directly to latent space
        # Eliminates redundant 3-layer conv encoder pass, saving 15-20s per epoch on GPU!
        self.router_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.router_fc = nn.Linear(hidden_dim * 4, 128)
        self.router = DistanceRouter(128, K)

        # Dynamic mode prototypes: K prototypes x max_layers x 3 (Plain, ResNet, ViT)
        self.mode_prototypes = nn.Parameter(torch.randn(K, max_layers, 3) * 0.1)

        self.layers = nn.ModuleList([
            HybridBlock(in_channels=hidden_dim, out_channels=hidden_dim,
                        num_heads=num_heads, norm_type=norm_type)
            for _ in range(max_layers)
        ])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, tau=1.0):
        B = x.shape[0]
        # 1. Stem computation (shared with router)
        h = self.stem(x)

        # 2. Ultra-fast routing from stem features
        z = F.relu(self.router_fc(self.router_pool(h).flatten(1)))
        pi_logits = self.router(z)
        pi = F.gumbel_softmax(pi_logits, tau=tau, hard=False, dim=-1) if self.training else F.softmax(pi_logits / tau, dim=-1)

        # Compute block mode routing for each layer: [B, max_layers, 3]
        mode_logits = (pi @ self.mode_prototypes.view(self.K, -1)).view(B, self.max_layers, 3)

        if self.training:
            mode_weights = F.gumbel_softmax(mode_logits, tau=tau, hard=True, dim=-1)
        else:
            mode_weights = F.one_hot(mode_logits.argmax(dim=-1), 3).float()

        # 3. Dynamic Hybrid layers
        for i, layer in enumerate(self.layers):
            w = mode_weights[:, i]  # [B, 3]
            h = layer(h, mode_weights=w)
            # Spatial downsampling at layer 2 and 4
            if i in (1, 3):
                h = F.max_pool2d(h, 2, 2)

        out = self.head(h)
        return out, mode_weights


def train_hybrid_aml(dataset='mnist', epochs=5, batch_size=64, lr=1e-3, steps_per_epoch=0,
                     norm_type='in'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training Hybrid AML (CNN + ViT) on dataset '{dataset}' ({device})...")

    # Data
    # Fast GPU Pipeline for MNIST: Load tensors directly to avoid CPU transforms bottleneck
    if dataset == 'mnist':
        mnist_raw_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        mnist_raw_test = torchvision.datasets.MNIST(root='./data', train=False, download=True)

        # Pre-process into [N, 3, 32, 32] tensors directly once on startup
        x_tr = F.interpolate(mnist_raw_train.data.unsqueeze(1).float() / 255.0, size=(32, 32), mode='bilinear').repeat(1, 3, 1, 1)
        x_tr = (x_tr - 0.1307) / 0.3081
        y_tr = mnist_raw_train.targets

        x_te = F.interpolate(mnist_raw_test.data.unsqueeze(1).float() / 255.0, size=(32, 32), mode='bilinear').repeat(1, 3, 1, 1)
        x_te = (x_te - 0.1307) / 0.3081
        y_te = mnist_raw_test.targets

        trainset = torch.utils.data.TensorDataset(x_tr, y_tr)
        testset = torch.utils.data.TensorDataset(x_te, y_te)
        num_workers = 0  # In-memory TensorDataset is fastest with 0 workers
    else:
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
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tf_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tf_test)
        num_workers = 2

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True if num_workers > 0 else False
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True if num_workers > 0 else False
    )

    model = HybridAMLNetwork(num_classes=10, max_layers=6, hidden_dim=48, K=8,
                            num_heads=4, norm_type=norm_type).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for step, (inputs, targets) in enumerate(trainloader):
            if steps_per_epoch and step >= steps_per_epoch:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs, mode_weights = model(inputs, tau=1.0)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * correct / total
        dt = time.time() - t0

        # Evaluation pass
        model.eval()
        test_correct, test_total = 0, 0
        layer_mode_counts = torch.zeros(6, 3, device=device)

        with torch.no_grad():
            for step, (inputs, targets) in enumerate(testloader):
                if steps_per_epoch and step >= steps_per_epoch:
                    break
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs, mode_weights = model(inputs, tau=1.0)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                layer_mode_counts += mode_weights.sum(dim=0)

        test_acc = 100.0 * test_correct / test_total
        mode_pct = (layer_mode_counts / test_total * 100.0).cpu()

        print(f"Epoch {epoch+1}/{epochs} | {dt:.1f}s | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        print("  -> Architecture Selection per Layer (Plain% / ResNet% / ViT%):")
        for lyr in range(6):
            p, r, v = mode_pct[lyr][0].item(), mode_pct[lyr][1].item(), mode_pct[lyr][2].item()
            print(f"     Layer {lyr+1}: Plain {p:5.1f}% | ResNet {r:5.1f}% | ViT {v:5.1f}%")

    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    out_path = f'checkpoints/aml_hybrid_{dataset}_best.pth'
    torch.save(model.state_dict(), out_path)
    print(f"[*] Saved hybrid model to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--steps', type=int, default=0, help='0 for full epoch')
    # B42: this script used to build BatchNorm blocks while train_continual_aml.py
    # built InstanceNorm ones, so two scripts documented as training the same
    # architecture trained different models. The default now matches the CLI
    # default everywhere else; 'bn' is kept reachable for the comparison.
    parser.add_argument('--norm', type=str, default='in', choices=['in', 'bn', 'gn'],
                        help='post-convolution normalisation (default: in)')
    args = parser.parse_args()
    train_hybrid_aml(dataset=args.dataset, epochs=args.epochs, batch_size=args.batch_size,
                     lr=args.lr, steps_per_epoch=args.steps, norm_type=args.norm)
