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
from mgn.generator import DistanceRouter


class HybridAMLNetwork(nn.Module):
    """Hybrid AML Network capable of routing each layer between:
    - Mode 0: PlainConvBlock
    - Mode 1: ResNetBlock
    - Mode 2: VisionTransformerBlock
    """
    def __init__(self, num_classes=10, max_layers=6, hidden_dim=48, K=8, num_heads=4):
        super().__init__()
        self.max_layers = max_layers
        self.hidden_dim = hidden_dim
        self.K = K

        # Vision encoder to extract image embedding for prototype selection
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(64 * 4, 128), nn.ReLU()
        )
        self.router = DistanceRouter(128, K)

        # Dynamic mode prototypes: K prototypes x max_layers x 3 (Plain, ResNet, ViT)
        self.mode_prototypes = nn.Parameter(torch.randn(K, max_layers, 3) * 0.1)

        # Layers: First conv stem, followed by dynamic HybridBlocks
        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.ReLU()
        )

        self.layers = nn.ModuleList([
            HybridBlock(in_channels=hidden_dim, out_channels=hidden_dim, num_heads=num_heads)
            for _ in range(max_layers)
        ])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, tau=1.0):
        B = x.shape[0]
        z = self.vision_encoder(x)
        pi_logits = self.router(z)
        pi = F.gumbel_softmax(pi_logits, tau=tau, hard=False, dim=-1) if self.training else F.softmax(pi_logits / tau, dim=-1)

        # Compute block mode routing for each layer: [B, max_layers, 3]
        # mode_prototypes: [K, max_layers, 3] -> (pi @ mode_prototypes.view(K, -1)).view(B, max_layers, 3)
        mode_logits = (pi @ self.mode_prototypes.view(self.K, -1)).view(B, self.max_layers, 3)

        if self.training:
            mode_weights = F.gumbel_softmax(mode_logits, tau=tau, hard=False, dim=-1)
        else:
            mode_weights = F.one_hot(mode_logits.argmax(dim=-1), 3).float()

        h = self.stem(x)
        for i, layer in enumerate(self.layers):
            w = mode_weights[:, i]  # [B, 3]
            h = layer(h, mode_weights=w)
            # Spatial downsampling at layer 2 and 4
            if i in (1, 3):
                h = F.max_pool2d(h, 2, 2)

        out = self.head(h)
        return out, mode_weights


def train_hybrid_aml(epochs=3, batch_size=64, lr=1e-3, steps_per_epoch=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training Hybrid AML (CNN + ViT) on {device}...")

    # Data
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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = HybridAMLNetwork(num_classes=10, max_layers=6, hidden_dim=48, K=8, num_heads=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for step, (inputs, targets) in enumerate(trainloader):
            if steps_per_epoch and step >= steps_per_epoch:
                break
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, mode_weights = model(inputs, tau=1.0)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

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
                inputs, targets = inputs.to(device), targets.to(device)
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
    out_path = 'checkpoints/aml_hybrid_cifar10_demo.pth'
    torch.save(model.state_dict(), out_path)
    print(f"[*] Saved hybrid model to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--steps', type=int, default=25)
    args = parser.parse_args()
    train_hybrid_aml(epochs=args.epochs, steps_per_epoch=args.steps)
