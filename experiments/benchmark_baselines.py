import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import time


class FixedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        seq = []
        in_d = input_dim
        for _ in range(num_layers):
            seq.append(nn.Linear(in_d, hidden_dim))
            seq.append(nn.ReLU())
            in_d = hidden_dim
        seq.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*seq)

    @property
    def P(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def P_eff(self):
        return self.P

    def forward(self, x):
        return self.net(x)


class FixedMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, output_dim):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_experts)
        ])
        self.router = nn.Linear(input_dim, num_experts)

    @property
    def P(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def P_eff(self):
        return self.P

    def forward(self, x):
        w = F.softmax(self.router(x), dim=-1)
        out = torch.stack([e(x) for e in self.experts], dim=1)
        return (w.unsqueeze(-1) * out).sum(dim=1)


class SmallCNN(nn.Module):
    def __init__(self, in_channels, input_dim, output_dim):
        super().__init__()
        side = int(input_dim ** 0.5)
        if side * side == input_dim:
            actual_channels = 1
        elif input_dim == 3072:
            actual_channels = 3
            side = 32
        else:
            actual_channels = 1
            side = int(input_dim ** 0.5)
        self.in_channels = actual_channels

        self.conv = nn.Sequential(
            nn.Conv2d(actual_channels, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, actual_channels, side, side)
            conv_out = self.conv(dummy).shape[1]

        self.fc = nn.Linear(conv_out, output_dim)

    @property
    def P(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def P_eff(self):
        return self.P

    def forward(self, x):
        if self.in_channels == 1:
            side = int(x.shape[1] ** 0.5)
            x_img = x.view(x.shape[0], 1, side, side)
        else:
            x_img = x.view(x.shape[0], 3, 32, 32)
        h = self.conv(x_img)
        return self.fc(h)


class ResNetBlockBaseline(nn.Module):
    """
    ResNet-8 Baseline with explicit Residual Blocks (Conv -> BN -> ReLU -> Conv -> BN + Skip Connection).
    Serves as the modern ResNet benchmark comparison against ConvAML.
    """
    def __init__(self, in_channels, input_dim, output_dim, base_channels=32):
        super().__init__()
        side = int(input_dim ** 0.5)
        if side * side == input_dim:
            actual_channels = 1
        elif input_dim == 3072:
            actual_channels = 3
            side = 32
        else:
            actual_channels = 1
            side = int(input_dim ** 0.5)
        self.in_channels = actual_channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(actual_channels, base_channels, 3, 1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )

        # ResNet Block 1 (base_channels -> base_channels)
        self.block1_conv1 = nn.Conv2d(base_channels, base_channels, 3, 1, padding=1)
        self.block1_bn1 = nn.BatchNorm2d(base_channels)
        self.block1_conv2 = nn.Conv2d(base_channels, base_channels, 3, 1, padding=1)
        self.block1_bn2 = nn.BatchNorm2d(base_channels)

        # ResNet Block 2 (base_channels -> 2*base_channels, stride 2)
        out_channels = base_channels * 2
        self.block2_conv1 = nn.Conv2d(base_channels, out_channels, 3, stride=2, padding=1)
        self.block2_bn1 = nn.BatchNorm2d(out_channels)
        self.block2_conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.block2_bn2 = nn.BatchNorm2d(out_channels)
        self.block2_skip = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 1, stride=2),
            nn.BatchNorm2d(out_channels)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, output_dim)

    @property
    def P(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def P_eff(self):
        return self.P

    def forward(self, x):
        if self.in_channels == 1:
            side = int(x.shape[1] ** 0.5)
            x_img = x.view(x.shape[0], 1, side, side)
        else:
            x_img = x.view(x.shape[0], 3, 32, 32)

        h = self.stem(x_img)

        # Block 1
        res1 = h
        h = F.relu(self.block1_bn1(self.block1_conv1(h)))
        h = self.block1_bn2(self.block1_conv2(h))
        h = F.relu(h + res1)

        # Block 2
        res2 = self.block2_skip(h)
        h = F.relu(self.block2_bn1(self.block2_conv1(h)))
        h = self.block2_bn2(self.block2_conv2(h))
        h = F.relu(h + res2)

        out = self.pool(h).view(x.shape[0], -1)
        return self.fc(out)



def train_torch(model, train_x, train_y, test_x, test_y, epochs=50, lr=1e-3, batch_size=128, device="cpu", label=""):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    import sys
    print_every = 1
    t0 = time.time()
    for epoch in range(epochs):
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss_fn(logits, yb).backward()
            opt.step()
            losses.append(loss_fn(logits, yb).item())
        if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
            mean_loss = sum(losses) / len(losses) if losses else 0.0
            print(f"  [{label or model.__class__.__name__}] Epoch {epoch+1}/{epochs} | loss={mean_loss:.4f}")
            sys.stdout.flush()

    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        def eval_acc(x_tensor, y_tensor):
            correct, total = 0, 0
            for i in range(0, len(x_tensor), 256):
                xb = x_tensor[i:i+256].to(device)
                yb = y_tensor[i:i+256].to(device)
                correct += (model(xb).argmax(-1) == yb).sum().item()
                total += len(yb)
            return correct / total if total > 0 else 0.0

        train_acc = eval_acc(train_x, train_y)
        test_acc = eval_acc(test_x, test_y)

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "P": model.P,
        "time": elapsed,
    }


def train_rf(train_x, train_y, test_x, test_y, n_estimators=200, label=""):
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    rf.fit(train_x.numpy(), train_y.numpy())
    elapsed = time.time() - t0
    train_acc = accuracy_score(train_y.numpy(), rf.predict(train_x.numpy()))
    test_acc = accuracy_score(test_y.numpy(), rf.predict(test_x.numpy()))
    n_params = n_estimators * sum(t.tree_.node_count for t in rf.estimators_)
    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "P": n_params,
        "time": elapsed,
    }


def search_best_mlp(train_x, train_y, test_x, test_y, input_dim, output_dim,
                    device="cpu", epochs=30, verbose=False):
    configs = []
    for L in [1, 2, 3]:
        for D in [32, 64, 128, 256, 512]:
            if input_dim * D + (L - 1) * D * D + D * output_dim > 5_000_000:
                continue
            configs.append((L, D))

    best = {"test_acc": 0, "L": 1, "D": 32, "P": 0}
    for L, D in configs:
        model = FixedMLP(input_dim, D, L, output_dim)
        res = train_torch(model, train_x, train_y, test_x, test_y,
                          epochs=epochs, device=device, label=f"MLP(L={L},D={D})")
        if verbose:
            print(f"    MLP L={L} D={D}: test={res['test_acc']:.4f} P={res['P']}")
        if res["test_acc"] > best["test_acc"]:
            best = {**res, "L": L, "D": D}
    return best
