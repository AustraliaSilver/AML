"""
benchmarks/benchmark_cl_comparison.py
=====================================
Continual Learning Benchmark: AML 18-Layer Hybrid Supernet vs Standard Neural Architectures
Compares under identical sequential training protocol without replay (Zero-Replay):
  1. Plain CNN (18-Layer Deep VGG-style CNN)
  2. Standard ResNet (ResNet-18)
  3. Vision Transformer (ViT: ConvStem + Self-Attention Layers + MLP)
  4. AML 18-Layer Hybrid Supernet (Ours: Dynamic Plain + ResNet + ViT + Snapshot Isolation)

Sequence: MNIST -> CIFAR-10 -> Fashion-MNIST
Outputs:
  - Accuracy Matrices A[i, j]
  - Average Final Accuracy (A_avg)
  - Backward Transfer (BWT) & Catastrophic Forgetting
  - Parameter Count (Total & Active), Model Size (MB), Latency
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mgn.blocks.hybrid_18 import ConvAML_18Layer_Hybrid
from train_continual_aml import get_task_data, ProtoIsolator

# =========================================================================
# 1. BASELINE ARCHITECTURES (Equalized Capacity ~ 1.5M - 12M params)
# =========================================================================

class BaselinePlainCNN(nn.Module):
    """18-Layer Plain CNN (VGG-style, no residual connections, no attention)."""
    def __init__(self, in_channels=3, num_classes=10, hidden_dim=48, num_layers=18):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if i in (5, 11):
                layers.append(nn.MaxPool2d(2, 2))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head_dim = hidden_dim * 16
        self.ln = nn.LayerNorm(self.head_dim)
        self.fc = nn.Linear(self.head_dim, num_classes)

    def forward(self, x):
        h = self.stem(x)
        h = self.features(h)
        h = self.ln(self.pool(h).flatten(1))
        return self.fc(h)


class BaselineResNet18(nn.Module):
    """Standard ResNet with residual skip connections across 18 layers."""
    def __init__(self, in_channels=3, num_classes=10, hidden_dim=48):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        # 8 residual blocks (each 2 convs = 16 convs) + stem + fc = 18 weighted layers
        self.stage1 = self._make_stage(hidden_dim, hidden_dim, blocks=3, stride=1) # 32x32
        self.stage2 = self._make_stage(hidden_dim, hidden_dim, blocks=3, stride=2) # 16x16
        self.stage3 = self._make_stage(hidden_dim, hidden_dim, blocks=2, stride=2) # 8x8
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head_dim = hidden_dim * 16
        self.ln = nn.LayerNorm(self.head_dim)
        self.fc = nn.Linear(self.head_dim, num_classes)

    def _make_stage(self, in_d, out_d, blocks, stride):
        layers = []
        # First block may downsample
        layers.append(self._ResBlock(in_d, out_d, stride))
        for _ in range(1, blocks):
            layers.append(self._ResBlock(out_d, out_d, 1))
        return nn.Sequential(*layers)

    class _ResBlock(nn.Module):
        def __init__(self, in_d, out_d, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_d, out_d, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_d)
            self.conv2 = nn.Conv2d(out_d, out_d, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_d)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_d != out_d:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_d, out_d, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_d)
                )

        def forward(self, x):
            res = F.relu(self.bn1(self.conv1(x)), inplace=True)
            res = self.bn2(self.conv2(res))
            return F.relu(res + self.shortcut(x), inplace=True)

    def forward(self, x):
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.ln(self.pool(h).flatten(1))
        return self.fc(h)


class BaselineViT(nn.Module):
    """Vision Transformer (ConvStem + TransformerEncoder self-attention layers)."""
    def __init__(self, in_channels=3, num_classes=10, hidden_dim=48, num_layers=8, num_heads=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=4, bias=False), # 32x32 -> 8x8 (64 tokens)
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.num_tokens = 8 * 8
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.stem(x) # [B, C, 8, 8]
        B, C, H, W = h.shape
        tokens = h.flatten(2).transpose(1, 2) # [B, 64, C]
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1) # [B, C]
        return self.fc(pooled)


# =========================================================================
# 2. CONTINUAL TRAINING & EVALUATION HARNESS
# =========================================================================

def train_baseline_continual(model, tasks, data_root, epochs_per_task=1, batch_cap=40, eval_cap=15, device="cuda"):
    model.to(device)
    T = len(tasks)
    A = [[0.0] * T for _ in range(T)]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for task_idx, t_name in enumerate(tasks):
        tr_loader, _ = get_task_data(t_name, data_root, workers=0)
        
        # Train on current task
        model.train()
        for ep in range(epochs_per_task):
            for step, (x, y) in enumerate(tr_loader):
                if step >= batch_cap:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out = model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                opt.step()

        # Evaluate on all seen tasks (0 .. task_idx)
        model.eval()
        for prev_idx in range(task_idx + 1):
            _, te_loader = get_task_data(tasks[prev_idx], data_root, workers=0)
            correct, total = 0, 0
            with torch.no_grad():
                for step, (x, y) in enumerate(te_loader):
                    if step >= eval_cap:
                        break
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    pred = model(x).argmax(-1)
                    correct += (pred == y).sum().item()
                    total += y.numel()
            A[task_idx][prev_idx] = (correct / total) * 100.0 if total > 0 else 0.0

    total_time = time.time() - t0
    return A, total_time


def train_aml_continual(tasks, data_root, epochs_per_task=1, batch_cap=40, eval_cap=15, device="cuda"):
    T = len(tasks)
    model = ConvAML_18Layer_Hybrid(K=12, max_dim=48, max_layers=18, num_classes=10).to(device)
    isolator = ProtoIsolator(model)
    dom_snaps = [None] * T
    A_live = [[0.0] * T for _ in range(T)]
    A_snap = [[0.0] * T for _ in range(T)]

    t0 = time.time()
    for task_idx, t_name in enumerate(tasks):
        s, e = isolator.begin_task(task_idx, T, reinit=False)
        tr_loader, te_loader = get_task_data(t_name, data_root, workers=0)
        
        opt = torch.optim.AdamW([
            {"params": model.stem.parameters(), "lr": 1e-3},
            {"params": model.layers.parameters(), "lr": 1e-3},
            # The encoder and the selector decide WHICH prototype rows are used,
            # so leaving them out of the optimiser kept the whole routing path
            # frozen at initialisation while the trunk kept moving.
            {"params": model.vision_encoder.parameters(), "lr": 1e-3},
            {"params": model.prototype_selector.parameters(), "lr": 1e-3},
            {"params": [model.proto_weights, model.structure_prototypes, model.block_mode_prototypes], "lr": 2e-3},
        ], weight_decay=1e-4)

        model.train()
        pi_mask = torch.zeros(model.K, dtype=torch.bool, device=device)
        pi_mask[s:e] = True

        for ep in range(epochs_per_task):
            for step, (x, y) in enumerate(tr_loader):
                if step >= batch_cap:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out, struct, pi, diag = model(x, tau=0.8, pi_mask=pi_mask)
                loss = F.cross_entropy(out, y)
                loss.backward()
                # begin_task() above only OPENS the isolation window. Without
                # mask_grad()/restore(), the previous tasks' prototype rows were
                # overwritten anyway, so this benchmark was measuring plain
                # finetuning with extra bookkeeping.
                isolator.mask_grad()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                isolator.restore()

        # Save snapshot for this task
        dom_snaps[task_idx] = {
            "stem": {k: v.detach().clone() for k, v in model.stem.state_dict().items()},
            "layers": {k: v.detach().clone() for k, v in model.layers.state_dict().items()},
            "ln": {k: v.detach().clone() for k, v in model.ln.state_dict().items()},
            "protos": model.proto_weights.detach().clone(),
            "struct_heads": {
                "sprotos": model.structure_prototypes.detach().clone(),
                "bmprotos": model.block_mode_prototypes.detach().clone(),
            },
            "enc": {k: v.detach().clone() for k, v in model.vision_encoder.state_dict().items()},
            "sel": {k: v.detach().clone() for k, v in model.prototype_selector.state_dict().items()},
        }

        # Evaluate on all seen tasks
        for prev_idx in range(task_idx + 1):
            _, te_prev = get_task_data(tasks[prev_idx], data_root, workers=0)
            # Prototype-row mask for the task this snapshot/eval targets: each
            # sample may only use the rows that its task owns.
            pi_mask_prev = torch.zeros(model.K, device=device, dtype=torch.bool)
            _s = round(prev_idx * model.K / T)
            _e = round((prev_idx + 1) * model.K / T)
            pi_mask_prev[_s:_e] = True
            
            # 1. Live mutable evaluation (no snapshot)
            correct_l, total_l = 0, 0
            model.eval()
            with torch.no_grad():
                for step, (x, y) in enumerate(te_prev):
                    if step >= eval_cap:
                        break
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    out, _, _, _ = model(x, tau=0.1, pi_mask=pi_mask_prev)
                    correct_l += (out.argmax(-1) == y).sum().item()
                    total_l += y.numel()
            A_live[task_idx][prev_idx] = (correct_l / total_l) * 100.0

            # 2. Snapshot Retrieval (Task-IL Oracle)
            # Temporarily restore FULL snapshot for prev_idx: every functional
            # block that was snapshotted (enc, sel, protos, stem, layers, ln,
            # structure heads) must be swapped, otherwise the snapshot degenerates
            # into a partial swap and the numbers are meaningless.
            cur_enc = {k: v.detach().clone() for k, v in model.vision_encoder.state_dict().items()}
            cur_sel = {k: v.detach().clone() for k, v in model.prototype_selector.state_dict().items()}
            cur_stem = {k: v.detach().clone() for k, v in model.stem.state_dict().items()}
            cur_layers = {k: v.detach().clone() for k, v in model.layers.state_dict().items()}
            cur_ln = ( {k: v.detach().clone() for k, v in model.ln.state_dict().items()}
                      if hasattr(model, "ln") else None)
            cur_protos = model.proto_weights.detach().clone()
            cur_sprotos = model.structure_prototypes.detach().clone()
            cur_bmprotos = model.block_mode_prototypes.detach().clone()

            model.vision_encoder.load_state_dict(dom_snaps[prev_idx]["enc"])
            model.prototype_selector.load_state_dict(dom_snaps[prev_idx]["sel"])
            model.stem.load_state_dict(dom_snaps[prev_idx]["stem"])
            model.layers.load_state_dict(dom_snaps[prev_idx]["layers"])
            if cur_ln is not None and "ln" in dom_snaps[prev_idx]:
                model.ln.load_state_dict(dom_snaps[prev_idx]["ln"])
            model.proto_weights.data.copy_(dom_snaps[prev_idx]["protos"])
            model.structure_prototypes.data.copy_(dom_snaps[prev_idx]["struct_heads"]["sprotos"])
            model.block_mode_prototypes.data.copy_(dom_snaps[prev_idx]["struct_heads"]["bmprotos"])

            correct_s, total_s = 0, 0
            with torch.no_grad():
                for step, (x, y) in enumerate(te_prev):
                    if step >= eval_cap:
                        break
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    out, _, _, _ = model(x, tau=0.1, pi_mask=pi_mask_prev)
                    correct_s += (out.argmax(-1) == y).sum().item()
                    total_s += y.numel()
            A_snap[task_idx][prev_idx] = (correct_s / total_s) * 100.0

            # Restore current weights
            model.vision_encoder.load_state_dict(cur_enc)
            model.prototype_selector.load_state_dict(cur_sel)
            model.stem.load_state_dict(cur_stem)
            model.layers.load_state_dict(cur_layers)
            if cur_ln is not None:
                model.ln.load_state_dict(cur_ln)
            model.proto_weights.data.copy_(cur_protos)
            model.structure_prototypes.data.copy_(cur_sprotos)
            model.block_mode_prototypes.data.copy_(cur_bmprotos)

    total_time = time.time() - t0
    p_total = sum(p.numel() for p in model.parameters())
    return A_live, A_snap, total_time, p_total


def compute_metrics(A):
    T = len(A)
    # Final average accuracy across all seen tasks
    acc_final = sum(A[T-1][j] for j in range(T)) / T
    # Backward transfer
    bwt = sum(A[T-1][j] - A[j][j] for j in range(T - 1)) / (T - 1)
    # Forgetting
    forg_list = [max(A[t][j] for t in range(j, T - 1)) - A[T-1][j] for j in range(T - 1)]
    forg = sum(forg_list) / len(forg_list)
    return acc_final, bwt, forg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 90)
    print(f" CONTINUAL LEARNING BENCHMARK: AML vs STANDARD ARCHITECTURES ({device})")
    print(" Protocol: 3-Task Sequence (MNIST -> CIFAR-10 -> Fashion-MNIST) | Zero-Replay")
    print("=" * 90)

    data_root = os.path.join(ROOT, "data")
    tasks = ["mnist", "cifar10", "fashion"]
    epochs = 1
    batch_cap = 45
    eval_cap = 20

    results = {}

    # 1. Plain CNN
    print("\n[1/4] Running Plain CNN (18 Layers)...")
    cnn = BaselinePlainCNN(in_channels=3, num_classes=10, hidden_dim=48, num_layers=18)
    p_cnn = sum(p.numel() for p in cnn.parameters())
    a_cnn, t_cnn = train_baseline_continual(cnn, tasks, data_root, epochs, batch_cap, eval_cap, device)
    acc_c, bwt_c, forg_c = compute_metrics(a_cnn)
    results["Plain CNN (18-Layer)"] = {"A": a_cnn, "ACC": acc_c, "BWT": bwt_c, "Forg": forg_c, "Time": t_cnn, "Params": p_cnn}

    # 2. ResNet-18
    print("\n[2/4] Running Standard ResNet (ResNet-18)...")
    resnet = BaselineResNet18(in_channels=3, num_classes=10, hidden_dim=48)
    p_res = sum(p.numel() for p in resnet.parameters())
    a_res, t_res = train_baseline_continual(resnet, tasks, data_root, epochs, batch_cap, eval_cap, device)
    acc_r, bwt_r, forg_r = compute_metrics(a_res)
    results["ResNet-18"] = {"A": a_res, "ACC": acc_r, "BWT": bwt_r, "Forg": forg_r, "Time": t_res, "Params": p_res}

    # 3. Vision Transformer (ViT)
    print("\n[3/4] Running Vision Transformer (ViT)...")
    vit = BaselineViT(in_channels=3, num_classes=10, hidden_dim=48, num_layers=8, num_heads=4)
    p_vit = sum(p.numel() for p in vit.parameters())
    a_vit, t_vit = train_baseline_continual(vit, tasks, data_root, epochs, batch_cap, eval_cap, device)
    acc_v, bwt_v, forg_v = compute_metrics(a_vit)
    results["Vision Transformer (ViT)"] = {"A": a_vit, "ACC": acc_v, "BWT": bwt_v, "Forg": forg_v, "Time": t_vit, "Params": p_vit}

    # 4. AML 18-Layer Hybrid Supernet (Ours)
    print("\n[4/4] Running AML 18-Layer Hybrid Supernet (Ours)...")
    a_aml_l, a_aml_s, t_aml, p_aml = train_aml_continual(tasks, data_root, epochs, batch_cap, eval_cap, device)
    acc_al, bwt_al, forg_al = compute_metrics(a_aml_l)
    acc_as, bwt_as, forg_as = compute_metrics(a_aml_s)
    results["AML Live Weights (No Replay)"] = {"A": a_aml_l, "ACC": acc_al, "BWT": bwt_al, "Forg": forg_al, "Time": t_aml, "Params": p_aml}
    results["AML Snapshot Retrieval (Ours)"] = {"A": a_aml_s, "ACC": acc_as, "BWT": bwt_as, "Forg": forg_as, "Time": t_aml, "Params": p_aml}

    # =========================================================================
    # DISPLAY FINAL COMPARISON TABLES
    # =========================================================================
    print("\n" + "=" * 105)
    print("                    ACCURACY MATRICES A[i, j] ACROSS ARCHITECTURES")
    print("=" * 105)
    for model_name, res in results.items():
        print(f"\n--- {model_name} ---")
        for i in range(len(tasks)):
            row = " | ".join(f"{res['A'][i][j]:6.2f}%" if j <= i else "   ---  " for j in range(len(tasks)))
            print(f"    After Stage {i+1} ({tasks[i]:<10}) : {row}")

    print("\n" + "=" * 105)
    print("                    COMPREHENSIVE BENCHMARK & PARAMETER COMPARISON TABLE")
    print("=" * 105)
    print(f"{'Architecture / Method':<32} | {'Final ACC':<10} | {'BWT':<10} | {'Forgetting':<11} | {'Total Params':<13} | {'Model (MB)':<10} | {'Time (s)':<8}")
    print("-" * 105)
    for model_name, res in results.items():
        mb_size = (res["Params"] * 4) / (1024 * 1024)
        print(f"{model_name:<32} | {res['ACC']:>6.2f}%   | {res['BWT']:>+6.2f}%   | {res['Forg']:>7.2f}%    | {res['Params']:>13,} | {mb_size:>8.2f}MB | {res['Time']:>6.1f}s")
    print("=" * 105)

if __name__ == "__main__":
    main()
