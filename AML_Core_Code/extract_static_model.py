"""
extract_static_model.py — Hard-prune the dynamic AML supernet into a
standalone static child model (pure nn.Sequential-style, zero Gumbel
overhead at inference).

Supports both mothers:
  - ConvAML_CIFAR_TrueDynamic (repo-root train.py, 18 layers, K=32)
  - ConvAML_CIFAR (legacy 6-layer, AML_Core_Code/train_cv_cifar.py)

The prototype weight layout comes exclusively from
train_cv_cifar_18.get_prototype_layout so extractor and trainers can
never drift apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from train_cv_cifar_18 import get_prototype_layout


class StaticLayer(nn.Module):
    """Static replica of fused_dynamic_conv2d + groupnorm + relu + gates."""

    def __init__(self, conv_w, conv_b, gate_c, dim_mask, g_plain, g_resnet, is_first_layer=False):
        super().__init__()
        self.is_first_layer = is_first_layer
        self.conv = nn.Conv2d(conv_w.shape[1], conv_w.shape[0], 3, padding=1)
        self.conv.weight.data = conv_w
        self.conv.bias.data = conv_b

        self.norm = nn.GroupNorm(1, conv_w.shape[0])

        self.register_buffer('gate_c', gate_c)
        self.register_buffer('dim_mask', dim_mask)
        self.register_buffer('g_plain', g_plain)
        self.register_buffer('g_resnet', g_resnet)

    def forward(self, x):
        h_conv = self.conv(x)
        h_norm = self.norm(h_conv)
        h_plain = F.relu(h_norm) * self.dim_mask

        if self.is_first_layer:
            h_expand = h_plain
        else:
            h_expand = x * self.dim_mask

        h_resnet = F.relu(h_expand + h_norm) * self.dim_mask

        h_active = self.g_plain * h_plain + self.g_resnet * h_resnet
        return self.gate_c * h_active + (1.0 - self.gate_c) * h_expand


class StaticChildModel(nn.Module):
    """
    Static subnet distilled from the supernet. No Gumbel-Softmax, no
    dynamic ops. `pool_after` is the set of layer indices after which the
    mother applies 2x2 max-pooling ({5, 11} for the 18-layer CIFAR net,
    {2} for the legacy 6-layer net).
    """

    def __init__(self, layers, fc_weight, fc_bias, pool_after=(5, 11)):
        super().__init__()
        self.features = nn.ModuleList(layers)
        self.pool_after = set(pool_after)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(fc_weight.shape[1], fc_weight.shape[0])
        self.fc.weight.data = fc_weight
        self.fc.bias.data = fc_bias

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in self.pool_after:
                h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.pool(h).view(x.shape[0], -1)
        h = F.layer_norm(h, [h.shape[-1]])
        h = F.dropout(h, p=0.2, training=self.training)
        return self.fc(h)


def _build_child(prototypes, pi, layer_gates, dim_masks, block_probs,
                 max_layers, max_dim, num_classes, pool_after, device,
                 layout_fn=get_prototype_layout):
    """Generic child builder from averaged structure tensors."""
    total_dim, offsets, layer_info, wo_offset, bo_offset = layout_fn(
        max_layers, max_dim, num_classes)
    assert prototypes.shape[1] == total_dim, \
        f"prototype width {prototypes.shape[1]} != layout {total_dim}"

    layers = []
    for i in range(max_layers):
        w_start, w_end, b_start, b_end, in_c = offsets[i]

        w_mix = (pi @ prototypes[:, w_start:w_end]).view(max_dim, in_c, 3, 3)
        b_mix = (pi @ prototypes[:, b_start:b_end]).view(max_dim)

        if i == 0:
            gate_c = torch.tensor([1.0], device=device)
        else:
            gate_c = layer_gates[:, i].view(1, 1, 1)

        mask = dim_masks[:, i].reshape(max_dim, 1, 1)

        if block_probs is not None:
            g_plain = block_probs[:, i, 0].reshape(1, 1, 1)
            g_resnet = block_probs[:, i, 1].reshape(1, 1, 1)
        else:
            g_plain = torch.tensor([1.0], device=device)
            g_resnet = torch.tensor([0.0], device=device)

        layers.append(StaticLayer(w_mix, b_mix, gate_c, mask, g_plain, g_resnet,
                                  is_first_layer=(i == 0)))

    w_out = (pi @ prototypes[:, wo_offset:bo_offset]).view(num_classes, 16 * max_dim)
    b_out = (pi @ prototypes[:, bo_offset:bo_offset + num_classes]).view(num_classes)

    child = StaticChildModel(layers, w_out, b_out, pool_after=pool_after)
    return child.to(device)


@torch.no_grad()
def extract_child_truedynamic(mother_model, x_dummy):
    """Extract a static child from ConvAML_CIFAR_TrueDynamic (train.py)."""
    mother_model.eval()
    B = x_dummy.shape[0]
    L = mother_model.max_layers
    D = mother_model.max_dim
    device = x_dummy.device

    # Deterministic structure: no Gumbel noise, batch-mean conditioning.
    z = mother_model.vision_encoder(x_dummy)

    pi = torch.softmax(mother_model.prototype_selector(z), dim=-1).mean(dim=0, keepdim=True)

    delta = mother_model.structure_delta(z)
    layer_soft = torch.sigmoid(delta[:, :L] + 2.0)
    layer_gates = (layer_soft > 0.5).float().mean(dim=0, keepdim=True)

    dim_logits = delta[:, L:].view(B, L, D)
    d_logit = dim_logits.mean(dim=-1, keepdim=True) + 2.0
    d_lengths = torch.clamp(torch.sigmoid(d_logit) * D, min=24.0)
    arange = torch.arange(D, device=device).float().view(1, 1, D)
    dim_masks = torch.sigmoid(10.0 * (d_lengths - arange)).mean(dim=0, keepdim=True)

    bm_logits = mother_model.block_mode_head(z).view(B, L, 2)
    block_probs = torch.softmax(bm_logits, dim=-1).mean(dim=0, keepdim=True)

    return _build_child(
        mother_model.proto_weights, pi, layer_gates, dim_masks, block_probs,
        max_layers=L, max_dim=D, num_classes=10, pool_after=(5, 11), device=device)


@torch.no_grad()
def extract_child_legacy(mother_model, x_dummy):
    """Extract a static child from the legacy 6-layer ConvAML_CIFAR."""
    mother_model.eval()
    B = x_dummy.shape[0]
    device = x_dummy.device

    gen = mother_model.gen
    if hasattr(gen, 'vision_encoder'):
        z = gen.vision_encoder(x_dummy)
    else:
        z = gen.encoder(x_dummy.view(x_dummy.shape[0], -1))

    pi = torch.softmax(gen.prototype_selector(z), dim=-1).mean(dim=0, keepdim=True)

    S = gen(x_dummy).structure
    layer_gates = S.layer_gates.float().mean(dim=0, keepdim=True)
    dim_masks = S.dim_masks.float().mean(dim=0, keepdim=True)
    block_probs = S.block_mode.float().mean(dim=0, keepdim=True) if S.block_mode is not None else None

    return _build_child(
        gen.prototype_lib.proto_weights, pi, layer_gates, dim_masks, block_probs,
        max_layers=mother_model.exec_.max_layers,
        max_dim=mother_model.exec_.max_dim,
        num_classes=mother_model.exec_.num_classes,
        pool_after=(2,), device=device,
        layout_fn=_legacy_layout)


def _legacy_layout(max_layers, max_dim, num_classes):
    """Replicates the inlined offset walk of the legacy 6-layer
    AcceleratedCVNetwork (train_cv_cifar.py), which still reserves
    per-layer attention slices."""
    offsets = []
    idx = 0
    for i in range(max_layers):
        in_c = 3 if i == 0 else max_dim
        w1_numel = max_dim * in_c * 9
        w_start = idx
        b_start = idx + w1_numel
        w_end = b_start
        b_end = b_start + max_dim
        idx = b_end
        attn_dim = max_dim // 4
        idx += (3 * attn_dim) * max_dim * 1 + 3 * attn_dim
        idx += max_dim * attn_dim * 1 + max_dim
        offsets.append((w_start, w_end, b_start, b_end, in_c))
    wo_offset = idx
    bo_offset = idx + (16 * max_dim) * num_classes
    return bo_offset + num_classes, offsets, None, wo_offset, bo_offset


def _logits_of(module, x):
    out = module(x)
    if isinstance(out, tuple):
        out = out[0]
    return out


def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_true = "--legacy" not in sys.argv
    if use_true:
        print("Loading ConvAML_CIFAR_TrueDynamic (18-layer)...")
        from train import ConvAML_CIFAR_TrueDynamic
        ckpt_path = os.environ.get("AML_CKPT",
                                   os.path.join(_ROOT, "aml_core_best.pth"))
        mother = ConvAML_CIFAR_TrueDynamic(K=32, max_dim=64, max_layers=18).to(device)
        if os.path.isfile(ckpt_path):
            mother.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True))
            print(f"  loaded checkpoint: {ckpt_path}")
        else:
            print(f"  [warn] checkpoint not found ({ckpt_path}), using random init")
        extract_fn = extract_child_truedynamic
    else:
        print("Loading legacy ConvAML_CIFAR (6-layer)...")
        from train_cv_cifar import ConvAML_CIFAR
        mother = ConvAML_CIFAR().to(device)
        extract_fn = extract_child_legacy

    mother.eval()
    x_test = torch.randn(1, 3, 32, 32, device=device)

    print("\nExtracting static child...")
    child = extract_fn(mother, x_test)
    child.eval()

    n_params = sum(p.numel() for p in child.parameters())
    print(f"  child parameters: {n_params/1e6:.3f}M")

    # 1. Output equivalence check (same conditioning sample)
    with torch.no_grad():
        out_mother = _logits_of(mother, x_test)
        out_child = child(x_test)

    diff = (out_mother - out_child).abs().max().item()
    print(f"Max |mother - child| logit diff: {diff:.6f}")
    print("  (small but nonzero expected: mother samples stochastic modes,")
    print("   child freezes their batch-mean expectation)")

    # 2. Inference speed
    iters = 200
    batch_size = 128
    x_bench = torch.randn(batch_size, 3, 32, 32, device=device)

    for _ in range(10):
        mother(x_bench)
        child(x_bench)
    torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            mother(x_bench)
    torch.cuda.synchronize()
    mother_time = time.time() - start

    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            child(x_bench)
    torch.cuda.synchronize()
    child_time = time.time() - start

    print(f"\n=== INFERENCE SPEED (batch {batch_size}, {iters} iters) ===")
    print(f"Mother (dynamic, fused CUDA): {mother_time:.3f}s ({(iters*batch_size)/mother_time:.0f} img/s)")
    print(f"Child  (static, native PyTorch): {child_time:.3f}s ({(iters*batch_size)/child_time:.0f} img/s)")

    if child_time < mother_time:
        print(f"-> Child is {mother_time/child_time:.1f}x faster!")
    else:
        print("-> Mother (CUDA kernels) remains faster than the naive static child.")


if __name__ == "__main__":
    benchmark()
