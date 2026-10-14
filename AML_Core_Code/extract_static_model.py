"""
extract_static_model.py — Hard-prune the dynamic AML supernet into a
standalone static child model (pure nn.Sequential-style, zero Gumbel
overhead at inference).

Supports both live mothers:
  - ConvAML_18Layer_Hybrid     (mgn/blocks/hybrid_18.py, 18 layers, K=12,
                                3 topologies: Plain / ResNet / ViT)
  - ConvAML_CIFAR_TrueDynamic  (repo-root train.py, 18 layers, K=32)

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

from train_cv_cifar_18 import get_prototype_layout, get_second_conv_layout


class StaticLayer(nn.Module):
    """Static replica of one 2-conv AML block (plain + residual mix)."""

    def __init__(self, conv_w, conv_b, conv2_w, conv2_b, gate_c, dim_mask,
                 g_plain, g_resnet, is_first_layer=False):
        super().__init__()
        self.is_first_layer = is_first_layer
        self.conv = nn.Conv2d(conv_w.shape[1], conv_w.shape[0], 3, padding=1)
        self.conv.weight.data = conv_w
        self.conv.bias.data = conv_b
        self.conv2 = nn.Conv2d(conv2_w.shape[1], conv2_w.shape[0], 3, padding=1)
        self.conv2.weight.data = conv2_w
        self.conv2.bias.data = conv2_b

        self.norm = nn.GroupNorm(1, conv_w.shape[0])
        self.norm2 = nn.GroupNorm(1, conv2_w.shape[0])

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

        h_resnet = F.relu(h_expand + self.norm2(self.conv2(h_plain))) * self.dim_mask

        h_active = self.g_plain * h_plain + self.g_resnet * h_resnet
        return self.gate_c * h_active + (1.0 - self.gate_c) * h_expand


class StaticHybridBlock(nn.Module):
    """Static replica of one HybridBlock: three topologies, one selected.

    The legacy StaticLayer fuses the plain and residual branches with two frozen
    weights, because the old supernet only had two candidates. The 18-layer
    hybrid has three, so the mode is a hard choice rather than a two-way mix:
    softmax(block_mode_prototypes) is averaged over the batch, and the argmax
    branch is compiled away. That matches HybridBlock.forward, which collapses
    to the same single branch once the weights are one-hot.

    The depth gate must be carried too. The mother computes

        h_new = gate_c * h_cand + (1.0 - gate_c) * h          # i > 0, gate open
        h_new = h                                             # gate <= gate_skip_eps
        h_new = layer(h)                                     # i == 0, gate ignored

    so a layer whose gate has closed contributes nothing, and a half-open layer
    contributes half. Dropping the gate and running every block at full strength
    would silently change the function the supernet learned - which is exactly
    what --depth-cost is there to shape.
    """

    def __init__(self, block, mode_idx, dim_mask, gate=1.0, is_first=False,
                 gate_skip_eps=1e-6):
        super().__init__()
        self.mode_idx = mode_idx
        self.is_first = is_first
        self.gate_skip_eps = gate_skip_eps
        # A closed gate is compiled out entirely, so the child never pays for a
        # block the mother would have skipped.
        self.is_active = bool(is_first or gate > gate_skip_eps)
        self.register_buffer('gate', torch.tensor(float(gate)))
        self.register_buffer('dim_mask', dim_mask)
        if mode_idx == 0:
            self.body = block.plain
        elif mode_idx == 1:
            self.body = block.resnet
        elif mode_idx == 2:
            self.body = block.vit
        else:
            raise ValueError(f"invalid mode {mode_idx}")

    def forward(self, x):
        if not self.is_active:
            return x
        out = self.body(x, dim_mask=self.dim_mask)
        if self.is_first:
            return out
        g = self.gate
        return g * out + (1.0 - g) * x


class StaticChildModel(nn.Module):
    """
    Static subnet distilled from the supernet. No Gumbel-Softmax, no
    dynamic ops. `pool_after` is the set of layer indices after which the
    mother applies 2x2 max-pooling ({5, 11} for the 18-layer CIFAR net,
    {2} for the legacy 6-layer net).
    """

    def __init__(self, layers, fc_weight, fc_bias, pool_after=(5, 11), ln=None,
                 stem=None, dropout_p=0.2):
        super().__init__()
        self.dropout_p = dropout_p
        # The 18-layer hybrid's trunk starts with a 3-channel conv stem; the
        # legacy supernet did not have one, so it stays optional.
        self.stem = stem
        self.features = nn.ModuleList(layers)
        self.pool_after = set(pool_after)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        # The 18-layer hybrid's trunk applies an affine LayerNorm before the head.
        # Falling back to a bare F.layer_norm here would silently drop those
        # parameters, so the module is carried over when the mother has one.
        self.ln = ln if ln is not None else nn.LayerNorm(fc_weight.shape[1])
        self.fc = nn.Linear(fc_weight.shape[1], fc_weight.shape[0])
        self.fc.weight.data = fc_weight
        self.fc.bias.data = fc_bias

    def forward(self, x):
        h = self.stem(x) if self.stem is not None else x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in self.pool_after:
                h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.pool(h).view(x.shape[0], -1)
        h = self.ln(h)
        # The hybrid trunk uses p=0.1 (hybrid_18.py:411); the legacy supernet used
        # 0.2. Carrying the wrong rate would make the child a different function
        # at train time than the model it was distilled from.
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        return self.fc(h)


def _build_child(prototypes, pi, layer_gates, dim_masks, block_probs,
                 max_layers, max_dim, num_classes, pool_after, device,
                 layout_fn=get_prototype_layout):
    """Generic child builder from averaged structure tensors."""
    total_dim, offsets, layer_info, wo_offset, bo_offset = layout_fn(
        max_layers, max_dim, num_classes)
    offsets2 = get_second_conv_layout(max_layers, max_dim, num_classes)
    assert prototypes.shape[1] == total_dim, \
        f"prototype width {prototypes.shape[1]} != layout {total_dim}"

    layers = []
    for i in range(max_layers):
        w_start, w_end, b_start, b_end, in_c = offsets[i]
        w2_start, w2_end, b2_start, b2_end = offsets2[i]

        w_mix = (pi @ prototypes[:, w_start:w_end]).view(max_dim, in_c, 3, 3)
        b_mix = (pi @ prototypes[:, b_start:b_end]).view(max_dim)
        w2_mix = (pi @ prototypes[:, w2_start:w2_end]).view(max_dim, max_dim, 3, 3)
        b2_mix = (pi @ prototypes[:, b2_start:b2_end]).view(max_dim)

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

        layers.append(StaticLayer(w_mix, b_mix, w2_mix, b2_mix, gate_c, mask,
                                  g_plain, g_resnet, is_first_layer=(i == 0)))

    w_out = (pi @ prototypes[:, wo_offset:bo_offset]).view(num_classes, 16 * max_dim)
    b_out = (pi @ prototypes[:, bo_offset:bo_offset + num_classes]).view(num_classes)

    child = StaticChildModel(layers, w_out, b_out, pool_after=pool_after)
    return child.to(device)


@torch.no_grad()
def extract_child_truedynamic(mother_model, x_dummy):
    """Extract a static child from ConvAML_CIFAR_TrueDynamic (train.py)."""
    mother_model.eval()
    L = mother_model.max_layers
    D = mother_model.max_dim
    device = x_dummy.device

    # Deterministic structure: no Gumbel noise, batch-mean conditioning.
    z = mother_model.vision_encoder(x_dummy)

    pi = mother_model.ood_router.get_pi(z, mother_model.rows_per_task).mean(dim=0, keepdim=True)

    delta = pi @ mother_model.structure_prototypes
    layer_soft = torch.sigmoid(delta[:, :L] + 2.0)
    layer_gates = (layer_soft > 0.5).float()

    dim_logits = delta[:, L:].view(1, L, D)
    d_logit = dim_logits.mean(dim=-1, keepdim=True) + 2.0
    d_lengths = torch.clamp(torch.sigmoid(d_logit) * D, min=24.0)
    arange = torch.arange(D, device=device).float().view(1, 1, D)
    dim_masks = torch.sigmoid(10.0 * (d_lengths - arange))

    bm_logits = (pi @ mother_model.block_mode_prototypes).view(1, L, 2)
    block_probs = torch.softmax(bm_logits, dim=-1)

    return _build_child(
        mother_model.proto_weights, pi, layer_gates, dim_masks, block_probs,
        max_layers=L, max_dim=D, num_classes=10, pool_after=(5, 11), device=device)


@torch.no_grad()
def extract_child_hybrid18(mother_model, x_dummy):
    """Extract a static child from ConvAML_18Layer_Hybrid (train_continual_aml).

    Structurally different from the legacy path, and the differences are not
    cosmetic:

    - Routing comes from `prototype_selector` (K x 256 -> K), not from
      `ood_router`.
    - Convolutional weights live in real submodules (`block.plain/.resnet/.vit`),
      not inside the `proto_weights` tensor, so there is no weighted sum to fold
      into a single kernel.
    - `block_mode_prototypes` is [K, L, 3] - three topologies, not two. The
      static child keeps the argmax branch per layer.

    The stem and the vision encoder are kept intact rather than pruned: they are
    shared, not per-layer, and the router reads the encoder, so replacing it
    would invalidate the domain router.
    """
    from mgn.blocks.hybrid_18 import ConvAML_18Layer_Hybrid  # noqa: F401
    from mgn.blocks.hybrid_router import HybridBlock  # noqa: F401

    # hybrid_18.py:411. The legacy supernet used 0.2; these constants must not
    # drift from the model they are meant to replicate.
    DROPOUT_P = 0.1

    if not isinstance(mother_model, ConvAML_18Layer_Hybrid):
        raise TypeError(f"expected ConvAML_18Layer_Hybrid, got "
                        f"{type(mother_model).__name__}")

    mother_model.eval()
    L, D = mother_model.max_layers, mother_model.max_dim
    device = x_dummy.device

    # Deterministic structure: no Gumbel noise, batch-mean conditioning.
    with torch.no_grad():
        z = mother_model.vision_encoder(x_dummy)
    pi = F.softmax(mother_model.prototype_selector(z), dim=-1).mean(0)   # [K]

    delta = (pi @ mother_model.structure_prototypes).reshape(-1)          # [L + L*D]
    # Depth gate: sigmoid(delta[:L] + 2.0), binarised exactly as the mother does
    # at eval (hybrid_18.py:238 then :260). Layer 0 is excluded - the mother
    # deliberately ignores its gate because it sets the feature scale.
    layer_gates = (torch.sigmoid(delta[:L] + 2.0) > 0.5).float()

    # Width floor is the MOTHER's (hybrid_18.py:244), not the legacy 24.0.
    # Copying the legacy constant would clamp this supernet to a narrower channel
    # set than it was trained with. The mother hardcodes the literal rather than
    # exposing it, so the value is mirrored here and the source line named.
    MIN_ACTIVE_DIM = 16.0
    d_lengths = torch.clamp(
        torch.sigmoid(delta[L:].view(L, D).mean(-1, keepdim=True) + 2.0) * D,
        min=MIN_ACTIVE_DIM)
    dim_masks = torch.sigmoid(10.0 * (d_lengths - torch.arange(
        D, device=device).float().view(1, D)))                          # [L, D]

    bm = torch.softmax(torch.einsum('k,klc->lc', pi,
                                    mother_model.block_mode_prototypes), dim=-1)
    modes = bm.argmax(-1)                                                 # [L]

    layers = []
    for i in range(L):
        blk = mother_model.layers[i]
        if not isinstance(blk, HybridBlock):
            raise TypeError(f"layer {i} is {type(blk).__name__}, expected HybridBlock")
        mask = dim_masks[i].view(D, 1, 1)
        layers.append(StaticHybridBlock(blk, int(modes[i]), mask,
                                        gate=float(layer_gates[i]),
                                        is_first=(i == 0),
                                        gate_skip_eps=mother_model.gate_skip_eps))

    # Head: the prototype-weighted classifier collapses the same way, because pi
    # has already been averaged over the batch.
    #
    # The offsets come from the MODEL, not from get_prototype_layout. The 18-layer
    # hybrid keeps its own layout - proto_weights is [K, wo_size + num_classes]
    # with wo_size = num_classes * head_dim - and the legacy layout in
    # train_cv_cifar_18.py describes a different tensor. Reading the offsets off
    # the model is what keeps the extractor from silently indexing the wrong
    # slice, which is how the two supernets drifted apart in the first place.
    proto = mother_model.proto_weights
    w_out = (pi @ proto[:, :mother_model.wo_size]).view(mother_model.num_classes,
                                                        mother_model.head_dim)
    b_out = (pi @ proto[:, mother_model.wo_size:
                          mother_model.wo_size + mother_model.num_classes])

    child = StaticChildModel(layers, w_out, b_out, pool_after=(5, 11),
                             ln=mother_model.ln, stem=mother_model.stem,
                             dropout_p=DROPOUT_P)
    return child.to(device).eval()


@torch.no_grad()
def extract_child(mother_model, x_dummy):
    """Extract a static child from the AML supernet.

    Public entry point. Dispatches on the actual model class:

      ConvAML_18Layer_Hybrid     -> extract_child_hybrid18   (mgn/blocks/hybrid_18.py)
      ConvAML_CIFAR_TrueDynamic  -> extract_child_truedynamic (repo-root train.py)

    The previous guard checked only for `structure_prototypes` and
    `proto_weights`, which the 18-layer hybrid also has, so passing it in
    sailed through validation and then died deep inside with
    "object has no attribute 'ood_router'". Dispatching on the type is the only
    check that actually distinguishes the two supernets.
    """
    from mgn.blocks.hybrid_18 import ConvAML_18Layer_Hybrid

    if isinstance(mother_model, ConvAML_18Layer_Hybrid):
        return extract_child_hybrid18(mother_model, x_dummy)

    if (hasattr(mother_model, "structure_prototypes")
            and hasattr(mother_model, "proto_weights")
            and hasattr(mother_model, "ood_router")):
        return extract_child_truedynamic(mother_model, x_dummy)

    raise TypeError(
        f"Unsupported mother model {type(mother_model).__name__}: expected "
        "ConvAML_18Layer_Hybrid or ConvAML_CIFAR_TrueDynamic."
    )


def _logits_of(module, x):
    out = module(x)
    if isinstance(out, tuple):
        out = out[0]
    return out


def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
