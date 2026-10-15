import torch
from mgn.blocks import (
    PlainConvBlock,
    ResNetBlock,
    VisionTransformerBlock,
    HybridBlock
)


def test_plain_conv_block():
    B, C, H, W = 4, 32, 16, 16
    x = torch.randn(B, C, H, W, requires_grad=True)
    block = PlainConvBlock(in_channels=C, out_channels=C)
    
    # Forward pass
    out = block(x)
    assert out.shape == (B, C, H, W)
    
    # Backward pass
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

    # Params & FLOPs
    params = block.compute_params()
    flops = block.compute_flops((H, W))
    assert params > 0
    assert flops > 0


def test_resnet_block():
    B, C, H, W = 4, 32, 16, 16
    x = torch.randn(B, C, H, W, requires_grad=True)
    block = ResNetBlock(in_channels=C, out_channels=C)
    
    # Forward pass with dim mask
    mask = torch.ones(B, C, 1, 1)
    out = block(x, dim_mask=mask)
    assert out.shape == (B, C, H, W)
    
    # Backward pass
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

    params = block.compute_params()
    flops = block.compute_flops((H, W))
    assert params > 0
    assert flops > 0


def test_vit_block():
    B, C, H, W = 4, 32, 8, 8
    x = torch.randn(B, C, H, W, requires_grad=True)
    block = VisionTransformerBlock(in_channels=C, out_channels=C, num_heads=4)
    
    # Forward pass
    out = block(x)
    assert out.shape == (B, C, H, W)
    
    # Backward pass
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

    params = block.compute_params()
    flops = block.compute_flops((H, W))
    assert params > 0
    assert flops > 0


def test_hybrid_block_routing():
    B, C, H, W = 4, 32, 8, 8
    block = HybridBlock(in_channels=C, out_channels=C, num_heads=4)

    # 1. Soft mixture mode (Bi-level DARTS)
    mode_weights = torch.softmax(torch.randn(B, 3), dim=-1)
    x = torch.randn(B, C, H, W, requires_grad=True)
    out_soft = block(x, mode_weights=mode_weights)
    assert out_soft.shape == (B, C, H, W)
    out_soft.sum().backward()
    assert x.grad is not None

    # 2. Hard selection: Plain Conv
    weights_plain = torch.zeros(B, 3)
    weights_plain[:, 0] = 1.0
    out_plain = block(x, mode_weights=weights_plain)
    assert out_plain.shape == (B, C, H, W)

    # 3. Hard selection: ResNet
    weights_resnet = torch.zeros(B, 3)
    weights_resnet[:, 1] = 1.0
    out_resnet = block(x, mode_weights=weights_resnet)
    assert out_resnet.shape == (B, C, H, W)

    # 4. Hard selection: ViT
    weights_vit = torch.zeros(B, 3)
    weights_vit[:, 2] = 1.0
    out_vit = block(x, mode_weights=weights_vit)
    assert out_vit.shape == (B, C, H, W)


def test_per_sample_dispatch_is_batch_composition_invariant():
    """A sample's output must not depend on which other samples share its batch.

    HybridBlock.forward has two hard fast paths: one when the whole batch agrees
    on a mode, one when it does not. Only the first multiplied the result by the
    sample's own routing weight; the per-sample dispatch returned the branch
    output unscaled. `all_hard` is `max > 0.99`, not `== 1.0`, so a near-one-hot
    weight (0.995) meant the same input produced 0.995*y or y depending on its
    batch neighbours - a silent ~0.5 % amplitude change that is a function of
    batching, not of the model.

    Exact one-hot weights (what the rest of this file uses) hide it: 1.0*y == y.
    """
    torch.manual_seed(0)
    B, C, H, W = 4, 8, 8, 8
    block = HybridBlock(in_channels=C, out_channels=C, num_heads=2).eval()
    x = torch.randn(B, C, H, W)

    # Near-one-hot but not exactly 1.0 -> `all_hard` is True.
    w = torch.tensor([[0.001, 0.004, 0.995]], dtype=torch.float32).expand(B, 3).clone()

    # (a) the whole batch agrees on mode 2 -> the whole-batch fast path
    out_uniform = block(x, mode_weights=w)

    # (b) the same weights, except one sample picks mode 0 -> not same_mode, so
    #     the per-sample dispatch runs. Samples 0..2 are unchanged inputs with
    #     unchanged weights, so their outputs must be bit-for-bit comparable.
    w2 = w.clone()
    w2[3] = torch.tensor([0.995, 0.004, 0.001])
    out_mixed = block(x, mode_weights=w2)

    assert torch.allclose(out_uniform[:3], out_mixed[:3], rtol=1e-5, atol=1e-6), (
        "samples 0..2 changed output when a sibling changed mode: the "
        "per-sample dispatch is not scaling by the routing weight "
        f"(max abs diff {(out_uniform[:3] - out_mixed[:3]).abs().max():.3e})")


def test_hybrid_channel_downsampling():
    B, C_in, C_out, H, W = 2, 16, 32, 8, 8
    block = HybridBlock(in_channels=C_in, out_channels=C_out, num_heads=4)
    x = torch.randn(B, C_in, H, W)
    out = block(x)
    assert out.shape == (B, C_out, H, W)


def test_vit_rejects_indivisible_num_heads():
    """Regression: --max-dim is a free CLI value and nothing reconciled it with
    num_heads. An indivisible pair used to fail deep inside forward() with
    "shape '[B, N, 3, 4, C // 4]' is invalid for input of size ..."; it must
    fail at construction with a message that names the two numbers."""
    try:
        VisionTransformerBlock(in_channels=32, out_channels=30, num_heads=4)
    except ValueError as e:
        assert "divisible by num_heads" in str(e), str(e)
        return
    raise AssertionError("indivisible num_heads must be rejected at build time")
