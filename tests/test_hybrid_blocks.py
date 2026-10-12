import torch
import pytest
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


def test_hybrid_channel_downsampling():
    B, C_in, C_out, H, W = 2, 16, 32, 8, 8
    block = HybridBlock(in_channels=C_in, out_channels=C_out, num_heads=4)
    x = torch.randn(B, C_in, H, W)
    out = block(x)
    assert out.shape == (B, C_out, H, W)
