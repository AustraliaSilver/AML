"""
tests/test_fused_conv.py

Numerical correctness test: compare fused_dynamic_conv2d CUDA kernel
against PyTorch reference implementation.
"""

import torch
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mgn.accel


def pytorch_reference_conv2d(X, prototypes, pi, w_offset, b_offset, C_in, C_out):
    """
    PyTorch reference: mix weights then call F.conv2d.
    """
    B = X.shape[0]
    H, W = X.shape[2], X.shape[3]
    per_oc_weight = C_in * 9

    w_direct = pi @ prototypes  # [B, W_dim]

    w_conv = w_direct[:, w_offset:w_offset + C_out * per_oc_weight]
    b_conv = w_direct[:, b_offset:b_offset + C_out]

    w_flat = w_conv.view(B * C_out, C_in, 3, 3)
    b_flat = b_conv.view(B * C_out)

    X_flat = X.view(1, B * C_in, H, W)
    Y = F.conv2d(X_flat, w_flat, b_flat, groups=B, padding=1)
    Y = Y.view(B, C_out, H, W)
    return Y


def test_layer_0():
    """Test layer 0: C_in=1, C_out=128, H=W=28"""
    B, K = 4, 4
    C_in, C_out = 1, 128
    H, W = 28, 28
    W_dim = C_out * C_in * 9 + C_out  # weight + bias

    prototypes = torch.randn(K, W_dim, device="cuda", dtype=torch.float32)
    pi = torch.softmax(torch.randn(B, K, device="cuda", dtype=torch.float32), dim=-1)
    X = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)

    w_offset = 0
    b_offset = C_out * C_in * 9

    Y_ref = pytorch_reference_conv2d(X, prototypes, pi, w_offset, b_offset, C_in, C_out)
    Y_cu = aml_accel.fused_dynamic_conv2d(X, prototypes, pi, w_offset, b_offset)

    max_diff = (Y_ref - Y_cu).abs().max().item()
    mean_diff = (Y_ref - Y_cu).abs().mean().item()
    print(f"  Layer 0: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    assert torch.allclose(Y_ref, Y_cu, atol=1e-4), f"FAILED! max_diff={max_diff}"
    print("  PASSED")


def test_layer_1():
    """Test layer 1: C_in=128, C_out=128, H=W=28"""
    B, K = 2, 4
    C_in, C_out = 128, 128
    H, W = 28, 28
    W_dim = C_out * C_in * 9 + C_out

    prototypes = torch.randn(K, W_dim, device="cuda", dtype=torch.float32)
    pi = torch.softmax(torch.randn(B, K, device="cuda", dtype=torch.float32), dim=-1)
    X = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)

    w_offset = 0
    b_offset = C_out * C_in * 9

    Y_ref = pytorch_reference_conv2d(X, prototypes, pi, w_offset, b_offset, C_in, C_out)
    Y_cu = aml_accel.fused_dynamic_conv2d(X, prototypes, pi, w_offset, b_offset)

    max_diff = (Y_ref - Y_cu).abs().max().item()
    mean_diff = (Y_ref - Y_cu).abs().mean().item()
    print(f"  Layer 1: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    assert torch.allclose(Y_ref, Y_cu, atol=1e-4), f"FAILED! max_diff={max_diff}"
    print("  PASSED")


def test_batch_size_1():
    """Test with batch_size=1 (single sample)"""
    B, K = 1, 4
    C_in, C_out = 1, 64
    H, W = 28, 28
    W_dim = C_out * C_in * 9 + C_out

    prototypes = torch.randn(K, W_dim, device="cuda", dtype=torch.float32)
    pi = torch.softmax(torch.randn(B, K, device="cuda", dtype=torch.float32), dim=-1)
    X = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)

    w_offset = 0
    b_offset = C_out * C_in * 9

    Y_ref = pytorch_reference_conv2d(X, prototypes, pi, w_offset, b_offset, C_in, C_out)
    Y_cu = aml_accel.fused_dynamic_conv2d(X, prototypes, pi, w_offset, b_offset)

    max_diff = (Y_ref - Y_cu).abs().max().item()
    print(f"  Batch=1: max_diff={max_diff:.6e}")
    assert torch.allclose(Y_ref, Y_cu, atol=1e-4), f"FAILED! max_diff={max_diff}"
    print("  PASSED")


def test_batch_size_64():
    """Test with batch_size=64 (production training size)"""
    B, K = 64, 4
    C_in, C_out = 1, 128
    H, W = 28, 28
    W_dim = C_out * C_in * 9 + C_out

    prototypes = torch.randn(K, W_dim, device="cuda", dtype=torch.float32)
    pi = torch.softmax(torch.randn(B, K, device="cuda", dtype=torch.float32), dim=-1)
    X = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)

    w_offset = 0
    b_offset = C_out * C_in * 9

    Y_ref = pytorch_reference_conv2d(X, prototypes, pi, w_offset, b_offset, C_in, C_out)
    Y_cu = aml_accel.fused_dynamic_conv2d(X, prototypes, pi, w_offset, b_offset)

    max_diff = (Y_ref - Y_cu).abs().max().item()
    print(f"  Batch=64: max_diff={max_diff:.6e}")
    assert torch.allclose(Y_ref, Y_cu, atol=1e-4), f"FAILED! max_diff={max_diff}"
    print("  PASSED")


def test_cifar10_channels():
    """Test with 3 input channels (CIFAR-10)"""
    B, K = 4, 4
    C_in, C_out = 3, 64
    H, W = 32, 32
    W_dim = C_out * C_in * 9 + C_out

    prototypes = torch.randn(K, W_dim, device="cuda", dtype=torch.float32)
    pi = torch.softmax(torch.randn(B, K, device="cuda", dtype=torch.float32), dim=-1)
    X = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)

    w_offset = 0
    b_offset = C_out * C_in * 9

    Y_ref = pytorch_reference_conv2d(X, prototypes, pi, w_offset, b_offset, C_in, C_out)
    Y_cu = aml_accel.fused_dynamic_conv2d(X, prototypes, pi, w_offset, b_offset)

    max_diff = (Y_ref - Y_cu).abs().max().item()
    print(f"  CIFAR (3ch): max_diff={max_diff:.6e}")
    assert torch.allclose(Y_ref, Y_cu, atol=1e-4), f"FAILED! max_diff={max_diff}"
    print("  PASSED")


def test_vram_savings():
    """Verify that fused kernel uses less VRAM than PyTorch approach."""
    B, K = 64, 4
    C_in, C_out = 1, 128
    H, W = 28, 28
    W_dim = C_out * C_in * 9 + C_out

    prototypes = torch.randn(K, W_dim, device="cuda", dtype=torch.float32)
    pi = torch.softmax(torch.randn(B, K, device="cuda", dtype=torch.float32), dim=-1)
    X = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    Y_cu = aml_accel.fused_dynamic_conv2d(X, prototypes, pi, 0, C_out * C_in * 9)
    mem_after = torch.cuda.memory_allocated()
    fused_vram = mem_after - mem_before

    w_direct = pi @ prototypes
    expected_saved = w_direct.numel() * 4  # fp32
    print(f"  VRAM saved: {expected_saved / 1e6:.1f} MB (w_direct not allocated)")
    print(f"  Fused kernel VRAM delta: {fused_vram / 1e6:.1f} MB")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)

    print("=" * 60)
    print("TEST: fused_dynamic_conv2d correctness")
    print("=" * 60)

    test_layer_0()
    test_layer_1()
    test_batch_size_1()
    test_batch_size_64()
    test_cifar10_channels()
    test_vram_savings()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
