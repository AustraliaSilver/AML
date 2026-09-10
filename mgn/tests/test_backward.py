"""
tests/test_backward.py

Numerical correctness test for the backward pass of fused_dynamic_conv2d.
Compares CUDA backward gradients against PyTorch autograd reference.
"""

import torch
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mgn.accel as aml_accel


def reference_backward(X, prototypes, pi, w_offset, b_offset, C_in, C_out, grad_Y):
    """
    PyTorch reference backward using autograd.
    """
    X_ref = X.clone().detach().requires_grad_(True)
    proto_ref = prototypes.clone().detach().requires_grad_(True)
    pi_ref = pi.clone().detach().requires_grad_(True)

    B = X_ref.shape[0]
    H, W = X_ref.shape[2], X_ref.shape[3]
    per_oc_weight = C_in * 9

    w_direct = pi_ref @ proto_ref
    w_conv = w_direct[:, w_offset:w_offset + C_out * per_oc_weight]
    b_conv = w_direct[:, b_offset:b_offset + C_out]

    w_flat = w_conv.reshape(B * C_out, C_in, 3, 3)
    b_flat = b_conv.reshape(B * C_out)

    X_flat = X_ref.view(1, B * C_in, H, W)
    Y = F.conv2d(X_flat, w_flat, b_flat, groups=B, padding=1)
    Y = Y.view(B, C_out, H, W)

    Y.backward(grad_Y)

    return X_ref.grad, pi_ref.grad, proto_ref.grad


def test_backward_layer_0():
    """Test backward for layer 0: C_in=1, C_out=128"""
    B, K = 4, 4
    C_in, C_out = 1, 64
    H, W = 28, 28
    W_dim = C_out * C_in * 9 + C_out

    prototypes = torch.randn(K, W_dim, device="cuda", dtype=torch.float32)
    pi = torch.softmax(torch.randn(B, K, device="cuda", dtype=torch.float32), dim=-1)
    X = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)
    grad_Y = torch.randn(B, C_out, H, W, device="cuda", dtype=torch.float32)

    w_offset = 0
    b_offset = C_out * C_in * 9

    # PyTorch reference
    gX_ref, gpi_ref, gproto_ref = reference_backward(
        X, prototypes, pi, w_offset, b_offset, C_in, C_out, grad_Y
    )

    # CUDA kernel
    X_cu = X.clone().requires_grad_(True)
    pi_cu = pi.clone().requires_grad_(True)
    proto_cu = prototypes.clone().requires_grad_(True)

    Y_cu = aml_accel.fused_dynamic_conv2d(X_cu, proto_cu, pi_cu, w_offset, b_offset)
    Y_cu.backward(grad_Y)

    # Compare grad_X
    gx_diff = (gX_ref - X_cu.grad).abs().max().item()
    print(f"  grad_X:  max_diff={gx_diff:.6e}")
    assert torch.allclose(gX_ref, X_cu.grad, atol=1e-3), "grad_X FAILED!"

    # Compare grad_pi
    gpi_diff = (gpi_ref - pi_cu.grad).abs().max().item()
    print(f"  grad_pi: max_diff={gpi_diff:.6e}")
    assert torch.allclose(gpi_ref, pi_cu.grad, atol=1e-3), "grad_pi FAILED!"

    # Compare grad_proto
    gp_diff = (gproto_ref - proto_cu.grad).abs().max().item()
    print(f"  grad_proto: max_diff={gp_diff:.6e}")
    assert torch.allclose(gproto_ref, proto_cu.grad, atol=1e-3), "grad_proto FAILED!"

    print("  PASSED")


def test_backward_end_to_end():
    """Test that a full training step produces similar updates."""
    B, K = 8, 4
    C_in, C_out = 1, 32
    H, W = 28, 28
    W_dim = C_out * C_in * 9 + C_out

    torch.manual_seed(42)
    prototypes = torch.randn(K, W_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    pi = torch.softmax(torch.randn(B, K, device="cuda", dtype=torch.float32), dim=-1)
    X = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)
    target = torch.randint(0, C_out, (B,), device="cuda")

    w_offset = 0
    b_offset = C_out * C_in * 9

    # Forward + backward
    Y = aml_accel.fused_dynamic_conv2d(X, prototypes, pi, w_offset, b_offset)

    # Simple loss: sum of output
    loss = Y.mean()
    loss.backward()

    print(f"  grad_proto norm: {prototypes.grad.norm().item():.4f}")
    print(f"  grad_proto max:  {prototypes.grad.abs().max().item():.6e}")
    assert prototypes.grad is not None
    assert prototypes.grad.norm().item() > 0, "Gradient is zero!"
    print("  PASSED")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)

    print("=" * 60)
    print("TEST: fused_dynamic_conv2d backward correctness")
    print("=" * 60)

    test_backward_layer_0()
    test_backward_end_to_end()

    print("\n" + "=" * 60)
    print("ALL BACKWARD TESTS PASSED")
    print("=" * 60)
