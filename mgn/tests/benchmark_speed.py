"""
tests/benchmark_speed.py

Benchmark: PyTorch baseline vs mgn.accel CUDA fused kernel.
Measures forward time, backward time, and VRAM usage.
"""

import torch
import torch.nn.functional as F
import time
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mgn.accel


def benchmark_fn(fn, n_warmup=10, n_iter=100):
    """Benchmark a callable, return average time in seconds."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter


def pytorch_fwd(X, prototypes, pi, C_in, C_out, w_offset, b_offset):
    B = X.shape[0]
    H, W = X.shape[2], X.shape[3]
    per_oc_weight = C_in * 9
    w_direct = pi @ prototypes
    w_conv = w_direct[:, w_offset:w_offset + C_out * per_oc_weight]
    b_conv = w_direct[:, b_offset:b_offset + C_out]
    w_flat = w_conv.reshape(B * C_out, C_in, 3, 3)
    b_flat = b_conv.reshape(B * C_out)
    X_flat = X.reshape(1, B * C_in, H, W)
    Y = F.conv2d(X_flat, w_flat, b_flat, groups=B, padding=1)
    return Y.reshape(B, C_out, H, W)


def pytorch_full_step(X, prototypes, pi, C_in, C_out, w_offset, b_offset):
    """Forward + backward (full training step)."""
    X_ref = X.clone().detach().requires_grad_(True)
    proto_ref = prototypes.clone().detach().requires_grad_(True)
    pi_ref = pi.clone().detach().requires_grad_(True)

    Y = pytorch_fwd(X_ref, proto_ref, pi_ref, C_in, C_out, w_offset, b_offset)
    loss = Y.mean()
    loss.backward()
    return Y


def cuda_full_step(X, prototypes, pi, w_offset, b_offset):
    """Forward + backward using fused CUDA kernel."""
    X_cu = X.clone().detach().requires_grad_(True)
    proto_cu = prototypes.clone().detach().requires_grad_(True)
    pi_cu = pi.clone().detach().requires_grad_(True)

    Y = mgn.accel.fused_dynamic_conv2d(X_cu, proto_cu, pi_cu, w_offset, b_offset)
    loss = Y.mean()
    loss.backward()
    return Y


def run_benchmark(B, C_in, C_out, H, W, K=4):
    print(f"\n--- B={B}, C_in={C_in}, C_out={C_out}, H=W={H} ---")

    W_dim = C_out * C_in * 9 + C_out
    prototypes = torch.randn(K, W_dim, device="cuda", dtype=torch.float32)
    pi = torch.softmax(torch.randn(B, K, device="cuda", dtype=torch.float32), dim=-1)
    X = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)

    w_offset = 0
    b_offset = C_out * C_in * 9

    # Forward only
    t_pytorch_fwd = benchmark_fn(
        lambda: pytorch_fwd(X, prototypes, pi, C_in, C_out, w_offset, b_offset)
    )
    t_cuda_fwd = benchmark_fn(
        lambda: mgn.accel.fused_dynamic_conv2d(X, prototypes, pi, w_offset, b_offset)
    )

    # Full step (fwd + bwd)
    t_pytorch_full = benchmark_fn(
        lambda: pytorch_full_step(X, prototypes, pi, C_in, C_out, w_offset, b_offset)
    )
    t_cuda_full = benchmark_fn(
        lambda: cuda_full_step(X, prototypes, pi, w_offset, b_offset)
    )

    # VRAM
    torch.cuda.reset_peak_memory_stats()
    _ = pytorch_full_step(X, prototypes, pi, C_in, C_out, w_offset, b_offset)
    vram_pytorch = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    _ = cuda_full_step(X, prototypes, pi, w_offset, b_offset)
    vram_cuda = torch.cuda.max_memory_allocated()

    w_direct_numel = B * W_dim
    vram_saved = w_direct_numel * 4  # fp32

    print(f"  Forward:  PyTorch={t_pytorch_fwd*1000:.2f}ms  CUDA={t_cuda_fwd*1000:.2f}ms  "
          f"Speedup={t_pytorch_fwd/t_cuda_fwd:.1f}x")
    print(f"  Full:     PyTorch={t_pytorch_full*1000:.2f}ms  CUDA={t_cuda_full*1000:.2f}ms  "
          f"Speedup={t_pytorch_full/t_cuda_full:.1f}x")
    print(f"  VRAM:     PyTorch_peak={vram_pytorch/1e6:.1f}MB  CUDA_peak={vram_cuda/1e6:.1f}MB  "
          f"Saved={vram_saved/1e6:.1f}MB")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(0)

    print("=" * 70)
    print("BENCHMARK: fused_dynamic_conv2d vs PyTorch")
    print("=" * 70)

    # Layer 0 configs (C_in=1)
    run_benchmark(B=4,   C_in=1,   C_out=128, H=28, W=28)
    run_benchmark(B=64,  C_in=1,   C_out=128, H=28, W=28)

    # Layer 1+ configs (C_in=128)
    run_benchmark(B=4,   C_in=128, C_out=128, H=28, W=28)
    run_benchmark(B=64,  C_in=128, C_out=128, H=28, W=28)

    # CIFAR-10 (C_in=3)
    run_benchmark(B=64,  C_in=3,   C_out=64,  H=32, W=32)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
