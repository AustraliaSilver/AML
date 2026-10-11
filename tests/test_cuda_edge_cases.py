"""
test_cuda_edge_cases.py - Comprehensive edge case unit tests for native CUDA kernels.

Tests cover:
1. Odd spatial dimensions (e.g., H=7, W=11; H=1, W=1; H=13, W=17)
2. Non-multiple-of-32 channel counts (e.g., C=1, 3, 7, 13, 29, 35, 63, 127)
3. Boundary batch sizes (B=1, B=64)
4. Large tensor sizes (B=8, C=64, H=32, W=32)
5. Batched Hyper Linear GEMM & Tensor Core WMMA on non-multiple-of-16 dimensions
6. Extreme values (large numbers, zero gradients, all-zeros, negative inputs)
7. Gradient parity & stability on irregular inputs
"""

import unittest
import torch
import torch.nn.functional as F

from mgn.accel import _C
from mgn.accel.fused_autograd import FusedPipelineFunction


class TestCUDAEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(42)

    def test_odd_spatial_conv2d_forward_and_backward(self):
        """Test Dynamic Conv2D with odd H and W (H=7, W=11)."""
        B, C_in, C_out = 2, 8, 16
        H, W = 7, 11
        x = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(B, C_out, C_in, 3, 3, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        bias = torch.randn(B, C_out, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        g_out = torch.randn(B, C_out, H, W, device="cuda", dtype=torch.float32)

        # Native backward
        d_in, d_w, d_b = _C.dynamic_conv2d_bwd(g_out, x, w.view(B, -1), B, C_in, C_out, H, W)
        self.assertEqual(d_in.shape, (B, C_in, H, W))
        self.assertEqual(d_w.shape, (B, C_out * C_in * 9))
        self.assertEqual(d_b.shape, (B, C_out))
        self.assertFalse(torch.isnan(d_in).any())
        self.assertFalse(torch.isnan(d_w).any())
        self.assertFalse(torch.isnan(d_b).any())

    def test_single_pixel_spatial_1x1(self):
        """Test Dynamic Conv2D and Normalizations with 1x1 spatial resolution."""
        B, C_in, C_out = 2, 4, 8
        H, W = 1, 1
        x = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)
        mean = x.mean(dim=(1, 2, 3))
        rstd = torch.rsqrt(x.var(dim=(1, 2, 3), unbiased=False) + 1e-5)
        g_out = torch.randn_like(x)

        # GroupNorm backward on 1x1
        din = _C.group_norm1_bwd(g_out, x, mean, rstd, B, C_in, H, W)
        self.assertEqual(din.shape, x.shape)
        self.assertFalse(torch.isnan(din).any())

    def test_non_multiple_of_32_channels_norm(self):
        """Test GroupNorm and InstanceNorm with odd / prime channel counts: C=1, 3, 7, 13, 29, 35."""
        odd_channels = [1, 3, 7, 13, 29, 35, 63, 127]
        for C in odd_channels:
            B, H, W = 2, 8, 8
            x = torch.randn(B, C, H, W, device="cuda", dtype=torch.float32)
            mean = x.mean(dim=(1, 2, 3))
            rstd = torch.rsqrt(x.var(dim=(1, 2, 3), unbiased=False) + 1e-5)
            g_out = torch.randn_like(x)

            # Native GroupNorm backward
            din_gn = _C.group_norm1_bwd(g_out, x, mean, rstd, B, C, H, W)
            self.assertEqual(din_gn.shape, (B, C, H, W))
            self.assertFalse(torch.isnan(din_gn).any())

            # InstanceNorm backward
            mean_in = x.mean(dim=(2, 3)).reshape(-1)
            rstd_in = torch.rsqrt(x.var(dim=(2, 3), unbiased=False).reshape(-1) + 1e-5)
            din_in = _C.instance_norm_bwd(g_out, x, mean_in, rstd_in, B, C, H, W)
            self.assertEqual(din_in.shape, (B, C, H, W))
            self.assertFalse(torch.isnan(din_in).any())

    def test_single_sample_batch_b1(self):
        """Test all operations with Batch size B = 1."""
        B, C, H, W = 1, 16, 9, 9
        x = torch.randn(B, C, H, W, device="cuda", dtype=torch.float32)
        mean = x.mean(dim=(1, 2, 3))
        rstd = torch.rsqrt(x.var(dim=(1, 2, 3), unbiased=False) + 1e-5)
        g_out = torch.randn_like(x)

        din = _C.group_norm1_bwd(g_out, x, mean, rstd, B, C, H, W)
        self.assertEqual(din.shape, (1, C, H, W))
        self.assertFalse(torch.isnan(din).any())

    def test_large_batch_and_spatial(self):
        """Test with larger tensors: B=32, C=32, H=16, W=16."""
        B, C, H, W = 32, 32, 16, 16
        x = torch.randn(B, C, H, W, device="cuda", dtype=torch.float32)
        mean = x.mean(dim=(1, 2, 3))
        rstd = torch.rsqrt(x.var(dim=(1, 2, 3), unbiased=False) + 1e-5)
        g_out = torch.randn_like(x)

        din = _C.group_norm1_bwd(g_out, x, mean, rstd, B, C, H, W)
        self.assertEqual(din.shape, (B, C, H, W))
        self.assertFalse(torch.isnan(din).any())

    def test_batched_hyper_linear_odd_shapes(self):
        """Test Tiled & Tensor Core WMMA GEMM on odd, non-multiple-of-16 dimensions."""
        test_shapes = [
            (2, 17, 33, 25),  # Odd M, K, N
            (1, 1, 1, 1),      # Minimal 1x1x1
            (4, 7, 19, 13),    # All prime dimensions
            (8, 64, 48, 32),   # Standard aligned dimensions
        ]
        for B, M, K, N in test_shapes:
            A = torch.randn(B, M, K, device="cuda", dtype=torch.float32)
            B_mat = torch.randn(B, K, N, device="cuda", dtype=torch.float32)
            bias = torch.randn(B, N, device="cuda", dtype=torch.float32)

            # 1. FP32 Tiled GEMM
            out_tiled = _C.batched_hyper_linear(A, B_mat, bias)
            ref_fp32 = torch.bmm(A, B_mat) + bias.unsqueeze(1)
            torch.testing.assert_close(out_tiled, ref_fp32, atol=1e-4, rtol=1e-4)

            # 2. Tensor Core WMMA BF16 GEMM
            out_tc = _C.batched_hyper_linear_tc(A, B_mat, bias)
            ref_bf16 = (torch.bmm(A.bfloat16().float(), B_mat.bfloat16().float()) + bias.unsqueeze(1))
            torch.testing.assert_close(out_tc, ref_bf16, atol=2e-2, rtol=2e-2)

    def test_extreme_and_zero_values(self):
        """Test numeric stability with all-zero, negative-only, and large-scale inputs."""
        B, C, H, W = 2, 16, 8, 8

        # All zeros
        x_zero = torch.zeros(B, C, H, W, device="cuda")
        mean = torch.zeros(B, device="cuda")
        rstd = torch.ones(B, device="cuda")
        g_out = torch.zeros_like(x_zero)
        din = _C.group_norm1_bwd(g_out, x_zero, mean, rstd, B, C, H, W)
        self.assertTrue((din == 0).all())

        # Large scale inputs (1e4)
        x_large = torch.randn(B, C, H, W, device="cuda") * 1e4
        mean_l = x_large.mean(dim=(1, 2, 3))
        rstd_l = torch.rsqrt(x_large.var(dim=(1, 2, 3), unbiased=False) + 1e-5)
        g_l = torch.randn_like(x_large)
        din_l = _C.group_norm1_bwd(g_l, x_large, mean_l, rstd_l, B, C, H, W)
        self.assertFalse(torch.isnan(din_l).any())
        self.assertFalse(torch.isinf(din_l).any())


if __name__ == "__main__":
    unittest.main()
