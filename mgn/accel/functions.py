"""
aml_accel.functions

Tiled conv2d: weight mixing via cuBLAS + tiled CUDA conv (no im2col).
Falls back to im2col+bmm if CUDA extension unavailable.
"""

import os
import torch
import torch.nn.functional as F
from torch.autograd import Function

_cuda_available = False
try:
    import mgn.accel._C as _C
    _cuda_available = True
except Exception:
    pass


class _TiledConvFn(Function):
    """Autograd Function wrapping tiled CUDA conv kernels."""

    @staticmethod
    def forward(ctx, X, prototypes, pi, w_offset, b_offset):
        B, C_in, H, W_ = X.shape
        per_oc_weight = C_in * 9
        C_out = (b_offset - w_offset) // per_oc_weight

        # Weight mixing via cuBLAS (fast)
        W_mix = torch.mm(pi, prototypes[:, w_offset:w_offset + C_out * per_oc_weight])
        W_mix = W_mix.view(B, C_out, C_in * 9).contiguous()
        b_mix = torch.mm(pi, prototypes[:, b_offset:b_offset + C_out]).contiguous()

        # CUDA conv (implicit GEMM for deep layers, tiled for layer0)
        Y = _C.conv_fwd(W_mix, X, b_mix)

        ctx.save_for_backward(X, prototypes, pi)
        ctx.w_offset = w_offset
        ctx.b_offset = b_offset
        ctx.B = B
        ctx.C_in = C_in
        ctx.C_out = C_out
        ctx.H = H
        ctx.W_ = W_
        ctx.W_mix = W_mix
        ctx.b_mix = b_mix

        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        X, prototypes, pi = ctx.saved_tensors
        B = ctx.B
        C_in = ctx.C_in
        C_out = ctx.C_out
        H = ctx.H
        W_ = ctx.W_
        w_offset = ctx.w_offset
        b_offset = ctx.b_offset
        per_oc_weight = C_in * 9

        W_mix = ctx.W_mix
        b_mix = ctx.b_mix

        # CUDA backward: grad_X, grad_W, grad_bias
        grad_X, grad_W_flat, grad_bias = _C.conv_bwd(
            grad_Y.contiguous(), W_mix, X.contiguous()
        )

        # Chain rule through weight mixing:
        # W_mix = pi @ proto[:, w:w+C_out*per_oc_weight]  → [B, C_out*per_oc_weight]
        # b_mix = pi @ proto[:, b:b+C_out]                → [B, C_out]
        proto_weight = prototypes[:, w_offset:w_offset + C_out * per_oc_weight]
        proto_bias = prototypes[:, b_offset:b_offset + C_out]

        # dL/dpi from weights: grad_W_flat[b] @ proto_weight.T → [B, K]
        grad_pi_weight = grad_W_flat.reshape(B, -1) @ proto_weight.T
        # dL/dpi from bias: grad_bias[b] @ proto_bias.T → [B, K]
        grad_pi_bias = grad_bias @ proto_bias.T
        grad_pi = grad_pi_weight + grad_pi_bias

        # dL/dproto from weights: pi.T @ grad_W_flat → [K, C_out*per_oc_weight]
        grad_proto_weight = pi.T @ grad_W_flat.reshape(B, -1)
        # dL/dproto from bias: pi.T @ grad_bias → [K, C_out]
        grad_proto_bias = pi.T @ grad_bias

        grad_proto = torch.zeros_like(prototypes)
        grad_proto[:, w_offset:w_offset + C_out * per_oc_weight] = grad_proto_weight
        grad_proto[:, b_offset:b_offset + C_out] = grad_proto_bias

        return grad_X, grad_proto, grad_pi, None, None


def fused_dynamic_conv2d(X, prototypes, pi, w_offset, b_offset):
    """
    Dynamic conv2d: mixes prototypes weighted by pi, then applies per-sample 3x3 conv.
    Uses optimized PyTorch Grouped Convolution (groups=B) for 100% robust dynamic weights.
    """
    B, C_in, H, W_ = X.shape
    per_oc_weight = C_in * 9
    C_out = (b_offset - w_offset) // per_oc_weight

    W_mix = torch.mm(pi, prototypes[:, w_offset:w_offset + C_out * per_oc_weight])
    W_mix = W_mix.view(B * C_out, C_in, 3, 3)
    b_mix = torch.mm(pi, prototypes[:, b_offset:b_offset + C_out]).view(B * C_out)

    X_grouped = X.reshape(1, B * C_in, H, W_)
    Y = F.conv2d(X_grouped, W_mix, b_mix, padding=1, groups=B)
    return Y.view(B, C_out, H, W_)
