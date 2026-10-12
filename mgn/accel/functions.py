"""
aml_accel.functions

Dynamic conv2d with per-sample prototype-mixed weights.
Uses optimized PyTorch Grouped Convolution (groups=B): robust and fast
enough that the pure-PyTorch path is the shipped implementation; the
fused CUDA kernels in mgn.accel._C cover the full-pipeline case instead
(see fused_autograd.py / train_cv_cifar_18.py).
"""

import torch
import torch.nn.functional as F


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


class NativeDynamicConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias=None):
        B, C_in, H, W = x.shape
        C_out = w.shape[1]
        ctx.save_for_backward(x, w)
        ctx.has_bias = (bias is not None)
        ctx.dims = (B, C_in, C_out, H, W)

        from mgn.accel import _C
        out = _C.dynamic_conv2d_fwd(x, w, bias)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        B, C_in, C_out, H, W = ctx.dims

        from mgn.accel import _C
        d_in, d_w, d_b = _C.dynamic_conv2d_bwd(
            grad_output.contiguous(),
            x.contiguous(),
            w.reshape(B, -1).contiguous(),
            B, C_in, C_out, H, W
        )
        d_w = d_w.reshape(B, C_out, C_in, 3, 3)
        grad_bias = d_b if ctx.has_bias else None
        return d_in, d_w, grad_bias


def native_dynamic_conv2d(x, w, bias=None):
    """
    Fast path for sample-specific dynamic conv2d (stride=1, padding=1, 3x3).
    Falls back gracefully to grouped convolution if CUDA or native _C is unavailable.
    """
    if x.is_cuda and w.is_cuda:
        try:
            from mgn.accel import _C
            if hasattr(_C, "dynamic_conv2d_fwd") and hasattr(_C, "dynamic_conv2d_bwd"):
                return NativeDynamicConv2dFunction.apply(x, w, bias)
        except Exception:
            pass

    B, C_in, H, W = x.shape
    C_out = w.shape[1]
    x_grouped = x.reshape(1, B * C_in, H, W)
    w_flat = w.reshape(B * C_out, C_in, 3, 3)
    b_flat = bias.reshape(B * C_out) if bias is not None else None
    out = F.conv2d(x_grouped, w_flat, b_flat, padding=1, groups=B)
    return out.view(B, C_out, H, W)

