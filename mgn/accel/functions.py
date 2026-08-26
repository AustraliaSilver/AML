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
