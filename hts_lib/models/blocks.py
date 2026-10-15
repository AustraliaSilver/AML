import torch.nn as nn
import torch.nn.functional as F


def _valid_groups(num_channels: int, preferred: int, cap: int) -> int:
    """`preferred` if it divides `num_channels`, else the largest divisor <= cap.

    GroupNorm requires num_channels % num_groups == 0. `min(8, out_d // 4)`
    returned 7 for out_d=30 (30 % 7 != 0 -> "num_channels must be divisible by
    num_groups") and 0 for out_d < 4.
    """
    if preferred >= 1 and num_channels % preferred == 0:
        return preferred
    for g in range(min(cap, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1  # unreachable: 1 always divides


def _norm_groups(num_channels: int, cap: int = 8) -> int:
    """GroupNorm groups for the stage blocks (gn2 and the 1x1 shortcut).

    Keeps the historical `min(cap, C // 4)` whenever it divides, so every
    default width is unchanged (24/48/96 -> 6/8/8).
    """
    return _valid_groups(num_channels, min(cap, num_channels // 4), cap)


def _stem_groups(num_channels: int, cap: int = 4) -> int:
    """GroupNorm groups matching gn1 and the native stems.

    Both `HtSResidualBlock.gn1` and `HtSContinualModel`'s stems were built with
    a fixed `GroupNorm(4, C)`. Keep exactly that while 4 divides C (the default
    widths 24/48/96 all do) and fall back to a divisor otherwise. Preferring
    `min(4, C // 4)` -- the pattern used for gn2 -- would silently change C=8/12
    from 4 groups to 2/3.
    """
    return _valid_groups(num_channels, cap, cap)


class HtSResidualBlock(nn.Module):
    """
    Lightweight Child Residual Block receiving dynamic kernels synthesized on-the-fly.
    
    Structure:
    - Main Convolutional Path (3x3): 100% dynamic, generated on-the-fly from HtS Basis.
    - Transition Shortcut Path (1x1): Lightweight static projection used only when downsampling (stride > 1 or in_d != out_d).
    - Normalization: GroupNorm parameters.
    """
    def __init__(self, in_d, out_d, stride=1):
        super().__init__()
        self.stride = stride
        self.in_d = in_d
        self.out_d = out_d
        
        self.gn1 = nn.GroupNorm(_stem_groups(in_d), in_d)
        self.gn2 = nn.GroupNorm(_norm_groups(out_d), out_d)
        
        if stride != 1 or in_d != out_d:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_d, out_d, 1, stride=stride, bias=False),
                nn.GroupNorm(_norm_groups(out_d), out_d)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, kernel):
        """
        Executes dynamic convolution with sample-specific weights.
        Args:
            x: Input tensor [B, in_d, H, W]
            kernel: Batch-specific kernel tensor [B, out_d, in_d, 3, 3]
        """
        B, C, H, W = x.shape
        res = F.gelu(self.gn1(x))
        
        if self.stride == 1:
            try:
                from mgn.accel.functions import native_dynamic_conv2d
                out = native_dynamic_conv2d(res, kernel)
            except Exception:
                x_r = res.reshape(1, B * self.in_d, H, W)
                k_r = kernel.reshape(B * self.out_d, self.in_d, 3, 3)
                out = F.conv2d(x_r, k_r, stride=1, padding=1, groups=B)
                out = out.reshape(B, self.out_d, H, W)
        else:
            x_r = res.reshape(1, B * self.in_d, H, W)
            k_r = kernel.reshape(B * self.out_d, self.in_d, 3, 3)
            out = F.conv2d(x_r, k_r, stride=self.stride, padding=1, groups=B)
            out = out.reshape(B, self.out_d, out.shape[2], out.shape[3])

        out = self.gn2(out)
        return F.gelu(out + self.shortcut(x))

