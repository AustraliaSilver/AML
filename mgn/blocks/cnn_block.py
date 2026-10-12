import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base import BaseDynamicBlock


class PlainConvBlock(BaseDynamicBlock):
    """Plain CNN block: Conv3x3 -> GroupNorm -> ReLU with dynamic channel masking.
    
    Standard feedforward convolutional block without residual skip connections.
    """
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 1):
        super().__init__(in_channels, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm = nn.GroupNorm(norm_groups, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, dim_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        if dim_mask is not None:
            out = out * dim_mask
        return out

    def compute_params(self, active_dim: Optional[float] = None) -> float:
        d = active_dim if active_dim is not None else float(self.out_channels)
        # in_c * d * 3 * 3 + d (bias) + 2*d (norm weight & bias)
        return float(self.in_channels * d * 9 + d + 2 * d)

    def compute_flops(self, spatial_size: Tuple[int, int], active_dim: Optional[float] = None) -> float:
        h, w = spatial_size
        d = active_dim if active_dim is not None else float(self.out_channels)
        conv_flops = 2.0 * self.in_channels * d * 9 * h * w
        norm_act_flops = 4.0 * d * h * w
        return conv_flops + norm_act_flops


class ResNetBlock(BaseDynamicBlock):
    """Residual CNN block: 2x (Conv3x3 -> GroupNorm -> ReLU) with identity skip connection.
    
    If in_channels != out_channels, a 1x1 projection is applied to the skip path.
    """
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 1):
        super().__init__(in_channels, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm1 = nn.GroupNorm(norm_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm2 = nn.GroupNorm(norm_groups, out_channels)
        self.act = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(norm_groups, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, dim_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        res = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        if dim_mask is not None:
            out = out * dim_mask

        out = self.conv2(out)
        out = self.norm2(out)
        
        out = self.act(out + res)
        if dim_mask is not None:
            out = out * dim_mask
        return out

    def compute_params(self, active_dim: Optional[float] = None) -> float:
        d = active_dim if active_dim is not None else float(self.out_channels)
        c1 = float(self.in_channels * d * 9 + d + 2 * d)
        c2 = float(d * d * 9 + d + 2 * d)
        c_sc = float(self.in_channels * d) if self.in_channels != self.out_channels else 0.0
        return c1 + c2 + c_sc

    def compute_flops(self, spatial_size: Tuple[int, int], active_dim: Optional[float] = None) -> float:
        h, w = spatial_size
        d = active_dim if active_dim is not None else float(self.out_channels)
        f1 = 2.0 * self.in_channels * d * 9 * h * w + 4.0 * d * h * w
        f2 = 2.0 * d * d * 9 * h * w + 4.0 * d * h * w
        f_sc = (2.0 * self.in_channels * d * h * w) if self.in_channels != self.out_channels else 0.0
        return f1 + f2 + f_sc
