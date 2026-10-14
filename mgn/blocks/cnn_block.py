import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import BaseDynamicBlock


def make_norm(norm_type: str, channels: int, norm_groups: int = 1) -> nn.Module:
    """Build the post-conv normalisation module for a block.

    A factory rather than an `if bn: ... else: GroupNorm` branch, because that
    binary branch silently swallowed every other value of `norm_type`:
    `--norm in` advertised InstanceNorm but produced GroupNorm, since the
    trainer maps `args.norm != 'bn'` to 'gn' before it ever reached here.

    'bn' is the only option with running statistics, which is why it is the only
    one that can go stale: see the note in train_continual_aml.py --arch-lr.
    """
    t = norm_type.lower()
    if t == "bn":
        return nn.BatchNorm2d(channels)
    if t in ("gn", "gn1", "group"):
        return nn.GroupNorm(norm_groups, channels)
    if t in ("in", "instance"):
        # affine=False: the conv already carries a bias for every non-'bn' norm,
        # so a second affine pair would be redundant parameters.
        return nn.InstanceNorm2d(channels, affine=False)
    raise ValueError(
        f"unknown norm_type {norm_type!r}; expected one of bn, gn1, in")


def conv_has_bias(norm_type: str) -> bool:
    """BatchNorm supplies its own shift, so the conv must not also have a bias."""
    return norm_type.lower() != "bn"


class PlainConvBlock(BaseDynamicBlock):
    """Plain CNN block: Conv3x3 -> Norm -> ReLU with dynamic channel masking.
    
    Standard feedforward convolutional block without residual skip connections.
    Supports norm_type 'bn' (cuDNN fused BatchNorm), 'gn1' (GroupNorm) and
    'in' (InstanceNorm).
    """
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "bn", norm_groups: int = 1):
        super().__init__(in_channels, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                              bias=conv_has_bias(norm_type))
        self.norm = make_norm(norm_type, out_channels, norm_groups)
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
        return float(self.in_channels * d * 9 + d + 2 * d)

    def compute_flops(self, spatial_size: Tuple[int, int], active_dim: Optional[float] = None) -> float:
        h, w = spatial_size
        d = active_dim if active_dim is not None else float(self.out_channels)
        conv_flops = 2.0 * self.in_channels * d * 9 * h * w
        norm_act_flops = 4.0 * d * h * w
        return conv_flops + norm_act_flops


class ResNetBlock(BaseDynamicBlock):
    """Residual CNN block: 2x (Conv3x3 -> Norm -> ReLU) with identity skip connection.
    
    Supports norm_type 'bn' (cuDNN fused BatchNorm), 'gn1' (GroupNorm) and 'in'
    (InstanceNorm).
    """
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "bn", norm_groups: int = 1):
        super().__init__(in_channels, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                               bias=conv_has_bias(norm_type))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                               bias=conv_has_bias(norm_type))
        self.norm1 = make_norm(norm_type, out_channels, norm_groups)
        self.norm2 = make_norm(norm_type, out_channels, norm_groups)
        self.act = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                make_norm(norm_type, out_channels, norm_groups)
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
