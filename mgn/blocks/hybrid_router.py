import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import BaseDynamicBlock
from .cnn_block import PlainConvBlock, ResNetBlock
from .vit_block import VisionTransformerBlock


class HybridBlock(BaseDynamicBlock):
    """Dynamic Hybrid Layer Block supporting 3 operational topologies:
      Mode 0: PlainConvBlock (VGG-style local spatial)
      Mode 1: ResNetBlock (Deep residual local spatial)
      Mode 2: VisionTransformerBlock (Global self-attention)
      
    Execution mixes the candidates according to continuous mode routing weights:
      out = mode_weights[0] * plain(x) + mode_weights[1] * resnet(x) + mode_weights[2] * vit(x)
    Supports soft DARTS training (differentiable bi-level) and hard one-hot argmax inference.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        norm_groups: int = 1
    ):
        super().__init__(in_channels, out_channels)
        self.plain = PlainConvBlock(in_channels, out_channels, norm_groups=norm_groups)
        self.resnet = ResNetBlock(in_channels, out_channels, norm_groups=norm_groups)
        self.vit = VisionTransformerBlock(in_channels, out_channels, num_heads=num_heads)

    def forward(
        self,
        x: torch.Tensor,
        mode_weights: Optional[torch.Tensor] = None,
        dim_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: [B, C_in, H, W]
        mode_weights: [B, 3] or [1, 3] (probabilities for [plain, resnet, vit])
        dim_mask: [B, C_out, 1, 1]
        """
        if mode_weights is None:
            # Default to resnet if unrouted
            return self.resnet(x, dim_mask=dim_mask)

        # Optimization: If one-hot / hard routed, execute ONLY the chosen block
        if mode_weights.ndim == 2 and (mode_weights == 1.0).any():
            # Check if entire batch shares the same hard choice
            mode_sum = mode_weights.sum(dim=0)
            B = x.shape[0]
            if mode_sum[0] == B:
                return self.plain(x, dim_mask=dim_mask)
            elif mode_sum[1] == B:
                return self.resnet(x, dim_mask=dim_mask)
            elif mode_sum[2] == B:
                return self.vit(x, dim_mask=dim_mask)

        # Soft continuous mixture for differentiable architecture search (DARTS)
        w_plain = mode_weights[:, 0].view(-1, 1, 1, 1)
        w_resnet = mode_weights[:, 1].view(-1, 1, 1, 1)
        w_vit = mode_weights[:, 2].view(-1, 1, 1, 1)

        out_plain = self.plain(x, dim_mask=dim_mask) if (w_plain > 1e-4).any() else 0.0
        out_resnet = self.resnet(x, dim_mask=dim_mask) if (w_resnet > 1e-4).any() else 0.0
        out_vit = self.vit(x, dim_mask=dim_mask) if (w_vit > 1e-4).any() else 0.0

        return w_plain * out_plain + w_resnet * out_resnet + w_vit * out_vit

    def compute_params(self, mode_idx: int = 1, active_dim: Optional[float] = None) -> float:
        if mode_idx == 0:
            return self.plain.compute_params(active_dim)
        elif mode_idx == 1:
            return self.resnet.compute_params(active_dim)
        elif mode_idx == 2:
            return self.vit.compute_params(active_dim)
        else:
            raise ValueError(f"Unknown mode index {mode_idx}")

    def compute_flops(
        self,
        spatial_size: Tuple[int, int],
        mode_idx: int = 1,
        active_dim: Optional[float] = None
    ) -> float:
        if mode_idx == 0:
            return self.plain.compute_flops(spatial_size, active_dim)
        elif mode_idx == 1:
            return self.resnet.compute_flops(spatial_size, active_dim)
        elif mode_idx == 2:
            return self.vit.compute_flops(spatial_size, active_dim)
        else:
            raise ValueError(f"Unknown mode index {mode_idx}")
