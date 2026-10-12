import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseDynamicBlock(nn.Module, ABC):
    """Abstract base class for all AML architecture candidate blocks.
    
    Guarantees consistent signature:
      forward(x: torch.Tensor, dim_mask: Optional[torch.Tensor] = None) -> torch.Tensor
    where:
      x: [B, C, H, W]
      dim_mask: [B, C, 1, 1] or [1, C, 1, 1]
      output: [B, C, H, W]
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x: torch.Tensor, dim_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process spatial feature map [B, C, H, W] and return [B, C, H, W]."""
        raise NotImplementedError

    @abstractmethod
    def compute_params(self, active_dim: Optional[float] = None) -> float:
        """Return the effective number of parameters in this block."""
        raise NotImplementedError

    @abstractmethod
    def compute_flops(self, spatial_size: Tuple[int, int], active_dim: Optional[float] = None) -> float:
        """Return estimated FLOPs for processing an input of shape (H, W)."""
        raise NotImplementedError
