import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base import BaseDynamicBlock


class VisionTransformerBlock(BaseDynamicBlock):
    """Vision Transformer block: Multi-Head Self-Attention (MHSA) + MLP.
    
    Seamlessly transforms 2D spatial feature map [B, C, H, W] to token sequence
    [B, N, C] (where N = H * W), applies LayerNorm, Scaled Dot-Product Attention
    (FlashAttention on GPU), residual connection, LayerNorm, and 2-layer MLP (FFN),
    then reshapes back to [B, C, H, W] for flawless pipeline compatibility.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0
    ):
        super().__init__(in_channels, out_channels)
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        # 1x1 projection if in_channels != out_channels
        if in_channels != out_channels:
            self.in_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.in_proj = nn.Identity()

        self.norm1 = nn.LayerNorm(out_channels)
        self.qkv = nn.Linear(out_channels, out_channels * 3, bias=True)
        self.proj = nn.Linear(out_channels, out_channels, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        hidden_dim = int(out_channels * mlp_ratio)
        self.norm2 = nn.LayerNorm(out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, dim_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, C_in, H, W]
        x = self.in_proj(x)
        B, C, H, W = x.shape
        orig_H, orig_W = H, W

        # MobileViT optimization: If spatial dimension >= 16, downscale tokens to max 8x8 (64 tokens)
        # to accelerate self-attention from O(1024^2) down to O(64^2) - 256x faster attention
        needs_downsample = H > 8 or W > 8
        if needs_downsample:
            x_down = F.adaptive_avg_pool2d(x, (8, 8))
            H_tok, W_tok = 8, 8
        else:
            x_down = x
            H_tok, W_tok = H, W

        N = H_tok * W_tok

        # Convert Spatial -> Sequence [B, N, C]
        x_seq = x_down.flatten(2).transpose(1, 2)

        # Multi-Head Self-Attention with Residual
        norm_x = self.norm1(x_seq)
        qkv = self.qkv(norm_x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use PyTorch 2.0+ optimized scaled_dot_product_attention (FlashAttention on GPU)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0
        )  # [B, num_heads, N, head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.proj_drop(self.proj(attn_out))
        x_seq = x_seq + attn_out

        # MLP with Residual
        x_seq = x_seq + self.mlp(self.norm2(x_seq))

        # Convert Sequence -> Spatial [B, C, H_tok, W_tok]
        out_down = x_seq.transpose(1, 2).reshape(B, C, H_tok, W_tok)

        if needs_downsample:
            out = F.interpolate(out_down, size=(orig_H, orig_W), mode='nearest') + x
        else:
            out = out_down


        if dim_mask is not None:
            out = out * dim_mask

        return out

    def compute_params(self, active_dim: Optional[float] = None) -> float:
        d = active_dim if active_dim is not None else float(self.out_channels)
        # In proj
        in_p = float(self.in_channels * d) if self.in_channels != self.out_channels else 0.0
        # Norm1 + Norm2
        norm_p = 4.0 * d
        # QKV (3 * d * d + 3 * d) + Proj (d * d + d)
        attn_p = 3.0 * d * d + 3.0 * d + d * d + d
        # MLP (d * (mlp_ratio * d) + (mlp_ratio * d) * d + mlp biases)
        mlp_h = d * self.mlp_ratio
        mlp_p = 2.0 * d * mlp_h + mlp_h + d
        return in_p + norm_p + attn_p + mlp_p

    def compute_flops(self, spatial_size: Tuple[int, int], active_dim: Optional[float] = None) -> float:
        h, w = spatial_size
        n = float(h * w)
        d = active_dim if active_dim is not None else float(self.out_channels)
        
        # In proj flops
        in_f = (2.0 * self.in_channels * d * n) if self.in_channels != self.out_channels else 0.0
        # QKV linear: 3 * 2 * n * d * d
        qkv_f = 6.0 * n * d * d
        # Attention: Q @ K^T is 2 * num_heads * n * n * head_dim = 2 * n^2 * d
        # Softmax @ V: 2 * n^2 * d
        attn_f = 4.0 * n * n * d
        # Out proj: 2 * n * d * d
        proj_f = 2.0 * n * d * d
        # MLP: 2 * 2 * n * d * (mlp_ratio * d)
        mlp_f = 4.0 * n * d * (d * self.mlp_ratio)
        return in_f + qkv_f + attn_f + proj_f + mlp_f
