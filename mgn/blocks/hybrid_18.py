"""18-Layer Hybrid AML Supernet Architecture (CNN + ResNet + ViT).

Modular 18-layer dynamic network with continuous / hard topological routing across:
  Mode 0: PlainConvBlock (VGG-style local spatial convolution)
  Mode 1: ResNetBlock (Deep residual local spatial convolution)
  Mode 2: VisionTransformerBlock (Global self-attention with FlashAttention)

Features:
  - 18 Dynamic Hybrid layers with cuDNN fused BatchNorm2d.
  - Spatial pooling at layers 5 and 11 (3 stages: 6 layers @ 32x32, 6 @ 16x16, 6 @ 8x8).
  - Dynamic depth gating (layer_soft & layer_gates) for runtime depth skipping.
  - Dynamic channel width masking.
  - Dynamic topology prototypes [K, 18, 3] for Plain, ResNet, and ViT selection.
  - Proto-isolated linear classification head and layer modulation for Zero-Forgetting continual learning.
  - Full compatibility with ProtoIsolator, ReplayBuffer, int8 snapshotting, and domain slot routing.
"""

import math
from typing import Optional, Tuple, Dict, Any, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid_router import HybridBlock
from mgn.generator import DistanceRouter


class DynamicStructure(NamedTuple):
    layer_gates: torch.Tensor
    dim_weights: torch.Tensor
    dim_masks: torch.Tensor
    block_mode: torch.Tensor
    max_layers: int
    max_dim: int
    output_dim: int


class ConvAML_18Layer_Hybrid(nn.Module):
    """18-Layer Hybrid AML Network with CNN, ResNet, and Vision Transformer candidate blocks."""
    def __init__(
        self,
        K: int = 12,
        latent_dim: int = 512,
        max_layers: int = 18,
        max_dim: int = 48,
        num_classes: int = 10,
        num_heads: int = 4,
        norm_type: str = "bn"
    ):
        super().__init__()
        self.K = K
        self.latent_dim = latent_dim
        self.max_layers = max_layers
        self.max_dim = max_dim
        self.num_classes = num_classes
        self.num_heads = num_heads

        self.head_dim = max_dim * 16

        # Ultra-fast GPU Vision Encoder: Single conv stem + pool to eliminate redundant deep conv passes
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, max_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(max_dim * 4, latent_dim),
            nn.ReLU(inplace=True)
        )

        # Prototype selector: Maps latent z to prototype distribution pi [B, K]
        self.prototype_selector = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            DistanceRouter(256, K)
        )

        # Structure prototypes: Layer depth gates and dynamic channel width fractions
        self.structure_prototypes = nn.Parameter(torch.zeros(K, max_layers + (max_layers * max_dim)))
        
        # Block mode prototypes: [K, max_layers, 3] for [Plain CNN, ResNet, ViT]
        self.block_mode_prototypes = nn.Parameter(torch.zeros(K, max_layers, 3))
        with torch.no_grad():
            # 3-Stage Hybrid Hierarchy (CoAtNet / MobileViT architectural principle):
            # Stage 1 (Layers 0..5 @ 32x32): ResNet + Plain CNN dominant, ViT suppressed (prevents downsampling blur)
            self.block_mode_prototypes.data[:, 0:6, 0] = 0.5   # Plain
            self.block_mode_prototypes.data[:, 0:6, 1] = 2.0   # ResNet
            self.block_mode_prototypes.data[:, 0:6, 2] = -3.0  # ViT

            # Stage 2 (Layers 6..11 @ 16x16): ResNet + Plain CNN dominant
            self.block_mode_prototypes.data[:, 6:12, 0] = 0.5  # Plain
            self.block_mode_prototypes.data[:, 6:12, 1] = 2.0  # ResNet
            self.block_mode_prototypes.data[:, 6:12, 2] = -1.0 # ViT

            # Stage 3 (Layers 12..17 @ 8x8): Vision Transformer + ResNet dominant
            self.block_mode_prototypes.data[:, 12:18, 0] = -1.0 # Plain
            self.block_mode_prototypes.data[:, 12:18, 1] = 1.0  # ResNet
            self.block_mode_prototypes.data[:, 12:18, 2] = 2.5  # ViT

        # Stem convolution (3 -> max_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(3, max_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max_dim),
            nn.ReLU(inplace=True)
        )

        # 18 Dynamic Hybrid Blocks (6 @ 32x32, 6 @ 16x16, 6 @ 8x8)
        self.layers = nn.ModuleList([
            HybridBlock(
                in_channels=max_dim,
                out_channels=max_dim,
                num_heads=num_heads,
                norm_type=norm_type
            )
            for _ in range(max_layers)
        ])

        # Feature pool (4x4 spatial pooling with LayerNorm for rich gradient flow)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.ln = nn.LayerNorm(self.head_dim)

        # Prototype weights: Linear classifier weights (wo, bo) isolated per task for Zero-Forgetting!
        self.wo_size = num_classes * self.head_dim
        self.bo_size = num_classes
        self.total_proto_dim = self.wo_size + self.bo_size

        self.proto_weights = nn.Parameter(torch.zeros(K, self.total_proto_dim))
        self._init_prototypes()

    def _init_prototypes(self):
        with torch.no_grad():
            std = math.sqrt(2.0 / self.head_dim)
            self.proto_weights.data[:, :self.wo_size].normal_(0.0, std)
            self.proto_weights.data[:, self.wo_size:].zero_()


    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
        struct_override: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pi_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, DynamicStructure, torch.Tensor, Dict[str, Any]]:
        B = x.shape[0]
        device = x.device

        # 1. Vision encoder pass to obtain latent vector z
        z = self.vision_encoder(x)

        # 2. Prototype routing
        pi_logits = self.prototype_selector(z)
        if pi_mask is not None:
            if pi_mask.dtype == torch.bool:
                pi_logits = pi_logits.masked_fill(~pi_mask.unsqueeze(0), -1e4)
            else:
                pi_logits = pi_logits.masked_fill((pi_mask == 0).unsqueeze(0), -1e4)

        if self.training:
            pi = F.gumbel_softmax(pi_logits, tau=tau, hard=False, dim=-1)
        else:
            pi = F.one_hot(pi_logits.argmax(-1), self.K).float()

        # 3. Dynamic architecture topology & depth derivation
        delta = pi @ self.structure_prototypes
        layer_logits = delta[:, :self.max_layers] + 2.0
        layer_soft = torch.sigmoid(layer_logits)
        if self.training:
            layer_soft = layer_soft + 0.05 * torch.randn_like(layer_soft)
        layer_soft = torch.clamp(layer_soft, 1e-4, 1.0)

        # Channel width fraction
        dim_logits = delta[:, self.max_layers:].view(B, self.max_layers, self.max_dim)
        d_logit = dim_logits.mean(dim=-1, keepdim=True) + 2.0
        dim_fractions = torch.sigmoid(d_logit)
        d_lengths = torch.clamp(dim_fractions * self.max_dim, min=16.0)
        arange = torch.arange(self.max_dim, device=device).float().view(1, 1, self.max_dim)
        dim_masks = torch.sigmoid(10.0 * (d_lengths - arange))

        # Mode logits for Plain, ResNet, ViT [B, max_layers, 3]
        bm_logits = (pi @ self.block_mode_prototypes.view(self.K, -1)).view(B, self.max_layers, 3)

        if struct_override is not None:
            layer_gates, block_mode = struct_override
            layer_gates = layer_gates.to(device=device, dtype=torch.float32)
            block_mode = block_mode.to(device=device, dtype=torch.float32)
            if block_mode.shape[-1] == 2:
                # Handle legacy 2-mode slot: [Plain, ResNet] -> [Plain, ResNet, 0]
                vit_zeros = torch.zeros(B, self.max_layers, 1, device=device)
                block_mode = torch.cat([block_mode, vit_zeros], dim=-1)
        else:
            layer_gates = (layer_soft > 0.5).float().detach() + layer_soft - layer_soft.detach()
            if self.training:
                # Batch-consistent Gumbel-Softmax: All samples in a batch select the same mode per layer,
                # triggering the fast path in HybridBlock (executing ONLY 1 branch instead of 3 branches),
                # yielding a 3x speedup while preserving full straight-through gradients!
                mean_bm_logits = bm_logits.mean(dim=0, keepdim=True)
                hard_mode = F.gumbel_softmax(mean_bm_logits, tau=tau, hard=True, dim=-1)
                block_mode = ((hard_mode - mean_bm_logits.detach()) + mean_bm_logits).expand(B, -1, -1)
            else:
                block_mode = F.one_hot(bm_logits.argmax(dim=-1), 3).float()


        dim_weights = d_lengths / self.max_dim

        struct = DynamicStructure(
            layer_gates=layer_gates,
            dim_weights=dim_weights,
            dim_masks=dim_masks,
            block_mode=block_mode,
            max_layers=self.max_layers,
            max_dim=self.max_dim,
            output_dim=self.num_classes
        )

        # 4. Supernet Forward Pass through 18 Hybrid Layers
        h = self.stem(x)

        for i, layer in enumerate(self.layers):
            w_mode = block_mode[:, i]  # [B, 3]
            d_mask = dim_masks[:, i].view(B, self.max_dim, 1, 1)
            gate_c = layer_gates[:, i:i+1].view(B, 1, 1, 1)

            # Apply candidate block with fast path
            h_cand = layer(h, mode_weights=w_mode, dim_mask=d_mask)

            # Dynamic depth gating (soft/hard skip)
            if i == 0:
                h = h_cand
            else:
                h = gate_c * h_cand + (1.0 - gate_c) * h

            # Spatial downsampling at layer 5 (-> 16x16) and layer 11 (-> 8x8)
            if i in (5, 11):
                h = F.max_pool2d(h, kernel_size=2, stride=2)

        # 5. Classifier Head
        h_pool = self.ln(self.pool(h).flatten(1))  # [B, head_dim]
        h_pool = F.dropout(h_pool, p=0.1, training=self.training)

        wo = (pi @ self.proto_weights[:, :self.wo_size]).view(B, self.num_classes, self.head_dim)
        bo = pi @ self.proto_weights[:, self.wo_size:self.wo_size + self.bo_size]
        out = (h_pool.unsqueeze(1) @ wo.transpose(1, 2)).squeeze(1) + bo


        diag = {
            "layer_soft": layer_soft,
            "bm_logits": bm_logits,
            "d_lengths": d_lengths,
            "z": z,
        }
        return out, struct, pi, diag
