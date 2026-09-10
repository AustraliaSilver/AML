from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F


@dataclass
class ModelStructure:
    layer_gates: torch.Tensor
    dim_weights: torch.Tensor
    residual: torch.Tensor
    dim_masks: Optional[torch.Tensor] = None
    block_mode: Optional[torch.Tensor] = None  # [batch, max_layers, 2] -> 0: PlainConv, 1: ResNetBlock
    input_dim: int = 2
    output_dim: int = 2
    max_dim: int = 32
    max_layers: int = 3
    dim_choices: Tuple[float, ...] = (8, 16, 32)
    use_conv: bool = False
    in_channels: int = 1

    @property
    def num_layers(self) -> torch.Tensor:
        return self.layer_gates.sum(dim=-1)

    @property
    def hidden_dim(self) -> torch.Tensor:
        if self.dim_weights.shape[-1] == 1:
            # Continuous mode
            return self.dim_weights.squeeze(-1) * self.max_dim
        else:
            choices = torch.tensor(self.dim_choices, device=self.dim_weights.device, dtype=torch.float32)
            return (self.dim_weights * choices).sum(dim=-1)

    def effective_params_formula(self) -> torch.Tensor:
        L = self.num_layers
        D = self.hidden_dim
        D_mean = D.mean(dim=-1) if D.dim() > 1 else D
        if getattr(self, "use_conv", False):
            c1 = getattr(self, "in_channels", 1) * D_mean * 9 + D_mean
            c2 = torch.clamp(L - 1.0, min=0.0) * (D_mean * D_mean * 9 + D_mean)
            # Classifier consumes 16*D features (AdaptiveAvgPool2d((4,4)))
            out = (16 * D_mean) * self.output_dim + self.output_dim
            return c1 + c2 + out
        else:
            weights = self.input_dim * D_mean + (L - 1) * D_mean * D_mean + D_mean * self.output_dim
            bias = L * D_mean + self.output_dim
            return weights + bias

    def flops_per_sample(self) -> torch.Tensor:
        L = self.num_layers
        D = self.hidden_dim
        D_mean = D.mean(dim=-1) if D.dim() > 1 else D
        if getattr(self, "use_conv", False):
            # CIFAR-32 input, max-pool after layers 5 and 11:
            # stage spatial sizes 32^2 / 16^2 / 8^2
            sp = (torch.clamp(L, max=5.0) * 1024.0
                  + torch.clamp(L - 5.0, min=0.0, max=6.0) * 256.0
                  + torch.clamp(L - 11.0, min=0.0) * 64.0)
            conv_flops = 9.0 * D_mean * D_mean * sp
            head_flops = (16.0 * D_mean) * self.output_dim
            return conv_flops + head_flops
        return self.input_dim * D_mean + (L - 1) * D_mean * D_mean + D_mean * self.output_dim

    def memory_bytes(self) -> torch.Tensor:
        return self.effective_params_formula() * 4


def project_architecture(S: ModelStructure, P_max: float):
    D = S.hidden_dim.detach()          # [B, L] in BOTH modes
    P = S.effective_params_formula().detach()   # [B]
    mask = P > P_max
    if not mask.any():
        return S
    scale = torch.sqrt(P_max / (P + 1e-8)).clamp(max=1.0)   # [B]
    D_proj = D.reshape(P.shape[0], -1) * scale.unsqueeze(-1)  # [B, L]
    keep = (~mask).reshape(-1, *([1] * (S.dim_weights.dim() - 1)))
    if S.dim_weights.dim() == 3 and S.dim_weights.shape[-1] == 1:
        # Continuous mode: dim_weights stores WIDTH FRACTIONS [B, L, 1].
        # (The old code scattered a discrete one-hot of shape [B, nd] into
        # this [B, L, 1] tensor -> shape error / silent wrong broadcast.)
        D_new = (D_proj / S.max_dim).reshape(S.dim_weights.shape)
    else:
        # Discrete mode: hidden_dim is per-LAYER [B, L]; snap each layer's
        # projected width to the nearest choice -> one-hot [B, L, nd].
        d_choices = torch.tensor(S.dim_choices, device=D.device, dtype=torch.float32)
        D_idx = (d_choices.view(1, 1, -1) - D_proj.unsqueeze(-1)).abs().argmin(dim=-1)
        D_new = F.one_hot(D_idx, num_classes=len(S.dim_choices)).to(S.dim_weights.dtype)
    S.dim_weights = torch.where(keep, S.dim_weights, D_new)
    return S


@dataclass
class GeneratedModel:
    structure: ModelStructure
    weights: Dict[str, torch.Tensor]
    embedding: Optional[torch.Tensor] = None
    pi_logits: Optional[torch.Tensor] = None

    def to(self, device):
        self.structure.layer_gates = self.structure.layer_gates.to(device)
        self.structure.dim_weights = self.structure.dim_weights.to(device)
        self.structure.residual = self.structure.residual.to(device)
        self.weights = {k: v.to(device) for k, v in self.weights.items()}
        if self.embedding is not None:
            self.embedding = self.embedding.to(device)
        return self
