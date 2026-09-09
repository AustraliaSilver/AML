from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F


@dataclass
class LayerSpec:
    layer_type: str = "mlp"
    in_dim: int = 2
    out_dim: int = 32
    num_heads: int = 0
    kernel_size: int = 0
    expansion_factor: float = 4.0

    def param_count(self) -> int:
        if self.layer_type == "mlp":
            return self.in_dim * self.out_dim + self.out_dim
        elif self.layer_type == "attention":
            d = self.in_dim
            return 4 * d * d + d * 4
        elif self.layer_type == "conv":
            k = self.kernel_size
            return self.in_dim * self.out_dim * k * k + self.out_dim
        return self.in_dim * self.out_dim + self.out_dim


@dataclass
class ModelStructure:
    layer_gates: torch.Tensor
    dim_weights: torch.Tensor
    residual: torch.Tensor
    dim_masks: Optional[torch.Tensor] = None
    block_mode: Optional[torch.Tensor] = None  # [batch, max_layers, 2] -> 0: PlainConv, 1: ResNetBlock
    type_weights: Optional[torch.Tensor] = None  # MLP vs Conv
    layer_types: List[str] = field(default_factory=lambda: ["mlp", "mlp", "mlp"])
    input_dim: int = 2
    output_dim: int = 2
    max_dim: int = 32
    max_layers: int = 3
    dim_choices: Tuple[float, ...] = (8, 16, 32)
    use_conv: bool = False
    in_channels: int = 1
    
    @property
    def layer_type(self) -> torch.Tensor:
        if self.type_weights is not None:
            return torch.softmax(self.type_weights, dim=-1)
        return torch.ones(self.layer_gates.shape[0], 2) * 0.5  # [mlp_prob, conv_prob]

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

    def effective_params(self) -> torch.Tensor:
        L = self.num_layers
        D = self.hidden_dim
        total = torch.zeros_like(D)
        for i in range(self.max_layers):
            gate = self.layer_gates[:, i]
            in_dim = self.input_dim if i == 0 else D
            out_dim = D if i < self.max_layers - 1 else self.output_dim
            layer = LayerSpec(layer_type="conv" if self.use_conv and i < 2 else (self.layer_types[i] if i < len(self.layer_types) else "mlp"),
                              in_dim=int(in_dim.mean().round().item()) if in_dim.dim() > 0 else int(in_dim),
                              out_dim=int(out_dim.mean().round().item()) if out_dim.dim() > 0 else int(out_dim))
            P_layer = layer.param_count()
            total = total + gate * P_layer
        return total + 0.01 * L * D

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
    L = S.num_layers.detach()
    D = S.hidden_dim.detach()
    P = S.effective_params_formula().detach()
    mask = P > P_max
    if not mask.any():
        return S
    scale = torch.sqrt(P_max / (P + 1e-8))
    D_sq = D.squeeze(-1) if D.dim() > 1 else D
    D_proj = D_sq * scale.clamp(max=1.0)
    L_proj = L * scale.clamp(max=1.0)
    keep = (~mask).reshape(-1, *([1] * (S.dim_weights.dim() - 1)))
    if S.dim_weights.dim() == 3 and S.dim_weights.shape[-1] == 1:
        # Continuous mode: dim_weights stores WIDTH FRACTIONS [B, L, 1].
        # (The old code scattered a discrete one-hot of shape [B, nd] into
        # this [B, L, 1] tensor -> shape error / silent wrong broadcast.)
        D_new = (D_proj / S.max_dim).reshape(S.dim_weights.shape)
    else:
        d_choices = torch.tensor(S.dim_choices, device=D.device, dtype=torch.float32)
        D_idx = (d_choices.unsqueeze(0) - D_proj.unsqueeze(1)).abs().argmin(dim=-1)
        D_onehot = F.one_hot(D_idx, num_samples=len(S.dim_choices)).to(S.dim_weights.dtype)
        D_new = D_onehot.unsqueeze(1).expand_as(S.dim_weights)
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
