import torch
import torch.nn as nn
import torch.nn.functional as F

from .architecture import ModelStructure, GeneratedModel


class PrototypeLibrary(nn.Module):
    def __init__(self, K: int, max_layers: int, num_dim_choices: int, weight_dim: int, proto_dim: int = 32, use_conv: bool = False, in_channels: int = 1, max_dim: int = 64, output_dim: int = 10):
        super().__init__()
        self.K = K
        self.use_conv = use_conv
        self.max_layers = max_layers
        self.max_dim = max_dim
        self.in_channels = in_channels
        self.output_dim = output_dim
        
        self.proto_embed = nn.Parameter(torch.randn(K, proto_dim))
        self.proto_weights = nn.Parameter(torch.zeros(K, weight_dim))
        self.layer_head = nn.Linear(proto_dim, max_layers)
        self.dim_head = nn.Linear(proto_dim, max_layers * num_dim_choices)
        self.residual_head = nn.Linear(proto_dim, 1)
        self.weight_head = nn.Linear(proto_dim, weight_dim)
        if use_conv:
            self.weight_head.requires_grad_(False)

        if use_conv:
            nn.init.constant_(self.layer_head.bias, 0.0)
            if num_dim_choices > 0:
                nn.init.constant_(self.dim_head.bias, 0.0)
                self.dim_head.bias.data.view(max_layers, num_dim_choices)[:, -1] = 3.0
            import math
            with torch.no_grad():
                for k in range(K):
                    idx = 0
                    for i in range(max_layers):
                        in_c = in_channels if i == 0 else max_dim
                        w1_numel = max_dim * in_c * 9
                        w1 = torch.empty(max_dim, in_c, 3, 3)
                        nn.init.kaiming_uniform_(w1, a=math.sqrt(5))
                        self.proto_weights.data[k, idx:idx + w1_numel] = w1.flatten()
                        idx += w1_numel
                        self.proto_weights.data[k, idx:idx + max_dim] = 0.0
                        idx += max_dim

                    wo_numel = (16 * max_dim) * output_dim
                    wo = torch.empty(output_dim, 16 * max_dim)
                    nn.init.kaiming_uniform_(wo, a=math.sqrt(5))
                    self.proto_weights.data[k, idx:idx + wo_numel] = wo.flatten()
                    idx += wo_numel
                    self.proto_weights.data[k, idx:idx + output_dim] = 0.0
        else:
            nn.init.normal_(self.proto_weights, std=0.02)

    def forward(self, pi: torch.Tensor, skip_weights: bool = False):
        emb = pi @ self.proto_embed
        if skip_weights:
            w_direct = None
        else:
            w_direct = pi @ self.proto_weights
        if self.use_conv:
            return (
                self.layer_head(emb),
                self.dim_head(emb),
                self.residual_head(emb),
                w_direct,
            )
        return (
            self.layer_head(emb),
            self.dim_head(emb),
            self.residual_head(emb),
            w_direct + self.weight_head(emb),
        )


class ModelGenerator(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
        max_layers: int = 3,
        max_dim: int = 32,
        dim_choices: tuple = (8, 16, 32),
        latent_dim: int = 64,
        noise_dim: int = 64,
        num_prototypes: int = 8,
        use_conv: bool = False,
        in_channels: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_layers = max_layers
        self.max_dim = max_dim
        self.dim_choices = dim_choices
        self.noise_dim = noise_dim if noise_dim == latent_dim else latent_dim
        self.latent_dim = latent_dim
        self.use_conv = use_conv
        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

        self.noise_var_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        if self.use_conv:
            self.noise_var_head.requires_grad_(False)

        self.prototype_selector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_prototypes),
        )

        if self.use_conv:
            weight_out = 0
            for i in range(max_layers):
                in_c = in_channels if i == 0 else max_dim
                weight_out += max_dim * in_c * 9 + max_dim
            weight_out += (16 * max_dim) * output_dim + output_dim
        else:
            weight_out = (
                input_dim * max_dim
                + max_dim
                + (max_layers - 1) * (max_dim * max_dim + max_dim)
                + max_dim * output_dim
                + output_dim
            )

        self.prototype_lib = PrototypeLibrary(
            K=num_prototypes,
            max_layers=max_layers,
            num_dim_choices=len(dim_choices),
            weight_dim=weight_out,
            use_conv=self.use_conv,
            in_channels=getattr(self, "in_channels", 1),
            max_dim=self.max_dim,
            output_dim=self.output_dim,
        )

        self.structure_delta = nn.Linear(latent_dim, max_layers + (max_layers * len(dim_choices)) + 1)
        # 2 executable block modes: Plain / ResNet
        self.block_mode_head = nn.Linear(latent_dim, max_layers * 2)
        if self.use_conv:
            nn.init.constant_(self.structure_delta.weight, 0.0)
            nn.init.constant_(self.structure_delta.bias, 0.0)
            nn.init.constant_(self.block_mode_head.weight, 0.0)
            nn.init.constant_(self.block_mode_head.bias, 0.0)
            with torch.no_grad():
                # ResNet-heavy init (see train.py note)
                self.block_mode_head.bias.data.view(max_layers, 2)[:, 1] = 2.0
        self.weight_delta = nn.Linear(latent_dim, weight_out)
        if self.use_conv:
            self.weight_delta.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
        noise_scale: float = 1.0,
    ) -> GeneratedModel:
        x_flat = x.view(x.shape[0], -1)
        z = self.encoder(x_flat)

        if self.use_conv:
            pi_logits = self.prototype_selector(z)
        else:
            log_var = self.noise_var_head(z).clamp(min=-10.0, max=10.0)
            sigma = torch.exp(0.5 * log_var).clamp(min=0.01, max=100.0)
            eps = torch.randn_like(sigma) * noise_scale
            z_noisy = z + eps * sigma
            pi_logits = self.prototype_selector(z_noisy)

        pi = F.gumbel_softmax(pi_logits, tau=tau, hard=False)

        base_layers, base_dims, base_res, w_direct = self.prototype_lib(pi, skip_weights=self.use_conv)

        if self.use_conv:
            delta = self.structure_delta(z)
            L_max = self.max_layers
            nd = len(self.dim_choices)
            layer_logits = base_layers + delta[:, :L_max]
            dim_logits = base_dims + delta[:, L_max : L_max + L_max * nd]
            res_logits = base_res + delta[:, -1:]
        else:
            delta = self.structure_delta(z_noisy)
            L_max = self.max_layers
            nd = len(self.dim_choices)
            layer_logits = base_layers + delta[:, :L_max]
            dim_logits = base_dims + delta[:, L_max : L_max + L_max * nd]
            res_logits = base_res + delta[:, -1:]

        layer_logits = layer_logits + 2.0
        layer_gates_soft = torch.sigmoid(layer_logits)
        layer_gates = (layer_gates_soft > 0.5).float().detach() + layer_gates_soft - layer_gates_soft.detach()
        
        dim_logits = dim_logits.view(-1, self.max_layers, len(self.dim_choices))
        d_logit = dim_logits.mean(dim=-1, keepdim=True) + 2.0
        dim_fractions = torch.sigmoid(d_logit)
        
        batch_size = x.shape[0]
        d_lengths = dim_fractions * self.max_dim
        arange = torch.arange(self.max_dim, device=x.device).float().view(1, 1, self.max_dim)
        
        dim_masks = torch.sigmoid(10.0 * (d_lengths - arange))

        if self.use_conv:
            block_logits = self.block_mode_head(z).view(-1, self.max_layers, 2)
            block_mode = F.gumbel_softmax(block_logits, tau=tau, hard=True)
        else:
            block_mode = None

        W = {}
        if self.use_conv:
            pass
        else:
            idx = 0
            for i in range(self.max_layers):
                if i == 0:
                    sz = self.max_dim * self.input_dim
                    W[f"fc{i+1}.weight"] = w_direct[:, idx:idx + sz].view(-1, self.max_dim, self.input_dim)
                else:
                    sz = self.max_dim * self.max_dim
                    W[f"fc{i+1}.weight"] = w_direct[:, idx:idx + sz].view(-1, self.max_dim, self.max_dim)
                idx += sz
                W[f"fc{i+1}.bias"] = w_direct[:, idx:idx + self.max_dim].view(-1, self.max_dim)
                idx += self.max_dim
            sz = self.max_dim * self.output_dim
            W["out.weight"] = w_direct[:, idx:idx + sz].view(-1, self.max_dim, self.output_dim)
            idx += sz
            W["out.bias"] = w_direct[:, idx:idx + self.output_dim].view(-1, self.output_dim)

        struct = ModelStructure(
            layer_gates=layer_gates,
            dim_weights=dim_fractions,
            residual=res_logits,
            dim_masks=dim_masks,
            use_conv=self.use_conv,
            block_mode=block_mode,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            max_layers=self.max_layers,
            max_dim=self.max_dim,
            dim_choices=self.dim_choices,
            in_channels=self.in_channels
        )
        return GeneratedModel(structure=struct, weights=W, pi_logits=pi_logits)
