import torch
import torch.nn as nn
import torch.nn.functional as F

from .architecture import ModelStructure, GeneratedModel
from .layout import get_prototype_layout, get_second_conv_layout
try:
    from hts_lib.models.basis import HtSLowRankAdapter
except ImportError:
    HtSLowRankAdapter = None


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
            # Init from the SAME layout that the slicers, extractor and CUDA
            # offsets consume (mgn.layout), instead of a second hand-mirrored
            # copy of the packing contract. Both convolutions of every layer
            # (main branch w1 + residual-branch w2) are initialized here so a
            # residual-mode layer draws genuinely trained second-conv weights.
            _total, layer_offsets, _info, wo_off, bo_off = get_prototype_layout(
                max_layers, max_dim, output_dim, in_channels)
            layer_offsets2 = get_second_conv_layout(
                max_layers, max_dim, output_dim, in_channels)
            with torch.no_grad():
                for k in range(K):
                    for (w_start, w_end, b_start, b_end, in_c) in layer_offsets:
                        w_t = torch.empty(max_dim, in_c, 3, 3)
                        nn.init.kaiming_uniform_(w_t, a=math.sqrt(5))
                        self.proto_weights.data[k, w_start:w_end] = w_t.flatten()
                        self.proto_weights.data[k, b_start:b_end] = 0.0
                    for (w2_start, w2_end, b2_start, b2_end) in layer_offsets2:
                        w2_t = torch.empty(max_dim, max_dim, 3, 3)
                        nn.init.kaiming_uniform_(w2_t, a=math.sqrt(5))
                        self.proto_weights.data[k, w2_start:w2_end] = w2_t.flatten()
                        self.proto_weights.data[k, b2_start:b2_end] = 0.0
                    wo = torch.empty(output_dim, 16 * max_dim)
                    nn.init.kaiming_uniform_(wo, a=math.sqrt(5))
                    self.proto_weights.data[k, wo_off:wo_off + wo.numel()] = wo.flatten()
                    self.proto_weights.data[k, bo_off:bo_off + output_dim] = 0.0
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


class DistanceRouter(nn.Module):
    def __init__(self, latent_dim, num_prototypes):
        super().__init__()
        import math
        self.keys = nn.Parameter(torch.empty(num_prototypes, latent_dim))
        nn.init.kaiming_uniform_(self.keys, a=math.sqrt(5))
        self.scale = nn.Parameter(torch.tensor([1.0]))
    def forward(self, z):
        diff = z.unsqueeze(1) - self.keys.unsqueeze(0)
        sq_dist = torch.sum(diff ** 2, dim=-1)
        return -sq_dist * F.softplus(self.scale)

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
        # Retained for API compatibility only; the reparameterized noise head
        # works in latent_dim space and reads nothing from this attribute.
        self.noise_dim = noise_dim
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
            _total, _offs, _info, _wo, _bo = get_prototype_layout(
                max_layers, max_dim, output_dim, in_channels)
            weight_out = _total
            self._proto_offsets = _offs
            self._proto_offsets2 = get_second_conv_layout(
                max_layers, max_dim, output_dim, in_channels)
            self._proto_head = (_wo, _bo)
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
            in_channels=self.in_channels,
            max_dim=self.max_dim,
            output_dim=self.output_dim,
        )

        # HtS Low-Rank Updates for CNN Layers
        if self.use_conv and HtSLowRankAdapter is not None:
            self.hts_adapters = nn.ModuleList()
            for i, (w_start, w_end, b_start, b_end, in_c) in enumerate(self._proto_offsets):
                self.hts_adapters.append(HtSLowRankAdapter(num_prototypes, self.max_dim, in_c))
            self.hts_adapters2 = nn.ModuleList()
            for i, (w_start, w_end, b_start, b_end) in enumerate(self._proto_offsets2):
                self.hts_adapters2.append(HtSLowRankAdapter(num_prototypes, self.max_dim, self.max_dim))


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

        # Optional exploration floor for depth gates (NAS-style path keep-alive).
        # While >0, layer_gates_soft is clamped from below before the
        # straight-through hardening, so deep paths stay gradient-alive (and,
        # once the floor crosses 0.5, forward-active) early in training.
        # Callers (e.g. MetaTrainer users) anneal it to 0 to hand control back
        # to the router. Default 0.0 reproduces the original behaviour.
        self.gate_floor = 0.0

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
            z_route = z
        else:
            log_var = self.noise_var_head(z).clamp(min=-10.0, max=10.0)
            sigma = torch.exp(0.5 * log_var).clamp(min=0.01, max=100.0)
            if self.training:
                # Reparameterized exploration noise -- TRAINING ONLY. Like
                # Gumbel, randn ignores module.eval(); two deterministic evals
                # of the same weights must not disagree.
                z_route = z + torch.randn_like(sigma) * noise_scale * sigma
            else:
                z_route = z
            pi_logits = self.prototype_selector(z_route)

        # Routing is stochastic only while training. Gumbel noise does not
        # respect module.eval() on its own, so gate it explicitly -- otherwise
        # evaluation measures accuracy through randomly perturbed weights.
        if self.training:
            pi = F.gumbel_softmax(pi_logits, tau=tau, hard=False)
        else:
            pi = F.softmax(pi_logits, dim=-1)

        # Always mix prototype rows into concrete per-sample weights. Passing
        # skip_weights=True here used to leave the conv model with no weights
        # at all, crashing TaskNetwork's conv branch with KeyError 'c1.weight'.
        base_layers, base_dims, base_res, w_direct = self.prototype_lib(pi)

        if self.use_conv:
            delta = self.structure_delta(z)
            L_max = self.max_layers
            nd = len(self.dim_choices)
            layer_logits = base_layers + delta[:, :L_max]
            dim_logits = base_dims + delta[:, L_max : L_max + L_max * nd]
            res_logits = base_res + delta[:, -1:]
        else:
            delta = self.structure_delta(z_route)
            L_max = self.max_layers
            nd = len(self.dim_choices)
            layer_logits = base_layers + delta[:, :L_max]
            dim_logits = base_dims + delta[:, L_max : L_max + L_max * nd]
            res_logits = base_res + delta[:, -1:]

        layer_logits = layer_logits + 2.0
        layer_gates_soft = torch.sigmoid(layer_logits)
        if self.gate_floor > 0 and self.training:
            layer_gates_soft = torch.clamp(layer_gates_soft, min=self.gate_floor)
        layer_gates = (layer_gates_soft > 0.5).float().detach() + layer_gates_soft - layer_gates_soft.detach()
        
        dim_logits = dim_logits.view(-1, self.max_layers, len(self.dim_choices))
        d_logit = dim_logits.mean(dim=-1, keepdim=True) + 2.0
        dim_fractions = torch.sigmoid(d_logit)

        d_lengths = dim_fractions * self.max_dim
        arange = torch.arange(self.max_dim, device=x.device).float().view(1, 1, self.max_dim)
        
        dim_masks = torch.sigmoid(10.0 * (d_lengths - arange))

        if self.use_conv:
            block_logits = self.block_mode_head(z).view(-1, self.max_layers, 2)
            if self.training:
                block_mode = F.gumbel_softmax(block_logits, tau=tau, hard=True)
            else:
                # Deterministic argmax mode at eval time.
                block_mode = F.one_hot(block_logits.argmax(dim=-1), 2).float()
        else:
            block_mode = None

        W = {}
        if self.use_conv:
            # Slice the prototype-mixed flat weight vector into per-layer
            # tensors matching TaskNetwork's contract:
            #   c{i}.weight [B, max_dim, in_c, 3, 3], c{i}.bias [B, max_dim],
            #   out.weight [B, output_dim, 16*max_dim], out.bias [B, output_dim].
            # Offsets come from the canonical mgn.layout (shared with init,
            # the executor and the static extractor).
            D = self.max_dim
            for i, (w_start, w_end, b_start, b_end, in_c) in enumerate(self._proto_offsets):
                w_aml = w_direct[:, w_start:w_end].view(-1, D, in_c, 3, 3)
                if hasattr(self, 'hts_adapters'):
                    w_hts = self.hts_adapters[i](pi)
                    W[f"c{i+1}.weight"] = w_aml + w_hts
                else:
                    W[f"c{i+1}.weight"] = w_aml
                W[f"c{i+1}.bias"] = w_direct[:, b_start:b_end]
            for i, (w2_start, w2_end, b2_start, b2_end) in enumerate(self._proto_offsets2):
                w2_aml = w_direct[:, w2_start:w2_end].view(-1, D, D, 3, 3)
                if hasattr(self, 'hts_adapters2'):
                    w2_hts = self.hts_adapters2[i](pi)
                    W[f"c{i+1}.weight2"] = w2_aml + w2_hts
                else:
                    W[f"c{i+1}.weight2"] = w2_aml
                W[f"c{i+1}.bias2"] = w_direct[:, b2_start:b2_end]
            wo_off, bo_off = self._proto_head
            sz_w = (16 * D) * self.output_dim
            W["out.weight"] = w_direct[:, wo_off:wo_off + sz_w].view(-1, self.output_dim, 16 * D)
            W["out.bias"] = w_direct[:, bo_off:bo_off + self.output_dim]
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
        # Persist the latent so ModelEvaluator.score_candidate can score the
        # candidate without re-running the encoder. pi/prototypes feed the
        # fused CUDA executor path (see GeneratedModel docs): pi is the SAME
        # tensor used to slice W, and prototypes is the raw library matrix.
        return GeneratedModel(structure=struct, weights=W, pi_logits=pi_logits,
                              embedding=z, pi=pi,
                              prototypes=(self.prototype_lib.proto_weights
                                          if self.use_conv else None))

