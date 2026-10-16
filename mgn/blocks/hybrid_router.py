import torch
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
        norm_type: str = "in",
        norm_groups: int = 1
    ):
        super().__init__(in_channels, out_channels)
        self.plain = PlainConvBlock(in_channels, out_channels, norm_type=norm_type, norm_groups=norm_groups)
        self.resnet = ResNetBlock(in_channels, out_channels, norm_type=norm_type, norm_groups=norm_groups)
        self.vit = VisionTransformerBlock(in_channels, out_channels, num_heads=num_heads)

    def forward(
        self,
        x: torch.Tensor,
        mode_weights: Optional[torch.Tensor] = None,
        dim_mask: Optional[torch.Tensor] = None,
        flags: Optional[Tuple[bool, bool, int]] = None
    ) -> torch.Tensor:
        """
        x: [B, C_in, H, W]
        mode_weights: [B, 3] or [1, 3] (probabilities for [plain, resnet, vit])
        dim_mask: [B, C_out, 1, 1]
        flags: this layer's `(all_hard, same_mode, chosen)`, precomputed by the
            caller via `_hoist_block_mode_flags`. Computing them here costs two
            device->host synchronisations (`_flags.tolist()` and
            `mode_idx[0].item()`), i.e. 36 per 18-layer forward, each of which
            drains the launch queue. A caller holding the whole `block_mode`
            [B, L, 3] tensor can read every layer's flags in one transfer and
            pass them down. The values come from the same comparisons on the
            same slice, so `flags` is a pure latency knob: passing it does not
            change the result. It MUST be left None whenever `mode_weights` is a
            sub-batch slice (the per-sample depth-skip path in
            `ConvAML_18Layer_Hybrid.forward`), because then row 0 of the slice is
            not row 0 of the batch.
        """
        if mode_weights is None:
            # Default to resnet if unrouted
            return self.resnet(x, dim_mask=dim_mask)

        if mode_weights.ndim == 2:
            if flags is None:
                mode_max, mode_idx = mode_weights.max(dim=-1)
                # One device->host transfer serves BOTH predicates below. The previous
                # code evaluated two separate Python `and`s, i.e. two syncs per layer.
                _flags = torch.stack(((mode_max > 0.99).all(),
                                      (mode_idx == mode_idx[0]).all()))
                all_hard, same_mode = _flags.tolist()
                del _flags
            else:
                all_hard, same_mode, chosen = flags
                # Only the index is still needed (per-sample MoE dispatch below).
                # argmax is device-side: it does not synchronise.
                mode_idx = mode_weights.argmax(dim=-1)

            if all_hard and same_mode:
                # Whole batch agrees: execute the single selected block once.
                if flags is None:
                    chosen = mode_idx[0].item()
                if chosen == 0:
                    cand = self.plain(x, dim_mask=dim_mask)
                elif chosen == 1:
                    cand = self.resnet(x, dim_mask=dim_mask)
                elif chosen == 2:
                    cand = self.vit(x, dim_mask=dim_mask)
                else:
                    raise ValueError(f"Invalid mode {chosen}")
                return cand * mode_weights[:, chosen].view(-1, 1, 1, 1)

            if all_hard and not self.training:
                # P3 - True per-sample MoE dispatch.
                #
                # The fall-through below would compute every active branch on the
                # FULL batch and then mask it out, so a batch split 50/50 between
                # Plain and ResNet pays 2 full-batch block evaluations instead of
                # 2 half-batch ones. Dispatching on the sub-batch that actually
                # uses each branch removes that waste.
                #
                # Equivalence: with one-hot mode_weights the soft mixture
                # sum_m w_m * branch_m(x) collapses to w_argmax * branch_argmax(x),
                # which is exactly what this computes. `all_hard` is what makes
                # that identity hold, so the check above is load-bearing.
                #
                # Eval-only ON PURPOSE: `out[mask] = branch(x[mask])` removes
                # mode_weights from the autograd graph entirely. The straight-
                # through estimator in ConvAML_18Layer_Hybrid.forward relies on
                # mode_weights being in the graph, otherwise the architecture
                # parameters receive an exactly-zero gradient and the whole
                # architecture search dies. Training keeps the differentiable
                # mixture below.
                out = None
                for m, branch in enumerate((self.plain, self.resnet, self.vit)):
                    mask = mode_idx == m
                    if bool(mask.any()):
                        y = branch(x[mask],
                                   dim_mask=(dim_mask[mask] if dim_mask is not None else None))
                        # Under autocast the branches do not all return the same
                        # dtype (conv -> bf16, LayerNorm/GELU -> fp32), so the
                        # buffer dtype has to follow the promoted result or
                        # index_put_ raises a dtype mismatch.
                        if out is None:
                            out = torch.zeros(x.shape, dtype=y.dtype, device=x.device)
                        # Scale by the sample's OWN weight for its chosen branch,
                        # exactly as the whole-batch fast path above does.
                        # `all_hard` is `max > 0.99`, not `== 1.0`, so skipping
                        # the multiplication made a sample's output depend on
                        # which other samples happened to occupy its batch.
                        w = mode_weights[mask, m].to(y.dtype).view(-1, 1, 1, 1)
                        out[mask] = y * w
                if out is None:                      # no sample matched (impossible)
                    out = torch.zeros_like(x)
                return out


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
