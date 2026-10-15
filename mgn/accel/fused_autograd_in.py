"""
fused_autograd_in.py — InstanceNorm variant of FusedPipelineFunction (v2).

Identical contract to mgn.accel.fused_autograd.FusedPipelineFunction but
routes through the `mgn.accel._Cin` extension whose pipeline uses
per-(sample, channel) normalization statistics (InstanceNorm) instead of
whole-volume GroupNorm(1). Backward algebra is shared with the GN variant via
`pipeline_backward` (only `_norm` dispatch and the maxpool op differ).
"""
import torch
import torch.nn.functional as F  # noqa: F401  (kept for parity of imports in traces)
from torch.autograd import Function
import torch._dynamo

# Same lazy-import contract as fused_autograd.py: the extension is a build
# artifact, so the module must import on a CPU-only checkout and fail only at
# the call that actually needs the kernel.
try:
    import mgn.accel._Cin as _Cin
    _CIN_IMPORT_ERROR = None
except ImportError as _e:            # pragma: no cover - depends on the host
    _Cin = None
    _CIN_IMPORT_ERROR = _e

from .fused_autograd import pipeline_backward


def _require_Cin():
    if _Cin is None:
        raise ImportError(
            "the InstanceNorm fused extension (mgn.accel._Cin) is not available; "
            "build it with `python build_ext.py _Cin` on a CUDA host, or use the "
            "pure-PyTorch path") from _CIN_IMPORT_ERROR


class FusedPipelineFunctionIN(Function):
    """Fused AML v2 forward with per-channel (InstanceNorm) statistics."""

    @staticmethod
    @torch._dynamo.disable
    def forward(ctx, X, prototypes, pi, layer_gates, dim_masks, block_mode,
                w_offsets, b_offsets, w2_offsets, b2_offsets,
                C_ins, C_outs, Hs, Ws, max_dim):
        L = len(C_ins)

        ctx.max_dim = max_dim
        ctx.L = L
        ctx.C_ins = C_ins
        ctx.C_outs = C_outs
        ctx.Hs = Hs
        ctx.Ws = Ws
        ctx.w_offsets = w_offsets
        ctx.b_offsets = b_offsets
        ctx.w2_offsets = w2_offsets
        ctx.b2_offsets = b2_offsets
        ctx.numel_proto_w = [C_outs[i] * C_ins[i] * 9 for i in range(L)]
        ctx.norm_mode = "in"

        _require_Cin()

        X_f = X.float().contiguous()
        proto_f = prototypes.float().contiguous()
        pi_f = pi.float().contiguous()
        lg_f = layer_gates.float().contiguous()
        dm_f = dim_masks.float().contiguous()
        bm_f = block_mode.float().contiguous()

        result = _Cin.in_fused_forward_and_save(
            X_f, proto_f, pi_f, lg_f, dm_f, bm_f,
            w_offsets, b_offsets, w2_offsets, b2_offsets,
            C_ins, C_outs, Hs, Ws, max_dim
        )

        output = result[0]
        hc1 = result[1:1 + L]
        hc2 = result[1 + L:1 + 2 * L]
        h_prev = result[1 + 2 * L:1 + 3 * L]
        w1 = result[1 + 3 * L:1 + 4 * L]
        w2 = result[1 + 4 * L:1 + 5 * L]

        ctx.save_for_backward(X_f, proto_f, pi_f, lg_f, dm_f, bm_f,
                              *hc1, *hc2, *h_prev, *w1, *w2)
        return output

    @staticmethod
    @torch._dynamo.disable
    def backward(ctx, grad_output):
        _require_Cin()
        return pipeline_backward(ctx, grad_output, _Cin.in_maxpool2x2_bwd,
                                 ctx.norm_mode)
