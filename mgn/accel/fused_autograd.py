"""
fused_autograd.py — Custom autograd.Function for the fused AML pipeline.

Forward+save: single C++ call (no Python loop).
Backward:     single C++ call (no Python loop).
"""
import torch
from torch.autograd import Function
import mgn.accel._C as _C


import torch._dynamo

class FusedPipelineFunction(Function):
    """Fused 6-layer AML forward with custom CUDA backward."""

    @staticmethod
    @torch._dynamo.disable
    def forward(ctx, X, prototypes, pi, layer_gates, dim_masks, block_mode,
                w_offsets, b_offsets, C_ins, C_outs, Hs, Ws, max_dim):
        L = len(C_ins)

        ctx.max_dim = max_dim
        ctx.L = L
        ctx.C_ins = C_ins
        ctx.C_outs = C_outs
        ctx.Hs = Hs
        ctx.Ws = Ws
        ctx.w_offsets = w_offsets
        ctx.b_offsets = b_offsets
        ctx.numel_proto_w = [C_outs[i] * C_ins[i] * 9 for i in range(L)]

        X_f = X.float().contiguous()
        proto_f = prototypes.float().contiguous()
        pi_f = pi.float().contiguous()
        lg_f = layer_gates.float().contiguous()
        dm_f = dim_masks.float().contiguous()
        bm_f = block_mode.float().contiguous()

        result = _C.fused_forward_and_save(
            X_f, proto_f, pi_f, lg_f, dm_f, bm_f,
            w_offsets, b_offsets, C_ins, C_outs, Hs, Ws, max_dim
        )

        output = result[0]
        saved_h_conv = result[1:1 + L]
        saved_h_prev = result[1 + L:1 + 2 * L]
        saved_w_mix = result[1 + 2 * L:1 + 3 * L]

        ctx.save_for_backward(X_f, proto_f, pi_f, lg_f, dm_f, bm_f,
                              *saved_h_conv, *saved_h_prev, *saved_w_mix)
        return output

    @staticmethod
    @torch._dynamo.disable
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        L = ctx.L

        grads = _C.fused_backward_full(
            grad_output.float().contiguous(),
            saved[0], saved[1], saved[2], saved[3], saved[4], saved[5],
            list(saved[6:6 + L]),
            list(saved[6 + L:6 + 2 * L]),
            list(saved[6 + 2 * L:]),
            ctx.w_offsets, ctx.b_offsets,
            ctx.C_ins, ctx.C_outs, ctx.Hs, ctx.Ws,
            ctx.numel_proto_w, ctx.max_dim
        )

        return (grads[0], grads[1], grads[2], grads[3], grads[4], grads[5],
                None, None, None, None, None, None, None)
