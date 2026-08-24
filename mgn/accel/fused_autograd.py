"""
fused_autograd.py — Custom autograd.Function for the fused AML pipeline.

Forward:  single C++ call (no Python loop).
Backward: per-layer loop over exported C++ ops using pre-allocated
          CONTIGUOUS gradient buffers, then copied back into the full
          structure-gradient tensors.

NOTE: the compiled `fused_backward_full` passed non-contiguous views
(select/narrow) of the structure-gradient tensors straight to CUDA
kernels that write linearly assuming contiguous [B], [B,2], [B,D]
layouts -> gradients for layer_gates / block_mode / dim_masks landed
in wrong memory locations. This implementation fixes that without
requiring a rebuild of the extension.
"""
import torch
from torch.autograd import Function
import mgn.accel._C as _C

import torch._dynamo


class FusedPipelineFunction(Function):
    """Fused AML forward (single C++ call) with corrected backward."""

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
        D = ctx.max_dim
        B = saved[0].shape[0]

        protos, pi, lg, dm, bm = saved[1], saved[2], saved[3], saved[4], saved[5]
        h_convs = list(saved[6:6 + L])
        h_prevs = list(saved[6 + L:6 + 2 * L])
        w_mixes = list(saved[6 + 2 * L:6 + 3 * L])

        dev = protos.device
        grad_pi = torch.zeros_like(pi)
        grad_protos = torch.zeros_like(protos)
        grad_lg_full = torch.zeros_like(lg)
        grad_dm_full = torch.zeros_like(dm)
        grad_bm_full = torch.zeros_like(bm)
        bm_has_vit = bm.dim() == 3 and bm.shape[-1] > 2

        grad = grad_output.float().contiguous()

        for i in range(L - 1, -1, -1):
            C_in = ctx.C_ins[i]
            C_out = ctx.C_outs[i]
            h_dim = ctx.Hs[i]
            w_dim = ctx.Ws[i]
            hp = h_prevs[i]
            hc = h_convs[i]

            dm_i = dm[:, i].contiguous()
            bm_i_full = bm[:, i].contiguous()
            bm_i = bm_i_full[:, :2].contiguous() if bm_has_vit else bm_i_full
            lg_i = lg[:, i].contiguous()

            # MaxPool backward at layers 5 and 11 (pool happens after post-conv)
            if i == 5 or i == 11:
                h_post = _C.fused_post_conv_oop(
                    hc, hc if i == 0 else hp, dm_i, bm_i, lg_i,
                    B, C_out, h_dim * w_dim, D, 0
                )
                grad = _C.maxpool2x2_bwd(
                    grad.contiguous(), h_post, B, C_out, h_dim, w_dim
                )

            # Contiguous per-layer gradient buffers (kernel writes linearly)
            gbm_t = torch.zeros(B, 2, device=dev, dtype=torch.float32)
            glg_t = torch.zeros(B, device=dev, dtype=torch.float32)
            gdm_t = torch.zeros(B, D, device=dev, dtype=torch.float32)

            dh_conv, dh_prev = _C.fused_post_conv_bwd(
                grad.contiguous(), hc, hp, dm_i, bm_i, lg_i,
                gbm_t, glg_t, gdm_t,
                B, C_out, h_dim * w_dim, D, 1 if i == 0 else 0
            )

            grad_lg_full[:, i] += glg_t
            grad_dm_full[:, i] += gdm_t
            if bm_has_vit:
                grad_bm_full[:, i, :2] += gbm_t
            else:
                grad_bm_full[:, i] += gbm_t

            # Grouped conv backward via cuDNN (weights only; bias grad manual)
            grads = torch.ops.aten.convolution_backward(
                dh_conv.reshape(1, B * C_out, h_dim, w_dim).to(torch.bfloat16),
                hp.reshape(1, B * C_in, h_dim, w_dim).to(torch.bfloat16),
                w_mixes[i].reshape(B * C_out, C_in, 3, 3).to(torch.bfloat16),
                None,
                [1, 1], [1, 1], [1, 1],
                False, [0, 0], B,
                [True, True, False],
            )
            dh_from_conv = grads[0].to(torch.float32).reshape(B, C_in, h_dim, w_dim)
            dW = grads[1].to(torch.float32).reshape(B, -1)
            db = dh_conv.sum(dim=(2, 3)).reshape(B, C_out)

            if i > 0:
                dh_from_conv.add_(dh_prev.reshape_as(dh_from_conv))

            # Chain through weight mixing: W_mix = pi @ proto_slice
            pw = ctx.numel_proto_w[i]
            wo = ctx.w_offsets[i]
            bo = ctx.b_offsets[i]
            proto_w = protos[:, wo:wo + pw]
            proto_b = protos[:, bo:bo + C_out]
            grad_pi.addmm_(dW, proto_w.t())
            grad_protos[:, wo:wo + pw].addmm_(pi.t(), dW)
            grad_pi.addmm_(db, proto_b.t())
            grad_protos[:, bo:bo + C_out].addmm_(pi.t(), db)

            grad = dh_from_conv

        return (grad, grad_protos, grad_pi, grad_lg_full, grad_dm_full, grad_bm_full,
                None, None, None, None, None, None, None)
