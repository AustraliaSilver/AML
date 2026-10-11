"""
fused_autograd.py — Custom autograd.Function for the fused AML pipeline (v2).

v2: the block is the REAL two-convolution ResNet (mgn.layout v2 slots):

    hc1 = conv1(h_in); n1 = GN(hc1); hp = relu(n1) * dm     (also conv2 input)
    hc2 = conv2(hp);   n2 = GN(hc2)
    he  = layer0 ? hp : h_in * dm
    hr  = relu(he + n2) * dm
    ha  = g_p * hp + g_r * hr
    out = layer0 ? ha : lg * ha + (1 - lg) * he

The C++ driver (mgn/csrc/in/../../bindings.cpp) executes exactly this chain via
the same ATen ops this module recomputes in backward, so forward and backward
cannot drift apart semantics-wise. The only custom kernel left in the hot path
is the elementwise mode-mix, whose derivative is written out closed-form below.

NOTE: the compiled `fused_backward_full` is retired (v1 single-conv algebra);
gradients for structure tensors (layer_gates / dim_masks / block_mode) are
accumulated here in fp32 contiguous buffers, mirroring the fix from the v1
postmortem: kernels must never write into non-contiguous select() views.
"""
import torch
import torch.nn.functional as F
from torch.autograd import Function
import mgn.accel._C as _C

# v2 fused mix-backward kernels (added 9/15). Optional: an older _C build
# without them falls back to the ATen-recompute path below, bit-identical
# to the pre-kernel code.
_CMIX_A = getattr(_C, "mix_bwd_v2_a", None)
_CMIX_B = getattr(_C, "mix_bwd_v2_b", None)

import torch._dynamo


EPS = 1e-5


def _norm(t, norm_mode):
    if norm_mode == "gn1":
        return F.group_norm(t, 1, eps=EPS)
    return F.instance_norm(t, eps=EPS)


def _norm_groups(C, norm_mode):
    # F.instance_norm without affine is exactly native_group_norm with
    # groups=C; both modes share the one fused ATen kernel.
    return 1 if norm_mode == "gn1" else C


def _norm_stats(t, norm_mode):
    """(normalized, mean, rstd) via aten::native_group_norm -- bit-identical
    to F.group_norm/F.instance_norm forward, but hands the per-group stats
    back so the backward never re-derives them from the raw tensor."""
    B, C, H, W = t.shape
    y, mean, rstd = torch.ops.aten.native_group_norm(
        t, None, None, B, C, H * W, _norm_groups(C, norm_mode), EPS)
    return y.view_as(t), mean, rstd


def _norm_bwd(g, raw, mean, rstd, norm_mode):
    """Fused d(raw) for the weightless norms: native CUDA backward kernel."""
    B, C, H, W = raw.shape
    if norm_mode == "gn1" and hasattr(_C, "group_norm1_bwd"):
        return _C.group_norm1_bwd(g, raw, mean, rstd, B, C, H, W)
    elif norm_mode == "in" and hasattr(_C, "instance_norm_bwd"):
        return _C.instance_norm_bwd(g, raw, mean, rstd, B, C, H, W)
    gg = torch.ops.aten.native_group_norm_backward(
        g.reshape(B, C * H * W), raw.reshape(B, C * H * W), mean, rstd,
        None, B, C, H * W, _norm_groups(C, norm_mode),
        [True, False, False])
    return gg[0].view(B, C, H, W)


def _dot_spatial(a, b, B):
    """Per-sample <a,b> over all dims: cuBLAS gemv, no [B,C,H,W] product
    materialization (vs a.mul().sum() = 25MB write + 25MB read per call)."""
    return torch.bmm(a.reshape(B, 1, -1), b.reshape(B, -1, 1)).view(B)


def _dot_channel(a, b, B, C):
    """Per-sample-per-channel <a,b> over (H,W): batched gemv [N,1,HW]x[N,HW,1]."""
    return torch.bmm(a.reshape(B * C, 1, -1),
                     b.reshape(B * C, -1, 1)).view(B, C)


def _conv_bwd(g_out, x_in, w_mix, B, C_out, C_in, hd, wd):
    """Direct per-sample dynamic conv backward via custom CUDA kernel."""
    if hasattr(_C, "dynamic_conv2d_bwd"):
        d_in, dW, db = _C.dynamic_conv2d_bwd(g_out, x_in, w_mix, B, C_in, C_out, hd, wd)
        return d_in, dW, db

    grads = torch.ops.aten.convolution_backward(
        g_out.reshape(1, B * C_out, hd, wd).to(torch.bfloat16),
        x_in.reshape(1, B * C_in, hd, wd).to(torch.bfloat16),
        w_mix.reshape(B * C_out, C_in, 3, 3).to(torch.bfloat16),
        None,
        [1, 1], [1, 1], [1, 1],
        False, [0, 0], B,
        [True, True, False],
    )
    dh_in = grads[0].to(torch.float32).reshape(B, C_in, hd, wd)
    dW = grads[1].to(torch.float32).reshape(B, -1)
    db = g_out.sum(dim=(2, 3)).reshape(B, C_out)
    return dh_in, dW, db


def _chain_proto(grad_pi, grad_protos, pi, protos, dW, db,
                 wo, pw, bo, C_out):
    proto_w = protos[:, wo:wo + pw]
    grad_pi.addmm_(dW, proto_w.t())
    grad_protos[:, wo:wo + pw].addmm_(pi.t(), dW)
    proto_b = protos[:, bo:bo + C_out]
    grad_pi.addmm_(db, proto_b.t())
    grad_protos[:, bo:bo + C_out].addmm_(pi.t(), db)


def pipeline_backward(ctx, grad_output, maxpool_bwd, norm_mode):
    """Shared backward for the GN and IN v2 pipelines (only maxpool op and
    normalization dispatch differ; both were swapped in via a C++ mirror)."""
    saved = ctx.saved_tensors
    L = ctx.L
    D = ctx.max_dim
    B = saved[0].shape[0]

    protos, pi, lg, dm, bm = saved[1], saved[2], saved[3], saved[4], saved[5]
    hc1s = saved[6:6 + L]
    hc2s = saved[6 + L:6 + 2 * L]
    hps = saved[6 + 2 * L:6 + 3 * L]
    w1s = saved[6 + 3 * L:6 + 4 * L]
    w2s = saved[6 + 4 * L:6 + 5 * L]

    grad_pi = torch.zeros_like(pi)
    grad_protos = torch.zeros_like(protos)
    grad_lg_full = torch.zeros_like(lg)
    grad_dm_full = torch.zeros_like(dm)
    grad_bm_full = torch.zeros_like(bm)

    grad = grad_output.float()

    for i in range(L - 1, -1, -1):
        C_in = ctx.C_ins[i]
        C_out = ctx.C_outs[i]
        hd = ctx.Hs[i]
        wd = ctx.Ws[i]
        hc1 = hc1s[i]
        hc2 = hc2s[i]
        hpv = hps[i]
        W1m = w1s[i]
        W2m = w2s[i]
        is0 = i == 0

        dmv = dm[:, i].view(B, D, 1, 1)
        gp = bm[:, i, 0].view(B, 1, 1, 1)
        gr = bm[:, i, 1].view(B, 1, 1, 1)
        lgv = lg[:, i].view(B, 1, 1, 1)

        C_out_i = C_out
        use_mix = (_CMIX_A is not None and _CMIX_B is not None
                   and C_out_i <= 256 and 256 % C_out_i == 0)

        # --- native norms (always): fused forward + the exact per-group
        # stats the fused backward kernels below consume ---
        xhat1, mean1, rstd1 = _norm_stats(hc1, norm_mode)
        xhat2, mean2, rstd2 = _norm_stats(hc2, norm_mode)
        pool_pre = (i == 5 or i == 11)

        if (not use_mix) or pool_pre:
            # full recompute chain: needed by the ATen fallback and, at the
            # two maxpool layers, by the argmax routing in maxpool_bwd.
            r1 = F.relu(xhat1)
            hp = r1 * dmv
            he = hp if is0 else hpv * dmv
            u2 = he + xhat2
            r2 = F.relu(u2)
            hr = r2 * dmv
            ha = gp * hp + gr * hr
            if is0:
                out_pre = ha
            else:
                out_pre = lgv * ha + (1.0 - lgv) * he
        else:
            hp = F.relu(xhat1) * dmv          # conv2 input for _conv_bwd

        # --- MaxPool backward at layers 5 and 11 (pools after the mix) ---
        if pool_pre:
            grad = maxpool_bwd(grad.contiguous(), out_pre, B, C_out, hd, wd)

        if use_mix:
            # one elementwise kernel per phase inverts the mix algebra; the
            # cuDNN conv bwd + native GN bwd stay on ATen exactly as before.
            dm2d = dm[:, i].contiguous()                  # [B, D]
            bm2d = bm[:, i].contiguous()                  # [B, 2]
            lg1d = lg[:, i].contiguous()                  # [B]
            g4 = grad.contiguous()
            gxc = xhat1.contiguous()
            hpvc = hpv.contiguous()
            (dn2, d_hp_pre, d_he, gdm_a,
             gbm0, gbm1, glg_i) = _CMIX_A(
                g4, gxc, xhat2.contiguous(), hpvc,
                dm2d, bm2d, lg1d, D, 1 if is0 else 0)
            dhc2 = _norm_bwd(dn2, hc2, mean2, rstd2, norm_mode)
            dhp_conv2, dW2, db2 = _conv_bwd(dhc2, hp, W2m, B, C_out, D, hd, wd)
            d_hp = d_hp_pre + dhp_conv2
            dn1, dh_skip, gdm_b = _CMIX_B(
                d_hp, d_he, gxc, hpvc, dm2d, D, 1 if is0 else 0)
            dhc1 = _norm_bwd(dn1, hc1, mean1, rstd1, norm_mode)
            dh_from, dW1, db1 = _conv_bwd(dhc1, hpv, W1m, B, C_out, C_in, hd, wd)
            if not is0:
                dh_from = dh_from + dh_skip
        else:
            # --- mix-algebra backward (ATen fallback path) ---
            dha = grad if is0 else grad * lgv
            d_he = torch.zeros_like(grad) if is0 else grad * (1.0 - lgv)
            m2 = (u2 > 0).to(grad.dtype)
            dhr = dha * gr
            dn2 = dhr * dmv * m2
            d_he = d_he + dhr * dmv * m2
            dh_skip = d_he * dmv              # grad of h_in via the skip
            d_hp = dha * gp + (d_he if is0 else 0)   # layer 0 skip source IS hp

            # --- conv2 branch: GN2 backward, cuDNN backward, weight grads ---
            dhc2 = _norm_bwd(dn2, hc2, mean2, rstd2, norm_mode)
            dhp_conv2, dW2, db2 = _conv_bwd(dhc2, hp, W2m, B, C_out, D, hd, wd)
            d_hp = d_hp + dhp_conv2

            # --- conv1 branch ---
            m1 = (xhat1 > 0).to(grad.dtype)
            dn1 = d_hp * dmv * m1
            dhc1 = _norm_bwd(dn1, hc1, mean1, rstd1, norm_mode)
            dh_from, dW1, db1 = _conv_bwd(dhc1, hpv, W1m, B, C_out, C_in, hd, wd)
            if not is0:
                dh_from = dh_from + dh_skip

        # --- prototype-mixing chain ---
        pw1 = ctx.numel_proto_w[i]
        pw2 = C_out * D * 9
        _chain_proto(grad_pi, grad_protos, pi, protos, dW1, db1,
                     ctx.w_offsets[i], pw1, ctx.b_offsets[i], C_out)
        _chain_proto(grad_pi, grad_protos, pi, protos, dW2, db2,
                     ctx.w2_offsets[i], pw2, ctx.b2_offsets[i], C_out)

        # --- structure gradients ---
        if use_mix:
            # emitted fused by the kernels (deterministic per-block tree
            # reductions; the same quantities the gemv path computes).
            grad_lg_full[:, i] += glg_i
            if grad_bm_full.shape[-1] == 2:
                grad_bm_full[:, i, 0] += gbm0
                grad_bm_full[:, i, 1] += gbm1
            else:
                grad_bm_full[:, i, :2] += torch.stack([gbm0, gbm1], dim=-1)
            grad_dm_full[:, i] += gdm_a + gdm_b
        else:
            if not is0:
                grad_lg_full[:, i] += _dot_spatial(grad, ha - he, B)
            gbm0 = _dot_spatial(dha, hp, B)
            gbm1 = _dot_spatial(dha, hr, B)
            if grad_bm_full.shape[-1] == 2:
                grad_bm_full[:, i, 0] += gbm0
                grad_bm_full[:, i, 1] += gbm1
            else:
                grad_bm_full[:, i, :2] += torch.stack([gbm0, gbm1], dim=-1)
            # grad of dim_mask: hp = relu(n1)*dm and hr = relu(he+n2)*dm apply dm;
            # he = h_prev*dm adds one more dm path for layers beyond the first.
            gdm = (_dot_channel(d_hp, r1, B, C_out)
                   + _dot_channel(dhr, r2, B, C_out))
            if not is0:
                gdm = gdm + _dot_channel(d_he, hpv, B, C_out)
            grad_dm_full[:, i] += gdm

        grad = dh_from

    return (grad, grad_protos, grad_pi, grad_lg_full, grad_dm_full, grad_bm_full,
            *[None] * 9)


class FusedPipelineFunction(Function):
    """Fused AML v2 forward (single C++ call) + ATen-recompute backward."""

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
        ctx.norm_mode = "gn1"

        X_f = X.float().contiguous()
        proto_f = prototypes.float().contiguous()
        pi_f = pi.float().contiguous()
        lg_f = layer_gates.float().contiguous()
        dm_f = dim_masks.float().contiguous()
        bm_f = block_mode.float().contiguous()

        result = _C.fused_forward_and_save(
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
        return pipeline_backward(ctx, grad_output, _C.maxpool2x2_bwd, ctx.norm_mode)
