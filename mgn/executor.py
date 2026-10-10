import torch
import torch.nn as nn
import torch.nn.functional as F

from .architecture import GeneratedModel


def _load_fused_pipeline():
    """Lazily import the fused v2 pipeline (first attempt only). Native
    extensions must NOT be loaded at `import mgn` time: CI/CPU-only
    consumers of this package never touch CUDA DLLs. Returns the
    FusedPipelineFunction class or None."""
    global _FUSED_PIPELINE
    if _FUSED_PIPELINE is _FUSED_UNTRIED:
        try:
            from .accel.dllsetup import bootstrap
            bootstrap()
            from .accel.fused_autograd import FusedPipelineFunction
            _FUSED_PIPELINE = FusedPipelineFunction
        except Exception:
            _FUSED_PIPELINE = None
    return _FUSED_PIPELINE


_FUSED_UNTRIED = object()
_FUSED_PIPELINE = _FUSED_UNTRIED


class TaskNetwork(nn.Module):
    def __init__(self, dropout=0.0, fused=False):
        """fused: route the conv-branch layer stack through the single-call
        fused CUDA pipeline (mgn.accel._C, v2 two-conv block algebra).
        Requires CUDA, a model carrying pi + raw prototypes, 2-mode block
        routing, <=5 layers (the pipeline pools internally after layers 5
        and 11) and the v2 prototype layout. When any precondition is not
        met the pure-eager path runs unchanged. Off by default: recorded
        benchmark numbers were produced eager."""
        super().__init__()
        self.dropout = dropout
        self.fused = fused
        self._fused_layout_cache = None

    def _fusable(self, x, model, S, W, L_max, batch):
        if not self.fused or not x.is_cuda or batch <= 0:
            return None
        pipeline = _load_fused_pipeline()
        if pipeline is None:
            return None
        protos = getattr(model, "prototypes", None)
        pi = getattr(model, "pi", None)
        if protos is None or pi is None or pi.shape[0] != batch:
            return None
        if S.dim_masks is None or S.dim_masks.dim() != 3:
            return None
        if S.block_mode is None or S.block_mode.shape[-1] != 2:
            return None
        if L_max > 5 or "c1.weight2" not in W:
            return None  # legacy single-conv layout has no v2 pipeline
        from .layout import get_prototype_layout, get_second_conv_layout
        key = (S.max_layers, S.max_dim, S.output_dim, S.in_channels)
        layout = self._fused_layout_cache
        if layout is None or layout[0] != key:
            total, offs, _info, _wo, _bo = get_prototype_layout(
                S.max_layers, S.max_dim, S.output_dim, S.in_channels)
            offs2 = get_second_conv_layout(
                S.max_layers, S.max_dim, S.output_dim, S.in_channels)
            layout = (key, total, offs, offs2)
            self._fused_layout_cache = layout
        if protos.shape[1] != layout[1]:
            return None  # layout mismatch -> eager, never corrupt math
        return pipeline, layout

    def forward(self, x: torch.Tensor, model: GeneratedModel) -> torch.Tensor:
        S = model.structure
        W = model.weights
        batch = x.shape[0]
        L_max = S.max_layers
        device = x.device

        arange = torch.arange(S.max_dim, device=device, dtype=torch.float32)
        D_eff = S.hidden_dim
        if D_eff.dim() > 1:
            D_eff = D_eff[:, 0]
        dim_mask_soft = torch.sigmoid((D_eff.unsqueeze(1) - arange.unsqueeze(0)) / 0.5)
        dim_mask = (dim_mask_soft > 0.5).float().detach() + dim_mask_soft - dim_mask_soft.detach()

        # Depth gates are INDEPENDENT per-layer Bernoulli gates, already
        # straight-through hardened at generation time (generator.py). The
        # previous F.gumbel_softmax(L_probs, dim=-1) normalized ACROSS the
        # layer dimension, turning them into a winner-take-all categorical
        # (≈1 active layer in train, often 0 at eval). Use them directly,
        # exactly like the conv branch below.
        each_layer_hard = S.layer_gates

        if getattr(S, "use_conv", False) or "c1.weight" in W:
            in_ch = getattr(S, "in_channels", 1)
            if x.dim() == 2:
                side = int((x.shape[1] // in_ch) ** 0.5)
                x = x.reshape(-1, in_ch, side, side)
            batch = x.shape[0]

            fused = self._fusable(x, model, S, W, L_max, batch)
            if fused is not None:
                pipeline, (  # noqa: E501 - early-return path, loop below untouched
                    _key, _total, offs, offs2) = fused
                H_img = x.shape[2]
                h = pipeline.apply(
                    x, model.prototypes, model.pi,
                    S.layer_gates, S.dim_masks, S.block_mode,
                    [o[0] for o in offs], [o[2] for o in offs],
                    [o[0] for o in offs2], [o[2] for o in offs2],
                    [o[4] for o in offs], [S.max_dim] * L_max,
                    [H_img] * L_max, [H_img] * L_max, S.max_dim,
                )
                # Head below is the exact copy of the eager branch's tail.
                h_pool = F.adaptive_avg_pool2d(h, (4, 4)).reshape(batch, -1)
                h_pool = F.layer_norm(h_pool, [h_pool.shape[-1]])
                h_pool = F.dropout(h_pool, p=0.3, training=self.training)
                if W["out.weight"].shape[0] == 1 and batch > 1:
                    return F.linear(h_pool, W["out.weight"][0], W["out.bias"][0])
                return (torch.bmm(h_pool.unsqueeze(1),
                                  W["out.weight"].transpose(1, 2)).squeeze(1)
                        + W["out.bias"])

            h = x
            
            for i in range(L_max):
                w_conv = W[f"c{i+1}.weight"]
                b_conv = W[f"c{i+1}.bias"]
                gate_c = S.layer_gates[:, i:i+1].view(-1, 1, 1, 1)
                dim_mask_c = S.dim_masks[:, i].view(-1, S.dim_masks.shape[-1], 1, 1) if S.dim_masks.dim() == 3 else S.dim_masks.view(-1, S.dim_masks.shape[-1], 1, 1)
                
                # In PyTorch, if batch size > 1 and we use bmm-style conv, we need group convolution.
                # W is dict of batched weights [batch, out_c, in_c, k, k] -> [batch * out_c, in_c, k, k]
                
                in_c = in_ch if i == 0 else w_conv.shape[2]
                out_c = w_conv.shape[1]
                
                if w_conv.shape[0] == 1 and batch > 1:
                    h_conv = F.conv2d(h, w_conv[0], b_conv[0], padding=1)
                else:
                    w_conv_flat = w_conv.reshape(batch * out_c, in_c, 3, 3)
                    b_conv_flat = b_conv.reshape(batch * out_c)
                    
                    h_flat = h.view(1, batch * in_c, h.shape[2], h.shape[3])
                    h_conv = F.conv2d(h_flat, w_conv_flat, b_conv_flat, groups=batch, padding=1)
                    h_conv = h_conv.view(batch, out_c, h_conv.shape[2], h_conv.shape[3])
                
                h_norm = F.layer_norm(h_conv, h_conv.shape[1:])
                h_plain_raw = F.relu(h_norm) * dim_mask_c
                h_plain = h_plain_raw
                h_expand = h_plain_raw if i == 0 else h * dim_mask_c
                
                if getattr(S, "block_mode", None) is not None:
                    g_plain = S.block_mode[:, i, 0].view(-1, 1, 1, 1)
                    g_resnet = S.block_mode[:, i, 1].view(-1, 1, 1, 1)
                    g_trans = S.block_mode[:, i, 2].view(-1, 1, 1, 1) if S.block_mode.shape[-1] > 2 else 0.0

                    # Real 2-conv residual block: the plain output feeds a
                    # second 3x3 conv (+GN); the skip is added AFTER it, so
                    # ResNet mode genuinely costs/computes more than Plain.
                    # Structures built without c2 weights (legacy) fall back
                    # to the old single-conv skip.
                    w2 = W.get(f"c{i+1}.weight2", None)
                    if w2 is not None:
                        b2 = W[f"c{i+1}.bias2"]
                        c2_in = h_plain_raw.shape[1]
                        if w2.shape[0] == 1 and batch > 1:
                            h_conv2 = F.conv2d(h_plain_raw, w2[0], b2[0], padding=1)
                        else:
                            w2_flat = w2.reshape(batch * c2_in, c2_in, 3, 3)
                            b2_flat = b2.reshape(batch * c2_in)
                            h2_flat = h_plain_raw.view(1, batch * c2_in,
                                                        h_plain_raw.shape[2], h_plain_raw.shape[3])
                            h_conv2 = F.conv2d(h2_flat, w2_flat, b2_flat,
                                               groups=batch, padding=1)
                            h_conv2 = h_conv2.view(batch, c2_in,
                                                   h_conv2.shape[2], h_conv2.shape[3])
                        h_norm2 = F.layer_norm(h_conv2, h_conv2.shape[1:])
                        h_resnet = F.relu(h_expand + h_norm2) * dim_mask_c
                    else:
                        h_resnet = F.relu(h_expand + h_norm) * dim_mask_c
                    w_qkv = W.get(f"attn{i+1}.qkv.weight", None)
                    if w_qkv is not None and g_trans.sum() > 0:
                        attn_dim = out_c // 4
                        b_qkv = W[f"attn{i+1}.qkv.bias"]
                        H_orig, W_orig = h_norm.shape[2], h_norm.shape[3]
                        if H_orig > 8: h_attn_in = F.adaptive_avg_pool2d(h_norm, (8, 8))
                        else: h_attn_in = h_norm
                        
                        if w_qkv.shape[0] == 1 and batch > 1:
                            qkv = F.conv2d(h_attn_in, w_qkv[0], b_qkv[0])
                        else:
                            w_qkv_flat = w_qkv.reshape(batch * 3 * attn_dim, out_c, 1, 1)
                            b_qkv_flat = b_qkv.reshape(batch * 3 * attn_dim)
                            qkv = F.conv2d(h_attn_in.view(1, batch * out_c, h_attn_in.shape[2], h_attn_in.shape[3]), w_qkv_flat, b_qkv_flat, groups=batch)
                        
                        qkv = qkv.view(batch, 3, attn_dim, h_attn_in.shape[2]*h_attn_in.shape[3])
                        Q = qkv[:, 0].transpose(1, 2)
                        K = qkv[:, 1]
                        V = qkv[:, 2].transpose(1, 2)
                        import math
                        attn = F.softmax(torch.bmm(Q, K) / math.sqrt(attn_dim), dim=-1)
                        attn = F.dropout(attn, p=0.1, training=self.training)
                        out_trans = torch.bmm(attn, V).transpose(1, 2).reshape(batch, attn_dim, h_attn_in.shape[2], h_attn_in.shape[3])
                        if H_orig > 8: out_trans = F.interpolate(out_trans, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
                        w_proj = W[f"attn{i+1}.proj.weight"]
                        b_proj = W[f"attn{i+1}.proj.bias"]
                        
                        if w_proj.shape[0] == 1 and batch > 1:
                            out_proj = F.conv2d(out_trans, w_proj[0], b_proj[0])
                        else:
                            w_proj_flat = w_proj.reshape(batch * out_c, attn_dim, 1, 1)
                            b_proj_flat = b_proj.reshape(batch * out_c)
                            out_proj = F.conv2d(out_trans.reshape(1, batch * attn_dim, out_trans.shape[2], out_trans.shape[3]), w_proj_flat, b_proj_flat, groups=batch)
                            
                        out_proj = out_proj.view(batch, out_c, out_trans.shape[2], out_trans.shape[3])
                        out_proj = F.layer_norm(out_proj, out_proj.shape[1:])
                        h_trans = F.relu(h_expand + h_norm + out_proj) * dim_mask_c
                    else:
                        h_trans = 0.0
                        
                    h_active = g_plain * h_plain + g_resnet * h_resnet + g_trans * h_trans
                else:
                    h_active = h_plain if i == 0 else F.relu(h_expand + h_norm) * dim_mask_c
                    
                if i == 0:
                    h = h_active
                else:
                    h = gate_c * h_active + (1.0 - gate_c) * h_expand

            # Adaptive pooling at the end (4x4 = 16 spatial elements matching 16 * max_dim head)
            h_pool = F.adaptive_avg_pool2d(h, (4, 4)).reshape(batch, -1)
            h_pool = F.layer_norm(h_pool, [h_pool.shape[-1]])
            h_pool = F.dropout(h_pool, p=0.3, training=self.training)
            
            if W["out.weight"].shape[0] == 1 and batch > 1:
                logits = F.linear(h_pool, W["out.weight"][0], W["out.bias"][0])
            else:
                logits = torch.bmm(h_pool.unsqueeze(1), W["out.weight"].transpose(1, 2)).squeeze(1) + W["out.bias"]
            return logits
        else:
            h = x
            for i in range(L_max):
                w_key = f"fc{i+1}.weight"
                b_key = f"fc{i+1}.bias"
                w = W[w_key]
                b = W[b_key]

                gate = each_layer_hard[:, i:i+1]

                # fc weights are stored [B, out, in] (generator view), so the
                # bmm must transpose: [B,1,in] x [B,in,out]. The old
                # h.unsqueeze(1) @ w only worked when h_dim == out_dim, i.e.
                # input_dim == max_dim.
                h_new = torch.bmm(h.unsqueeze(1), w.transpose(1, 2)).squeeze(1) + b

                if i == 0:
                    h = h_new
                else:
                    h_new = h_new * dim_mask
                    h = gate * (h_new + h) + (1 - gate) * h

                if i < L_max - 1:
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)

            w_out = W["out.weight"]
            b_out = W["out.bias"]
            h = torch.bmm(h.unsqueeze(1), w_out).squeeze(1) + b_out

            return h
