import torch
import torch.nn as nn
import torch.nn.functional as F

from .architecture import GeneratedModel


class TaskNetwork(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = dropout

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
                if x.shape[1] == 3072:
                    x = x.reshape(-1, 3, 32, 32)
                elif x.shape[1] == 784:
                    x = x.reshape(-1, 1, 28, 28)
                else:
                    side = int((x.shape[1] // in_ch) ** 0.5)
                    x = x.reshape(-1, in_ch, side, side)
            batch = x.shape[0]
            
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
                
                h_norm = F.group_norm(h_conv, num_groups=1)
                h_plain_raw = F.relu(h_norm) * dim_mask_c
                h_plain = h_plain_raw
                h_expand = h_plain_raw if i == 0 else h * dim_mask_c
                
                if getattr(S, "block_mode", None) is not None:
                    g_plain = S.block_mode[:, i, 0].view(-1, 1, 1, 1)
                    g_resnet = S.block_mode[:, i, 1].view(-1, 1, 1, 1)
                    g_trans = S.block_mode[:, i, 2].view(-1, 1, 1, 1) if S.block_mode.shape[-1] > 2 else 0.0
                    
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
                        out_proj = F.group_norm(out_proj, num_groups=1)
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
