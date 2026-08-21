import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.checkpoint import checkpoint

CONV_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}\
    .get(os.environ.get("AML_CONV_DTYPE", "bf16"), torch.bfloat16)

class AcceleratedCVNetwork(nn.Module):
    def __init__(self, max_layers=6, max_dim=64, num_classes=10, in_channels=3):
        super().__init__()
        self.max_layers = max_layers
        self.max_dim = max_dim
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.eager = True
        self._shared_hit = 0
        self._total_calls = 0

        ad = max(1, max_dim // 4)
        self.mlp_net = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(max_dim, max_dim * 2, 1), nn.GELU(),
                nn.Conv2d(max_dim * 2, max_dim, 1),
            ) for _ in range(max_layers)
)
        self.vit_qkv = nn.ModuleList(
            nn.Conv2d(max_dim, 3 * ad, 1) for _ in range(max_layers)
        )
        self.vit_proj = nn.ModuleList(
            nn.Conv2d(ad, max_dim, 1) for _ in range(max_layers)
        )
        self.vit_pnorm = nn.ModuleList(
            nn.GroupNorm(1, max_dim) for _ in range(max_layers)
        )
        from AML_Core_Code.train_cv_cifar_18 import get_prototype_layout
        self.total_dim, self.offsets, self._layer_info, self.wo_offset, self.bo_offset = get_prototype_layout(self.max_layers, self.max_dim, self.num_classes, in_channels=self.in_channels)
        self._build_slices()

    def _build_slices(self):
        total_end = self.bo_offset + self.num_classes
        self.register_buffer('_proto_end', torch.tensor(total_end), persistent=False)
        self._layer_info = []
        for i in range(self.max_layers):
            w_start, w_end, b_start, b_end, in_c = self.offsets[i]
            C_out = self.max_dim
            self._layer_info.append((w_start, w_end, b_start, b_end, in_c, C_out))

    def _forward_layer(self, h, W_mix, b_mix, shared_conv, gate_c, dim_mask_c, bm_i, i, use_mlp, use_vit, use_resnet, use_plain):
        B = h.shape[0]
        C_out = W_mix.shape[1]
        in_c = W_mix.shape[2]
        if CONV_DTYPE == torch.float32:
            if shared_conv:
                h_conv = F.conv2d(h, W_mix[0], b_mix.view(-1), padding=1)
            else:
                h_conv = F.conv2d(h, W_mix, b_mix, groups=B, padding=1)
        else:
            if shared_conv:
                W_bf16 = W_mix.reshape(C_out, in_c, 3, 3).to(CONV_DTYPE)
                h_conv = F.conv2d(h.to(CONV_DTYPE), W_bf16, b_mix.reshape(-1).to(CONV_DTYPE), padding=1).to(torch.float32).view(B, C_out, h.shape[2], h.shape[3])
            else:
                h_bf16 = h.reshape(1, B * in_c, h.shape[2], h.shape[3]).to(CONV_DTYPE)
                W_bf16 = W_mix.reshape(B * C_out, in_c, 3, 3).to(CONV_DTYPE)
                b_bf16 = b_mix.reshape(-1).to(CONV_DTYPE)
                h_conv = F.conv2d(h_bf16, W_bf16, b_bf16, groups=B, padding=1).to(torch.float32).view(B, C_out, h.shape[2], h.shape[3])

        h_norm = F.group_norm(h_conv, num_groups=1)
        h_plain = F.relu(h_norm) * dim_mask_c
        h_expand = h_plain if i == 0 else h * dim_mask_c

        h_active = torch.zeros_like(h_plain)
        if use_plain:
            h_active = h_active + bm_i[:, 0].view(-1, 1, 1, 1) * h_plain
        if use_resnet:
            h_resnet = F.relu(h_expand + h_norm) * dim_mask_c
            h_active = h_active + bm_i[:, 1].view(-1, 1, 1, 1) * h_resnet
        if use_vit:
            ad = max(1, self.max_dim // 4)
            hh, ww = h_norm.shape[2], h_norm.shape[3]
            tv = F.adaptive_avg_pool2d(h_norm, (4, 4)) if hh > 4 else h_norm
            qkv = self.vit_qkv[i](tv).view(B, 3, ad, -1)
            Q, K, V = qkv[:, 0].transpose(1, 2), qkv[:, 1], qkv[:, 2].transpose(1, 2)
            att = F.softmax(torch.bmm(Q, K) / math.sqrt(ad), dim=-1)
            t = torch.bmm(att, V).transpose(1, 2).reshape(B, ad, tv.shape[2], tv.shape[3])
            if hh > 4: t = F.interpolate(t, size=(hh, ww), mode="bilinear", align_corners=False)
            h_vit = F.relu(h_expand + h_norm + self.vit_pnorm[i](self.vit_proj[i](t))) * dim_mask_c
            h_active = h_active + bm_i[:, 2].view(-1, 1, 1, 1) * h_vit
        if use_mlp:
            h_mlp = F.relu(self.mlp_net[i](h_norm)) * dim_mask_c
            h_active = h_active + bm_i[:, 3].view(-1, 1, 1, 1) * h_mlp

        if i == 0:
            h_out = h_active
        else:
            h_out = gate_c * h_active + (1.0 - gate_c) * h_expand

        return h_out

    def forward(self, x, prototypes, pi, model_structure, delta=None):
        S, B = model_structure, x.shape[0]
        h = x
        # Fix 1: batches whose entire pi is identical share one conv kernel (groups=1)
        shared_conv = False
        if os.environ.get("AML_NO_SHARED") != "1":
            try:
                shared_conv = bool((pi[1:] == pi[:1]).all().item())
            except Exception:
                shared_conv = False
        if hasattr(self, '_shared_hit') and shared_conv:
            self._shared_hit += 1
        if hasattr(self, '_total_calls'):
            self._total_calls += 1
        for i in range(self.max_layers):
            w_start, w_end, _, b_end, in_c, C_out = self._layer_info[i]
            gate_c = S.layer_gates[:, i:i+1].view(-1, 1, 1, 1)
            dim_mask_c = S.dim_masks[:, i].view(-1, S.dim_masks.shape[-1], 1, 1)
            bm_i = S.block_mode[:, i, :] if S.block_mode is not None else torch.zeros(B, 4, device=x.device)

            # Fix 2: one fused GEMM over [W | b] per layer (numerically identical to 2 mm's)
            proto_slice = prototypes[:, w_start:b_end]
            W_out_all = (pi[:1] if shared_conv else pi) @ proto_slice
            if delta is not None:
                W_out_all = W_out_all + (delta[:1] if shared_conv else delta)[:, w_start:b_end]
            W_conv = W_out_all[:, :w_end - w_start].view(B if not shared_conv else 1, C_out, in_c, 3, 3)
            b_conv = W_out_all[:, w_end - w_start:]

            # Fix 3: run only the block modes actually present in the batch
            # (AML_NO_UNIQUE=1 disables the prune and runs all 4 experts).
            if os.environ.get("AML_NO_UNIQUE") == "1":
                use_plain, use_resnet, use_vit, use_mlp = True, True, True, True
            else:
                uniq_i = torch.unique(bm_i.argmax(dim=-1)).tolist()
                use_plain, use_resnet, use_vit, use_mlp = (0 in uniq_i), (1 in uniq_i), (2 in uniq_i), (3 in uniq_i)

            if os.environ.get("AML_NO_CKPT") == "1":
                h = self._forward_layer(h, W_conv, b_conv, shared_conv, gate_c, dim_mask_c, bm_i, i, use_mlp, use_vit, use_resnet, use_plain)
            else:
                h = checkpoint(self._forward_layer, h, W_conv, b_conv, shared_conv, gate_c, dim_mask_c, bm_i, i, use_mlp, use_vit, use_resnet, use_plain, use_reentrant=False)

            if i == 5 or i == 11:
                h = F.max_pool2d(h, kernel_size=2, stride=2)

        h_pool = F.dropout(F.layer_norm(self.pool(h).reshape(B, -1), [self.max_dim * 16]), p=0.2, training=self.training)
        W_out = torch.mm(pi, prototypes[:, self.wo_offset:self.bo_offset]).view(B, self.num_classes, 16 * self.max_dim)
        b_out = torch.mm(pi, prototypes[:, self.bo_offset:self.bo_offset + self.num_classes]).view(B, self.num_classes)
        if delta is not None:
            W_out = W_out + delta[:, self.wo_offset:self.bo_offset].view(B, self.num_classes, 16 * self.max_dim)
            b_out = b_out + delta[:, self.bo_offset:self.bo_offset + self.num_classes]
        return torch.bmm(W_out, h_pool.unsqueeze(2)).squeeze(2) + b_out

def get_prototype_layout(max_layers, max_dim, num_classes, in_channels=3):
    total_dim = 0
    offsets = []
    in_c = in_channels
    for i in range(max_layers):
        C_out = max_dim
        per_oc = in_c * 9
        w_start = total_dim
        w_end = w_start + C_out * per_oc
        b_start = w_end
        b_end = b_start + C_out
        offsets.append((w_start, w_end, b_start, b_end, in_c))
        total_dim = b_end
        in_c = C_out
    wo_offset = total_dim
    total_dim += num_classes * (max_dim * 16)
    bo_offset = total_dim
    total_dim += num_classes
    return total_dim, offsets, [], wo_offset, bo_offset