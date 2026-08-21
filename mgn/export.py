import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ExportedLayer(nn.Module):
    def __init__(self, i, in_c, out_c, w_conv, b_conv, mode, w_qkv=None, b_qkv=None, w_proj=None, b_proj=None):
        super().__init__()
        self.i = i
        self.mode = mode # 0: Plain, 1: ResNet, 2: ViT
        
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv.weight.data = w_conv.clone()
        self.conv.bias.data = b_conv.clone()
        self.norm = nn.GroupNorm(1, out_c)
        
        if self.mode == 2:
            attn_dim = out_c // 4
            self.qkv = nn.Conv2d(out_c, 3 * attn_dim, kernel_size=1)
            self.qkv.weight.data = w_qkv.clone()
            self.qkv.bias.data = b_qkv.clone()
            self.proj = nn.Conv2d(attn_dim, out_c, kernel_size=1)
            self.proj.weight.data = w_proj.clone()
            self.proj.bias.data = b_proj.clone()
            self.proj_norm = nn.GroupNorm(1, out_c)
            self.attn_dim = attn_dim
            
    def forward(self, x, h_prev):
        h_in = x if self.i == 0 else h_prev
        h_conv = self.conv(h_in)
        h_norm = self.norm(h_conv)
        h_plain_raw = F.relu(h_norm)
        
        h_expand = h_plain_raw if self.i == 0 else h_prev
        
        if self.mode == 0:
            return h_plain_raw
        elif self.mode == 1:
            return F.relu(h_expand + h_norm)
        elif self.mode == 2:
            H, W = h_norm.shape[2], h_norm.shape[3]
            h_attn_in = F.adaptive_avg_pool2d(h_norm, (8, 8)) if H > 8 else h_norm
            qkv = self.qkv(h_attn_in)
            batch = qkv.shape[0]
            qkv = qkv.view(batch, 3, self.attn_dim, -1)
            Q = qkv[:, 0].transpose(1, 2)
            K = qkv[:, 1]
            V = qkv[:, 2].transpose(1, 2)
            
            attn = F.softmax(torch.bmm(Q, K) / math.sqrt(self.attn_dim), dim=-1)
            out_trans = torch.bmm(attn, V).transpose(1, 2).reshape(batch, self.attn_dim, h_attn_in.shape[2], h_attn_in.shape[3])
            
            if H > 8:
                out_trans = F.interpolate(out_trans, size=(H, W), mode='bilinear', align_corners=False)
                
            out_proj = self.proj_norm(self.proj(out_trans))
            return F.relu(h_expand + h_norm + out_proj)

class ExportedModel(nn.Module):
    def __init__(self, layers, fc_w, fc_b, in_channels=1):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(fc_w.shape[1], fc_w.shape[0])
        self.fc.weight.data = fc_w.clone()
        self.fc.bias.data = fc_b.clone()
        self.in_channels = in_channels
        self.ln = nn.LayerNorm(fc_w.shape[1])
        
    def forward(self, x):
        if x.dim() == 2:
            side = int((x.shape[1] // self.in_channels) ** 0.5)
            x = x.reshape(-1, self.in_channels, side, side)
            
        h_prev = None
        for layer in self.layers:
            h_prev = layer(x, h_prev)
            
        h_pool = F.adaptive_avg_pool2d(h_prev, (7, 7))
        h_pool = h_pool.reshape(x.shape[0], -1)
        h_pool = self.ln(h_pool)
        return self.fc(h_pool)

def export_static_model(model, sample_idx=0, in_channels=1):
    """
    Trích xuất mô hình được sinh ra bởi HyperNetwork thành một mô hình PyTorch tĩnh
    chỉ chứa đúng cấu trúc và trọng số thực tế (Hard Pruning).
    """
    S = model.structure
    W = model.weights
    
    # Lấy thông tin cấu trúc của sample_idx
    L_max = S.layer_gates.shape[1]
    gates = S.layer_gates[sample_idx] # [L_max]
    
    # Tính L_eff (số lớp thực sự hoạt động, chặn >= 0.5)
    L_eff = (gates > 0.5).sum().item()
    if L_eff == 0: L_eff = 1 # Ít nhất 1 lớp
    
    # Tính D_eff (số kênh)
    dim_mask = S.dim_masks[sample_idx] # [L_max, D_max]
    
    exported_layers = []
    
    for i in range(L_eff):
        d_mask = dim_mask[i]
        out_c = (d_mask > 0.5).sum().item()
        if out_c == 0: out_c = 1
        
        in_c = in_channels if i == 0 else exported_layers[-1].conv.out_channels
        
        w_conv = W[f"c{i+1}.weight"][sample_idx] # [D_max, in_c_max, 3, 3]
        b_conv = W[f"c{i+1}.bias"][sample_idx] # [D_max]
        
        # Cắt xén (Slice) trọng số theo D_eff
        w_conv = w_conv[:out_c, :in_c, :, :]
        b_conv = b_conv[:out_c]
        
        # Xác định Block Mode (0: Plain, 1: ResNet, 2: ViT)
        mode = S.block_mode[sample_idx, i].argmax().item()
        
        w_qkv, b_qkv, w_proj, b_proj = None, None, None, None
        if mode == 2:
            attn_dim = out_c // 4
            if attn_dim == 0: attn_dim = 1
            w_qkv_full = W[f"attn{i+1}.qkv.weight"][sample_idx] # [3*attn_max, D_max, 1, 1]
            b_qkv_full = W[f"attn{i+1}.qkv.bias"][sample_idx]
            
            w_qkv = w_qkv_full[:3*attn_dim, :out_c, :, :]
            b_qkv = b_qkv_full[:3*attn_dim]
            
            w_proj_full = W[f"attn{i+1}.proj.weight"][sample_idx]
            b_proj_full = W[f"attn{i+1}.proj.bias"][sample_idx]
            
            w_proj = w_proj_full[:out_c, :attn_dim, :, :]
            b_proj = b_proj_full[:out_c]
            
        layer = ExportedLayer(i, in_c, out_c, w_conv, b_conv, mode, w_qkv, b_qkv, w_proj, b_proj)
        exported_layers.append(layer)
        
    # Fully Connected Layer
    fc_w_full = W["out.weight"][sample_idx] # [out_classes, 49 * D_max]
    fc_b_full = W["out.bias"][sample_idx]
    
    last_out_c = exported_layers[-1].conv.out_channels
    # fc_w is arranged as [out_c, 49 * D_max]. It is flattened from (D_max, 7, 7).
    # We must reshape it to (out_classes, D_max, 7, 7), slice it to (out_classes, last_out_c, 7, 7), then flatten again
    fc_w = fc_w_full.view(fc_w_full.shape[0], -1, 7, 7)
    fc_w = fc_w[:, :last_out_c, :, :].reshape(fc_w_full.shape[0], -1)
    
    exported_model = ExportedModel(exported_layers, fc_w, fc_b_full, in_channels)
    return exported_model
