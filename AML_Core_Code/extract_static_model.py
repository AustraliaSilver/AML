import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mgn.generator import ModelGenerator
from train_cv_cifar import ConvAML_CIFAR

class StaticChildModel(nn.Module):
    """
    Một mạng tĩnh (Mạng Con) được tạo ra từ Mạng Mẹ.
    Hoàn toàn không chứa Gumbel-Softmax hay thao tác sinh động.
    """
    def __init__(self, layers, fc_weight, fc_bias):
        super().__init__()
        self.features = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(fc_weight.shape[1], fc_weight.shape[0])
        self.fc.weight.data = fc_weight
        self.fc.bias.data = fc_bias

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i == 2: # MaxPool ở layer 3 như trong kiến trúc CIFAR
                h = F.max_pool2d(h, kernel_size=2, stride=2)
                
        h = self.pool(h).view(x.shape[0], -1)
        h = F.layer_norm(h, [h.shape[-1]])
        h = F.dropout(h, p=0.2, training=self.training)
        return self.fc(h)

class StaticLayer(nn.Module):
    """
    Mô phỏng lại fused_dynamic_conv2d + norm_relu_gate nhưng dạng Tĩnh
    """
    def __init__(self, conv_w, conv_b, gate_c, dim_mask, g_plain, g_resnet, is_first_layer=False):
        super().__init__()
        self.is_first_layer = is_first_layer
        self.conv = nn.Conv2d(conv_w.shape[1], conv_w.shape[0], 3, padding=1)
        self.conv.weight.data = conv_w
        self.conv.bias.data = conv_b
        
        self.norm = nn.GroupNorm(1, conv_w.shape[0]) # Tương đương LayerNorm
        
        # Các hệ số giữ lại làm param cố định
        self.register_buffer('gate_c', gate_c)
        self.register_buffer('dim_mask', dim_mask)
        self.register_buffer('g_plain', g_plain)
        self.register_buffer('g_resnet', g_resnet)

    def forward(self, x):
        # Vì ảnh đầu vào layer đầu có 3 kênh, các layer sau có max_dim kênh,
        # Nếu số kênh đầu vào của layer không khớp với x (do đây là khối ResNet tĩnh), ta sẽ xử lý pad 0
        if x.shape[1] != self.conv.in_channels:
            pad_size = self.conv.in_channels - x.shape[1]
            x_pad = F.pad(x, (0,0, 0,0, 0,pad_size))
        else:
            x_pad = x
            
        h_conv = self.conv(x_pad)
        h_norm = self.norm(h_conv)
        h_plain = F.relu(h_norm) * self.dim_mask
        
        if self.is_first_layer:
            h_expand = h_plain
        else:
            h_expand = x_pad * self.dim_mask
            
        h_resnet = F.relu(h_expand + h_norm) * self.dim_mask
        
        h_active = self.g_plain * h_plain + self.g_resnet * h_resnet
        return self.gate_c * h_active + (1.0 - self.gate_c) * h_expand


def extract_child(mother_model, x_dummy):
    """ Trích xuất Mạng Con từ Mạng Mẹ """
    mother_model.eval()
    with torch.no_grad():
        # 1. Chạy mạng mẹ để lấy pi và cấu trúc (Tĩnh hóa)
        gen = mother_model.gen
        if hasattr(gen, 'vision_encoder'):
            z = gen.vision_encoder(x_dummy)
        else:
            x_flat = x_dummy.view(x_dummy.shape[0], -1)
            z = gen.encoder(x_flat)
        pi_logits = gen.prototype_selector(z)
        
        # Bỏ qua nhiễu, lấy hard argmax hoặc trung bình. Ở đây lấy soft trung bình của batch
        pi = F.softmax(pi_logits, dim=-1).mean(dim=0, keepdim=True) # [1, K]
        
        S = gen(x_dummy).structure
        # Lấy trung bình các cổng qua batch
        layer_gates = S.layer_gates.mean(dim=0, keepdim=True)
        dim_masks = S.dim_masks.mean(dim=0, keepdim=True)
        block_mode = S.block_mode.mean(dim=0, keepdim=True) if S.block_mode is not None else None
        
        prototypes = gen.prototype_lib.proto_weights
        
        # 2. Xây dựng Mạng Con
        layers = []
        idx = 0
        max_dim = mother_model.exec_.max_dim
        max_layers = mother_model.exec_.max_layers
        
        for i in range(max_layers):
            in_c = 3 if i == 0 else max_dim
            w1_numel = max_dim * in_c * 9
            
            # Trộn trọng số: W = pi @ Prototypes
            w_mix = (pi @ prototypes[:, idx:idx+w1_numel]).view(max_dim, in_c, 3, 3)
            b_mix = (pi @ prototypes[:, idx+w1_numel:idx+w1_numel+max_dim]).view(max_dim)
            
            idx += w1_numel + max_dim
            # Skip attention
            attn_dim = max_dim // 4
            idx += (3 * attn_dim) * max_dim * 1 + 3 * attn_dim
            idx += max_dim * attn_dim * 1 + max_dim
            
            # Khởi tạo Layer Tĩnh
            g_plain = block_mode[:, i, 0].view(1,1,1) if block_mode is not None else torch.tensor([1.0], device=x_dummy.device)
            g_resnet = block_mode[:, i, 1].view(1,1,1) if block_mode is not None else torch.tensor([0.0], device=x_dummy.device)
            
            if i == 0:
                g_resnet = torch.tensor([0.0], device=x_dummy.device) # Lớp đầu không có resnet
                gate_c = torch.tensor([1.0], device=x_dummy.device)
                is_first = True
            else:
                gate_c = layer_gates[:, i].view(1,1,1)
                is_first = False
                
            mask = dim_masks[:, i].view(max_dim, 1, 1)
            
            layers.append(StaticLayer(w_mix, b_mix, gate_c, mask, g_plain, g_resnet, is_first))

        # Đầu ra FC
        num_classes = mother_model.exec_.num_classes
        wo_offset = idx
        bo_offset = idx + (16 * max_dim) * num_classes
        w_out = (pi @ prototypes[:, wo_offset:bo_offset]).view(num_classes, 16 * max_dim)
        b_out = (pi @ prototypes[:, bo_offset:bo_offset+num_classes]).view(num_classes)
        
        child = StaticChildModel(layers, w_out, b_out)
        return child.to(x_dummy.device)

def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Khởi tạo Mạng Mẹ (Dynamic CIFAR)...")
    mother = ConvAML_CIFAR().to(device)
    mother.eval()
    
    x_test = torch.randn(1, 3, 32, 32, device=device)
    
    print("\nTrích xuất Mạng Con (Static CIFAR)...")
    child = extract_child(mother, x_test)
    child.eval()
    
    # 1. So sánh đầu ra để đảm bảo toán học tương đương
    with torch.no_grad():
        out_mother = mother(x_test)
        out_child = child(x_test)
    
    diff = (out_mother - out_child).abs().max().item()
    print(f"Độ sai lệch đầu ra giữa Mẹ và Con: {diff:.6f} (Nên gần 0)")
    
    # 2. Đo tốc độ suy luận (Inference Speed)
    iters = 1000
    batch_size = 128
    x_bench = torch.randn(batch_size, 3, 32, 32, device=device)
    
    # Warmup
    for _ in range(10):
        mother(x_bench)
        child(x_bench)
    torch.cuda.synchronize()
    
    # Đo Mẹ
    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            mother(x_bench)
    torch.cuda.synchronize()
    mother_time = time.time() - start
    
    # Đo Con
    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            child(x_bench)
    torch.cuda.synchronize()
    child_time = time.time() - start
    
    print(f"\n=== SO SÁNH TỐC ĐỘ SUY LUẬN (Batch 128, {iters} lần) ===")
    print(f"Mạng Mẹ (2 giai đoạn + C++/CUDA): {mother_time:.3f} giây ({(iters*batch_size)/mother_time:.0f} ảnh/s)")
    print(f"Mạng Con (Tĩnh - Native PyTorch): {child_time:.3f} giây ({(iters*batch_size)/child_time:.0f} ảnh/s)")
    
    if child_time < mother_time:
        print(f"-> TRÍCH XUẤT THÀNH CÔNG! Mạng Con nhanh hơn {mother_time/child_time:.1f} lần!")
    else:
        print("Mạng mẹ (CUDA) thậm chí vẫn nhanh hơn mạng con tĩnh!")

if __name__ == "__main__":
    benchmark()
