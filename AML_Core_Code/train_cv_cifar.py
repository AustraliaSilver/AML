import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os
import sys

cuda_toolkit_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin'
if os.path.isdir(cuda_toolkit_bin):
    os.add_dll_directory(cuda_toolkit_bin)
torch_lib = r'C:\Users\hunga\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib'
if os.path.isdir(torch_lib):
    os.add_dll_directory(torch_lib)

import torch._inductor.codecache as _cc
import pathlib as _pathlib
_orig_write_atomic = _cc.write_atomic
def _patched_write_atomic(path, data, **kwargs):
    try:
        _orig_write_atomic(path, data, **kwargs)
    except FileExistsError:
        _pathlib.Path(path).unlink(missing_ok=True)
        _orig_write_atomic(path, data, **kwargs)
_cc.write_atomic = _patched_write_atomic

import torch._dynamo
torch._dynamo.config.suppress_errors = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mgn.generator import ModelGenerator
import mgn.accel.functions as aml_ops

try:
    import mgn.accel._C as _C
    _C_available = True
except ImportError:
    _C_available = False

try:
    from mgn.accel.fused_autograd import FusedPipelineFunction
    _fused_autograd_available = True
except ImportError:
    _fused_autograd_available = False


class AcceleratedCVNetwork(nn.Module):
    def __init__(self, max_layers=6, max_dim=64, num_classes=10):
        super().__init__()
        self.max_layers = max_layers
        self.max_dim = max_dim
        self.num_classes = num_classes
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        offsets = []
        idx = 0
        for i in range(max_layers):
            in_c = 3 if i == 0 else max_dim
            w1_numel = max_dim * in_c * 9
            w_start = idx
            b_start = idx + w1_numel
            w_end = b_start
            b_end = b_start + max_dim
            idx = b_end
            attn_dim = max_dim // 4
            idx += (3 * attn_dim) * max_dim * 1 + 3 * attn_dim
            idx += max_dim * attn_dim * 1 + max_dim
            offsets.append((w_start, w_end, b_start, b_end, in_c))
        self.offsets = offsets
        self.wo_offset = idx
        self.bo_offset = idx + (16 * max_dim) * num_classes

        self._build_slices()

    def _build_slices(self):
        total_end = self.bo_offset + self.num_classes
        self.register_buffer('_proto_end', torch.tensor(total_end), persistent=False)

        self._layer_info = []
        for i in range(self.max_layers):
            w_start, w_end, b_start, b_end, in_c = self.offsets[i]
            C_out = self.max_dim
            self._layer_info.append((w_start, w_end, b_start, b_end, in_c, C_out))

    def forward(self, x, prototypes, pi, model_structure):
        S = model_structure
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]

        # Robust dynamic execution path
        h = x
        for i in range(self.max_layers):
            w_start, w_end, b_start, b_end, in_c, C_out = self._layer_info[i]

            h_conv = aml_ops.fused_dynamic_conv2d(h, prototypes, pi, w_start, b_start)

            gate_c = S.layer_gates[:, i:i+1].view(-1, 1, 1, 1)
            dim_mask_c = S.dim_masks[:, i].view(-1, S.dim_masks.shape[-1], 1, 1)

            h_norm = F.group_norm(h_conv, num_groups=1)
            h_plain = F.relu(h_norm) * dim_mask_c
            h_expand = h_plain if i == 0 else h * dim_mask_c

            g_plain = S.block_mode[:, i, 0].view(-1, 1, 1, 1) if S.block_mode is not None else 1.0
            g_resnet = S.block_mode[:, i, 1].view(-1, 1, 1, 1) if S.block_mode is not None else 0.0

            h_resnet = F.relu(h_expand + h_norm) * dim_mask_c
            h_active = g_plain * h_plain + g_resnet * h_resnet

            if i == 0:
                h = h_active
            else:
                h = gate_c * h_active + (1.0 - gate_c) * h_expand

            if i == 2:
                h = F.max_pool2d(h, kernel_size=2, stride=2)
                H, W = H // 2, W // 2

        h_pool = self.pool(h).reshape(B, -1)
        h_pool = F.layer_norm(h_pool, [h_pool.shape[-1]])
        h_pool = F.dropout(h_pool, p=0.2, training=self.training)

        W_out = torch.mm(pi, prototypes[:, self.wo_offset:self.bo_offset]).view(B, self.num_classes, 16 * self.max_dim)
        b_out = torch.mm(pi, prototypes[:, self.bo_offset:self.bo_offset + self.num_classes]).view(B, self.num_classes)

        out = torch.bmm(W_out, h_pool.unsqueeze(2)).squeeze(2) + b_out
        return out


class VisionModelGenerator(ModelGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.GroupNorm(1, 16), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.GroupNorm(1, 32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.GroupNorm(1, 64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, self.latent_dim),
            nn.ReLU()
        )
        self.encoder = nn.Identity()

    def forward(self, x, tau=1.0, noise_scale=1.0):
        z_vision = self.vision_encoder(x)
        self._cached_z = z_vision
        return super().forward(z_vision, tau=tau, noise_scale=noise_scale)


class ConvAML_CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = VisionModelGenerator(
            input_dim=3 * 32 * 32,
            output_dim=10,
            max_layers=6,
            max_dim=128,
            dim_choices=(64, 96, 128),
            latent_dim=64,
            noise_dim=64,
            num_prototypes=12,
            use_conv=True,
            in_channels=3
        )

        self.exec_ = AcceleratedCVNetwork(max_layers=6, max_dim=128, num_classes=10)

        total_proto_params = self.gen.prototype_lib.proto_weights.shape[1]
        self.num_hts_proto = 8

        self.hts_memory = nn.Parameter(torch.randn(self.num_hts_proto, total_proto_params) * 0.02)

        self.hts_router = nn.Sequential(
            nn.Linear(self.gen.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_hts_proto)
        )

        self.memory_correction = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 10)
        )
        self.memory_gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x):
        model = self.gen(x)
        pi_aml = F.gumbel_softmax(model.pi_logits, tau=1.0, hard=False)

        z = self.gen._cached_z
        pi_hts = F.softmax(self.hts_router(z), dim=-1)

        mixed_prototypes = torch.cat([self.gen.prototype_lib.proto_weights, self.hts_memory], dim=0)
        mixed_pi = torch.cat([pi_aml, pi_hts], dim=-1)

        mixed_prototypes = mixed_prototypes.to(mixed_pi.dtype)

        out_aml = self.exec_(x, mixed_prototypes, mixed_pi, model.structure)

        gate = torch.sigmoid(self.memory_gate)
        correction = self.memory_correction(out_aml.detach())
        out = out_aml + gate * correction

        return out, model.structure


def train_cifar():
    torch.backends.cuda.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_amp and amp_dtype == torch.float16 else None
    if device == "cuda":
        torch.cuda.empty_cache()

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3 if device == "cuda" else 0
    print(f"GPU: {gpu_name} | VRAM: {gpu_mem:.1f}GB | AMP: {use_amp} | cuDNN.benchmark: ON", flush=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    print(f"Loading CIFAR10 from {os.path.abspath(data_root)}...", flush=True)
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    print("Dataset loaded. Initializing model...", flush=True)

    model = ConvAML_CIFAR().to(device)
    print("Model initialized in pure PyTorch eager mode.", flush=True)

    gen_params = [p for p in model.gen.parameters() if p.requires_grad]
    hts_params = [p for p in (list(model.hts_router.parameters()) + [model.hts_memory] + list(model.memory_correction.parameters()) + [model.memory_gate]) if p.requires_grad]

    optimizer_gen = optim.AdamW(gen_params, lr=1e-4, weight_decay=1e-4)
    optimizer_hts = optim.AdamW(hts_params, lr=5e-4, weight_decay=1e-4)

    epochs = 10
    scheduler_gen = optim.lr_scheduler.CosineAnnealingLR(optimizer_gen, T_max=epochs)
    scheduler_hts = optim.lr_scheduler.CosineAnnealingLR(optimizer_hts, T_max=epochs)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        start_time = time.time()
        for inputs, labels in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer_gen.zero_grad(set_to_none=True)
            optimizer_hts.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                outputs, structure = model(inputs)
                loss = criterion(outputs, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer_gen)
                scaler.unscale_(optimizer_hts)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer_gen)
                scaler.step(optimizer_hts)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_gen.step()
                optimizer_hts.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_time = time.time() - start_time
        train_acc = 100. * correct / total

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    outputs, _ = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "aml_core_best.pth")

        mem_gate = torch.sigmoid(model.memory_gate).item()
        mem_norm = model.hts_memory.norm().item()
        print(f"Epoch [{epoch+1:02d}/{epochs}] | {epoch_time:.1f}s ({total/epoch_time:.0f} img/s) | Loss: {total_loss/len(trainloader):.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | MemGate: {mem_gate:.3f} | MemNorm: {mem_norm:.4f}", flush=True)

        scheduler_gen.step()
        scheduler_hts.step()

    print(f"=== BEST ACCURACY: {best_acc:.2f}% ===")

if __name__ == "__main__":
    train_cifar()
