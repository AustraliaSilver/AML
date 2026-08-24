import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os
import argparse
import sys
import copy

cuda_toolkit_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin'
if os.path.isdir(cuda_toolkit_bin): os.add_dll_directory(cuda_toolkit_bin)
torch_lib = r'C:\Users\hunga\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib'
if os.path.isdir(torch_lib): os.add_dll_directory(torch_lib)

import torch._inductor.codecache as _cc
import pathlib as _pathlib
_orig_write_atomic = _cc.write_atomic
def _patched_write_atomic(path, data, **kwargs):
    try: _orig_write_atomic(path, data, **kwargs)
    except FileExistsError:
        _pathlib.Path(path).unlink(missing_ok=True)
        _orig_write_atomic(path, data, **kwargs)
_cc.write_atomic = _patched_write_atomic

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AML_Core_Code'))
from train_cv_cifar import ConvAML_CIFAR, AcceleratedCVNetwork
from mgn.architecture import ModelStructure
import torch.nn.functional as F
import math


class CutoutTransform:
    def __init__(self, length=8):
        self.length = length
    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        y = torch.randint(h, (1,)).item()
        x = torch.randint(w, (1,)).item()
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        img = img.clone()
        img[:, y1:y2, x1:x2] = 0.0
        return img


def cutmix_data(x, y, alpha=1.0):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    bbx1 = max(0, cx - cut_w // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(W, cx + cut_w // 2)
    bby2 = min(H, cy + cut_h // 2)
    return bbx1, bby1, bbx2, bby2


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class ConvAML_CIFAR_v3(nn.Module):
    """v3: K=32, latent=128, CAF attention, better router, HTS=16"""
    def __init__(self, K=32, latent_dim=128, max_layers=6, max_dim=128, num_hts=16):
        super().__init__()
        self.K = K
        self.max_layers = max_layers
        self.max_dim = max_dim

        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(1, 32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(1, 64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(1, 128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )

        self.prototype_selector = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, K)
        )

        weight_dim = 0
        for i in range(max_layers):
            in_c = 3 if i == 0 else max_dim
            weight_dim += max_dim * in_c * 9 + max_dim
            attn_dim = max_dim // 4
            weight_dim += (3 * attn_dim) * max_dim + 3 * attn_dim
            weight_dim += max_dim * attn_dim + max_dim
        weight_dim += (16 * max_dim) * 10 + 10

        self.proto_embed = nn.Parameter(torch.randn(K, 64) * 0.02)
        self.proto_weights = nn.Parameter(torch.zeros(K, weight_dim))
        self._init_prototypes()

        self.hts_memory = nn.Parameter(torch.randn(num_hts, weight_dim) * 0.02)
        self.hts_router = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, num_hts)
        )
        self.memory_correction = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10)
        )
        self.memory_gate = nn.Parameter(torch.tensor(-1.0))

        self.exec_ = AcceleratedCVNetwork(max_layers=max_layers, max_dim=max_dim, num_classes=10)

    def _init_prototypes(self):
        with torch.no_grad():
            for k in range(self.K):
                idx = 0
                for i in range(self.max_layers):
                    in_c = 3 if i == 0 else self.max_dim
                    w1 = torch.empty(self.max_dim, in_c, 3, 3)
                    nn.init.kaiming_uniform_(w1, a=math.sqrt(5))
                    self.proto_weights.data[k, idx:idx + w1.numel()] = w1.flatten()
                    idx += w1.numel()
                    self.proto_weights.data[k, idx:idx + self.max_dim] = 0.0
                    idx += self.max_dim
                    attn_dim = self.max_dim // 4
                    w_qkv = torch.empty(3 * attn_dim, self.max_dim, 1, 1)
                    nn.init.kaiming_uniform_(w_qkv, a=math.sqrt(5))
                    self.proto_weights.data[k, idx:idx + w_qkv.numel()] = w_qkv.flatten()
                    idx += w_qkv.numel()
                    self.proto_weights.data[k, idx:idx + 3 * attn_dim] = 0.0
                    idx += 3 * attn_dim
                    w_proj = torch.empty(self.max_dim, attn_dim, 1, 1)
                    nn.init.kaiming_uniform_(w_proj, a=math.sqrt(5))
                    self.proto_weights.data[k, idx:idx + w_proj.numel()] = w_proj.flatten()
                    idx += w_proj.numel()
                    self.proto_weights.data[k, idx:idx + self.max_dim] = 0.0
                    idx += self.max_dim
                wo = torch.empty(10, 16 * self.max_dim)
                nn.init.kaiming_uniform_(wo, a=math.sqrt(5))
                self.proto_weights.data[k, idx:idx + wo.numel()] = wo.flatten()
                idx += wo.numel()
                self.proto_weights.data[k, idx:idx + 10] = 0.0

    def caf_attention(self, logits):
        """Contrasting-Driven Attention Function (KernelWarehouse ICML 2024).
        alpha_ij = z_ij / sum(|z_ip|) — allows negative weights."""
        denom = logits.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        alpha = logits / denom
        return alpha

    def forward(self, x):
        B = x.shape[0]
        device = x.device

        z = self.vision_encoder(x)

        pi_aml = self.caf_attention(self.prototype_selector(z))
        pi_hts = F.softmax(self.hts_router(z), dim=-1)

        mixed_prototypes = torch.cat([self.proto_weights, self.hts_memory], dim=0)
        mixed_pi = torch.cat([pi_aml, pi_hts], dim=-1)

        fixed_struct = ModelStructure(
            layer_gates=torch.ones(B, self.max_layers, device=device),
            dim_weights=torch.ones(B, self.max_layers, 1, device=device),
            residual=torch.zeros(B, 1, device=device),
            dim_masks=torch.ones(B, self.max_layers, self.max_dim, device=device),
            block_mode=F.one_hot(torch.full((B, self.max_layers), 1, dtype=torch.long, device=device), 3).float(),
            use_conv=True, in_channels=3, max_layers=self.max_layers, max_dim=self.max_dim,
            dim_choices=(64, 96, 128), output_dim=10
        )

        out_aml = self.exec_(x, mixed_prototypes, mixed_pi, fixed_struct)
        gate = torch.sigmoid(self.memory_gate)
        correction = self.memory_correction(out_aml.detach())
        return out_aml + gate * correction


class ConvAML_CIFAR_v3_no_hts(nn.Module):
    """v3 without HTS: K=32, CAF attention, better router, no memory correction"""
    def __init__(self, K=32, latent_dim=128, max_layers=6, max_dim=128):
        super().__init__()
        self.K = K
        self.max_layers = max_layers
        self.max_dim = max_dim

        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(1, 32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(1, 64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(1, 128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )

        self.prototype_selector = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, K)
        )

        weight_dim = 0
        for i in range(max_layers):
            in_c = 3 if i == 0 else max_dim
            weight_dim += max_dim * in_c * 9 + max_dim
            attn_dim = max_dim // 4
            weight_dim += (3 * attn_dim) * max_dim + 3 * attn_dim
            weight_dim += max_dim * attn_dim + max_dim
        weight_dim += (16 * max_dim) * 10 + 10

        self.proto_embed = nn.Parameter(torch.randn(K, 64) * 0.02)
        self.proto_weights = nn.Parameter(torch.zeros(K, weight_dim))
        self._init_prototypes()

        self.exec_ = AcceleratedCVNetwork(max_layers=max_layers, max_dim=max_dim, num_classes=10)

    def _init_prototypes(self):
        with torch.no_grad():
            for k in range(self.K):
                idx = 0
                for i in range(self.max_layers):
                    in_c = 3 if i == 0 else self.max_dim
                    w1 = torch.empty(self.max_dim, in_c, 3, 3)
                    nn.init.kaiming_uniform_(w1, a=math.sqrt(5))
                    self.proto_weights.data[k, idx:idx + w1.numel()] = w1.flatten()
                    idx += w1.numel()
                    self.proto_weights.data[k, idx:idx + self.max_dim] = 0.0
                    idx += self.max_dim
                    attn_dim = self.max_dim // 4
                    w_qkv = torch.empty(3 * attn_dim, self.max_dim, 1, 1)
                    nn.init.kaiming_uniform_(w_qkv, a=math.sqrt(5))
                    self.proto_weights.data[k, idx:idx + w_qkv.numel()] = w_qkv.flatten()
                    idx += w_qkv.numel()
                    self.proto_weights.data[k, idx:idx + 3 * attn_dim] = 0.0
                    idx += 3 * attn_dim
                    w_proj = torch.empty(self.max_dim, attn_dim, 1, 1)
                    nn.init.kaiming_uniform_(w_proj, a=math.sqrt(5))
                    self.proto_weights.data[k, idx:idx + w_proj.numel()] = w_proj.flatten()
                    idx += w_proj.numel()
                    self.proto_weights.data[k, idx:idx + self.max_dim] = 0.0
                    idx += self.max_dim
                wo = torch.empty(10, 16 * self.max_dim)
                nn.init.kaiming_uniform_(wo, a=math.sqrt(5))
                self.proto_weights.data[k, idx:idx + wo.numel()] = wo.flatten()
                idx += wo.numel()
                self.proto_weights.data[k, idx:idx + 10] = 0.0

    def caf_attention(self, logits):
        denom = logits.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return logits / denom

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        z = self.vision_encoder(x)
        pi = self.caf_attention(self.prototype_selector(z))

        fixed_struct = ModelStructure(
            layer_gates=torch.ones(B, self.max_layers, device=device),
            dim_weights=torch.ones(B, self.max_layers, 1, device=device),
            residual=torch.zeros(B, 1, device=device),
            dim_masks=torch.ones(B, self.max_layers, self.max_dim, device=device),
            block_mode=F.one_hot(torch.full((B, self.max_layers), 1, dtype=torch.long, device=device), 3).float(),
            use_conv=True, in_channels=3, max_layers=self.max_layers, max_dim=self.max_dim,
            dim_choices=(64, 96, 128), output_dim=10
        )

        return self.exec_(x, self.proto_weights, pi, fixed_struct)


class ConvAML_CIFAR_v2(nn.Module):
    """v2: K=16, latent=128, fixed ResNet arch, HTS kept, lr=1e-3, no torch.compile"""
    def __init__(self, K=16, latent_dim=128, max_layers=6, max_dim=128, num_hts=8):
        super().__init__()
        self.K = K
        self.max_layers = max_layers
        self.max_dim = max_dim

        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(1, 32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(1, 64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(1, 128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )

        self.prototype_selector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, K)
        )

        weight_dim = 0
        for i in range(max_layers):
            in_c = 3 if i == 0 else max_dim
            weight_dim += max_dim * in_c * 9 + max_dim
            attn_dim = max_dim // 4
            weight_dim += (3 * attn_dim) * max_dim + 3 * attn_dim
            weight_dim += max_dim * attn_dim + max_dim
        weight_dim += (16 * max_dim) * 10 + 10

        self.proto_embed = nn.Parameter(torch.randn(K, 64) * 0.02)
        self.proto_weights = nn.Parameter(torch.zeros(K, weight_dim))
        self._init_prototypes()

        self.hts_memory = nn.Parameter(torch.randn(num_hts, weight_dim) * 0.02)
        self.hts_router = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, num_hts)
        )
        self.memory_correction = nn.Sequential(
            nn.Linear(10, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 10)
        )
        self.memory_gate = nn.Parameter(torch.tensor(-1.0))

        self.exec_ = AcceleratedCVNetwork(max_layers=max_layers, max_dim=max_dim, num_classes=10)

    def _init_prototypes(self):
        with torch.no_grad():
            for k in range(self.K):
                idx = 0
                for i in range(self.max_layers):
                    in_c = 3 if i == 0 else self.max_dim
                    w1 = torch.empty(self.max_dim, in_c, 3, 3)
                    nn.init.kaiming_uniform_(w1, a=math.sqrt(5))
                    self.proto_weights.data[k, idx:idx + w1.numel()] = w1.flatten()
                    idx += w1.numel()
                    self.proto_weights.data[k, idx:idx + self.max_dim] = 0.0
                    idx += self.max_dim
                    attn_dim = self.max_dim // 4
                    w_qkv = torch.empty(3 * attn_dim, self.max_dim, 1, 1)
                    nn.init.kaiming_uniform_(w_qkv, a=math.sqrt(5))
                    self.proto_weights.data[k, idx:idx + w_qkv.numel()] = w_qkv.flatten()
                    idx += w_qkv.numel()
                    self.proto_weights.data[k, idx:idx + 3 * attn_dim] = 0.0
                    idx += 3 * attn_dim
                    w_proj = torch.empty(self.max_dim, attn_dim, 1, 1)
                    nn.init.kaiming_uniform_(w_proj, a=math.sqrt(5))
                    self.proto_weights.data[k, idx:idx + w_proj.numel()] = w_proj.flatten()
                    idx += w_proj.numel()
                    self.proto_weights.data[k, idx:idx + self.max_dim] = 0.0
                    idx += self.max_dim
                wo = torch.empty(10, 16 * self.max_dim)
                nn.init.kaiming_uniform_(wo, a=math.sqrt(5))
                self.proto_weights.data[k, idx:idx + wo.numel()] = wo.flatten()
                idx += wo.numel()
                self.proto_weights.data[k, idx:idx + 10] = 0.0

    def forward(self, x):
        B = x.shape[0]
        device = x.device

        z = self.vision_encoder(x)
        pi_aml = F.gumbel_softmax(self.prototype_selector(z), tau=1.0, hard=False)
        pi_hts = F.softmax(self.hts_router(z), dim=-1)

        mixed_prototypes = torch.cat([self.proto_weights, self.hts_memory], dim=0)
        mixed_pi = torch.cat([pi_aml, pi_hts], dim=-1)

        fixed_struct = ModelStructure(
            layer_gates=torch.ones(B, self.max_layers, device=device),
            dim_weights=torch.ones(B, self.max_layers, 1, device=device),
            residual=torch.zeros(B, 1, device=device),
            dim_masks=torch.ones(B, self.max_layers, self.max_dim, device=device),
            block_mode=F.one_hot(torch.full((B, self.max_layers), 1, dtype=torch.long, device=device), 3).float(),
            use_conv=True, in_channels=3, max_layers=self.max_layers, max_dim=self.max_dim,
            dim_choices=(64, 96, 128), output_dim=10
        )

        out_aml = self.exec_(x, mixed_prototypes, mixed_pi, fixed_struct)
        gate = torch.sigmoid(self.memory_gate)
        correction = self.memory_correction(out_aml.detach())
        return out_aml + gate * correction


def train_model(model_name, epochs=50):
    torch.backends.cuda.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_amp and amp_dtype == torch.float16 else None

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        CutoutTransform(length=8),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)

    print(f"\n========== TRAINING {model_name.upper()} ==========")
    if model_name == 'aml':
        model = ConvAML_CIFAR().to(device)
        gen_params = [p for p in model.gen.parameters() if p.requires_grad]
        hts_params = [p for p in (list(model.hts_router.parameters()) + [model.hts_memory] + list(model.memory_correction.parameters()) + [model.memory_gate]) if p.requires_grad]
        optimizer = optim.AdamW([{'params': gen_params, 'lr': 1e-4}, {'params': hts_params, 'lr': 5e-4}], weight_decay=1e-4)
    elif model_name == 'aml_v2':
        model = ConvAML_CIFAR_v2().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif model_name == 'aml_v3':
        model = ConvAML_CIFAR_v3().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif model_name == 'aml_v3_no_hts':
        model = ConvAML_CIFAR_v3_no_hts().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif model_name == 'aml_k128':
        model = ConvAML_CIFAR_K128(K=16, max_dim=64).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif model_name == 'aml_layerwise':
        model = ConvAML_CIFAR_layerwise().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif model_name == 'aml_direct':
        model = ConvAML_CIFAR_direct().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif model_name == 'resnet18':
        model = torchvision.models.resnet18(num_classes=10).to(device)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        model.maxpool = nn.Identity().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif model_name == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(num_classes=10).to(device)
        model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    ema = None

    use_cutmix = False
    cutmix_prob = 0.5

    total_time = 0
    torch.cuda.reset_peak_memory_stats()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        for batch in trainloader:
            inputs = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)

            if use_cutmix and torch.rand(1).item() < cutmix_prob:
                inputs, y_a, y_b, lam = cutmix_data(inputs, labels, alpha=1.0)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    outputs = model(inputs)
                    if isinstance(outputs, tuple): outputs = outputs[0]
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    outputs = model(inputs)
                    if isinstance(outputs, tuple): outputs = outputs[0]
                    loss = criterion(outputs, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if ema:
                ema.update()

        scheduler.step()
        torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        total_time += epoch_time

        if ema:
            ema.apply_shadow()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in testloader:
                inputs = batch[0].to(device, non_blocking=True)
                labels = batch[1].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    outputs = model(inputs)
                    if isinstance(outputs, tuple): outputs = outputs[0]
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        acc = 100. * correct / total

        if ema:
            ema.restore()

        if acc > best_acc:
            best_acc = acc

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | {epoch_time:.1f}s | {len(trainloader.dataset)/epoch_time:.0f} img/s | LR: {current_lr:.6f} | Test Acc: {acc:.2f}% | Best: {best_acc:.2f}%", flush=True)

    max_mem = torch.cuda.max_memory_allocated() / 1024**2
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- {model_name.upper()} SUMMARY ---")
    print(f"Avg Time/Epoch : {total_time/epochs:.2f} s")
    print(f"Avg Throughput : {len(trainloader.dataset)/(total_time/epochs):.0f} img/s")
    print(f"Peak VRAM      : {max_mem:.1f} MB")
    print(f"Total Params   : {total_params/1e6:.2f}M")
    print(f"Trainable      : {trainable_params/1e6:.2f}M")
    print(f"Best Accuracy  : {best_acc:.2f}%")
    print("===================================\n", flush=True)

# ============================================================
# Approach 1: K=128 (global prototypes, 128 effective DOF)
# ============================================================
class ConvAML_CIFAR_K128(nn.Module):
    """K=128, CAF attention, 3-layer router, no HTS"""
    def __init__(self, K=128, latent_dim=128, max_layers=6, max_dim=128):
        super().__init__()
        self.K = K
        self.max_layers = max_layers
        self.max_dim = max_dim

        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(128, latent_dim), nn.ReLU()
        )
        self.prototype_selector = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, K)
        )
        weight_dim = 0
        for i in range(max_layers):
            in_c = 3 if i == 0 else max_dim
            weight_dim += max_dim * in_c * 9 + max_dim
            attn_dim = max_dim // 4
            weight_dim += (3 * attn_dim) * max_dim + 3 * attn_dim
            weight_dim += max_dim * attn_dim + max_dim
        weight_dim += (16 * max_dim) * 10 + 10
        self.proto_weights = nn.Parameter(torch.zeros(K, weight_dim))
        self._init_prototypes()
        self.exec_ = AcceleratedCVNetwork(max_layers=max_layers, max_dim=max_dim, num_classes=10)

    def _init_prototypes(self):
        with torch.no_grad():
            for k in range(self.K):
                idx = 0
                for i in range(self.max_layers):
                    in_c = 3 if i == 0 else self.max_dim
                    w = torch.empty(self.max_dim, in_c, 3, 3)
                    nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                    w.mul_(math.sqrt(self.K))
                    self.proto_weights.data[k, idx:idx + w.numel()] = w.flatten()
                    idx += w.numel(); self.proto_weights.data[k, idx:idx + self.max_dim] = 0.0; idx += self.max_dim
                    attn_dim = self.max_dim // 4
                    w_qkv = torch.empty(3 * attn_dim, self.max_dim, 1, 1)
                    nn.init.kaiming_uniform_(w_qkv, a=math.sqrt(5))
                    w_qkv.mul_(math.sqrt(self.K))
                    self.proto_weights.data[k, idx:idx + w_qkv.numel()] = w_qkv.flatten()
                    idx += w_qkv.numel(); self.proto_weights.data[k, idx:idx + 3 * attn_dim] = 0.0; idx += 3 * attn_dim
                    w_proj = torch.empty(self.max_dim, attn_dim, 1, 1)
                    nn.init.kaiming_uniform_(w_proj, a=math.sqrt(5))
                    w_proj.mul_(math.sqrt(self.K))
                    self.proto_weights.data[k, idx:idx + w_proj.numel()] = w_proj.flatten()
                    idx += w_proj.numel(); self.proto_weights.data[k, idx:idx + self.max_dim] = 0.0; idx += self.max_dim
                wo = torch.empty(10, 16 * self.max_dim)
                nn.init.kaiming_uniform_(wo, a=math.sqrt(5))
                wo.mul_(math.sqrt(self.K))
                self.proto_weights.data[k, idx:idx + wo.numel()] = wo.flatten()
                idx += wo.numel(); self.proto_weights.data[k, idx:idx + 10] = 0.0

    def caf_attention(self, logits):
        denom = logits.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return logits / denom

    def forward(self, x):
        B = x.shape[0]; device = x.device
        z = self.vision_encoder(x)
        pi = self.caf_attention(self.prototype_selector(z))
        struct = ModelStructure(
            layer_gates=torch.ones(B, self.max_layers, device=device),
            dim_weights=torch.ones(B, self.max_layers, 1, device=device),
            residual=torch.zeros(B, 1, device=device),
            dim_masks=torch.ones(B, self.max_layers, self.max_dim, device=device),
            block_mode=F.one_hot(torch.full((B, self.max_layers), 1, dtype=torch.long, device=device), 3).float(),
            use_conv=True, in_channels=3, max_layers=self.max_layers, max_dim=self.max_dim,
            dim_choices=(64, 96, 128), output_dim=10
        )
        return self.exec_(x, self.proto_weights, pi, struct)


# ============================================================
# Approach 2: Layer-wise prototypes (3 groups * K=16 = 48 DOF)
# ============================================================
class ConvAML_CIFAR_layerwise(nn.Module):
    """3 layer groups * K=16, CAF attention, no HTS"""
    def __init__(self, K_per_group=16, latent_dim=128, max_layers=6, max_dim=128):
        super().__init__()
        self.K_per_group = K_per_group
        self.max_layers = max_layers
        self.max_dim = max_dim
        self.groups = [(0, 2), (2, 4), (4, 6)]  # layer ranges for each group

        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(1, 32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(1, 64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(1, 128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(128, latent_dim), nn.ReLU()
        )

        # Compute weight dims per layer
        self.layer_weight_dims = []
        for i in range(max_layers):
            in_c = 3 if i == 0 else max_dim
            dim = max_dim * in_c * 9 + max_dim
            attn_dim = max_dim // 4
            dim += (3 * attn_dim) * max_dim + 3 * attn_dim
            dim += max_dim * attn_dim + max_dim
            self.layer_weight_dims.append(dim)
        out_dim = (16 * max_dim) * 10 + 10
        self.total_weight_dim = sum(self.layer_weight_dims) + out_dim

        # Compute weight offset for each layer
        self.layer_offsets = [0]
        for dim in self.layer_weight_dims:
            self.layer_offsets.append(self.layer_offsets[-1] + dim)
        self.out_offset = self.layer_offsets[-1]

        # Create per-group prototypes (with zeros for non-group layers)
        self.proto_weights = nn.Parameter(torch.zeros(3 * K_per_group, self.total_weight_dim))
        self._init_group_prototypes()

        # Per-group routers
        self.prototype_selectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 256), nn.LayerNorm(256), nn.GELU(),
                nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
                nn.Linear(256, K_per_group)
            ) for _ in range(3)
        ])

        self.exec_ = AcceleratedCVNetwork(max_layers=max_layers, max_dim=max_dim, num_classes=10)

    def _init_group_prototypes(self):
        with torch.no_grad():
            for g, (l_start, l_end) in enumerate(self.groups):
                for k in range(self.K_per_group):
                    proto_idx = g * self.K_per_group + k
                    for i in range(l_start, l_end):
                        in_c = 3 if i == 0 else self.max_dim
                        off = self.layer_offsets[i]
                        w = torch.empty(self.max_dim, in_c, 3, 3)
                        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                        self.proto_weights.data[proto_idx, off:off + w.numel()] = w.flatten()
                        off += w.numel()
                        self.proto_weights.data[proto_idx, off:off + self.max_dim] = 0.0
                        off += self.max_dim
                        attn_dim = self.max_dim // 4
                        w_qkv = torch.empty(3 * attn_dim, self.max_dim, 1, 1)
                        nn.init.kaiming_uniform_(w_qkv, a=math.sqrt(5))
                        self.proto_weights.data[proto_idx, off:off + w_qkv.numel()] = w_qkv.flatten()
                        off += w_qkv.numel()
                        self.proto_weights.data[proto_idx, off:off + 3 * attn_dim] = 0.0
                        off += 3 * attn_dim
                        w_proj = torch.empty(self.max_dim, attn_dim, 1, 1)
                        nn.init.kaiming_uniform_(w_proj, a=math.sqrt(5))
                        self.proto_weights.data[proto_idx, off:off + w_proj.numel()] = w_proj.flatten()
                        off += w_proj.numel()
                        self.proto_weights.data[proto_idx, off:off + self.max_dim] = 0.0
                    # Init output layer for all groups
                    wo = torch.empty(10, 16 * self.max_dim)
                    nn.init.kaiming_uniform_(wo, a=math.sqrt(5))
                    self.proto_weights.data[proto_idx, self.out_offset:self.out_offset + wo.numel()] = wo.flatten()

    def caf_attention(self, logits):
        denom = logits.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return logits / denom

    def forward(self, x):
        B = x.shape[0]; device = x.device
        z = self.vision_encoder(x)
        pi_groups = [self.caf_attention(sel(z)) for sel in self.prototype_selectors]
        pi = torch.cat(pi_groups, dim=-1)
        struct = ModelStructure(
            layer_gates=torch.ones(B, self.max_layers, device=device),
            dim_weights=torch.ones(B, self.max_layers, 1, device=device),
            residual=torch.zeros(B, 1, device=device),
            dim_masks=torch.ones(B, self.max_layers, self.max_dim, device=device),
            block_mode=F.one_hot(torch.full((B, self.max_layers), 1, dtype=torch.long, device=device), 3).float(),
            use_conv=True, in_channels=3, max_layers=self.max_layers, max_dim=self.max_dim,
            dim_choices=(64, 96, 128), output_dim=10
        )
        return self.exec_(x, self.proto_weights, pi, struct)


# ============================================================
# Approach 3: Direct per-component weight generation (no prototypes)
# ============================================================
class ConvAML_CIFAR_direct(nn.Module):
    """Per-component weight generation with r=16 bottleneck, no prototypes"""
    def __init__(self, latent_dim=128, max_layers=6, max_dim=128, r=16):
        super().__init__()
        self.max_layers = max_layers
        self.max_dim = max_dim
        self.r = r

        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(1, 32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(1, 64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(1, 128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(128, latent_dim), nn.ReLU()
        )

        # Compute weight dim for each component (conv weight + bias per layer)
        component_dims = []
        layer_info = []
        for i in range(max_layers):
            in_c = 3 if i == 0 else max_dim
            w_dim = max_dim * in_c * 9
            b_dim = max_dim
            layer_info.append((in_c, w_dim, b_dim))
            component_dims.append(w_dim + b_dim)
        out_w_dim = (16 * max_dim) * 10
        out_b_dim = 10
        component_dims.append(out_w_dim + out_b_dim)

        # Learned expansion matrices M and per-component MLP heads
        self.M = nn.ParameterList()
        self.coeff_heads = nn.ModuleList()
        for dim in component_dims:
            self.M.append(nn.Parameter(torch.randn(r, dim) * 0.02))
            self.coeff_heads.append(nn.Sequential(
                nn.Linear(latent_dim, 64), nn.ReLU(),
                nn.Linear(64, r)
            ))

        self.layer_info = layer_info  # list of (in_c, w_dim, b_dim)
        self._init_M()

    def _init_M(self):
        with torch.no_grad():
            for i, dim in enumerate([d for m in self.M for d in [m.shape[1]]]):
                nc = math.isqrt(dim // 9) if i < 6 else math.isqrt(dim // 10)
                if nc > 0 and dim > 0:
                    std = math.sqrt(5.0 / nc)
                    self.M[i].data.normal_(0, std)

    def forward(self, x):
        B = x.shape[0]; device = x.device
        z = self.vision_encoder(x)
        H, W = x.shape[2], x.shape[3]

        h = x
        idx = 0
        for i in range(self.max_layers):
            in_c, w_dim, b_dim = self.layer_info[i]
            coeff = self.coeff_heads[idx](z)  # [B, r]
            weights = coeff @ self.M[idx]  # [B, w_dim + b_dim]
            W_conv = weights[:, :w_dim].view(B, self.max_dim, in_c, 3, 3)
            b_conv = weights[:, w_dim:].view(B, self.max_dim)
            idx += 1

            h_conv = F.conv2d(
                h.reshape(1, B * in_c, h.shape[2], h.shape[3]),
                W_conv.reshape(B * self.max_dim, in_c, 3, 3),
                b_conv.reshape(-1),
                padding=1, groups=B,
            ).view(B, self.max_dim, h.shape[2], h.shape[3])
            h = F.group_norm(h_conv, self.max_dim)
            h = F.relu(h)
            if i == 2:
                h = F.max_pool2d(h, 2, 2)
                H, W = H // 2, W // 2

        coeff = self.coeff_heads[idx](z)
        weights = coeff @ self.M[idx]
        W_out = weights[:, :(16 * self.max_dim) * 10].view(B, 10, 16 * self.max_dim)
        b_out = weights[:, (16 * self.max_dim) * 10:].view(B, 10)

        h_pool = F.adaptive_avg_pool2d(h, (4, 4)).reshape(B, -1)
        h_pool = F.layer_norm(h_pool, [h_pool.shape[-1]])
        out = torch.bmm(W_out, h_pool.unsqueeze(2)).squeeze(2) + b_out
        return out


# ============================================================
# Training entry
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['aml', 'aml_v2', 'aml_v3', 'aml_v3_no_hts', 'aml_k128', 'aml_layerwise', 'aml_direct', 'resnet18', 'mobilenet_v2'])
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    train_model(args.model, epochs=args.epochs)
