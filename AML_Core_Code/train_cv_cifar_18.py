import sys
import os
try:
    from hts_lib.models.basis import HtSLowRankAdapter
except ImportError:
    HtSLowRankAdapter = None
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# Register CUDA/torch DLL directories and harden the inductor cache before
# any native extension import. Paths are derived from the live environment
# (CUDA_PATH, installed torch) rather than hardcoded.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mgn.accel.dllsetup import bootstrap

bootstrap()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mgn.generator import ModelGenerator
# Canonical flat prototype layout lives in mgn.layout; re-exported here so
# long-standing importers (extract_static_model, validate_ext_in, train.py)
# keep resolving get_prototype_layout from this module.
from mgn.layout import get_prototype_layout, get_second_conv_layout
import mgn.accel.functions as aml_ops

# The fused CUDA pipelines (fused_forward / block drivers) implement the v2
# REAL 2-convolution ResNet block (ATen conv+norm chain per branch, custom
# elementwise mix kernel); their backward recomputes the identical ATen chain
# in mgn/accel/fused_autograd.py. Certified by validate_ext_in.py: forward
# parity at L=18 (GN 0.0034 / IN 0.0435 rel) and gradient parity within
# 1e-2 at L=2..4, L=18 drift below the reference's own noise floor. Flip to
# False to force the pure-eager ATen path.
_FUSED_TWO_CONV_READY = True

try:
    import mgn.accel._C as _C
    _C_available = True
except ImportError:
    _C_available = False

# InstanceNorm variant of the fused pipeline (per-channel norm statistics).
try:
    import mgn.accel._Cin as _Cin  # noqa: F401  (import success sets the capability flag; bindings_in owns the symbols)
    from mgn.accel.fused_autograd_in import FusedPipelineFunctionIN
    _Cin_available = True
except ImportError:
    _Cin_available = False

try:
    from mgn.accel.fused_autograd import FusedPipelineFunction
    _fused_autograd_available = True
except ImportError:
    _fused_autograd_available = False


class AcceleratedCVNetwork(nn.Module):
    def __init__(self, max_layers=6, max_dim=64, num_classes=10, norm_mode="gn1", K=32):
        super().__init__()
        self.max_layers = max_layers
        self.max_dim = max_dim
        self.num_classes = num_classes
        self.norm_mode = norm_mode
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Build layer offsets for exactly 18 layers
        # Using 64 channels for layer 1..6, 128 for 7..12, 128 for 13..18 to keep size manageable.
        # Wait, max_dim is the max dimension allowed. We just set in_channels and out_channels to max_dim everywhere for simplicity!
        self.total_dim, self.offsets, self._layer_info, self.wo_offset, self.bo_offset = get_prototype_layout(self.max_layers, self.max_dim, self.num_classes)
        
        self._build_slices()
        
        # HtS Low-Rank Integration
        self.use_hts_low_rank = (HtSLowRankAdapter is not None)
        if self.use_hts_low_rank:
            # K is passed via __init__
            self.hts_adapters = nn.ModuleList()
            for i in range(self.max_layers):
                in_c = self.offsets[i][4]
                self.hts_adapters.append(HtSLowRankAdapter(K, self.max_dim, in_c))
            self.hts_adapters2 = nn.ModuleList()
            for i in range(self.max_layers):
                self.hts_adapters2.append(HtSLowRankAdapter(K, self.max_dim, self.max_dim))


    def _build_slices(self):
        total_end = self.bo_offset + self.num_classes
        self.register_buffer('_proto_end', torch.tensor(total_end), persistent=False)

        self._offsets2 = get_second_conv_layout(self.max_layers, self.max_dim,
                                                self.num_classes, 3)
        # Offset vectors for the v2 fused pipeline (conv2 weight/bias slots).
        self._w2_off = [o2[0] for o2 in self._offsets2]
        self._b2_off = [o2[2] for o2 in self._offsets2]
        self._layer_info = []
        for i in range(self.max_layers):
            w_start, w_end, b_start, b_end, in_c = self.offsets[i]
            C_out = self.max_dim
            self._layer_info.append((w_start, w_end, b_start, b_end, in_c, C_out))

    def forward(self, x, prototypes, pi, model_structure):
        S = model_structure
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]

        if getattr(self, 'use_hts_low_rank', False):
            # Bypass fused ops when using Low-Rank updates
            h = x
            w_mix = pi @ prototypes
            for i in range(self.max_layers):
                w_start, w_end, b_start, b_end, in_c, C_out = self._layer_info[i]
                
                # Get base weights
                w_base = w_mix[:, w_start:w_end].view(B, C_out, in_c, 3, 3)
                b_base = w_mix[:, b_start:b_end]
                
                # Add Low-Rank update
                w_hts = self.hts_adapters[i](pi) # [B, C_out, in_c, 3, 3]
                w_final = w_base + w_hts
                
                # Group convolution trick for batched weights
                h_flat = h.view(1, B * in_c, h.shape[2], h.shape[3])
                w_flat = w_final.reshape(B * C_out, in_c, 3, 3)
                b_flat = b_base.reshape(B * C_out)
                
                h_conv = F.conv2d(h_flat, w_flat, b_flat, groups=B, padding=1)
                h_conv = h_conv.view(B, C_out, h_conv.shape[2], h_conv.shape[3])

                gate_c = S.layer_gates[:, i:i+1].view(-1, 1, 1, 1)
                dim_mask_c = S.dim_masks[:, i].view(-1, S.dim_masks.shape[-1], 1, 1)

                if self.norm_mode == "in":
                    h_norm = F.instance_norm(h_conv)
                else:
                    h_norm = F.group_norm(h_conv, num_groups=1)
                h_plain = F.relu(h_norm) * dim_mask_c
                h_expand = h_plain if i == 0 else h * dim_mask_c

                w2_start = self._w2_off[i]
                b2_start = self._b2_off[i]
                w2_base = w_mix[:, w2_start:w2_start + self.max_dim * self.max_dim * 9].view(B, self.max_dim, self.max_dim, 3, 3)
                b2_base = w_mix[:, b2_start:b2_start + self.max_dim]
                
                w2_hts = self.hts_adapters2[i](pi)
                w2_final = w2_base + w2_hts
                
                h_plain_flat = h_plain.view(1, B * self.max_dim, h_plain.shape[2], h_plain.shape[3])
                w2_flat = w2_final.reshape(B * self.max_dim, self.max_dim, 3, 3)
                b2_flat = b2_base.reshape(B * self.max_dim)
                
                h_conv2 = F.conv2d(h_plain_flat, w2_flat, b2_flat, groups=B, padding=1)
                h_conv2 = h_conv2.view(B, self.max_dim, h_conv2.shape[2], h_conv2.shape[3])

                if self.norm_mode == "in":
                    h_norm2 = F.instance_norm(h_conv2)
                else:
                    h_norm2 = F.group_norm(h_conv2, num_groups=1)
                h_resnet = F.relu(h_expand + h_norm2) * dim_mask_c

                g_plain = S.block_mode[:, i, 0].view(-1, 1, 1, 1)
                g_resnet = S.block_mode[:, i, 1].view(-1, 1, 1, 1)
                h_active = g_plain * h_plain + g_resnet * h_resnet

                if i == 0:
                    h = h_active
                else:
                    h = gate_c * h_active + (1.0 - gate_c) * h_expand
                    
                if i == 5 or i == 11:
                    if h.shape[2] > 1:
                        h = F.max_pool2d(h, 2, 2)
            
            # Classifier
            h_pool = self.pool(h).reshape(B, -1)
            wo = w_mix[:, self.wo_offset:self.wo_offset + self.num_classes * 16 * self.max_dim].view(B, self.num_classes, 16 * self.max_dim)
            bo = w_mix[:, self.bo_offset:self.bo_offset + self.num_classes]
            out = torch.bmm(h_pool.unsqueeze(1), wo.transpose(1, 2)).squeeze(1) + bo
            return out
        elif self.norm_mode == "in" and _Cin_available and x.is_cuda and _FUSED_TWO_CONV_READY:
            w_offsets = [self.offsets[i][0] for i in range(self.max_layers)]
            b_offsets = [self.offsets[i][2] for i in range(self.max_layers)]
            C_ins = [self.offsets[i][4] for i in range(self.max_layers)]
            C_outs = [self.max_dim] * self.max_layers
            Hs = [H]
            Ws = [W]
            for i in range(1, self.max_layers):
                if i == 6 or i == 12:
                    H = H // 2
                    W = W // 2
                Hs.append(H)
                Ws.append(W)
            bm = S.block_mode.float() if S.block_mode is not None else torch.ones(B, self.max_layers, 2, device=x.device, dtype=torch.float32) * 0.5
            with torch.amp.autocast("cuda", enabled=False):
                h = FusedPipelineFunctionIN.apply(
                    x.float(), prototypes.float(), pi.float(),
                    S.layer_gates.float(), S.dim_masks.float(), bm.float(),
                    w_offsets, b_offsets, self._w2_off, self._b2_off,
                    C_ins, C_outs, Hs, Ws, self.max_dim
                )
        elif _C_available and x.is_cuda and _FUSED_TWO_CONV_READY:
            w_offsets = [self.offsets[i][0] for i in range(self.max_layers)]
            b_offsets = [self.offsets[i][2] for i in range(self.max_layers)]
            C_ins = [self.offsets[i][4] for i in range(self.max_layers)]
            C_outs = [self.max_dim] * self.max_layers
            # Hs/Ws are the INPUT spatial sizes for layer i.
            # Pooling happens AFTER layers 5 and 11, so inputs are:
            #   layers 0..5  -> H
            #   layers 6..11 -> H/2
            #   layers 12..17-> H/4
            Hs = [H]
            Ws = [W]
            for i in range(1, self.max_layers):
                if i == 6 or i == 12:
                    H = H // 2
                    W = W // 2
                Hs.append(H)
                Ws.append(W)
            bm = S.block_mode.float() if S.block_mode is not None else torch.ones(B, self.max_layers, 2, device=x.device, dtype=torch.float32) * 0.5
            with torch.amp.autocast("cuda", enabled=False):
                if _fused_autograd_available:
                    h = FusedPipelineFunction.apply(
                        x.float(), prototypes.float(), pi.float(),
                        S.layer_gates.float(), S.dim_masks.float(), bm.float(),
                        w_offsets, b_offsets, self._w2_off, self._b2_off,
                        C_ins, C_outs, Hs, Ws, self.max_dim
                    )
                else:
                    h = _C.fused_forward(
                        x.float(), prototypes.float(), pi.float(),
                        S.layer_gates.float(), S.dim_masks.float(), bm.float(),
                        w_offsets, b_offsets, self._w2_off, self._b2_off,
                        C_ins, C_outs, Hs, Ws, self.max_dim
                    )
        else:
            h = x
            for i in range(self.max_layers):
                w_start, w_end, b_start, b_end, in_c, C_out = self._layer_info[i]

                h_conv = aml_ops.fused_dynamic_conv2d(h, prototypes, pi, w_start, b_start)

                gate_c = S.layer_gates[:, i:i+1].view(-1, 1, 1, 1)
                dim_mask_c = S.dim_masks[:, i].view(-1, S.dim_masks.shape[-1], 1, 1)

                if self.norm_mode == "in":
                    h_norm = F.instance_norm(h_conv)
                else:
                    h_norm = F.group_norm(h_conv, num_groups=1)
                h_plain = F.relu(h_norm) * dim_mask_c
                h_expand = h_plain if i == 0 else h * dim_mask_c

                # Real 2-conv residual branch (see mgn.layout v2): the plain
                # activation feeds a second per-layer conv+norm, the skip is
                # ADDED to its output before the final ReLU. This mirrors
                # mgn/executor.py block-for-block.
                w2_start, w2_end, b2_start, b2_end = self._offsets2[i]
                h_conv2 = aml_ops.fused_dynamic_conv2d(h_plain, prototypes, pi,
                                                       w2_start, b2_start)
                if self.norm_mode == "in":
                    h_norm2 = F.instance_norm(h_conv2)
                else:
                    h_norm2 = F.group_norm(h_conv2, num_groups=1)

                g_plain = S.block_mode[:, i, 0].view(-1, 1, 1, 1) if S.block_mode is not None else 1.0
                g_resnet = S.block_mode[:, i, 1].view(-1, 1, 1, 1) if S.block_mode is not None else 0.0

                h_resnet = F.relu(h_expand + h_norm2) * dim_mask_c
                h_active = g_plain * h_plain + g_resnet * h_resnet

                if i == 0:
                    h = h_active
                else:
                    h = gate_c * h_active + (1.0 - gate_c) * h_expand

                if i == 5 or i == 11:
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
        if self.training:
            pi_aml = F.gumbel_softmax(model.pi_logits, tau=1.0, hard=False)
        else:
            # Deterministic routing at eval: Gumbel noise ignores module.eval()
            # and otherwise makes repeated test evaluations disagree (same seed,
            # different draw order) -- mirrors the fix in mgn/generator.py.
            pi_aml = F.softmax(model.pi_logits, dim=-1)

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
    torch.backends.cudnn.benchmark = True
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
    train_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
    train_clean = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_test)

    # Held-out validation split: model selection must not read the test set.
    n_val = 5000
    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(42))
    tr_idx, va_idx = perm[:-n_val].tolist(), perm[-n_val:].tolist()
    trainset = torch.utils.data.Subset(train_full, tr_idx)
    valset = torch.utils.data.Subset(train_clean, va_idx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=256, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)

    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)
    print("Dataset loaded. Initializing model...", flush=True)

    model = ConvAML_CIFAR().to(device)

    model = torch.compile(model, mode='reduce-overhead')
    print("Model compiled with torch.compile (reduce-overhead)", flush=True)

    gen_params = [p for p in model.gen.parameters() if p.requires_grad]
    hts_params = [p for p in (list(model.hts_router.parameters()) + [model.hts_memory] + list(model.memory_correction.parameters()) + [model.memory_gate]) if p.requires_grad]

    optimizer_gen = optim.AdamW(gen_params, lr=1e-4, weight_decay=1e-4)
    optimizer_hts = optim.AdamW(hts_params, lr=5e-4, weight_decay=1e-4)

    epochs = 10
    scheduler_gen = optim.lr_scheduler.CosineAnnealingLR(optimizer_gen, T_max=epochs)
    scheduler_hts = optim.lr_scheduler.CosineAnnealingLR(optimizer_hts, T_max=epochs)

    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    best_state = None
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aml_core_best.pth")

    def run_eval(loader, n_total):
        correct = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    outputs, _ = model(inputs)
                correct += outputs.argmax(1).eq(labels).sum().item()
        return 100. * correct / n_total

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
        val_acc = run_eval(valloader, len(valset))
        test_acc = run_eval(testloader, len(testset))
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), ckpt_path)

        mem_gate = torch.sigmoid(model.memory_gate).item()
        mem_norm = model.hts_memory.norm().item()
        print(f"Epoch [{epoch+1:02d}/{epochs}] | {epoch_time:.1f}s ({total/epoch_time:.0f} img/s) | Loss: {total_loss/len(trainloader):.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Test: {test_acc:.2f}% | MemGate: {mem_gate:.3f} | MemNorm: {mem_norm:.4f}", flush=True)

        scheduler_gen.step()
        scheduler_hts.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    final_test = run_eval(testloader, len(testset))
    print(f"=== BEST VAL ACCURACY: {best_val:.2f}% | Test acc of best-val model: {final_test:.2f}% ===")

if __name__ == "__main__":
    train_cifar()
