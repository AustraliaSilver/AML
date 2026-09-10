import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os
import sys

# Register CUDA/torch DLL directories and harden the inductor cache before
# any native extension import. Paths are derived from the live environment
# (CUDA_PATH, installed torch) rather than hardcoded.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mgn.accel.dllsetup import bootstrap

bootstrap()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AML_Core_Code'))


def get_cifar10(batch_size=128):
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
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    return trainloader, testloader


class CIFAR_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(512 * 16),
            nn.Dropout(0.2),
            nn.Linear(512 * 16, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CIFAR_ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, dim=192, num_heads=4, num_layers=6, num_classes=10):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.head(x[:, 0])


def make_resnet18():
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def train_model(name, model, epochs=10, lr=1e-3):
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_amp and amp_dtype == torch.float16 else None

    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params:,}", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    trainloader, testloader = get_cifar10()

    model.train()
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            loss = criterion(model(inputs), labels)
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
        if i >= 5: break

    torch.cuda.reset_peak_memory_stats()
    total_time = 0
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        start = time.time()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss = criterion(model(inputs), labels)
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

        torch.cuda.synchronize()
        epoch_time = time.time() - start
        total_time += epoch_time
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        acc = 100. * correct / total
        if acc > best_acc: best_acc = acc
        throughput = len(trainloader.dataset) / epoch_time
        print(f"  Epoch {epoch+1:02d}/{epochs} | {epoch_time:.1f}s | {throughput:.0f} img/s | Test: {acc:.2f}%", flush=True)

    max_mem = torch.cuda.max_memory_allocated() / 1024**2
    avg_time = total_time / epochs
    throughput = len(trainloader.dataset) / avg_time
    print(f"  --- {name} --- Params: {params:,} | Best: {best_acc:.2f}% | Avg: {avg_time:.1f}s/ep | {throughput:.0f} img/s | VRAM: {max_mem:.1f}MB", flush=True)
    return {'name': name, 'best_acc': best_acc, 'last_acc': acc, 'time': avg_time, 'throughput': throughput, 'vram': max_mem, 'params': params}


if __name__ == '__main__':
    torch.manual_seed(42)

    models = [
        ('CNN', lambda: CIFAR_CNN()),
        ('ViT', lambda: CIFAR_ViT()),
        ('ResNet-18', make_resnet18),
    ]

    results = []
    for name, model_fn in models:
        print(f"\n{'='*60}")
        print(f"  TRAINING: {name}")
        print(f"{'='*60}")
        torch.cuda.empty_cache()
        model = model_fn()
        r = train_model(name, model, epochs=10)
        results.append(r)
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("  RESULTS (10 epochs, CIFAR-10, batch=128, lr=1e-3, AdamW)")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Best%':>7} {'Last%':>7} {'Params':>10} {'s/ep':>7} {'img/s':>8} {'VRAM':>8}")
    print(f"{'-'*15} {'-'*7} {'-'*7} {'-'*10} {'-'*7} {'-'*8} {'-'*8}")
    for r in results:
        print(f"{r['name']:<15} {r['best_acc']:>6.2f}% {r['last_acc']:>6.2f}% {r['params']:>10,} {r['time']:>6.1f}s {r['throughput']:>7.0f} {r['vram']:>7.1f}MB")
    print(f"{'='*70}")
