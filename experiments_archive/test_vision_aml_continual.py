import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pure_vision_aml_benchmark import AMLVisionArchitectureSearchModel, set_seed

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                logits, _ = model(x, tau=0.3)
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total

def run_vision_aml_continual_mnist():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== KIEM TRA KHA NANG GHI NHO CUA VISION AML: CIFAR-10 -> MNIST ({device}) ===")
    
    # 1. Load Data
    transform_cifar_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar_test)
    cifar_test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False)
    
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    ])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    mnist_test  = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    
    mnist_train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    mnist_test_loader  = DataLoader(mnist_test, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    # 2. Load Pretrained Vision AML Checkpoint (66.18% Accuracy on CIFAR-10)
    model = AMLVisionArchitectureSearchModel(dim=192, depth=6, K=16).to(device)
    ckpt_path = "best_vision_aml_arch_search.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    
    cifar_initial_acc = evaluate_accuracy(model, cifar_test_loader, device)
    print(f"\n[DIEM XUAT PHAT] Do chinh xac CIFAR-10 ban dau: {cifar_initial_acc:.2f}%\n")
    
    # 3. Train tiep tuc tren MNIST (3 Epochs)
    print("--- BAT DAU HUAN LUYEN TIEP TUC TREN MNIST (3 EPOCHS) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    epochs = 3
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for x, y in mnist_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _ = model(x, tau=0.5)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        mnist_acc = evaluate_accuracy(model, mnist_test_loader, device)
        cifar_retest_acc = evaluate_accuracy(model, cifar_test_loader, device)
        print(f"Epoch {epoch}/{epochs} ({time.time()-t0:4.1f}s) -> MNIST Test Acc: {mnist_acc:5.2f}% | CIFAR-10 Retest Acc: {cifar_retest_acc:5.2f}%")

    # 4. Do dac do phan bo Router sau khi hoc MNIST
    cifar_final_acc = evaluate_accuracy(model, cifar_test_loader, device)
    mnist_final_acc = evaluate_accuracy(model, mnist_test_loader, device)
    
    print("\n" + "="*85)
    print("TONG KET KIEM TRA GHI NHO (CATASTROPHIC FORGETTING AUDIT TREN VISION AML)")
    print("="*85)
    print(f"1. CIFAR-10 Ban Dau (Goc)                : {cifar_initial_acc:5.2f}%")
    print(f"2. MNIST Test Accuracy (Task Moi)        : {mnist_final_acc:5.2f}%")
    print(f"3. CIFAR-10 Kiem Tra Lai (Sau Khi Hoc)   : {cifar_final_acc:5.2f}%")
    delta_forgetting = cifar_initial_acc - cifar_final_acc
    print(f"4. Muc Do Quen Tri Thuc (Delta Forgetting): {delta_forgetting:+5.2f}% (Giam tu {cifar_initial_acc:.2f}% xuong {cifar_final_acc:.2f}%)")
    print("="*85)

if __name__ == '__main__':
    run_vision_aml_continual_mnist()
