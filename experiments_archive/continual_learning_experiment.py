import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from aml_architecture_arena import FastMultiArchTransformer, set_seed

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                logits, _ = model(x)
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total

def get_routing_distribution(model, data_loader, device):
    model.eval()
    arch_weights_all = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            with torch.amp.autocast('cuda'):
                _, arch_probs = model(x)
            arch_weights_all.append(arch_probs.cpu().numpy())
    arch_arr = np.concatenate(arch_weights_all, axis=0) # [N, 6, 4]
    mean_prefs = np.mean(arch_arr, axis=0) * 100 # [6, 4] %
    return mean_prefs

def run_continual_learning_experiment():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== CONTINUAL LEARNING EXPERIMENT: CIFAR-10 -> MNIST (Catastrophic Forgetting Test) ({device}) ===")
    
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
    cifar_test  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
    
    cifar_train_loader = DataLoader(cifar_train, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    cifar_test_loader  = DataLoader(cifar_test, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
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
    
    model = FastMultiArchTransformer(img_size=32, patch_size=4, in_chans=3, num_classes=10, dim=192, depth=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # PHASE 1: CIFAR-10
    print("\n" + "="*75)
    print("PHASE 1: TRAINING ON CIFAR-10 (NATURAL COLOR IMAGES)")
    print("="*75)
    epochs_cifar = 5
    for ep in range(1, epochs_cifar + 1):
        model.train()
        for x, y in cifar_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _ = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        cifar_acc = evaluate_accuracy(model, cifar_test_loader, device)
        print(f"Epoch {ep}/{epochs_cifar} -> CIFAR-10 Test Acc: {cifar_acc:5.2f}%")
        
    cifar_acc_initial = evaluate_accuracy(model, cifar_test_loader, device)
    cifar_routing_initial = get_routing_distribution(model, cifar_test_loader, device)
    print(f"\n[PHASE 1 COMPLETED] Initial CIFAR-10 Test Accuracy: {cifar_acc_initial:.2f}%")

    # PHASE 2: MNIST
    print("\n" + "="*75)
    print("PHASE 2: SEQUENTIAL CONTINUAL TRAINING ON MNIST (GRAYSCALE DIGITS)")
    print("="*75)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    epochs_mnist = 3
    
    for ep in range(1, epochs_mnist + 1):
        model.train()
        for x, y in mnist_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _ = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        mnist_acc = evaluate_accuracy(model, mnist_test_loader, device)
        cifar_acc_retest = evaluate_accuracy(model, cifar_test_loader, device)
        print(f"Epoch {ep}/{epochs_mnist} -> MNIST Test Acc: {mnist_acc:5.2f}% | CIFAR-10 Retest Acc: {cifar_acc_retest:5.2f}%")
        
    cifar_acc_final = evaluate_accuracy(model, cifar_test_loader, device)
    mnist_acc_final = evaluate_accuracy(model, mnist_test_loader, device)
    cifar_routing_final = get_routing_distribution(model, cifar_test_loader, device)
    mnist_routing_final = get_routing_distribution(model, mnist_test_loader, device)
    
    # FINAL REPORT
    print("\n" + "="*85)
    print("CATASTROPHIC FORGETTING AUDIT REPORT")
    print("="*85)
    delta_forgetting = cifar_acc_initial - cifar_acc_final
    print(f"1. Initial CIFAR-10 Accuracy (Phase 1)   : {cifar_acc_initial:5.2f}%")
    print(f"2. MNIST Test Accuracy (Phase 2)         : {mnist_acc_final:5.2f}%")
    print(f"3. Final CIFAR-10 Accuracy (After MNIST) : {cifar_acc_final:5.2f}%")
    print(f"4. Catastrophic Forgetting Drop (Delta)  : {delta_forgetting:+5.2f}% (Drop Rate: {delta_forgetting/cifar_acc_initial*100:.1f}%)")
    
    print("\n" + "="*85)
    print("ROUTING PREFERENCE SHIFT (% AVERAGE ACROSS NETWORK)")
    print("="*85)
    print(f"{'Architecture Paradigm':<25} {'CIFAR-10 (Original)':<22} {'MNIST (New Task)':<22} {'CIFAR-10 (After)'}")
    print("-" * 85)
    arch_names = ["CNN (Conv2D)", "ViT (Self-Attention)", "MoE (Sparse SwiGLU)", "ResNet (MLP)"]
    c_init_avg = np.mean(cifar_routing_initial, axis=0)
    m_final_avg = np.mean(mnist_routing_final, axis=0)
    c_final_avg = np.mean(cifar_routing_final, axis=0)
    for k in range(4):
        print(f"{arch_names[k]:<25} {c_init_avg[k]:<20.2f}% {m_final_avg[k]:<20.2f}% {c_final_avg[k]:<20.2f}%")
    print("="*85)

if __name__ == '__main__':
    run_continual_learning_experiment()
