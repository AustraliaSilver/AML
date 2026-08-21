"""
CONTINUE PHASE 2: TRAIN CONTINUAL MNIST DIRECTLY FROM CHECKPOINT
================================================================
Load checkpoint hts_cifar_best.pth (CIFAR-10 = 78.81%) va chay tiep Phase 2 MNIST
tren kien truc HtS Dynamic Hypernetwork goc.
"""

import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from hts_exact_original_baseline import OriginalHtSModel, evaluate, set_seed

def run_phase2():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== TIEP TUC PHASE 2: CONTINUAL MNIST TREN HtS GOC ({device}) ===")

    # Load Model & Checkpoint Phase 1
    model = OriginalHtSModel(in_channels=3, d=32, num_layers=8, num_classes=10).to(device)
    model.load_state_dict(torch.load("hts_cifar_best.pth", map_location=device, weights_only=True))
    
    # Datasets
    tf_c_te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    tf_m_tr = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,)*3, (0.3081,)*3)
    ])
    tf_m_te = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,)*3, (0.3081,)*3)
    ])

    c_te = DataLoader(datasets.CIFAR10('./data', False, download=True, transform=tf_c_te), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    m_tr = DataLoader(datasets.MNIST('./data', True, download=True, transform=tf_m_tr), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    m_te = DataLoader(datasets.MNIST('./data', False, download=True, transform=tf_m_te), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    cifar_init = evaluate(model, c_te, device)
    print(f"\n[XAC NHAN CHECKPOINT] CIFAR-10 Test Acc Truoc Khi Train MNIST: {cifar_init:.2f}%\n")

    criterion = nn.CrossEntropyLoss()
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=5, eta_min=1e-5)

    print("============================================================")
    print("PHASE 2: TRAIN CONTINUAL MNIST (5 Epochs) TREN HtS GOC")
    print("============================================================")
    for ep in range(1, 6):
        model.train()
        t0 = time.time()
        for x, y in m_tr:
            x, y = x.to(device), y.to(device)
            optimizer2.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer2.step()
        scheduler2.step()
        
        m_acc = evaluate(model, m_te, device)
        c_ret = evaluate(model, c_te, device)
        print(f"Epoch {ep}/5 ({time.time()-t0:.1f}s) -> MNIST Acc: {m_acc:.2f}% | CIFAR-10 Retest: {c_ret:.2f}%")

    cifar_final = evaluate(model, c_te, device)
    mnist_final = evaluate(model, m_te, device)

    print("\n" + "="*65)
    print("TONG KET CONTINUAL LEARNING TREN KIEN TRUC HtS GOC")
    print("="*65)
    print(f"CIFAR-10 Ban Dau (Phase 1)    : {cifar_init:.2f}%")
    print(f"MNIST Dat Duoc (Phase 2)      : {mnist_final:.2f}%")
    print(f"CIFAR-10 Retest (Sau Phase 2) : {cifar_final:.2f}%")
    print(f"Delta Forgetting              : {cifar_init - cifar_final:+.2f}%")
    print("="*65)

if __name__ == '__main__':
    run_phase2()
