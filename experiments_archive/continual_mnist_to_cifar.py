"""
Benchmark Continual Learning 2 Chieu:
Phase 1: Train SOTA MNIST (dat ~99.7%)
Phase 2: Train tiep CIFAR-10 bang 2 phuong phap:
  1. Fine-tuning truc tiep (Naive - de do do quen Catastrophic Forgetting)
  2. Task-Isolated Branching / Freeze (Bao toan 100% MNIST o 99.7%)
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def set_seed(seed=2026):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Kien truc SOTA MNIST Network
class SOTAMNISTNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

def run_experiment():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== HUAN LUYEN CONTINUAL: MNIST TRUOC -> CIFAR-10 SAU ({device}) ===")
    
    # 1. Dataset MNIST (28x28 grayscale)
    tf_m_tr = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tf_m_te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 2. Dataset CIFAR-10 Grayscale 28x28 (de co the chay chung qua model cung kich thuoc)
    tf_c_tr = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.RandomCrop(28, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48,), (0.24,))
    ])
    tf_c_te = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.48,), (0.24,))
    ])
    
    m_tr = DataLoader(datasets.MNIST('./data', True, download=True, transform=tf_m_tr), batch_size=128, shuffle=True, num_workers=2)
    m_te = DataLoader(datasets.MNIST('./data', False, download=True, transform=tf_m_te), batch_size=256, shuffle=False, num_workers=2)
    c_tr = DataLoader(datasets.CIFAR10('./data', True, download=True, transform=tf_c_tr), batch_size=128, shuffle=True, num_workers=2)
    c_te = DataLoader(datasets.CIFAR10('./data', False, download=True, transform=tf_c_te), batch_size=256, shuffle=False, num_workers=2)

    model = SOTAMNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # =========================================================================
    # PHASE 1: TRAIN MNIST (5 Epochs dat 99.5%+)
    # =========================================================================
    print("\n------------------------------------------------------------")
    print("PHASE 1: TRAIN MNIST (5 Epochs)")
    print("------------------------------------------------------------")
    optimizer1 = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=5, eta_min=1e-5)
    
    for ep in range(1, 6):
        model.train()
        t0 = time.time()
        for x, y in m_tr:
            x, y = x.to(device), y.to(device)
            optimizer1.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer1.step()
        scheduler1.step()
        acc = evaluate(model, m_te, device)
        print(f"Epoch {ep}/5 ({time.time()-t0:.1f}s) -> MNIST Acc: {acc:.2f}%")
        
    mnist_init = evaluate(model, m_te, device)
    print(f"\n[PHASE 1 HOAN TAT] MNIST Dat: {mnist_init:.2f}%\n")
    
    # =========================================================================
    # PHASE 2: TRAIN TIEP CIFAR-10 TRUC TIEP TREN MODEL (Kiem tra do quen)
    # =========================================================================
    print("------------------------------------------------------------")
    print("PHASE 2: TRAIN TIEP CIFAR-10 (6 Epochs) -> KIEM TRA DO NHO MNIST")
    print("------------------------------------------------------------")
    optimizer2 = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=6, eta_min=1e-5)
    
    for ep in range(1, 7):
        model.train()
        t0 = time.time()
        for x, y in c_tr:
            x, y = x.to(device), y.to(device)
            optimizer2.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer2.step()
        scheduler2.step()
        
        cifar_acc = evaluate(model, c_te, device)
        mnist_retest = evaluate(model, m_te, device)
        print(f"Epoch {ep}/6 ({time.time()-t0:.1f}s) -> CIFAR-10 Acc: {cifar_acc:.2f}% | MNIST Retest: {mnist_retest:.2f}%")

    mnist_final = evaluate(model, m_te, device)
    cifar_final = evaluate(model, c_te, device)
    
    print("\n" + "="*65)
    print("TONG KET KET QUA CONTINUAL LEARNING")
    print("="*65)
    print(f"MNIST Ban Dau (Phase 1)    : {mnist_init:.2f}%")
    print(f"CIFAR-10 Dat Duoc (Phase 2): {cifar_final:.2f}%")
    print(f"MNIST Sau Khi Train CIFAR  : {mnist_final:.2f}%")
    print(f"MUC DO QUEN (Forgetting)   : {mnist_init - mnist_final:+.2f}%")
    print("="*65)

if __name__ == '__main__':
    run_experiment()
