import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

def train_sota_mnist():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing SOTA MNIST Architecture on {device}...")
    
    # Chuẩn MNIST SOTA: Không resize lên 32x32 làm nhòe nét, giữ 28x28 1 channel
    # Data Augmentation nhẹ chống overfit
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=train_transform), 
                              batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=test_transform), 
                             batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Kiến trúc chuyên dụng đạt >99.5% - 99.8% MNIST (Ensemble / Deep Conv + BN + Dropout)
    class SOTAMNISTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1: 28x28 -> 14x14
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2), # stride 2 thay cho maxpool giữ thông tin
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.25),
                
                # Block 2: 14x14 -> 7x7
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
                
                # Block 3: 7x7 -> 4x4
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
                nn.Linear(128, 10)
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = SOTAMNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    best_acc = 0.0
    for epoch in range(1, 11):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = 100.0 * correct / total
        if acc > best_acc:
            best_acc = acc
        print(f"Epoch {epoch:2d}/10 ({time.time()-t0:.1f}s) -> Test Accuracy: {acc:.2f}% (Best: {best_acc:.2f}%)")

if __name__ == '__main__':
    train_sota_mnist()
