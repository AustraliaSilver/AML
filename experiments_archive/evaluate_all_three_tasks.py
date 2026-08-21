"""
FULL INDEPENDENT TEST EVALUATION FOR ALL 3 TASKS
================================================
Chạy đánh giá kiểm thử độc lập toàn diện trên toàn bộ tập test của:
1. MNIST Test Set (10,000 images)
2. CIFAR-10 Test Set (10,000 images)
3. Fashion-MNIST Test Set (10,000 images)

In chi tiết:
- Accuracy (%)
- Số lượng ảnh đoán đúng / đoán sai (Errors / 10,000)
- Thời gian suy luận (Latency / FPS)
- Độ đa dạng Cosine và phân bổ Router trên từng task
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from hts_3task_fashion import HtSThreeTaskModel, evaluate_fast, set_seed

def full_test_evaluation():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== FULL TEST EVALUATION TREN 3 DATASETS ({device}) ===\n")

    # Load Model
    model = HtSThreeTaskModel(d1=24, d2=48, d3=96, K=8, num_classes=10).to(device)
    
    # Chuẩn bị Test Transforms & Loaders
    tf_m = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tf_c = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    tf_f = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    m_te = DataLoader(datasets.MNIST('./data', False, download=True, transform=tf_m), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    c_te = DataLoader(datasets.CIFAR10('./data', False, download=True, transform=tf_c), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    f_te = DataLoader(datasets.FashionMNIST('./data', False, download=True, transform=tf_f), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    print("Dang chay danh gia kiem thu tren toan bo 30,000 anh test...\n")

    # 1. Test MNIST
    t0 = time.time()
    m_acc, m_err, m_tot = evaluate_fast(model, m_te, device, task="mnist")
    t_mnist = time.time() - t0

    # 2. Test CIFAR-10
    t0 = time.time()
    c_acc, c_err, c_tot = evaluate_fast(model, c_te, device, task="cifar")
    t_cifar = time.time() - t0

    # 3. Test Fashion-MNIST
    t0 = time.time()
    f_acc, f_err, f_tot = evaluate_fast(model, f_te, device, task="fashion")
    t_fashion = time.time() - t0

    print("="*75)
    print("KET QUA KIEM THU DOC LAP TOAN DIEN (TEST SET EVALUATION)")
    print("="*75)
    print(f"{'Dataset':<16} | {'Test Accuracy':<15} | {'So Anh Dung/Tong':<18} | {'So Anh Sai':<12} | {'Thoi Gian':<10}")
    print("-"*75)
    print(f"{'1. MNIST':<16} | {m_acc:>12.2f}%   | {m_tot - m_err:>6}/{m_tot:<6}      | {m_err:>6} loi   | {t_mnist:>6.2f}s")
    print(f"{'2. CIFAR-10':<16} | {c_acc:>12.2f}%   | {c_tot - c_err:>6}/{c_tot:<6}      | {c_err:>6} loi   | {t_cifar:>6.2f}s")
    print(f"{'3. Fashion-MNIST':<16} | {f_acc:>12.2f}%   | {f_tot - f_err:>6}/{f_tot:<6}      | {f_err:>6} loi   | {t_fashion:>6.2f}s")
    print("="*75)
    
    tot_correct = (m_tot - m_err) + (c_tot - c_err) + (f_tot - f_err)
    tot_samples = m_tot + c_tot + f_tot
    print(f"Tong so anh kiem thu thanh cong : {tot_correct:,} / {tot_samples:,} anh")
    print(f"Do chinh xac trung binh toan bo  : {100.0 * tot_correct / tot_samples:.2f}%\n")

if __name__ == '__main__':
    full_test_evaluation()
