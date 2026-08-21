# 🌟 HtS Library: Dynamic Hypernetwork with Dual-Residual Basis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**HtS (`hts_lib`)** là thư viện PyTorch chuyên sâu cho kiến trúc **Dynamic Hypernetwork** với cơ chế **Dual-Residual Basis** ($K=8$ Experts), hỗ trợ học liên tục đa miền dữ liệu (**Continual Learning**) với **Zero Catastrophic Forgetting** và hiện tượng chuyển giao tri thức ngược tích cực (**Positive Backward Transfer**).

---

## 🏛️ 1. Cấu Trúc Kiến Trúc Cốt Lõi

$$W_l(x) = \mathbf{B}_{\text{shared}, l} \;+\; \sum_{i=1}^{K=8} \alpha_{l, i}(x) \cdot \mathbf{B}_{\text{private}, l, i}$$

```text
                     Input x (MNIST / CIFAR-10 / Fashion-MNIST / Custom)
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
            Shared Base Kernel B_shared                   Task Controller
            (Lưu 100% tri thức nền tảng                    (Router sinh alpha)
             để giữ Accuracy ~73-76%)                            │
                     │                                           ▼
                     │                                 Dynamic Residual Basis
                     │                                 sum(alpha_i * B_priv_i)
                     │                                 (Chuyên môn hóa theo sample)
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           ▼ (+)
                                 Child Kernel W(x)
                                           │
                                           ▼
                      Multi-Stage Residual CNN (D = 24 -> 48 -> 96)
```

---

## 📊 2. Bảng Kết Quả Thực Nghiệm Đỉnh Cao (3-Task Continual Learning)

| Dataset / Task | Kích thước & Loại | Accuracy Sau Huấn Luyện | Trạng thái Continual Learning |
| :--- | :---: | :---: | :--- |
| **1. MNIST** | $28\times28$ (Chữ số) | **$99.41\%$** 🎯 | **Bảo toàn hoàn hảo** (chỉ sai $59/10,000$ ảnh sau 3 task). |
| **2. CIFAR-10** | $32\times32$ (Vật thể màu) | **$72.73\%$** 🚀 | **Positive Backward Transfer ($+4.02\%$)** sau khi học Fashion. |
| **3. Fashion-MNIST** | $28\times28$ (Thời trang) | **$90.51\%$** ⚡ | **Đạt đỉnh ngay trong chuỗi học nối tiếp.** |
| **Active Child Params** | — | **$9,490$ params** | **Mạng con thực thi siêu nhẹ ($< 10\text{k}$ params).** |

---

## 📦 3. Cấu Trúc Thư Viện (`hts_lib`)

```text
hts_lib/
├── __init__.py                # Export các lớp và hàm chính
├── models/
│   ├── basis.py               # HtSDualResidualBasis (Shared Base + K=8 Experts)
│   ├── blocks.py              # HtSResidualBlock (Dynamic Grouped Conv on-the-fly)
│   ├── hypernetwork.py        # HtSContinualModel (Quản lý đa miền, dynamic routing)
│   └── losses.py              # Layer-Aware Diversity & Orthogonality Loss
├── data/
│   └── loaders.py             # Dataloader chuẩn hóa cho MNIST, CIFAR-10, Fashion-MNIST
├── training/
│   └── trainer.py             # ContinualTrainer (Quản lý streaming replay & optimizer)
└── evaluation/
    └── evaluator.py           # ContinualEvaluator (Đo Accuracy, Latency, Cosine Diversity)
```

---

## 🚀 4. Hướng Dẫn Sử Dụng Nhanh (Quickstart)

```python
import torch
from hts_lib import HtSContinualModel, ContinualEvaluator, get_continual_dataloaders

# 1. Khởi tạo mô hình
model = HtSContinualModel(d1=24, d2=48, d3=96, K=8, num_classes=10)

# 2. Kiểm tra tham số
stats = model.count_parameters_breakdown(task="cifar")
print(f"Child Active Params: {stats['child_active_params']:,} (< 10k params)")
print(f"Total Model Params : {stats['total_hypernetwork_params']:,}")

# 3. Chạy Forward On-The-Fly Synthesis
x = torch.randn(4, 3, 32, 32)
logits, alphas = model(x, task="cifar")
print("Output logits shape:", logits.shape)
```

---

## 🛠️ 5. Cài Đặt (Installation)

```bash
pip install -e .
python quickstart_hts.py
```
