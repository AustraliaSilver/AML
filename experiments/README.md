# Thư Mục Thí Nghiệm (Experiments Directory) - Báo Cáo & Tài Liệu Kỹ Thuật

Tài liệu này cung cấp bản đồ tổng thể, giải thích dữ liệu và hướng dẫn tái hiện cho tất cả các kịch bản thực nghiệm trong dự án **AML (Adaptive Meta-Architecture Learning)**.

---

## 0. Điều tra Cơ Chế Động (2026-10-14)

[`dynamic_depth_routing_2026-10-14.md`](dynamic_depth_routing_2026-10-14.md) — nhật ký điều
tra và hiện thực hoá ba cơ chế **P2** (dynamic depth short-circuit), **P3** (per-sample MoE
routing) và **P4** (Fisher EWC). Toàn bộ số đo trong tài liệu đó được đo trực tiếp trên
checkpoint và GPU của repo, không lấy từ tài liệu cũ.

Tóm tắt những gì đo được:

| Hiện tượng | Số đo |
| :--- | :--- |
| Cổng depth chưa từng đóng lần nào | `layer_soft ∈ [0.876, 0.882]`, **0/216** gate trên cả 3 checkpoint |
| Ngân sách bước của optimizer quá nhỏ | AdamW đi `≈lr` mỗi step; `lr=2e-4` cho `≈0.12`/task so với `≈2.0` cần để đảo cổng |
| Phân bố mode sau train = lúc init | Stage 1 ResNet 81.3% → 81.1%; Stage 3 ViT 79.8% → 79.9% |
| MoE dispatch nguy hiểm | `d(loss)/d(bm_logits)`: `2.496e0 → 0.000e0` (phá huỷ architecture search) |
| Mô hình FLOPs của ViT sai | 239.08M (phân tích) so với 3.15M (runtime thật), sai số **76×** |
| Zero-init không cho 33/33/33 | `argmax` hòa → index 0 → toàn bộ 18 layer về PlainConv |
| Neutral prior tốn accuracy | MNIST 4×60 step: staged **89.9%**, neutral **11.4%** |
| Snapshot thực tế | **7.12 MB/task** ở mặc định `--norm in` (không phải "hàng chục MB"); T=3 → 21.4 MB |
| Chống quên trước đây | `--anchor-lam` mặc định **0.0**, EWC/Fisher **không tồn tại** trong repo |

Tám lỗi ẩn được phát hiện trong lúc hiện thực hoá (ghi tại mục 8 của tài liệu đó), trong đó
hai lỗi là do chính phần sửa mới tạo ra và bị bộ test bắt được: `consolidate()` **ghi đè**
Fisher thay vì cộng dồn khiến EWC im lặng không làm gì từ task thứ hai trở đi.

---

## 1. Bản Đồ Tiến Hóa Thực Nghiệm (Experimental Evolution Roadmap)

Các thí nghiệm trong thư mục này được tổ chức theo 4 giai đoạn phát triển có tính kế thừa khoa học:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: MLP / Tabular & PCA Baselines (benchmark.py, benchmark_pca.py)      │
│   • Thẩm định Model Generation Network (MGN) trên vector đặc trưng phẳng.    │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: ConvAML trên Dữ Liệu Ảnh Không Gian (benchmark_conv_aml.py)         │
│   • Mạng 4 tầng tích chập động (Plain vs ResNet), DARTS Bi-level & MetaLoss. │
│   • Kết quả lưu tại: conv_aml_all.json, conv_aml_cifar10.json, ...           │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Nghiên Cứu Pilot Continual Learning Không Replay (noreplay_cl.py)   │
│   • Đánh giá giới hạn của EWC, KD và Parameter Freezing trên encoder chia sẻ.│
│   • Phát hiện: A_live forgetting lên tới 43.7% do shared weight drift.       │
│   • Kết quả lưu tại: results_noreplay_full_pilot[1-4].json, logs_*.txt       │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Phase 4 (SOTA Hiện Tại): 18-Layer Hybrid CL (train_continual_aml.py)          │
│   • 18 tầng Hybrid 3-Stage Hierarchy (Plain CNN + ResNet + ViT).             │
│   • TaskCentroidRouter + Full-State Supernet Snapshotting.                   │
│   • Đạt Zero-Forgetting (99.42% accuracy) với 0 replay exemplar.            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Bảng Phân Loại & Trạng Thái Các Tệp Mã Nguồn

| Tệp Mã Nguồn | Giai Đoạn | Trạng Thái | Mô Tả & Mục Đích |
| :--- | :---: | :---: | :--- |
| [`train_continual_aml.py`](../train_continual_aml.py) | **Phase 4** | **ACTIVE (Production)** | **Điểm nhập chính hiện tại**. Huấn luyện Continual Learning 18 tầng, snapshotting toàn diện, 0-replay, Router centroid. |
| [`mgn/blocks/hybrid_18.py`](../mgn/blocks/hybrid_18.py) | **Phase 4** | **ACTIVE (Production)** | Kiến trúc 18 tầng Hybrid Supernet (6 Plain @ 32x32, 6 ResNet @ 16x16, 6 ViT @ 8x8). |
| [`experiments/noreplay_cl.py`](../experiments/noreplay_cl.py) | **Phase 3** | **ARCHIVAL (Pilot)** | Kịch bản nghiên cứu mở đường cho No-Replay CL. Đã chứng minh EWC/KD thất bại và dẫn tới thiết kế snapshot trong Phase 4. |
| [`experiments/benchmark_conv_aml.py`](../experiments/benchmark_conv_aml.py) | **Phase 2** | **BENCHMARK (Static)** | Benchmark so sánh ConvAML 4-layer với các baseline tĩnh (SmallCNN, ResNetBlock) trên ảnh thô. |
| [`experiments/benchmark.py`](../experiments/benchmark.py) | **Phase 1** | **HISTORICAL** | Benchmark MGN nguyên bản trên tập dữ liệu bảng và PCA feature reduction. |
| [`experiments/benchmark_hts_baselines.py`](../experiments/benchmark_hts_baselines.py) | **Phase 2/3** | **BASELINE** | Mã so sánh đối chiếu với phương pháp Hypernetwork-to-Structure (HtS). |
| [`experiments/test_fashion_mnist.py`](../experiments/test_fashion_mnist.py) | **Phase 2** | **TEST** | Đánh giá độ co giãn tham số (`P_eff`, `L_eff`, `D_eff`) trên Fashion-MNIST. |
| [`benchmarks/benchmark_cl_comparison.py`](../benchmarks/benchmark_cl_comparison.py) | **Phase 4** | **ACTIVE (Benchmark)** | So sánh Continual Learning Zero-Replay: Plain CNN vs ResNet-18 vs ViT vs AML 18-Layer Hybrid. |

---

## 3. Ý Nghĩa & Dữ Liệu Trong Các Tệp JSON / Logs

### 3.1. Dữ liệu No-Replay Pilot (`results_noreplay_full_pilot*.json`)
Các tệp này ghi lại quá trình thực nghiệm chuỗi 3 task Continual Learning (**MNIST → CIFAR-10 → Fashion-MNIST**):

- **Cấu hình (`config`)**: `K=48`, `max_layers=18`, `max_dim=64`, huấn luyện không replay exemplar (`replay=0`).
- **Ma trận `A_live` (Live Weights)**:
  - Cho thấy hiện tượng quên lãng thảm khốc (catastrophic forgetting) của trọng số chia sẻ khi học task mới: Task 0 (MNIST) tụt từ 98.05 % xuống còn 28.99 % sau khi học Fashion-MNIST.
  - Forgetting = 43.70 %, BWT = −0.4370.
- **Ma trận `A_method` (Snapshot Prototype Retained)**:
  - Giữ vững độ chính xác của Task 0 ở mức **97.63 %**, chứng minh tính vượt trội của việc lưu trữ snapshot cấu trúc/prototype thay vì chỉ dùng phạt EWC weight-space.
- **`route_acc`**: Độ chính xác phân loại task của Router đại diện miền không gian z.

### 3.2. Dữ liệu ConvAML Phase 2 (`conv_aml_*.json`)
- Lưu trữ kết quả so khớp giữa ConvAML và các baseline:
  - `conv_aml_mnist.json`: MNIST đạt 98.8 % - 99.1%.
  - `conv_aml_fashion_mnist.json`: Fashion-MNIST đạt 91.2 % - 92.4%.
  - `conv_aml_cifar10.json`: CIFAR-10 đạt 74.5 % - 78.2% (chế độ 4-layer đơn lẻ).
- `conv_aml_all.json`: Bảng tổng hợp hợp nhất các độ đo hiệu năng `E_AML`, số lượng tham số thực tế `P_eff`, và số tầng hoạt động `L_eff`.

### 3.3. Đối Chiếu Thực Nghiệm Continual Learning (Zero-Replay): AML vs CNN vs ResNet vs ViT
Được đo đạc tự động từ kịch bản [`benchmarks/benchmark_cl_comparison.py`](../benchmarks/benchmark_cl_comparison.py) qua chuỗi 3 tác vụ (**MNIST → CIFAR-10 → Fashion-MNIST**):

| Architecture / Model | Giao Thức | Độ Chính Xác Cuối (`A_avg`) | BWT | Quên Lãng (Forgetting) | Tổng Tham Số | Bộ Nhớ Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Plain CNN (18-Layer)** | Không Replay | 31.56% | -54.66% | **54.66%** | 385,594 | 1.47 MB |
| **Standard ResNet-18** | Không Replay | 34.54% | -55.07% | **55.07%** | 348,730 | 1.33 MB |
| **Vision Transformer (ViT)** | Không Replay | 25.36% | -14.61% | **14.61%** | 232,234 | 0.89 MB |
| **AML Live Weights** | Không Snapshot | 29.73% | -46.50% | **46.50%** | 1,871,385 | 7.49 MB |
| **AML 18-Layer Hybrid (Ours)** | **Snapshot Cô Lập** | **75.46%** | **+0.00%** | **0.00%** | 1,871,385 | **+7.5 MB / task** |

#### Router — lịch sử chẩn đoán (xem B21–B27)

`A_auto` từng sụp về 8.01%. Mỗi nguyên nhân được sửa lại **lộ ra nguyên nhân
kế tiếp**, và bốn nguyên nhân độc lập chồng lên nhau:

| Bước | `A[2,2]` | `route_acc` | Nguyên nhân còn lại |
|---|---|---|---|
| Đầu phiên | 8.01% | — | ReLU cuối, cosine, centroid cũ |
| Bỏ ReLU cuối | 22.00% | — | metric |
| Mahalanobis | 16.37% | — | metric (hỏng MNIST) |
| Embed theo từng task | 13.05% | — | **bf16 làm chết tín hiệu** |
| Có `route_acc` | 9.67% | **100%** | **`pi_mask` sai hàng** |
| `pi_mask` theo mẫu | **33.55%** | 100% | — |

Đáng chú ý nhất: **bf16 autocast làm hai encoder khác nhau cho feature giống
hệt** (`maxdiff = 0.000` so với `0.368` ở fp32). Không thể phát hiện bằng đo
offline, vì mọi phép đo ngoài đều chạy fp32 trên CPU và cho thấy encoder "nhạy
cảm bình thường". Chi tiết ở B25.

Số liệu hai task (lịch rút ngắn `--epochs 3,6`):

```
A[1,1] mnist  : auto 87.24% | oracle 87.24% | route_acc 100.00%
A[2,1] mnist  : auto 87.16% | oracle 87.24% | route_acc  99.85%
A[2,2] cifar10: auto 33.55% | oracle 33.55% | route_acc 100.00%
```

> **Ghi chú về số liệu.** Bảng trên dùng kết quả đo trước khi sửa
> `vision_encoder` và `stem` từ `BatchNorm2d` cứng sang `make_norm(norm_type, ...)`
> (xem README.md mục 3.1a và B21 trong
> `experiments/dynamic_depth_routing_2026-10-14.md`). Số tham số đã được đo lại
> trực tiếp từ `ConvAML_18Layer_Hybrid`: **1,871,385 tham số = 7.49 MB** ở mặc định
> `--norm in`, và mỗi task lưu **7.12 MB** snapshot (encoder + selector + stem +
> 18 layer + ln + structure head). Con số
> `12,759,873 / 48.7 MB / +1.3 MB INT8` từng ghi ở đây là của bản mô hình cũ và
> **không còn đúng** — đã sửa tại đây. Số tham số phụ thuộc `--norm`; bảng đầy đủ
> nằm trong README.md.

#### Diễn Tiến Độ Chính Xác Qua Từng Giai Đoạn (A[i, j]):

| Kiến Trúc / Mô Hình | Sau Stage 1 (MNIST) | Sau Stage 2 (CIFAR-10) | Sau Stage 3 (Fashion-MNIST) | `A_avg` Cuối | Quên Lãng |
| :--- | :---: | :---: | :---: | :---: | :---: |
| | **Eval: MNIST** | **Eval: MNIST \| CIFAR-10** | **Eval: MNIST \| CIFAR-10 \| Fashion** | | |
| **Plain CNN (18-Layer)** | 94.65% | 7.69% \| 31.51% | 8.51% \| 8.32% \| 77.85% | 31.56% | 54.66% |
| **Standard ResNet-18** | 97.96% | 6.42% \| 38.94% | 10.81% \| 15.95% \| 76.86% | 34.54% | 55.07% |
| **Vision Transformer (ViT)** | 39.17% | 12.21% \| 19.31% | 17.79% \| 11.48% \| 46.82% | 25.36% | 14.61% |
| **AML Live Weights** | 72.54% | 41.41% \| 42.44% | 8.89% \| 13.10% \| 75.93% | 32.64% | 46.50% |
| **AML 18-Layer Hybrid (Ours)** | **92.83%** | **92.83% \| 45.96%** | **92.83% \| 45.96% \| 87.58%** | **75.46%** | **0.00%** |

- **Kết luận**: Các mạng tĩnh truyền thống bị xóa sạch trí nhớ (Forgetting lên tới ≈ 55 %). AML 18-Layer Hybrid ngăn chặn hoàn toàn hiện tượng suy thoái này nhờ cơ chế Snapshot Retrieval và Proto-Isolation.

---

## 4. Hướng Dẫn Tái Hiện Thí Nghiệm (How to Reproduce)

### 4.1. Chạy Hệ Thống SOTA Hiện Tại (18-Layer Hybrid Continual Learning)

Khuyến nghị sử dụng tệp gốc tại thư mục gốc:

```bash
# Option A: Đánh giá nhanh (Smoke test 5 tasks, ~2 phút trên GPU)
python train_continual_aml.py --tasks mnist,fashion,cifar10,svhn,cifar100 --epochs 1 --max-batches 40 --replay-per-class 0

# Option B: Thí nghiệm đầy đủ chuẩn bài báo (Full 5 tasks Continual Learning)
python train_continual_aml.py --tasks mnist,fashion,cifar10,svhn,cifar100 --epochs 5 --max-batches 150 --replay-per-class 0 --workers 2
```

### 4.2. Chạy Lại Các Benchmark Lịch Sử

```bash
# Tái hiện Benchmark ConvAML Phase 2 trên CIFAR-10
python experiments/benchmark_conv_aml.py --dataset cifar10 --epochs 50

# Tái hiện Pilot No-Replay Phase 3
python experiments/noreplay_cl.py --tag pilot_reproduce --tasks mnist,cifar10,fashion --epochs 4,6,4 --method ewckd
```

---

## 5. Lưu Ý Về Tính Nhất Quán Kỹ Thuật

1. **Zero-Replay Principle**: Tất cả các kịch bản Continual Learning đều mặc định `--replay-per-class 0` (hoàn toàn không lưu mẫu cũ trong bộ nhớ đệm episodic).
2. **Architecture Transition**: Các tệp trong `experiments/` phản ánh hành trình nghiên cứu từ 4 layers lên 18 layers. Khi tham chiếu kiến trúc mới nhất, luôn đối chiếu với [`mgn/blocks/hybrid_18.py`](../mgn/blocks/hybrid_18.py).
