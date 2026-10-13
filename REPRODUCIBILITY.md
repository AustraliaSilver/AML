# Reproducibility Guide: Zero-Replay Continual Learning on 18-Layer Hybrid AML Supernet

This document provides exact, verified reproduction steps for the **18-Layer Hybrid Supernet (CNN + ResNet + ViT)** and its **Zero-Replay Continual Learning Benchmark** on MNIST $\rightarrow$ CIFAR-10 $\rightarrow$ Fashion-MNIST.

---

## 1. System Requirements & Hardware

- **GPU**: NVIDIA GPU with $\ge$ 4 GB VRAM (Tested on NVIDIA GeForce RTX 3050 Laptop GPU, 6GB VRAM, compute capability 8.6).
- **OS**: Windows 10/11 or Ubuntu 20.04/22.04.
- **Python**: 3.10 to 3.12.
- **PyTorch**: 2.0+ with CUDA and native `bfloat16` or `float16` AMP support.

---

## 2. Installation & Verification

```bash
# 1. Clone repository & install dependencies
git clone https://github.com/AustraliaSilver/AML.git
cd AML
pip install -e .

# 2. Run unit & integration test suite (65 tests, ~10s runtime)
pytest tests/test_hybrid_18_advanced.py tests/test_hybrid_18.py
python -m unittest discover tests
```

Expected result: All unit tests pass cleanly (`Ran 50 tests in ~9s -> OK`, `5 passed in ~8s`).

---

## 3. Continual Learning Benchmarks (Zero-Replay)

### Option A: Fast Verification Schedule (~20 - 25 minutes)

Verifies the mathematical Zero-Forgetting property ($BWT = 0.00\%$) across all 3 tasks with the 18-layer hybrid architecture:

```bash
python train_continual_aml.py \
  --model-type hybrid18 \
  --tasks mnist,cifar10,fashion \
  --epochs 3,10,6 \
  --replay-per-class 0 \
  --router-type centroid
```

#### Expected Benchmark Outputs:

- **Accuracy Matrix $A_{\text{oracle}}$ (Task-IL Snapshot Retrieval)**:
  ```text
  after \ eval on |    mnist |  cifar10 |  fashion
              mnist |    92.83 |        — |        —
            cifar10 |    92.83 |    45.96 |        —
            fashion |    92.83 |    45.96 |    87.58

  [oracle]     BWT: +0.00% | Forgetting: 0.00%
  ```

- **Accuracy Matrix $A_{\text{live}}$ (Standard Mutable Weights)**:
  ```text
  [live]       BWT: -46.50% | Forgetting: 46.50%
  ```

---

### Option B: Full Production Schedule (~55 - 65 minutes)

Extended schedule to maximize CIFAR-10 feature convergence:

```bash
python train_continual_aml.py \
  --model-type hybrid18 \
  --tasks mnist,cifar10,fashion \
  --epochs 5,28,16 \
  --replay-per-class 0 \
  --router-type centroid
```

---

## 4. Architecture Specifications

- **Total Layers**: 18 dynamic blocks (`HybridBlock`).
- **Modality Choices per Layer**:
  - `Mode 0`: Plain Convolution (VGG-style local spatial convolution).
  - `Mode 1`: ResNet Block (Two distinct 3x3 convolutions with residual identity).
  - `Mode 2`: Vision Transformer Block (Global self-attention with token pooling).
- **3-Stage CoAtNet/MobileViT Hierarchy**:
  - **Stage 1 (Layers 0–5 @ 32x32)**: ResNet (81%) + Plain CNN (13%), ViT suppressed to eliminate downsampling blur.
  - **Stage 2 (Layers 6–11 @ 16x16)**: ResNet (80%) + Plain CNN (15%).
  - **Stage 3 (Layers 12–17 @ 8x8)**: Vision Transformer (72%) + ResNet (18%).
- **Classifier Head**: Spatial $4 \times 4$ adaptive pooling (768-d) $\rightarrow$ `LayerNorm(768)` $\rightarrow$ `Dropout(0.1)` $\rightarrow$ Prototype Linear Classifier.
- **Continual Learning Memory**: Full-state functional supernet snapshot ($\mathbf{W}_{\text{stem}}, \mathbf{W}_{\text{layers}}^{1..18}, \mathbf{W}_{\text{protos}}$) + `TaskCentroidRouter` unit hypersphere vector centroids $\mu_t \in \mathbb{R}^{512}$.

---

## 5. Historical Benchmarks & Pilot Logs

For earlier experimental phases (Phase 1 MLP/PCA benchmarks, Phase 2 ConvAML spatial benchmarks, and Phase 3 No-Replay pilot runs with raw JSON outputs), refer to the dedicated documentation in [`experiments/README.md`](experiments/README.md).
