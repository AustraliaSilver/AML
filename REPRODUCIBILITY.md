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

### Option D: Dynamic Depth + Per-Sample Routing + EWC (2026-10-14)

Adds three mechanisms that were previously inert. Full measurements and the bugs
found while validating them are in
[`experiments/dynamic_depth_routing_2026-10-14.md`](experiments/dynamic_depth_routing_2026-10-14.md).

```bash
python train_continual_aml.py \
  --model-type hybrid18 \
  --tasks mnist,cifar10,fashion \
  --epochs 3,10,6 \
  --replay-per-class 0 \
  --router-type centroid \
  --arch-lr 3e-3 \
  --depth-cost 0.02 \
  --flops-cost 0.01 \
  --ewc-lambda 100.0
```

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--arch-lr` | `3e-3` | LR for `structure_prototypes` / `block_mode_prototypes`. Was `2e-4`, which capped the per-task travel of the depth logit at ≈0.12 against the ≈2.0 needed to close a gate. |
| `--depth-cost` | `0.02` | Sparsity penalty on the depth gates. **Required** for the short-circuit to have anything to skip — without it all 18 gates stay open. |
| `--depth-target` | `0.0` | `>0` turns the penalty into a hinge `relu(mean − target)`, bounding pruning to `1−target` so the trunk can never collapse. |
| `--flops-cost` | `0.0` | Complexity term, in GFLOP/sample. ViT is the cheapest mode at 32×32 and 16×16 (tokens are pooled to 8×8), so this pushes those stages toward ViT — the opposite of "Conv at shallow layers". At 8×8 Plain is cheapest again. |
| `--arch-prior` | `staged` | `staged` = CoAtNet-style 3-stage prior. `neutral` = small symmetric noise. Measured at a 4×60-step budget: staged 89.9 %, neutral 11.4 %. |
| `--ewc-lambda` | `0.0` | Elastic Weight Consolidation on the shared trunk. `>0` is what lets forgetting be bounded without storing an 18-layer snapshot per task. |
| `--amb-topk` | `0` | Restrict the ambiguous-sample retry (router margin `< --route-tau`) to the k most likely domains. `0` retries under every seen snapshot, which re-loads enc + sel + stem + 18 layers + ln (~1.5 M parameters) per domain per batch. `2` is the usual choice. |

Depth telemetry is printed every epoch:

```text
[mnist] ep 3/3 | 264.7s | loss 1.31 | mnist acc 87.31% | [Plain: 12%, ResNet: 59%, ViT: 29%] | [active L: 15.20/18, 331.5 MFLOP/s]
```

`active L` is the mean number of open gates per sample; `MFLOP/s` is the
differentiably expected cost of the selected structure.

#### Regression tests

```bash
python -m pytest tests/test_dynamic_depth_routing.py -q     # 18 tests
python scratch/test_p2_p3.py                                 # equivalence + gradient guards
```

These encode the specific failures observed: uninitialised memory leaking into
per-sample-skipped rows, dtype mismatch under autocast, the MoE dispatch zeroing
the architecture gradient, and EWC's Fisher being reset instead of accumulated.

---

### Option C: Baseline Comparison (Plain CNN vs ResNet-18 vs ViT vs AML) (~3 minutes)

Directly replicates the side-by-side continual learning benchmark under identical zero-replay conditions:

```bash
python benchmarks/benchmark_cl_comparison.py
```

Expected output confirms severe memory erasure in standard static networks vs AML preservation:

| Architecture / Model | After Stage 1 (MNIST) | After Stage 2 (CIFAR-10) | After Stage 3 (Fashion-MNIST) | Final ($A_{\text{avg}}$) | Forgetting |
| :--- | :---: | :---: | :---: | :---: | :---: |
| | **Eval: MNIST** | **Eval: MNIST \| CIFAR-10** | **Eval: MNIST \| CIFAR-10 \| Fashion** | | |
| **Plain CNN (18-Layer)** | 94.65% | 7.69% \| 31.51% | 8.51% \| 8.32% \| 77.85% | 31.56% | 54.66% |
| **Standard ResNet-18** | 97.96% | 6.42% \| 38.94% | 10.81% \| 15.95% \| 76.86% | 34.54% | 55.07% |
| **Vision Transformer (ViT)** | 39.17% | 12.21% \| 19.31% | 17.79% \| 11.48% \| 46.82% | 25.36% | 14.61% |
| **AML Live Weights** | 72.54% | 41.41% \| 42.44% | 8.89% \| 13.10% \| 75.93% | 32.64% | 46.50% |
| **AML 18-Layer Hybrid (Ours)** | **92.83%** | **92.83% \| 45.96%** | **92.83% \| 45.96% \| 87.58%** | **75.46%** | **0.00%** |

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
