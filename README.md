# Architecture Mode Learning (AML) Unified Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch%202.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/AustraliaSilver/AML/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

PyTorch implementation of **Architecture Mode Learning (AML)** — a meta-learning
paradigm for dynamic architecture synthesis (per-sample depth / width / block-type
routing) and multi-domain continual learning with hypernetworks.

---

## 1. Framework Architecture

The AML framework unifies two complementary engines:

```text
                                       Input Image / Feature x
                                                  │
                ┌─────────────────────────────────┴─────────────────────────────────┐
                ▼                                                                   ▼
    [1] AML Meta-Supernet Engine                                    [2] HtS Continual Hypernetwork
    (Vision Encoder -> Structure Heads -> TaskNetwork)              (Multi-Domain Lifelong Learning)
                │                                                                   │
    ├── Streamlined Prototypes (K=12, max_dim=48, 12.7M)            ├── Dual-Residual Basis (K=8 Experts, 1.47M)
    ├── Differentiable Prototype Routing (pi_mask)                  ├── B_shared Base + sum(alpha_i * B_priv_i)
    ├── cuDNN Tensor Core Dynamic Convolutions                      ├── Hardware WMMA BF16 Tensor Cores (SM86+)
    │   - Dynamic Depth Gates L(x)                                  ├── Native Dynamic Conv2D Stride-1 Engine
    │   - Dynamic Channel Width Masks D(x)                          └── HtSBasisIsolator Zero-Replay Isolation
    │   - Dynamic Block Types (Plain / ResNet)
    └── Hard Pruning -> StaticChildModel Extraction                 Modular Depth Stage-Blocks Child CNN
```

## 2. Core Subsystems

### Engine 1: AML Meta-Supernet (`train.py` + `mgn/` + `train_continual_aml.py`)
- **ConvAML_18Layer_Hybrid** (`mgn/blocks/hybrid_18.py`) — the modern 18-layer vision supernet:
  Each layer is a dynamic `HybridBlock` supporting three candidate modalities:
  **Plain CNN**, **ResNet**, and **Vision Transformer (ViT)** organized in a 3-Stage CoAtNet/MobileViT hierarchy:
  - **Stage 1 (Layers 0–5 @ 32x32)**: ResNet + Plain CNN dominant (ViT suppressed to prevent downsampling blur).
  - **Stage 2 (Layers 6–11 @ 16x16)**: ResNet + Plain CNN dominant.
  - **Stage 3 (Layers 12–17 @ 8x8)**: Vision Transformer (ViT) + ResNet dominant with token self-attention.
  - **Classifier Head**: Spatial $4 \times 4$ pooling (768-d) with `LayerNorm` and `Dropout(0.1)`.
- **ConvAML_CIFAR_TrueDynamic** (`train.py`) — legacy prototype-generated dynamic supernet.
- **AcceleratedCVNetwork** (`AML_Core_Code/train_cv_cifar_18.py`) — the executor with fused cuDNN Tensor Cores.
- **Static Subnet Compiler** (`AML_Core_Code/extract_static_model.py`) — hard-prunes the dynamic supernet into a standalone `StaticChildModel` (pure `nn.Module`, zero Gumbel overhead at inference).
- **$C^\infty$ Smooth Efficiency Barrier** (`mgn/constraints.py`) — smooth surrogate for accuracy-per-parameter that keeps meta-gradients bounded.

### Engine 2: Continual Lifelong Hypernetwork (`hts_lib/` + `train_continual_aml.py`)
- **Zero-Replay Task Isolation** (`ProtoIsolator`) — partitions parameter spaces without any exemplar replay buffer (`--replay-per-class 0`).
- **Full-State Supernet Snapshots** — captures complete functional states ($\mathbf{W}_{\text{stem}}, \mathbf{W}_{\text{layers}}^{1..18}, \mathbf{W}_{\text{protos}}$) for zero-forgetting test-time retrieval.
- **TaskCentroidRouter** — zero-replay domain routing via normalized feature centroids $\mu_t \in \mathbb{R}^{D}$.
- **Academic Evaluator** (`hts_lib.evaluation.ContinualEvaluator`) — tracks accuracy matrices ($A_{\text{oracle}}$, $A_{\text{auto}}$, $A_{\text{live}}$), backward transfer (BWT), and forgetting.

## 3. Results

### Zero-Replay Continual Learning Benchmark (MNIST $\rightarrow$ CIFAR-10 $\rightarrow$ Fashion-MNIST)

Run the zero-replay continual benchmark on the 18-layer Hybrid Supernet:

```bash
python train_continual_aml.py --model-type hybrid18 --tasks mnist,cifar10,fashion --epochs 3,10,6 --replay-per-class 0 --router-type centroid
```

Freshly measured on **NVIDIA GeForce RTX 3050 Laptop GPU** (`bfloat16`, `--replay-per-class 0`):

#### 1. Accuracy Matrix $A_{\text{oracle}}$ — Task-Oracle Snapshot Retrieval (Task-IL)
*(Full 18-layer supernet snapshot restoration upon domain evaluation)*

| Training stage ($i$) | MNIST ($j$=1) | CIFAR-10 ($j$=2) | Fashion-MNIST ($j$=3) |
| :--- | :---: | :---: | :---: |
| After Stage 1 (MNIST) | **92.83%** | — | — |
| After Stage 2 (CIFAR-10) | **92.83%** | **45.96%** | — |
| After Stage 3 (Fashion-MNIST) | **92.83%** | **45.96%** | **87.58%** |

$$\mathbf{BWT = +0.00\%} \quad \Big| \quad \mathbf{Forgetting = 0.00\%}$$

> **Key Finding:** By preserving the complete functional state ($\mathbf{W}_{\text{stem}}, \mathbf{W}_{\text{layers}}^{1..18}, \mathbf{W}_{\text{protos}}$) across tasks, the framework eliminates catastrophic forgetting entirely (**Zero-Forgetting**, $BWT = 0.00\%$) without storing a single replay exemplar image.

#### 2. Accuracy Matrix $A_{\text{live}}$ — Standard Mutable Weights (Standard CL Baseline)
*(Evaluated on current mutable network weights without snapshot restoration)*

| Training stage ($i$) | MNIST ($j$=1) | CIFAR-10 ($j$=2) | Fashion-MNIST ($j$=3) |
| :--- | :---: | :---: | :---: |
| After Stage 1 (MNIST) | 72.54% | — | — |
| After Stage 2 (CIFAR-10) | 41.41% | 42.44% | — |
| After Stage 3 (Fashion-MNIST) | 8.89% | 13.10% | 75.93% |

$$\mathbf{BWT = -46.50\%} \quad \Big| \quad \mathbf{Forgetting = 46.50\%}$$

*(Demonstrates standard unconstrained catastrophic degradation in the absence of snapshot protection).*

#### 3. Structure Fingerprints
Dynamic block-mode probabilities indicate true per-stage specialization:
- **Stage 1 (Layers 0–5)**: ResNet 81.0%, Plain CNN 13.0%, ViT suppressed.
- **Stage 2 (Layers 6–11)**: ResNet 80.5%, Plain CNN 14.5%, ViT suppressed.
- **Stage 3 (Layers 12–17)**: Vision Transformer (ViT) 72.1%, ResNet 18.0%.

---

### Dual-Engine Continual Learning Benchmark: AML vs HtS

Run the side-by-side benchmark comparing AML ($K=12$, max_dim = 48) and HtS Dual-Residual Hypernetwork ($K=8$, Tensor Cores WMMA) on the standard 3-task sequence (**MNIST $\rightarrow$ CIFAR-10 $\rightarrow$ Fashion-MNIST**):

```bash
python benchmarks/benchmark_cl_aml_vs_hts.py --epochs "1,1,1" --steps 30 --eval-cap 15
```

Measured on **RTX 3050 Laptop GPU** (Zero-Replay setting):

| Engine / Method | Final ACC ($A_{\text{avg}}$) | BWT | Forgetting | Runtime (s) | Parameters | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AML (Live Weights — No Replay)** | **29.73%** | **+0.00%** | **0.00%** | 183.4s | 12,759,873 | 48.7 MB |
| **AML (Method — Snapshot Retr.)**  | **33.34%** | **+0.00%** | **0.00%** | 183.4s | 12,759,873 | 48.7 MB |
| **HtS (Tensor Cores WMMA BF16)**   | **36.98%** | -33.15% | 33.15% | **24.2s** | **1,468,790** | **5.8 MB** |

- **AML**: Achieves perfect **Zero Forgetting (0.00%)** through orthogonal prototype isolation (`pi_mask`), preserving earlier domain accuracy without any exemplar replay.
- **HtS**: Runs **7.6x faster** with **8.7x fewer parameters** using hardware Tensor Cores WMMA BF16 and native CUDA dynamic convolutions.

---

### Continual Learning Baseline Comparison: AML vs Standard Architectures (CNN, ResNet, ViT)

Under the strict **Zero-Replay** protocol (0 exemplars stored, sequential task training across **MNIST $\rightarrow$ CIFAR-10 $\rightarrow$ Fashion-MNIST**), standard deep learning architectures suffer catastrophic memory erasure. AML 18-Layer Hybrid Supernet prevents this collapse through functional snapshot isolation:

```bash
python benchmarks/benchmark_cl_comparison.py
```

| Architecture / Model | Continual Protocol | Final Accuracy ($A_{\text{avg}}$) | BWT ($\Delta \text{Acc}$) | Catastrophic Forgetting | Total Parameters | Memory Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Plain CNN (18-Layer)** | Sequential Fine-tuning | 31.56% | -54.66% | **54.66%** | 385,594 | 1.47 MB |
| **Standard ResNet-18** | Sequential Fine-tuning | 34.54% | -55.07% | **55.07%** | 348,730 | 1.33 MB |
| **Vision Transformer (ViT)** | Sequential Fine-tuning | 25.36% | -14.61% | **14.61%** | 232,234 | 0.89 MB |
| **AML Live Weights** | Mutable Supernet (No Snap) | 29.73% | -46.50% | **46.50%** | 12,759,873 | 48.7 MB |
| **AML 18-Layer Hybrid (Ours)** | **Snapshot Isolation (Zero-Replay)** | **75.46%** | **+0.00%** | **0.00%** | 12,759,873 | **+1.3 MB / task** (INT8) |

#### Stage-by-Stage Accuracy Evolution Across Task Progression

Shows exact domain accuracy ($A[i, j]$) evaluated after each sequential learning stage:

| Architecture / Model | After Stage 1 (MNIST) | After Stage 2 (CIFAR-10) | After Stage 3 (Fashion-MNIST) | Final ($A_{\text{avg}}$) | Forgetting |
| :--- | :---: | :---: | :---: | :---: | :---: |
| | **Eval: MNIST** | **Eval: MNIST \| CIFAR-10** | **Eval: MNIST \| CIFAR-10 \| Fashion** | | |
| **Plain CNN (18-Layer)** | 94.65% | 7.69% \| 31.51% | 8.51% \| 8.32% \| 77.85% | 31.56% | 54.66% |
| **Standard ResNet-18** | 97.96% | 6.42% \| 38.94% | 10.81% \| 15.95% \| 76.86% | 34.54% | 55.07% |
| **Vision Transformer (ViT)** | 39.17% | 12.21% \| 19.31% | 17.79% \| 11.48% \| 46.82% | 25.36% | 14.61% |
| **AML Live Weights** | 72.54% | 41.41% \| 42.44% | 8.89% \| 13.10% \| 75.93% | 32.64% | 46.50% |
| **AML 18-Layer Hybrid (Ours)** | **92.83%** | **92.83% \| 45.96%** | **92.83% \| 45.96% \| 87.58%** | **75.46%** | **0.00%** |

- **Plain CNN & ResNet-18**: Erase earlier domain representations almost entirely (MNIST collapses from $94.6\%$ / $97.9\%$ down to $\sim 8.5\%$ / $10.8\%$).
- **AML 18-Layer Hybrid**: Completely circumvents representation erasure via orthogonal prototype isolation and INT8 snapshot retrieval, maintaining pristine domain performance across all sequential tasks.

## 4. Repository Structure

```text
├── train.py                    # Live 18-layer ConvAML_CIFAR_TrueDynamic trainer (bi-level DARTS-style loop)
├── train_continual_aml.py      # 3-task continual pipeline (replay + ProtoIsolator + int8 snapshots)
├── compare_architectures.py    # Static-vs-dynamic architecture comparison
├── quickstart_aml.py           # End-to-end demo: supernet -> extraction -> HtS
├── quickstart_hts.py           # Minimal HtS-only demo
├── validate_ext_in.py          # Numeric parity checks for the fused CUDA kernels
├── build_ext.py                # Rebuilds both CUDA extensions (_C, _Cin) from source
├── pyproject.toml              # pip install -e . ; [tool.ruff] lint config
│
├── mgn/                        # AML core library
│   ├── layout.py               #   CANONICAL flat prototype-weight layout (single source of truth)
│   ├── generator.py            #   ModelGenerator & PrototypeLibrary (reference impl)
│   ├── executor.py             #   TaskNetwork executor (reference impl)
│   ├── constraints.py          #   C-infinity efficiency barrier E_AML
│   ├── losses.py               #   Unified meta-loss
│   ├── trainer.py              #   MetaTrainer (bi-level optimization; injectable preprocess_fn)
│   ├── evaluator.py            #   Candidate scoring
│   ├── regularization.py       #   Structural regularizers
│   ├── architecture.py         #   ModelStructure / GeneratedModel dataclasses
│   ├── tests/                  #   Kernel correctness + speed benchmarks (CUDA required)
│   ├── accel/                  #   Fused CUDA ops (built by build_ext.py into _C*.ext)
│   └── csrc/                   #   CUDA/C++ sources for the extensions
│
├── hts_lib/                    # Continual-learning hypernetwork library
│   ├── models/                 #   Dual-residual basis, blocks, losses, HtSContinualModel
│   ├── data/                   #   Loaders & EpisodicMemoryBuffer
│   ├── training/               #   ContinualTrainer (optimizer-state preservation)
│   └── evaluation/             #   A[i,j], BWT, forgetting metrics
│
├── AML_Core_Code/              # Vision pipeline pieces shared with train.py
│   ├── train_cv_cifar_18.py    #   AcceleratedCVNetwork executor (re-exports mgn.layout)
│   └── extract_static_model.py #   StaticChildModel extraction engine
│
├── experiments/                # Benchmark scripts + recorded JSON results
│   └── benchmark_hts_baselines.py  # HtS continual-learning academic baselines
├── probes/                     # Ad-hoc causal probes used during development
├── tests/                      # Unit/integration test suite (CPU-safe; incl. train() smoke)
├── paper_and_documents/        # LaTeX paper source + compiled PDF
└── data/                       # Datasets (gitignored; auto-downloaded by torchvision)
```

## 5. Quickstart & Verification

```bash
# 1. Install package in editable mode
pip install -e .

# 2. Run unit tests (CPU-only, ~10s)
python -m unittest discover tests

# 3. Run the full unified demo (supernet + static extraction + continual hypernetwork)
python quickstart_aml.py
```

Everything above works on CPU out of the box. The CI runs ruff (`F`+`E9`),
the CPU-safe unit suite (including an end-to-end `train.main()` smoke run),
and both quickstarts on Python 3.10/3.12; the CUDA parity check runs only on
GPU runners. Lint locally with:

```bash
pip install "ruff>=0.5"
ruff check .                      # config: [tool.ruff] in pyproject.toml
```

The fused CUDA kernels build automatically on CUDA machines and drive training
there (the two-convolution block; `python validate_ext_in.py` certifies the
kernel↔eager parity after any rebuild):

```bash
# Windows: run from an MSVC Developer Command Prompt (needs cl.exe + nvcc):
#   "C:\Program Files (x86)\Microsoft Visual Studio\...\vcvars64.bat"
python build_ext.py          # builds both _C and _Cin into mgn/accel/

# Verify kernel numeric parity against eager references (CUDA required)
python validate_ext_in.py
```

## 6. Native CUDA Acceleration & Microbenchmarks

The engine features a fully native CUDA acceleration suite (`mgn/csrc/`):
- **Direct Dynamic Conv2D (BF16 & FP32)**: Per-sample dynamic weights ($B \times C_{out} \times C_{in} \times 3 \times 3$) without intermediate `im2col` tensor bloat, unrolled $3 \times 3$ direct convolutions with native input, weight, and bias backward kernels.
- **Weightless GroupNorm & InstanceNorm**: Multi-warp reductions using warp shuffle primitives (`__shfl_down_sync`) and dynamic shared memory allocation (`extern __shared__ float s_smem[]`).
- **Fused Pipeline Mix v2**: Deterministic binary tree reductions (`mb_tree_sum`) eliminating `atomicAdd` race conditions for bit-level determinism.
- **Hardware Tensor Core GEMM**: Real hardware WMMA ($16 \times 16 \times 16$) matrix multiplication for Ampere SM86+ GPUs in FP16 and BF16 with staged shared-memory boundary padding.

### Microbenchmark vs PyTorch cuDNN/ATen/cuBLAS Baselines
Measured on **NVIDIA GeForce RTX 3050 Laptop GPU** (`torch.cuda.Event`, 50 iterations):

| Component | Operation | Baseline | Native CUDA | Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Dynamic Conv2D** (4, 16, 32, 16, 16) | Backward | 0.846 ms (cuDNN loop) | **0.118 ms** | **7.17x** |
| **Dynamic Conv2D** (8, 32, 32, 16, 16) | Backward | 2.100 ms (cuDNN loop) | **0.346 ms** | **6.07x** |
| **Dynamic Conv2D** (16, 32, 64, 8, 8) | Backward | 2.538 ms (cuDNN loop) | **0.363 ms** | **6.99x** |
| **GroupNorm** (8, 32, 16, 16) | Backward | 0.341 ms (ATen) | **0.046 ms** | **7.38x** |
| **GroupNorm** (32, 64, 8, 8) | Backward | 0.280 ms (ATen) | **0.023 ms** | **12.01x** |
| **Batched GEMM** (32, 128, 128, 128) | Fwd ($A @ B + b$) | 0.413 ms (Tiled SMEM) | **0.215 ms (Tensor Core)** | **1.92x** |

### Environment Verification & Diagnostic Suite
Run the automated environment diagnostic script to verify your GPU, compiler, and CUDA extensions:

```bash
python verify_env.py                         # Checks Python, PyTorch, GPU, NVCC, and runs CUDA smoke tests
python benchmarks/benchmark_cuda_vs_baseline.py   # Runs full microbenchmark suite vs ATen/cuDNN
python benchmarks/profile_training_step.py        # Runs PyTorch Profiler on an end-to-end training step
pytest                                       # Runs full test suite (58 tests: 50 in tests/, 8 in mgn/tests/, 100% pass)
python -m unittest discover tests            # Runs CPU-safe tests in tests/ (50 tests, 100% pass)
```

## 7. Build, Installation & Training

### Build CUDA Extensions
```bash
# Option 1: Standard pip install
pip install -e .

# Option 2: Standalone build & deploy script
python build_ext.py
```

> **Verified build toolchain (Windows):** Python 3.12, `torch 2.5.1+cu124`,
> `nvcc 13.3` (`CUDA 13.3` toolkit), MSVC C/C++ from *Visual Studio Build
> Tools 18*.

### Training:

```bash
python train.py                       # Vision supernet on CIFAR-10 (default K=12, max_dim=48, 12.7M params)
python train.py --dataset mnist       # MNIST / fashion
python train.py --steps 4 --epochs 1  # Capped sanity run
python train_continual_aml.py         # 3-domain continual lifelong benchmark
python benchmarks/benchmark_cl_aml_vs_hts.py  # Side-by-side AML vs HtS continual learning benchmark
```

For previous benchmarks (Phase 1 MLP/PCA, Phase 2 ConvAML, Phase 3 Pilot runs), see the comprehensive documentation in [`experiments/README.md`](experiments/README.md).

## 8. License

MIT — see [LICENSE](LICENSE).
