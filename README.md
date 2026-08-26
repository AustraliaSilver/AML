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
    ├── PrototypeLibrary (K=32 prototype rows)                      ├── Dual-Residual Basis (K=8 Experts)
    ├── Gumbel-Softmax Continuous Routing                           ├── B_shared Base + sum(alpha_i * B_priv_i)
    │   - Dynamic Depth Gates L(x)                                  ├── Layer-Aware Diversity & Orthogonality
    │   - Dynamic Channel Width Masks D(x)                          └── Episodic Memory Buffer + Routing Isolation
    │   - Dynamic Block Types (Plain / ResNet)
    └── Hard Pruning -> StaticChildModel Extraction                 Synthesized Child CNN (<10k active params)
```

## 2. Core Subsystems

### Engine 1: AML Meta-Supernet (`train.py` + `mgn/` + `AML_Core_Code/`)
- **ConvAML_CIFAR_TrueDynamic** (`train.py`) — the live 18-layer vision supernet:
  a CNN encoder produces a latent `z`, from which per-sample depth gates, width
  masks and Plain/ResNet block modes are generated; weights are mixed from a
  shared prototype library via attention over routing logits `pi`.
- **AcceleratedCVNetwork** (`AML_Core_Code/train_cv_cifar_18.py`) — the executor.
  Uses the fused CUDA kernels in `mgn/accel/_C*` when available and falls back to
  native PyTorch ops otherwise (identical math, slower).
- **Static Subnet Compiler** (`AML_Core_Code/extract_static_model.py`) — hard-prunes
  the dynamic supernet into a standalone `StaticChildModel` (pure `nn.Module`, zero
  Gumbel overhead at inference). Entry point: `extract_child(mother, x_dummy)`.
- **$C^\infty$ Smooth Efficiency Barrier** (`mgn/constraints.py`) — smooth surrogate
  for accuracy-per-parameter that keeps meta-gradients bounded.

### Engine 2: Continual Lifelong Hypernetwork (`hts_lib/`)
- **Dual-Residual Basis** (`hts_lib.models.HtSDualResidualBasis`) — decouples an
  invariant shared base from sample-adaptive private residual experts.
- **Continual pipeline** (`train_continual_aml.py`) — task-incremental learning on
  MNIST → CIFAR-10 → Fashion-MNIST with prototype-row isolation, episodic replay,
  int8 snapshots of routing state, and post-training consolidation.
- **Academic Evaluator** (`hts_lib.evaluation.ContinualEvaluator`) — accuracy matrix
  $A_{i,j}$, backward transfer (BWT), average forgetting.

## 3. Results

Run the continual benchmark yourself:

```bash
python train_continual_aml.py                  # full default schedule (5,14,8 epochs + consolidation)
python train_continual_aml.py --epochs 1,1,1 --consolidate-epochs 1   # quick smoke run
```

Freshly measured on the current tree (RTX 3050 6GB, seed 42, full default schedule,
~75 min wall-clock):

### Accuracy matrix $A_{i,j}$ (accuracy on task $j$ after training stage $i$)

| Training stage ($i$) | MNIST ($j$=1) | CIFAR-10 ($j$=2) | Fashion-MNIST ($j$=3) |
| :--- | :---: | :---: | :---: |
| After Stage 1 (MNIST) | **99.32%** | — | — |
| After Stage 2 (CIFAR-10) | **99.34%** | **74.49%** | — |
| After Stage 3 (Fashion-MNIST) | **99.34%** | **74.20%** | **90.36%** |

**Backward transfer: −0.14% · Average forgetting: 0.15%** — i.e. near-zero
catastrophic forgetting across the 3-domain sequence. Per-domain structure
fingerprints confirm the generated architectures genuinely diverge per domain
(pairwise gate/bmRes L1 divergence 0.7–2.1 rather than ~0).

> Note on historical numbers: earlier revisions of this README quoted
> "BWT = +1.95%" from a specific v14/v15 run whose exact configuration is not
> reproducible from the current tree; the repo's own test suite encoded a
> different value. That claim was removed. Treat freshly generated runs as the
> source of truth.

Vision supernet benchmarks (including static baselines) are recorded in
`experiments/*.json`.

## 4. Repository Structure

```text
├── train.py                    # Live 18-layer ConvAML_CIFAR_TrueDynamic trainer (bi-level DARTS-style loop)
├── train_continual_aml.py      # 3-task continual pipeline (replay + ProtoIsolator + int8 snapshots)
├── compare_architectures.py    # Static-vs-dynamic architecture comparison
├── benchmark_baselines.py      # HtS continual-learning baselines
├── quickstart_aml.py           # End-to-end demo: supernet -> extraction -> HtS
├── quickstart_hts.py           # Minimal HtS-only demo
├── validate_ext_in.py          # Numeric parity checks for the fused CUDA kernels
├── build_ext.py                # Rebuilds both CUDA extensions (_C, _Cin) from source
├── pyproject.toml              # pip install -e .
│
├── mgn/                        # AML core library
│   ├── generator.py            #   ModelGenerator & PrototypeLibrary (reference impl)
│   ├── executor.py             #   TaskNetwork executor (reference impl)
│   ├── constraints.py          #   C-infinity efficiency barrier E_AML, budget projection
│   ├── losses.py               #   Unified meta-loss
│   ├── trainer.py              #   MetaTrainer (bi-level optimization)
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
│   ├── train_cv_cifar_18.py    #   AcceleratedCVNetwork executor + prototype layout
│   └── extract_static_model.py #   StaticChildModel extraction engine
│
├── experiments/                # Benchmark scripts + recorded JSON results
├── probes/                     # Ad-hoc causal probes used during development
├── tests/                      # Unit/integration test suite (CPU-safe)
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

Everything above works on CPU out of the box. The fused CUDA kernels are used
automatically on CUDA machines when built:

```bash
# Windows: run from an MSVC Developer Command Prompt (needs cl.exe + nvcc)
python build_ext.py          # builds both _C and _Cin into mgn/accel/

# Verify kernel numeric parity against eager references (CUDA required)
python validate_ext_in.py
```

Training:

```bash
python train.py                       # vision supernet on CIFAR-10 (fused kernels if available)
python train_continual_aml.py         # 3-domain continual benchmark
```

## 6. License

MIT — see [LICENSE](LICENSE).
