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
  On CUDA it runs the fused pipeline from `mgn/accel/_C*`; the block algebra is
  the real two-convolution ResNet (shared conv1 + residual conv2), recomputed
  identically by the ATen/eager fallback so both paths stay bit-swap verifiable
  (`validate_ext_in.py` certifies kernel↔eager parity).
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

> These numbers were freshly measured **on the correctness-fixed tree**
> (`4791131`+): deterministic evaluation, validation-split model selection,
> CUDA race fixes, and the two-protocol reporting below. Older revisions of
> this README quoted figures measured with evaluation-time Gumbel noise and
> test-set selection; treat them as superseded.
>
> All figures below were re-measured on the **two-convolution ResNet block**:
> a residual-mode layer executes two distinct `3×3` convs and is charged
> accordingly in `effective_params_formula` / `flops_per_sample`
> (`mgn/architecture.py`); the fused CUDA kernels implement the same block and
> the eager/fused/eager-reference paths agree to within the validated parity
> tolerances (`validate_ext_in.py`).

Run the continual benchmark yourself:

```bash
python train_continual_aml.py                  # full default schedule (5,28,16 epochs + consolidation)
python train_continual_aml.py --epochs 1,1,1 --consolidate-epochs 1   # quick smoke run
```

Freshly measured on the current tree (RTX 3050 6GB, seed 42, full default schedule,
~6 h wall-clock):

### Accuracy matrix $A_{i,j}$ — LIVE weights (standard CL protocol, comparable to ER/EWC/DER literature)

| Training stage ($i$) | MNIST ($j$=1) | CIFAR-10 ($j$=2) | Fashion-MNIST ($j$=3) |
| :--- | :---: | :---: | :---: |
| After Stage 1 (MNIST) | **99.26%** | — | — |
| After Stage 2 (CIFAR-10) | 97.82% | **77.47%** | — |
| After Stage 3 (Fashion-MNIST) | 98.01% | 71.21% | **90.85%** |

**LIVE: BWT −3.76% · average forgetting 3.76%**

### Accuracy matrix $A_{i,j}$ — METHOD (per-domain int8 snapshots + router, incl. test-time retrieval of ambiguous samples)

| Training stage ($i$) | MNIST | CIFAR-10 | Fashion-MNIST |
| :--- | :---: | :---: | :---: |
| After Stage 1 (MNIST) | **99.23%** | — | — |
| After Stage 2 (CIFAR-10) | 99.24% | **77.43%** | — |
| After Stage 3 | **99.24%** | **77.28%** | **90.91%** |

**METHOD: BWT −0.07% · average forgetting 0.07%** — the gap versus the LIVE
column (~+6.1pt on CIFAR-10) is the measured contribution of snapshot
retrieval, reported separately rather than blended into one headline number.

Per-domain structure fingerprints confirm the generated architectures genuinely
diverge per domain (pairwise gate L1 divergence **2.41–5.47** rather than ~0).

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

> The `mgn` benchmark trainer (`experiments/benchmark_conv_aml.py`) uses the
> pure-eager `TaskNetwork` by default so recorded JSONs stay reproducible.
> Pass `--fused-executor` to route its conv branch through the same fused v2
> pipeline. After the two-phase `fused_mix_bwd_v2` custom CUDA kernels
> (coalesced grid-stride backward of the mix algebra; `mgn/csrc/kernels/
> fused_pipeline.cu`), the pipeline backward costs 89 ms vs 169 ms for the
> original ATen-recompute chain (-47%). Measured on the RTX 3050 laptop:
> end-to-end `--fast` per seed 232→119 s ConvAML (1.95x) / 300→89 s No-O
> (3.4x); the `train.py`/continual path shares the same backward kernels.
> Fused falls back to eager automatically on any unmet precondition.
> Gradients carry the pipeline's bf16 conv-backward drift (~1e-3 rel,
> gated by `validate_ext_in`), so fused runs are NOT bit-identical to the
> recorded numbers.

> **Verified build toolchain (Windows):** Python 3.12, `torch 2.5.1+cu124`,
> `nvcc 13.3` (`CUDA 13.3` toolkit), MSVC C/C++ from *Visual Studio Build
> Tools 18*. The wheels bundle their own cu12 runtime, so a newer host `nvcc`
> still links correctly; other combinations are unverified. The built
> `.pyd`s in `mgn/accel/` are build artifacts and NOT tracked in git (see
> `.gitignore`): regenerate them with `python build_ext.py` (needs the
> toolchain above); every consumer falls back to the pure-eager path when
> they are absent, so a plain `pip install -e .` + `python train.py` still
> runs -- just slower on CUDA.

Training:

```bash
python train.py                       # vision supernet on CIFAR-10 (fused kernels if available)
python train.py --dataset mnist       # MNIST / fashion (both mapped to 3x32x32)
python train.py --steps 4 --epochs 1  # capped sanity run (no full-epoch cost)
python train_continual_aml.py         # 3-domain continual benchmark
```

## 6. License

MIT — see [LICENSE](LICENSE).
