# Architecture Mode Learning (AML) Unified Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch%202.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/AustraliaSilver/AML/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

PyTorch implementation of **Architecture Mode Learning (AML)** — a meta-learning paradigm for
dynamic architecture synthesis (per-sample depth, width and block-type routing) and
multi-domain continual learning with hypernetworks.

> **Contents**
> [1. Architecture](#1-architecture) · [2. Core subsystems](#2-core-subsystems) ·
> [3. Results](#3-results) · [4. Repository layout](#4-repository-layout) ·
> [5. Quickstart](#5-quickstart) · [6. CUDA acceleration](#6-native-cuda-acceleration--microbenchmarks) ·
> [7. Build & train](#7-build-installation--training) · [8. License](#8-license)

---

## 1. Architecture

Two complementary engines:

```text
                                       Input Image / Feature x
                                                  │
                ┌─────────────────────────────────┴───────────────────────────────┐
                ▼                                                                    ▼
    [1] AML Meta-Supernet Engine                          [2] HtS Continual Hypernetwork
    (Vision Encoder -> Structure Heads -> TaskNetwork)    (Multi-Domain Lifelong Learning)
                │                                                                    │
    ├── Streamlined Prototypes (K=12, max_dim=48, 1.87M) ├── Dual-Residual Basis (K=8 Experts, 1.47M)
    ├── Differentiable Prototype Routing (pi_mask)       ├── B_shared + Σ (alpha_i · B_priv_i)
    ├── cuDNN Tensor Core Dynamic Convolutions           ├── Hardware WMMA BF16 Tensor Cores (SM86+)
    │   - Dynamic Depth Gates L(x)                      ├── Native Dynamic Conv2D Stride-1 Engine
    │   - Dynamic Channel Width Masks D(x)              └── HtSBasisIsolator Zero-Replay Isolation
    │   - Dynamic Block Types (Plain / ResNet / ViT)
    └── Hard Pruning -> StaticChildModel Extraction      Modular Depth Stage-Blocks Child CNN
```

## 2. Core subsystems

### Engine 1 — AML Meta-Supernet

`train.py` · `mgn/` · `train_continual_aml.py`

| Component | What it is |
| :-- | :-- |
| **`ConvAML_18Layer_Hybrid`** (`mgn/blocks/hybrid_18.py`) | The modern 18-layer vision supernet. Each layer is a `HybridBlock` choosing between **Plain CNN**, **ResNet** and **Vision Transformer** in a 3-stage CoAtNet/MobileViT hierarchy. |
| **`ConvAML_CIFAR_TrueDynamic`** (`train.py`) | Legacy prototype-generated dynamic supernet, 12.76M params. |
| **`AcceleratedCVNetwork`** (`AML_Core_Code/train_cv_cifar_18.py`) | Executor with fused cuDNN Tensor Cores. |
| **Static Subnet Compiler** (`AML_Core_Code/extract_static_model.py`) | Hard-prunes the supernet into a standalone `StaticChildModel` — pure `nn.Module`, zero Gumbel overhead at inference. |
| **C-infinity smooth efficiency barrier** (`mgn/constraints.py`) | Smooth surrogate for accuracy-per-parameter that keeps meta-gradients bounded. |

Stage hierarchy of the hybrid supernet:

| Stage | Layers | Resolution | Dominant mode |
| :-- | :-- | :-- | :-- |
| 1 | 0–5 | 32 × 32 | ResNet + Plain CNN (ViT suppressed to avoid downsampling blur) |
| 2 | 6–11 | 16 × 16 | ResNet + Plain CNN |
| 3 | 12–17 | 8 × 8 | ViT + ResNet, token self-attention |

Classifier head: 4 × 4 spatial pooling (768-d) → `LayerNorm` → `Dropout(0.1)`.

### Engine 2 — Continual Lifelong Hypernetwork

`hts_lib/` · `train_continual_aml.py`

| Component | What it is |
| :-- | :-- |
| **Zero-Replay Task Isolation** (`ProtoIsolator`) | Partitions parameter space across tasks with **no** exemplar buffer (`--replay-per-class 0`). |
| **Full-State Supernet Snapshots** | Captures `W_stem`, `W_layers[1..18]`, `W_ln`, `W_protos` for zero-forgetting test-time retrieval. **7.49 MB per task** at the default `--norm in`. |
| **Fisher EWC** | Two accumulators over the shared trunk — A = Σ F⁽ᵗ⁾ and B = Σ F⁽ᵗ⁾θ★⁽ᵗ⁾ — so the exact multi-task quadratic is kept in O(1) task count instead of O(T) snapshots. Disabled by default (`--ewc-lambda 0`); see §3.4. |
| **TaskCentroidRouter** | Zero-replay domain routing via normalized feature centroids μₜ ∈ ℝᴰ. |
| **Academic Evaluator** (`hts_lib.evaluation.ContinualEvaluator`) | Accuracy matrices `A_oracle` / `A_auto` / `A_live`, backward transfer (BWT), forgetting. |

---

## 3. Results

### 3.1 Zero-replay continual learning

**MNIST → CIFAR-10 → Fashion-MNIST**, no replay exemplars.

```bash
python train_continual_aml.py --model-type hybrid18 \
  --tasks mnist,cifar10,fashion --epochs 5,28,16 \
  --replay-per-class 0 --router-type centroid --ewc-lambda 100
```

Measured on **NVIDIA GeForce RTX 3050 Laptop GPU**, `bfloat16`, `--replay-per-class 0`.

### 3.1a Full production run (2026-10-14)

`--epochs 5,28,16 --ewc-lambda 100`, seed 42, InstanceNorm.

**`A_oracle` — task-oracle snapshot retrieval**

| after \ eval on | MNIST | CIFAR-10 | Fashion |
| :-- | --: | --: | --: |
| after MNIST | **97.94 %** | — | — |
| after CIFAR-10 | **97.94 %** | **67.91 %** | — |
| after Fashion | **97.94 %** | **67.91 %** | **92.11 %** |

**BWT = +0.00 % · Forgetting = 0.00 %** — MNIST is bit-stable at 97.94 % through
both later tasks.

**`A_live` — standard mutable weights**

| after \ eval on | MNIST | CIFAR-10 | Fashion |
| :-- | --: | --: | --: |
| after MNIST | 98.09 % | — | — |
| after CIFAR-10 | 28.00 % | 68.03 % | — |
| after Fashion | 16.23 % | 31.43 % | 92.09 % |

**BWT = −59.23 % · Forgetting = 59.23 %**

**`A_auto` — autonomous centroid router, no task identity**

| after \ eval on | MNIST | CIFAR-10 | Fashion |
| :-- | --: | --: | --: |
| after MNIST | 97.94 % | — | — |
| after CIFAR-10 | **8.01 %** | 67.72 % | — |
| after Fashion | 26.33 % | 43.19 % | 74.63 % |

**BWT = −48.07 % · Forgetting = 48.07 %**

> **Two negative results from this run, stated plainly.**
>
> **1. EWC did not help.** `--ewc-lambda 100` gives `A_live` forgetting 59.23 %,
> *worse* than the recorded 46.50 % baseline. Earlier EWC and shared-backbone
> numbers in this README came from `--steps 50` runs and **do not hold at full
> scale** — they are kept below but are no longer evidence of anything. The
> Fisher anchor is evidently too weak against a 5 460-step CIFAR schedule.
>
> > **Caveat added 2026-10-15 — this measurement predates the EWC fix.** The
> > penalty used here paired a running sum of Fisher with a *single* `θ*` that
> > was re-pinned to the most recent task, so from task 2 on it anchored task 1's
> > Fisher to weights that had already forgotten task 1 (audit entry B30). The
> > implementation is now the exact multi-task quadratic in two accumulators,
> > checked against a brute-force reference. **So this table is evidence that
> > *that implementation* does not help, not evidence that diagonal-Fisher EWC
> > does not help.** It has not been re-measured since; anyone citing "EWC is
> > refuted at scale" is citing a result from the broken version.
>
> **2. The router is the binding constraint.** `A_auto` collapses to 8.01 % on
> MNIST after CIFAR-10 while `A_oracle` holds 97.94 % — the snapshot is fine, the
> router simply picks the wrong domain. This is now the single largest source of
> error in the system, and it is a routing-accuracy problem, not a storage one.

#### Historical `A_oracle` (pre-2026-10-14, `--norm bn`, `--epochs 3,10,6`)

| after \ eval on | MNIST | CIFAR-10 | Fashion |
| :-- | --: | --: | --: |
| after MNIST | **92.83 %** | — | — |
| after CIFAR-10 | **92.83 %** | **45.96 %** | — |
| after Fashion | **92.83 %** | **45.96 %** | **87.58 %** |

**BWT = +0.00 %  ·  Forgetting = 0.00 %**

> **Key finding.** By preserving the complete functional state — `W_stem`,
> `W_layers[1..18]`, `W_protos` — across tasks, the framework eliminates
> catastrophic forgetting entirely (**zero-forgetting**, BWT = 0.00 %) without
> storing a single replay exemplar image.

#### Historical `A_live` (same older configuration)

| after \ eval on | MNIST | CIFAR-10 | Fashion |
| :-- | --: | --: | --: |
| after MNIST | 72.54 % | — | — |
| after Fashion | 8.89 % | 13.10 % | 75.93 % |

**BWT = −46.50 %  ·  Forgetting = 46.50 %**

Standard unconstrained catastrophic degradation without snapshot protection.

### 3.2 Normalization — the default was the accuracy ceiling

`--norm` now defaults to **`in` (InstanceNorm)**. Previously it defaulted to
**BatchNorm**, which was silently capping every accuracy number in this file.

Controlled A/B — MNIST, 5 epochs, seed 42, `--arch-lr 3e-3 --depth-cost 0.02`,
identical configuration, only the normalization differs:

| `--norm` | per-epoch accuracy | A[1,1] | s/epoch | curve shape |
| :-- | :-- | --: | --: | :-- |
| `bn` | 88.86 → 84.40 → 84.75 → 83.93 → 87.09 | **84.75 %** | 100 | peaks, then **decays** |
| `gn1` | 85.28 → 93.06 → 94.57 → 95.87 → 96.38 | **96.26 %** | 116 | monotonic |
| `in` | 92.09 → 95.85 → 97.21 → 97.77 → 97.91 | **98.03 %** | 108 | monotonic |

InstanceNorm wins on accuracy, stability **and** wall-clock. Confirmed on the
full 3-task schedule: MNIST **98.09 %**, CIFAR-10 **68.03 %** (BatchNorm peaked at
40 % then slid to 21 %).

**Why BatchNorm fails specifically here.** This supernet holds three candidate
branches per layer, each with its own BN — 54 BN modules and 108 running values
across the 18 layers. Training is batch-consistent: **one branch per layer per
batch**, so a branch's `running_mean` / `running_var` updates *only while that
branch is routed to*. Raising `--arch-lr` from `2e-4` to `3e-3` was necessary to
unblock architecture search, but it also makes weights drift faster than the
stored statistics describe them. The old `2e-4` default kept weights nearly
frozen, which is exactly why the historic 92.83 % was reproducible — by being slow.

Decisive test on the shipped MNIST checkpoint, same weights both times:

```text
eval()  — BN uses RUNNING stats :  73.81 %
train() — BN uses BATCH stats   :  97.04 %
```

A 23-point gap. The network was never the problem; the inference-time statistics
were. BatchNorm also **failed to reproduce**: two runs of the same config with the
same seed gave 72.75 % and 87.09 %, with live and routed accuracy swapping order.

**Guarantee vs measurement.** What is *guaranteed* by construction is that IN/GN
cannot suffer stale statistics — for a fixed input, `train()` and `eval()` return
bit-identical output (`max |Δ| = 0.000e+00`, vs `3.571e+00` for BN). What is *not*
guaranteed is that they are optimal here; that remains empirical.

**`--norm in` never actually worked** until this change: `cnn_block.py` branched on
`if norm_type == "bn": ... else: GroupNorm` and the trainer mapped any non-`bn`
value to `"gn"`, so `--norm in` silently built GroupNorm while `choices` advertised
three options. Norm construction is now a `make_norm()` factory that rejects
unknown values.

### 3.3 Dynamic depth, per-sample routing and stability

| Mechanism | What it does | Guard against regressing |
| :-- | :-- | :-- |
| **P2 depth short-circuit** | A closed gate branches *around* the block at inference instead of computing it and multiplying by 0 | exact-equality test vs a full-compute reference, incl. per-sample skips |
| **P3 per-sample MoE dispatch** | Runs each candidate block only on the sub-batch that selected it | numerical equivalence for one-hot routing + non-zero architecture gradient test |
| **P4 Fisher EWC** | Two diagonal-Fisher accumulators over the shared trunk, so the exact multi-task penalty is O(1) in task count instead of a 7.49 MB snapshot per task | gradient checked against a brute-force one-copy-per-task reference, and the minimum must be the Fisher-weighted mean of every task's optimum, not the last one |

```bash
python -m unittest tests.test_dynamic_depth_routing    # CI-enforced regression guards
```

Measured wall-clock effect of the depth short-circuit (RTX 3050, batch 64, bf16),
verified bit-identical against the old always-evaluate loop:

| layers closed | executed | latency | speed-up |
| --: | --: | --: | --: |
| 0 | 18 | 276.3 ms | 1.00× |
| 8 | 10 | 235.8 ms | 1.17× |
| 12 | 6 | 200.6 ms | 1.38× |
| 17 | 1 | 40.4 ms | **6.83×** |

The curve is non-linear because early high-resolution layers dominate wall-clock.
Per-sample skipping alone is nearly worthless (1.04×), since the block still runs
for the open half of the batch.

#### Structure fingerprints — read these carefully

Reported per-stage block-mode probabilities:

| Stage | ResNet | Plain CNN | ViT |
| :-- | --: | --: | --: |
| 1 (layers 0–5) | 81.0 % | 13.0 % | suppressed |
| 2 (layers 6–11) | 80.5 % | 14.5 % | suppressed |
| 3 (layers 12–17) | 18.0 % | — | 72.1 % |

> **A 2026-10-14 audit showed these are the initialisation, not a learned
> outcome.** Softmax of the hard-coded prior is 18.1 / **81.3** / 0.5 % for stage 1
> and 2.4 / 17.8 / **79.8 %** for stage 3; re-measuring on the three shipped
> checkpoints gave 18.4 / **81.1** / 0.6 % and 2.4 / 17.7 / **79.9 %**. Training
> moved the distribution by **less than half a percentage point**. The 72.1 % above
> is the nearest value in the literature-facing tables — it is not evidence of
> discovered structure.
>
> The depth gates were worse: `layer_soft` sat in `[0.876, 0.882]` on every row of
> every checkpoint — **0 of 216** gates ever closed. Root cause was the optimiser
> step budget: AdamW moves a parameter by ≈ `lr` per step, and the architecture LR
> of `2e-4` allowed ≈ 0.12 of travel per task against the ≈ 2.0 needed to flip a
> gate. With `--arch-lr 3e-3` a gate needs `2.0 / arch_lr` architecture steps.

Full investigation log with every measurement:
[`experiments/dynamic_depth_routing_2026-10-14.md`](experiments/dynamic_depth_routing_2026-10-14.md)

#### Shared backbone (P4-3A) — short-schedule result, not confirmed

> **Superseded.** The figures below come from a `--steps 50` schedule (CIFAR-10
> saw a quarter of one epoch). The full run above shows EWC failing at scale, so
> these numbers should not be cited. Kept only to document what was tried.

`--shared-backbone 1` stores no per-task copy of the trunk and leaves EWC solely
responsible for retention. Same schedule, same seed, only that flag differs:

| | per-task snapshot | shared backbone |
| :-- | --: | --: |
| per-task state | 7.49 MB | **≈ 1.2 MB** |
| `A_live` forgetting | 23.05 % | **13.91 %** |
| `A_auto` forgetting | 9.84 % | 9.23 % |
| `A_oracle` forgetting | **0.00 %** | 9.00 % |

It **halves** live forgetting (MNIST after CIFAR-10: 37.28 % → 65.81 %) and cuts
per-task storage 6×, but it **cannot** keep oracle zero-forgetting — with one
shared trunk there is no second trunk to retrieve. Enable it when deployment does
not supply a task id. These are single-seed numbers at a short schedule; a
multi-seed comparison is needed before treating the gain as real.

#### Corrections to the original design proposals

1. The proposed per-sample dispatch **zeroes the architecture gradient**
   (`2.50e0 → 0.00e0`) because `out[mask] = branch(x[mask])` removes
   `mode_weights` from the autograd graph. It is now eval-only.
2. `VisionTransformerBlock.compute_flops` overstated ViT cost by **76×** — it used
   `h*w` tokens while `forward` pools to 64. ViT is the cheapest mode at 32×32 and
   16×16, the opposite of the "Conv is cheaper when shallow" premise.
3. A zero-valued `block_mode_prototypes` does **not** give a 33/33/33 split.
   Inference takes `argmax`, ties resolve to index 0, and all 18 layers collapse to
   PlainConv. Measured cost of the noise-based neutral prior: **89.9 % → 11.4 %**
   on MNIST at a 4×60-step budget.

### 3.4 Dual-engine benchmark — AML vs HtS

```bash
python benchmarks/benchmark_cl_aml_vs_hts.py --epochs "1,1,1" --steps 30 --eval-cap 15
```

Zero-replay, RTX 3050 Laptop GPU:

| Engine / Method | Final ACC | BWT | Forgetting | Runtime | Parameters | Model size |
| :-- | --: | --: | --: | --: | --: | --: |
| **AML (live weights, no replay)** | **29.73 %** | **+0.00 %** | **0.00 %** | 183.4 s | 1,871,385 | 7.49 MB |
| **AML (snapshot retrieval)** | **33.34 %** | **+0.00 %** | **0.00 %** | 183.4 s | 1,871,385 | 7.49 MB |
| **HtS (Tensor Cores WMMA BF16)** | **36.98 %** | −33.15 % | 33.15 % | **24.2 s** | **1,468,790** | **5.8 MB** |

- **AML** achieves zero forgetting through orthogonal prototype isolation (`pi_mask`).
- **HtS** runs **7.6× faster** with **8.7× fewer parameters** using hardware WMMA BF16 tensor cores.

### 3.5 Baseline comparison — AML vs standard architectures

```bash
python benchmarks/benchmark_cl_comparison.py
```

Strict zero-replay, MNIST → CIFAR-10 → Fashion-MNIST:

| Architecture / Model | Continual protocol | Final ACC | BWT | Forgetting | Parameters | Memory |
| :-- | :-- | --: | --: | --: | --: | --: |
| Plain CNN (18-layer) | sequential fine-tuning | 31.56 % | −54.66 % | **54.66 %** | 385,594 | 1.47 MB |
| Standard ResNet-18 | sequential fine-tuning | 34.54 % | −55.07 % | **55.07 %** | 348,730 | 1.33 MB |
| Vision Transformer (ViT) | sequential fine-tuning | 25.36 % | −14.61 % | **14.61 %** | 232,234 | 0.89 MB |
| AML live weights | mutable supernet, no snapshot | 29.73 % | -46.50 % | **46.50 %** | 1,871,385 | 7.49 MB |
| **AML 18-layer hybrid (ours)** | **snapshot isolation (zero-replay)** | **75.46 %** | **+0.00 %** | **0.00 %** | 1,871,385 | **+7.5 MB / task** |

Stage-by-stage evolution, `A[i,j]` after each sequential stage:

| Architecture / Model | after MNIST | after CIFAR-10 | after Fashion |
| :-- | :-- | :-- | :-- |
| Plain CNN (18-layer) | 94.65 % | 7.69 % \| 31.51 % | 8.51 % \| 8.32 % \| 77.85 % |
| Standard ResNet-18 | 97.96 % | 6.42 % \| 38.94 % | 10.81 % \| 15.95 % \| 76.86 % |
| Vision Transformer (ViT) | 39.17 % | 12.21 % \| 19.31 % | 17.79 % \| 11.48 % \| 46.82 % |
| AML live weights | 72.54 % | 41.41 % \| 42.44 % | 8.89 % \| 13.10 % \| 75.93 % |
| **AML 18-layer hybrid (ours)** | **92.83 %** | **92.83 % \| 45.96 %** | **92.83 % \| 45.96 % \| 87.58 %** |

- Plain CNN and ResNet-18 erase earlier domains almost entirely (MNIST collapses
  from 94.6 % / 97.9 % down to ≈ 8.5 % / 10.8 %).
- AML circumvents representation erasure via orthogonal prototype isolation and
  INT8 snapshot retrieval, maintaining pristine performance across all tasks.

> **Parameter-count correction (2026-10-14).** These tables previously credited the
> hybrid18 model with **12,759,873** parameters. That figure belongs to the legacy
> `ConvAML_CIFAR_TrueDynamic` (`train.py`), which materialises every convolutional
> kernel inside `proto_weights` and is therefore **6.82×** larger.
> `ConvAML_18Layer_Hybrid` — the model every number in this file measures — has
> **1,871,385** parameters (7.49 MB fp32) at the default `--norm in`. Its deployed
> cost in snapshot mode is `+7.5 MB` per task.
>
> The count depends on `--norm`, because each norm type carries a different number
> of affine parameters. Measured directly, not derived:
>
> | `--norm` | Parameters | Size |
> | :-- | --: | --: |
> | `bn` (BatchNorm) | 1,874,073 | 7.50 MB |
> | **`in` (InstanceNorm, default)** | **1,871,385** | **7.49 MB** |
> | `gn1` (GroupNorm) | 1,876,761 | 7.51 MB |
>
> The earlier `1,874,073` in this file was the BatchNorm figure, measured before
> `--norm` was made to reach the vision encoder and the stem — see B21 in
> `experiments/dynamic_depth_routing_2026-10-14.md`.

---

## 4. Repository layout

```text
├── train.py                    # Live 18-layer ConvAML_CIFAR_TrueDynamic trainer (bi-level DARTS loop)
├── train_continual_aml.py      # 3-task continual pipeline (ProtoIsolator + int8 snapshots + EWC)
├── compare_architectures.py    # Static-vs-dynamic architecture comparison
├── quickstart_aml.py           # End-to-end demo: supernet -> extraction -> HtS
├── quickstart_hts.py           # Minimal HtS-only demo
├── validate_ext_in.py          # Numeric parity checks for the fused CUDA kernels
├── build_ext.py                # Rebuilds both CUDA extensions (_C, _Cin) from source
├── verify_env.py               # GPU / compiler / extension diagnostic
├── pyproject.toml              # pip install -e . ; [tool.ruff] lint config
│
├── mgn/                        # AML core library
│   ├── blocks/                 #   hybrid_18.py, hybrid_router.py, cnn_block.py, vit_block.py
│   ├── layout.py               #   CANONICAL flat prototype-weight layout (single source of truth)
│   ├── generator.py            #   ModelGenerator & PrototypeLibrary (reference impl)
│   ├── executor.py             #   TaskNetwork executor (reference impl)
│   ├── constraints.py          #   C-infinity efficiency barrier
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
│   ├── train_cv_cifar_18.py    #   AcceleratedCVNetwork executor
│   └── extract_static_model.py #   StaticChildModel extraction engine
│
├── experiments/                # Benchmark scripts + recorded JSON results
│   ├── dynamic_depth_routing_2026-10-14.md   # Full audit + measurement log
│   └── benchmark_hts_baselines.py             # HtS continual-learning baselines
├── benchmarks/                 # Standalone comparison + profiling scripts
├── probes/                     # Ad-hoc causal probes used during development
├── tests/                      # Unit/integration suite (CPU-safe, incl. train() smoke)
├── paper_and_documents/        # LaTeX paper source + compiled PDF
└── data/                       # Datasets (gitignored; auto-downloaded by torchvision)
```

## 5. Quickstart

```bash
# 1. Install in editable mode
pip install -e .

# 2. Run unit tests (CPU-only, ~30s)
python -m unittest discover tests

# 3. Run the full unified demo (supernet -> static extraction -> continual hypernetwork)
python quickstart_aml.py
```

Everything above works on CPU out of the box. CI runs ruff (`F` + `E9`), the
CPU-safe unit suite (including an end-to-end `train.main()` smoke run), and both
quickstarts on Python 3.10 / 3.12; the CUDA parity check runs only on GPU runners.

```bash
pip install "ruff>=0.5"
ruff check .                      # config: [tool.ruff] in pyproject.toml
```

The fused CUDA kernels build automatically on CUDA machines and drive training
there — `python validate_ext_in.py` certifies kernel ↔ eager parity after any rebuild:

```bash
# Windows: run from an MSVC Developer Command Prompt (needs cl.exe + nvcc):
#   "C:\Program Files (x86)\Microsoft Visual Studio\...\vcvars64.bat"
python build_ext.py          # builds both _C and _Cin into mgn/accel/

python validate_ext_in.py    # verify kernel numeric parity (CUDA required)
```

## 6. Native CUDA acceleration

The engine ships a fully native CUDA suite (`mgn/csrc/`):

- **Direct Dynamic Conv2D (BF16 & FP32)** — per-sample dynamic weights of shape
  `B × C_out × C_in × 3 × 3` with no intermediate `im2col` tensor bloat; unrolled
  3 × 3 direct convolution with native input, weight and bias backward kernels.
- **Weightless GroupNorm & InstanceNorm** — multi-warp reductions using warp
  shuffle primitives (`__shfl_down_sync`) and dynamic shared memory
  (`extern __shared__ float s_smem[]`).
- **Fused Pipeline Mix v2** — deterministic binary-tree reductions (`mb_tree_sum`)
  eliminating `atomicAdd` races for bit-level determinism.
- **Hardware Tensor Core GEMM** — real WMMA 16 × 16 × 16 matrix multiplication for
  Ampere SM86+ GPUs in FP16 and BF16, with staged shared-memory boundary padding.

### Microbenchmark vs PyTorch cuDNN / ATen / cuBLAS

RTX 3050 Laptop GPU, `torch.cuda.Event`, 50 iterations:

| Component | Operation | Baseline | Native CUDA | Speedup |
| :-- | :-- | --: | --: | --: |
| Dynamic Conv2D (4, 16, 32, 16, 16) | backward | 0.846 ms (cuDNN loop) | **0.118 ms** | **7.17×** |
| Dynamic Conv2D (8, 32, 32, 16, 16) | backward | 2.100 ms (cuDNN loop) | **0.346 ms** | **6.07×** |
| Dynamic Conv2D (16, 32, 64, 8, 8) | backward | 2.538 ms (cuDNN loop) | **0.363 ms** | **6.99×** |
| GroupNorm (8, 32, 16, 16) | backward | 0.341 ms (ATen) | **0.046 ms** | **7.38×** |
| GroupNorm (32, 64, 8, 8) | backward | 0.280 ms (ATen) | **0.023 ms** | **12.01×** |
| Batched GEMM (32, 128, 128, 128) | forward (A·B + b) | 0.413 ms (tiled SMEM) | **0.215 ms (tensor core)** | **1.92×** |

### Environment diagnostics

```bash
python verify_env.py                              # Python, PyTorch, GPU, NVCC, CUDA smoke tests
python benchmarks/benchmark_cuda_vs_baseline.py   # full microbenchmark suite vs ATen/cuDNN
python benchmarks/profile_training_step.py        # PyTorch profiler on an end-to-end step
pytest                                            # full suite (58 tests: 50 in tests/, 8 in mgn/tests/)
python -m unittest discover tests                 # CPU-safe tests in tests/
```

## 7. Build, installation & training

```bash
# Build CUDA extensions - either
pip install -e .
python build_ext.py
```

> **Verified toolchain (Windows):** Python 3.12, `torch 2.5.1+cu124`,
> `nvcc 13.3` (CUDA 13.3 toolkit), MSVC C/C++ from Visual Studio Build Tools 18.

```bash
python train.py                       # legacy ConvAML_CIFAR_TrueDynamic on CIFAR-10 (12.76M params)
python train.py --dataset mnist       # MNIST / Fashion
python train.py --steps 4 --epochs 1  # capped sanity run

python train_continual_aml.py         # 3-domain continual lifelong benchmark
python benchmarks/benchmark_cl_aml_vs_hts.py   # AML vs HtS side-by-side
```

Earlier benchmark phases (Phase 1 MLP/PCA, Phase 2 ConvAML, Phase 3 pilot runs)
are documented in [`experiments/README.md`](experiments/README.md).

## 8. License

MIT — see [LICENSE](LICENSE).