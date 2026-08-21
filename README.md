# 🌟 Architecture Mode Learning (AML) Unified Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **Architecture Mode Learning (AML)** — a meta-learning paradigm for dynamic continuous architecture synthesis, efficient subnetwork generation, and multi-domain continual learning.

---

## 🏛️ 1. Framework Architecture

The AML framework unifies two complementary engines:

```text
                                       Input Image / Feature x
                                                  │
                ┌─────────────────────────────────┴─────────────────────────────────┐
                ▼                                                                   ▼
    [1] AML Meta-Supernet Engine                                    [2] HtS Continual Hypernetwork
    (ModelGenerator G_phi -> TaskNetwork T_W)                       (Multi-Domain Lifelong Learning)
                │                                                                   │
    ├── PrototypeLibrary (K=32 Modes)                               ├── Dual-Residual Basis (K=8 Experts)
    ├── Gumbel-Softmax Continuous Routing                           ├── B_shared Base + sum(alpha_i * B_priv_i)
    │   - Dynamic Depth Gates L(x)                                  ├── Layer-Aware Diversity & Orthogonality
    │   - Dynamic Channel Width Masks D(x)                          └── Zero Catastrophic Forgetting
    │   - Dynamic Block Types (Plain / ResNet / ViT)                                │
    └── Hard Pruning -> StaticChildModel Extraction                                 └── Synthesized Child CNN (<10k params)
```

---

## 🚀 2. Core AML Subsystems

### 🔹 Engine 1: Model Generator Supernet (`mgn/` & `AML_Core_Code/`)
- **Meta-Generator (`mgn.generator.ModelGenerator`)**: Dynamically samples computational topologies and weights conditioned on input properties.
- **TaskNetwork Executor (`mgn.executor.TaskNetwork`)**: Executes dynamically masked and routed graphs with per-sample group convolutions.
- **Static Subnet Compiler (`AML_Core_Code/extract_static_model.py`)**: Hard-prunes the dynamic supernet into a standalone `StaticChildModel` (pure `nn.Sequential` with zero runtime Gumbel overhead).
- **$C^\infty$ Smooth Efficiency Barrier (`mgn.constraints.compute_aml_efficiency`)**: Mathematical objective $\mathcal{E}_{\text{AML}}$ guaranteeing non-exploding meta-gradients.

### 🔹 Engine 2: Continual Lifelong Hypernetwork (`hts_lib/`)
- **Dual-Residual Basis (`hts_lib.models.HtSDualResidualBasis`)**: Decouples invariant core knowledge ($\mathbf{B}_{\text{shared}}$) from sample-adaptive residual modulations ($\mathbf{B}_{\text{private}}$).
- **Task-Incremental Lifelong Learning**: Evaluated on 3-domain sequence ($\text{MNIST} \to \text{CIFAR-10} \to \text{Fashion-MNIST}$) with Positive Backward Transfer ($\text{BWT} = +1.95\%$) and empirical zero forgetting ($F = 0.06\%$).
- **Episodic Memory Buffer (`hts_lib.data.EpisodicMemoryBuffer`)**: Formal memory-budgeted reservoir sampling and rebalancing for constrained continual learning.

---

## 📊 3. Academic Continual Learning Benchmark

### Accuracy Matrix $A_{i, j}$ (Accuracy on Task $j$ after Training Stage $i$):

| Training Stage ($i$) | Evaluated on MNIST ($j=1$) | Evaluated on CIFAR-10 ($j=2$) | Evaluated on Fashion-MNIST ($j=3$) |
| :--- | :---: | :---: | :---: |
| **After Stage 1 (MNIST)** | **$99.53\%$** | — | — |
| **After Stage 2 (CIFAR-10)** | **$99.35\%$** | **$68.71\%$** | — |
| **After Stage 3 (Fashion-MNIST)** | **$99.41\%$** | **$72.73\%$** 🚀 | **$90.51\%$** ⚡ |

---

## 📦 4. Repository Structure

```text
AustraliaSilver/AML (main branch)
├── 📦 mgn/                    # Model Generator Network (AML Core Supernet)
│   ├── generator.py           # ModelGenerator & PrototypeLibrary
│   ├── executor.py            # Dynamic TaskNetwork executor
│   ├── constraints.py         # C^infinity efficiency metric E_AML & soft barrier
│   ├── export.py              # Supernet to static model compiler
│   └── losses.py              # Meta-loss & structural regularization
│
├── 📦 AML_Core_Code/          # AML Vision pipelines & Static Model Extractor
│   ├── train_cv_cifar.py      # ConvAML CIFAR Supernet training
│   └── extract_static_model.py# StaticChildModel extraction engine
│
├── 📦 hts_lib/                # Continual Learning Hypernetwork Engine
│   ├── models/                # Dual-Residual Basis, Dynamic Blocks, Losses
│   ├── data/                  # Loaders & EpisodicMemoryBuffer
│   ├── training/              # ContinualTrainer (Optimizer preservation)
│   └── evaluation/            # Academic Evaluator (A[i,j], BWT, Forgetting)
│
├── 🧪 tests/                  # Unit and integration test suite
├── 🚀 quickstart_aml.py       # End-to-end demo of Supernet & Subnet extraction
├── 📊 benchmark_baselines.py  # Standard baseline comparisons
├── 📄 setup.py                # Unified package installer (pip install -e .)
└── 📁 paper_and_documents/    # Research paper, LaTeX sources, and proofs
```

---

## ⚡ 5. Quickstart & Verification

```bash
# 1. Install package in editable mode
pip install -e .

# 2. Run unit tests
python -m unittest discover tests

# 3. Run full AML unified demo (Supernet generation + Static extraction + Continual learning)
python quickstart_aml.py
```
