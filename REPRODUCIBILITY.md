# Reproducibility Guide: Zero-Replay Continual Learning on 18-Layer Hybrid AML Supernet

This document provides exact, verified reproduction steps for the **18-Layer Hybrid Supernet (CNN + ResNet + ViT)** and its **Zero-Replay Continual Learning Benchmark** on MNIST → CIFAR-10 → Fashion-MNIST.

---

## 1. System Requirements & Hardware

- **GPU**: NVIDIA GPU with ≥ 4 GB VRAM (Tested on NVIDIA GeForce RTX 3050 Laptop GPU, 6GB VRAM, compute capability 8.6).
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

# 2. Run unit & integration test suite (~1 min runtime)
pytest tests/test_hybrid_18_advanced.py tests/test_hybrid_18.py
python -m unittest discover tests
```

Expected result: everything passes cleanly — `10 passed in ~7s` for the first
command, `Ran 279 tests in ~62s` / `OK` for the second (the exact counts grow as
tests are added; zero failures is the contract). GPU-only tests skip on a
CPU-only machine.

---

## 3. Continual Learning Benchmarks (Zero-Replay)

### Option A: Fast Verification Schedule (~20 - 25 minutes)

Verifies the mathematical Zero-Forgetting property (BWT = 0.00 %) across all 3 tasks with the 18-layer hybrid architecture:

```bash
python train_continual_aml.py \
  --model-type hybrid18 \
  --tasks mnist,cifar10,fashion \
  --epochs 3,10,6 \
  --replay-per-class 0 \
  --router-type centroid
```

#### Expected Benchmark Outputs:

- **Accuracy Matrix `A_oracle` (Task-IL Snapshot Retrieval)**:
  ```text
  after \ eval on |    mnist |  cifar10 |  fashion
              mnist |    92.83 |        — |        —
            cifar10 |    92.83 |    45.96 |        —
            fashion |    92.83 |    45.96 |    87.58

  [oracle]     BWT: +0.00% | Forgetting: 0.00%
  ```

- **Accuracy Matrix `A_live` (Standard Mutable Weights)**:
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
| `--auto-live-thresh` | `0.0` | Open-set fallback for the autonomous protocol: evaluate with live weights when no snapshot reaches this confidence. **Disabled by default** — correctness is verified (it cannot alter `A_oracle`) but an accuracy benefit has not been demonstrated. See the investigation log. |
| `--shared-backbone` | `0` | P4-3A: store no per-task copy of stem / 18 layers / ln (5.86 MB/task) and let EWC alone retain earlier domains. **Superseded.** The A/B below ran at `--steps 50`, where CIFAR-10 saw a quarter of one epoch; at the production schedule EWC does not hold the shared trunk at all and live forgetting stays at **59.23 %** (B19). Per-task state still drops to ~1.34 MB at 10 classes (the sum of the surviving keys in the B64 breakdown: `enc` 0.401 + `sel` 0.805 + `protos` 0.092 + `struct_heads` 0.045 MB), but the retention the flag was supposed to buy is not there. Do not cite the 23.05 % → 13.91 % pair. |
| `--grad-checkpoint` | `off` | Recompute each of the 18 blocks' activations in backward instead of keeping them (`torch.utils.checkpoint`, non-reentrant). **Bitwise identical** to the ordinary path — verified for one forward+backward (CPU and the bf16 production path) and for three chained training steps from a single seed, with losses and final weights both `torch.equal` — so it is a pure speed/memory trade and no result depends on it. It costs ~17 % of step time where the step already fits, and turns a step that does not fit into one that does: 64×64 batch 256 on a 6 GiB card goes 10526 ms / 9737 MiB (spilling) → **1858 ms / 3594 MiB** (5.7×), while 32×32 batch 256 goes 448 → 524 ms with peak 2706 → 982 MiB. `on` forces it; `auto` runs the first step normally and switches it on if the peak exceeds 85 % of the card. The case that matters is Tiny ImageNet (64×64): **batch 128 fits as it stands (4893 MiB), batch 256 needs the flag.** See B69g. |
| `--width-mask-grad` | `1` | Whether the channel **width** mask carries a gradient; `0` detaches `dim_masks` (`mgn/blocks/hybrid_18.py`). Named with the "width" prefix to keep it distinct from `TrainIsolator.mask_grad()`, the unrelated no-op marker that gates the prototype selector's gradient via backward hooks — they are different mechanisms and previously shared a name. Measured on the production step (32×32, batch 256, bf16, K=12/max_dim=48/norm=in): the shipped mask costs **+57.4 ms and +536 MiB** per step, and detaching removes only the architecture gradient through the mask multiply — **~23 ms and ~29 MiB**. The remaining ~508 MiB is autograd retaining activations for the *weight* gradient, which `--grad-checkpoint` does remove (masked vs unmasked checkpointed steps differ by **1 MiB**). So `0` is not a memory fix, and width routing then learns only from the `--depth-cost` term, which does change the chosen topology. Default `1` leaves every published number untouched. |
| `--input-norm` | `per-task` | How **INPUT** images are normalised — distinct from `--norm`, which is the post-conv layer norm. `per-task` applies each domain's own mean/std from `mgn.datasets.TASK_SPECS`; that needs the task identity, which the oracle and live protocols have and the autonomous one does not — yet `A_auto` was produced with it, because the loader is built per task (closed by B71 — see below). `A_auto` measured under `per-task` normalisation is therefore an **upper bound**: the task-agnostic `pooled` mode measures 40.46 % vs 66.05 % on the production schedule (`mnist,cifar10,fashion` epochs 5,28,16), while the oracle is unchanged (85.90 → 86.61 %), confirming the router — not the model — is what suffers under task-agnostic input normalisation. `pooled` = one constant for the whole sequence, weighted by training-set size (something a deployer can compute from the unlabelled pile); `imagewise` = each image's own per-channel statistics; `none` = raw `[0,1]`. Any value other than `per-task` changes every number in every column, so it is opt-in and gets its own row, and it is refused together with `--replay-per-class > 0` (replay exemplars are stored with the per-task constants, so the two streams would be normalised differently). See B71. |

### 3A decision

`--shared-backbone 1` is **not** a strict improvement. It buys 6× less per-task
storage by giving up the task-oracle zero-forgetting claim, which is unavoidable
when there is only one trunk to retrieve.

**Correction (B19).** An earlier version of this section also claimed it buys
lower average error, on the strength of a `--steps 50` A/B. That does not survive
the production schedule: EWC is not holding the trunk, so the shared-backbone
configuration is *worse* on retention, not better. The honest statement of what
3A offers is "6× less per-task storage at the cost of the zero-forgetting claim",
with no retention benefit. Enable it when deployment does not supply a task id;
keep the default otherwise.

Depth telemetry is printed every epoch:

```text
[mnist] ep 3/3 | 264.7s | loss 1.31 | mnist acc 87.31% | [Plain: 12%, ResNet: 59%, ViT: 29%] | [active L: 15.20/18, 331.5 MFLOP/s]
```

`active L` is the mean number of open gates per sample; `MFLOP/s` is the
differentiably expected cost of the selected structure.

#### Regression tests

```bash
python -m pytest tests -q                                  # everything, CPU-safe
python -m pytest tests/test_dynamic_depth_routing.py -q    # depth / P2 / P3
python -m pytest tests/test_cost_and_autoprotocol.py -q     # cost model + protocol
python validate_ext_in.py                                  # CUDA kernels (needs a GPU)
```

These encode the specific failures observed: uninitialised memory leaking into
per-sample-skipped rows, dtype mismatch under autocast, the MoE dispatch zeroing
the architecture gradient, and EWC's Fisher being reset instead of accumulated.

> The one-off diagnostic scripts that produced those measurements lived in
> `scratch/`, which is git-ignored and therefore **not** shipped. Where a check
> mattered it was rewritten as one of the tests above; the rest was a throwaway
> and is not claimed as reproducible.

---

### Option C: Baseline Comparison (Plain CNN vs ResNet-18 vs ViT vs AML) (~3 minutes)

Directly replicates the side-by-side continual learning benchmark under identical zero-replay conditions:

```bash
python benchmarks/benchmark_cl_comparison.py
```

Expected output confirms severe memory erasure in standard static networks vs AML preservation:

| Architecture / Model | After Stage 1 (MNIST) | After Stage 2 (CIFAR-10) | After Stage 3 (Fashion-MNIST) | Final (`A_avg`) | Forgetting |
| :--- | :---: | :---: | :---: | :---: | :---: |
| | **Eval: MNIST** | **Eval: MNIST \| CIFAR-10** | **Eval: MNIST \| CIFAR-10 \| Fashion** | | |
| **Plain CNN (18-Layer)** | 94.65% | 7.69% \| 31.51% | 8.51% \| 8.32% \| 77.85% | 31.56% | 54.66% |
| **Standard ResNet-18** | 97.96% | 6.42% \| 38.94% | 10.81% \| 15.95% \| 76.86% | 34.54% | 55.07% |
| **Vision Transformer (ViT)** | 39.17% | 12.21% \| 19.31% | 17.79% \| 11.48% \| 46.82% | 25.36% | 14.61% |
| **AML Live Weights** | 72.54% | 41.41% \| 42.44% | 8.89% \| 13.10% \| 75.93% | 32.64% | 46.50% |
| **AML 18-Layer Hybrid (Ours)** | **92.83%** | **92.83% \| 45.96%** | **92.83% \| 45.96% \| 87.58%** | **75.46%** | **0.00%** |

---

### Option E: Extended sequence — CIFAR-10 → CIFAR-100 → Tiny ImageNet

A harder sequence: three tasks with different class counts (10 / 100 / 200) and a
resolution change (32x32 → 32x32 → 64x64).

```bash
python train_continual_aml.py \
  --tasks cifar10,cifar100,tinyimagenet \
  --epochs 20,30,10 \
  --replay-per-class 0 --norm in --encoder-tail-act none \
  --arch-lr 3e-3 --depth-cost 0.02 \
  --grad-checkpoint auto \
  --data-root data
```

The supernet keeps **one shared classifier head** of width `max(n_classes)` = 200;
each task trains and is read back on its own class range `[0, n)`, and each task
keeps its native resolution (its class count and input size come from
`mgn/datasets.py`). `--data-root` may be an `os.pathsep`-separated list, so the
datasets can live in separate mounts; a Kaggle 2xT4 walkthrough is in
`notebooks/kaggle_continual_aml.ipynb`.

> **Add `--grad-checkpoint auto` on any card under ~12 GiB.** The last note in
> this section used to quote ~12.5 s/step for Tiny ImageNet, and that was the
> *spilling* figure: at 64×64 batch 256 with 200 classes the step peaks at
> **9889 MiB**, so on a 6 GiB card it either crawls or dies. Measured on an
> RTX 3050 Laptop (6 GiB), median of 9 seeded steps:

> | 64×64, batch 256, 200 classes | ms/step | peak |
> | :-- | --: | --: |
> | no checkpoint | 6697 | 9889 MiB |
> | `--grad-checkpoint on` | **1746** | **3643 MiB** |

> Checkpointing is bitwise identical (`--grad-checkpoint`'s help, verified in
> `tests/test_grad_checkpoint.py`), so adding it changes no result — it only
> makes the sequence runnable, and 3.8× faster because the step stops thrashing.
> Steady-state trainer timing including the router and `--depth-cost 0.02` is
> **1.75 s/step** at 64×64 and ~0.45 s/step at 32×32, so budget ~3.3 hours for
> `20,30,10` rather than the many hours the old figure implied.

> This sequence is **not** part of the paper's protocol and its numbers are not
> reported anywhere in this repository yet. The matrices in §3 remain the
> MNIST/CIFAR-10/Fashion run.

---

## 3Z. Two things that break a long run on a laptop

Both were hit and diagnosed during the 2026-10-16 verification run of the
production schedule. Neither is a property of the model; both would otherwise
be re-learned the expensive way.

### 3Z.1 A fixed `--seed` does NOT make a run reproducible

`train_continual_aml.py` sets `torch.backends.cudnn.benchmark = True`, so cuDNN
picks convolution algorithms **by timing them at run time**. The choice can
differ between two runs on the same machine (load, thermals, driver state),
and a different algorithm rounds differently. The seed fixes the architecture
sampling and the data order, but **not** the algorithm, so run-to-run
reproduction is not bit-exact.

This is not a theoretical caveat — it is measured. Three runs of the production
schedule, same seed, same flags; the last two at the **same** commit:

| quantity | B71 `8c16a65` | `d80c764` 09:26 | `d80c764` 11:59 | spread |
| :-- | --: | --: | --: | --: |
| A[1,1] auto | 98.49 % | 98.51 % | 98.37 % | 0.14 |
| A[2,2] auto | 67.16 % | 68.12 % | 67.51 % | **0.96** |
| A[3,2] auto | 57.95 % | *(run died)* | 47.31 % | **10.64** |
| **A_avg auto** | **66.05 %** | *(run died)* | **62.75 %** | **3.30** |
| A_avg oracle | 85.90 % | *(run died)* | 85.99 % | 0.09 |
| A_avg live | 44.14 % | *(run died)* | 48.73 % | 4.59 |

The 09:26 run was lost to §3Z.2 before reaching task 3, which is why it has no
`A_avg`.

**The middle column is the important one: it is the same commit as the right
column, so the 0.96 spread on A[2,2] has no code change to attribute it to.**
Anything smaller than that is inside the noise.

Two consequences worth stating plainly:

* **The drift is not uniform across the three columns, and that is the whole
  story.** `A_avg oracle` moves 0.09 points: training itself reproduces. The two
  columns that depend on a *routing decision* move 3.3 and 4.6 points, and
  essentially all of the `A_auto` movement is in one cell:

  | cell | B71 | `d80c764` | Δ |
  | :-- | --: | --: | --: |
  | A[3,2] `A_oracle` | 67.16 % | 67.52 % | +0.36 |
  | A[3,2] `route_acc` | 83.26 % | 63.85 % | **−19.41** |
  | A[3,2] `route_hist` | `[0, 8326, 1674]` | `[0, 6385, 3615]` | 1 941 queries flipped |

  The oracle cell is stable, so the CIFAR-10 representation is fine — what moved
  is *which snapshot gets selected*. 1 941 CIFAR-10 queries sat close enough to
  the cifar10/fashion decision boundary that a rounding difference flipped them
  to Fashion. **The router's boundary is the fragile part, not the learner.**

* **Do not verify "this change is number-neutral" by re-running training.**
  The comparison you want is already stronger and cheaper: run both forms
  **inside one process**, where cuDNN's algorithm choice is shared. That is what
  `benchmarks/capture_router_identity.py` and the tests in
  `tests/test_grad_checkpoint.py` do, and it is bitwise.
* **Cite the commit a number came from.** A published figure describes one run
  on one machine, not a protocol invariant.

**So how precisely is `A_avg auto` reproducible?** To a few points, not to two.
66.05 % is the measured value of one run and stays reported as such; a rerun
landed on 62.75 %. The `oracle ≫ auto ≫ live` ordering survives every run, and
the routing statistics in §3.1b should be read the same way — one run's
measurement, not a stable constant.

To make runs bit-reproducible instead, set `torch.backends.cudnn.benchmark =
False`, `cudnn.deterministic = True` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
and accept the slowdown. That is not the default here because the throughput
loss is real and nothing in the shipped protocol depends on bit-exact replay.

### 3Z.2 The machine sleeps and takes the run with it

A verification run died after ~88 minutes with an **empty stderr** and no exit
code. The log showed the stall directly — one step logged at `39.1 s`, the next
at `5363.0 s` — and the System event log confirmed it: `Kernel-Power 506` at
10:25:23 local (entering Modern Standby) and `507` at 11:53:28 local (exiting,
reason "Power Button"). Windows terminates background process trees on the way
into and out of Modern Standby.

An empty stderr is the tell: a Python crash or OOM writes something. Silence
plus a vanished process means something outside the process ended it.

To run anything longer than a few minutes:

* **Keep the laptop lid open.** Closing it suspends the hardware regardless of
  any power request.
* Hold a system-awake request in the *training* thread. The request is
  per-process, so a side-car guardian does not cover the trainer:

  ```python
  import ctypes
  ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
  ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
  ```

* Launch detached so a terminal or harness restart cannot orphan it. On Windows
  `Win32_Process.Create` leaves the child parented to `WmiPrvSE`, which
  survives the session. Note that `ReturnValue = 0` means the spawn was
  *accepted*, not that the process survived — check that the log is growing.

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
- **Classifier Head**: Spatial 4 × 4 adaptive pooling (768-d) → `LayerNorm(768)` → `Dropout(0.1)` → Prototype Linear Classifier.
- **Continual Learning Memory**: Full-state functional supernet snapshot (`W_stem`, `W_layers[1..18]`, `W_protos`) + `TaskCentroidRouter` unit hypersphere vector centroids μₜ ∈ ℝ⁵¹².

---

## 5. Historical Benchmarks & Pilot Logs

For earlier experimental phases (Phase 1 MLP/PCA benchmarks, Phase 2 ConvAML spatial benchmarks, and Phase 3 No-Replay pilot runs with raw JSON outputs), refer to the dedicated documentation in [`experiments/README.md`](experiments/README.md).
