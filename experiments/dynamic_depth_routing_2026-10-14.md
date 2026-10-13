# Dynamic Depth, Per-Sample Routing and Continual Storage — investigation log

**Date:** 2026-10-13/14 · **Codebase:** `7a9b920` → working tree
**Scope:** review + implementation of the P2 (depth), P3 (routing), P4 (continual)
proposals, plus the hidden bugs found while validating them.

Every number below was measured on this repository (RTX 3050 6 GB, torch 2.x,
bf16 autocast). Nothing is quoted from the original write-up unless marked
*(claimed)*.

---

## 1. Summary

| # | Finding | Impact |
|---|---------|--------|
| 1 | `layer_soft` never leaves `[0.876, 0.882]` after training; **0 of 216** row×layer gates close | The depth mechanism was inert: the P2 short-circuit had nothing to skip |
| 2 | `structure_prototypes` moves at most `0.04` from its zero init | Architecture search had no budget to act |
| 3 | Post-training mode share equals the init mode share to <0.5 pt | Routing was effectively frozen at initialisation |
| 4 | Naive MoE dispatch zeroes the architecture gradient (`2.50e0 → 0.00e0`) | Would have silently killed architecture search |
| 5 | `VisionTransformerBlock.compute_flops` overstated ViT cost by **76×** | Any FLOPs-based loss would penalise the wrong mode |
| 6 | Neutral (zero) prior collapses all 18 layers to PlainConv via argmax tie-break | The proposed "33/33/33" init is not reachable |
| 7 | Neutral prior costs **79 accuracy points** vs the staged prior at equal budget | Removing the inductive bias is harmful at this training budget |
| 8 | Full snapshot is **7.2 MB/task**, not "tens of MB" | Overstated ~3× in the proposal |
| 9 | `A_live` collapses 98.05 → 3.52 → 28.99; `--anchor-lam` defaults to **0.0** | There was no working anti-forgetting mechanism at all |
| 10 | Stale non-editable `mgn` copy in site-packages shadowed the repo | Scripts run from subdirectories silently imported old code |

---

## 2. Why the depth gates never closed

### 2.1 The mechanism

`mgn/blocks/hybrid_18.py`:

```
delta       = pi @ structure_prototypes                 # [B, L + L*D]
layer_soft  = sigmoid(delta[:, :L] + 2.0)
gate        = (layer_soft > 0.5)                         # hard, per sample
```

`structure_prototypes` is initialised to exactly zero, so `layer_soft = sigmoid(2) = 0.8808`
and the hard gate is **open at every layer of every prototype row**.

### 2.2 Measured on the three shipped checkpoints

| checkpoint | min `layer_soft` | max | pruned rows×layers |
|---|---|---|---|
| `aml_continual_after_mnist.pth`   | 0.8792 | 0.8817 | **0 / 216** |
| `aml_continual_after_cifar10.pth` | 0.8767 | 0.8817 | **0 / 216** |
| `aml_continual_after_fashion.pth` | 0.8765 | 0.8817 | **0 / 216** |

And the parameters themselves barely moved:

| checkpoint | `structure_prototypes` \|max\| |
|---|---|
| mnist | 0.0155 |
| cifar10 | 0.0387 |
| fashion | 0.0403 |

### 2.3 Root cause: the optimiser step budget

`AdamW` moves a parameter by approximately `lr` per step regardless of gradient
magnitude. To flip `sigmoid(delta + 2)` across 0.5 the logit must travel ≈ 2.0.

* old `opt_arch` LR: **2e-4**
* CIFAR-10 task budget: 195 steps/epoch × 3 arch-trainable epochs ≈ 585 steps
* reachable travel: `585 × 2e-4 ≈ 0.12` — **16× short of the 2.0 needed**

Instrumented directly (`scratch/diag_arch_grad.py`, 12 real steps):

```
step |g| bm    |g| sp    |d bm|     |d sp|
  0   4.35e-01  1.05e-01  2.000e-04  2.000e-04
 11   3.95e-01  1.01e-01  1.230e-04  1.491e-04
GradScaler scale reductions (skipped steps): 0/12
total displacement after 12 steps: bm 1.50e-03  sp 1.93e-03
```

The gradient **is** present and well-scaled (‖g‖ ≈ 0.4); the step is clamped
exactly at `lr`. GradScaler was *not* skipping steps, so that hypothesis is
eliminated. Raising the LR to 3e-3 gives ≈ 1.75 of travel per task.

---

## 3. Why routing never changed

Softmax of the hard-coded init, and of the trained parameters:

| stage | init Plain / ResNet / ViT | after training |
|---|---|---|
| 1 (layers 0-5) | 18.1% / **81.3%** / 0.5% | 18.4% / **81.1%** / 0.6% |
| 2 (layers 6-11) | 17.5% / **78.6%** / 3.9% | 17.9% / **78.1%** / 3.9% |
| 3 (layers 12-17) | 2.4% / 17.8% / **79.8%** | 2.4% / 17.7% / **79.9%** |
| overall | 12.7% / 59.2% / 28.1% | 12.9% / 59.0% / 28.1% |

The proposal quoted "81 % and 72 %". The 81.3 % figure is real (stage 1 ResNet);
**72 % does not correspond to anything** — the stage-3 ViT share is 79.8 %, and
the overall ResNet share is 59.2 %. The two distributions are indistinguishable
before and after training, i.e. the structure is the initialisation.

---

## 4. Per-sample MoE dispatch: correct only as an inference optimisation

### 4.1 It is currently worth nothing

Measured on the shipped `fashion` checkpoint, 64 samples:

```
distinct modes chosen per layer across 64 samples:
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
=> existing single-branch fast path fires on 18/18 layers
```

Because `pi` is one-hot at eval and every prototype row carries the *same*
init pattern, `bm_logits` is identical across samples, so the existing fast path
in `hybrid_router.forward` already runs one branch. A per-sample dispatch adds
nothing until routing is genuinely non-uniform.

Where it *would* matter (forced 3-way mixed batch, one layer):

| spatial | current mixed path | proposed dispatch | uniform fast path | ratio |
|---|---|---|---|---|
| 32×32 | 8.70 ms | 4.55 ms | 2.06 ms | **1.91×** |
| 16×16 | 3.21 ms | 2.81 ms | 0.71 ms | 1.14× |
| 8×8 | 2.34 ms | 2.61 ms | 0.48 ms | **0.90× (slower)** |

The small-spatial regression is real: masked gather/scatter plus three
`.any()` syncs cost more than they save once the tensor is tiny.

### 4.2 The proposed code silently kills architecture search

`out[mask] = branch(x[mask])` removes `mode_weights` from the autograd graph.
The straight-through estimator in `hybrid_18.forward`
(`((hard - bm_logits.detach()) + bm_logits)`) then delivers **exactly zero**
gradient to `block_mode_prototypes` / `structure_prototypes`:

```
CURRENT code          : d(loss)/d(bm_logits) = 2.496e+00
PROPOSED MoE dispatch : d(loss)/d(bm_logits) = 0.000e+00
```

Implemented version is therefore gated on `not self.training`, and is guarded
by an `all_hard` check — with non-one-hot weights the identity
`Σ wₘ·branchₘ(x) = w_argmax·branch_argmax(x)` does not hold, so dispatch would
change the output. Numerically it is exact for one-hot routing
(max diff `0.00e+00`, `4.77e-07` in a separate fp32 check).

---

## 5. The FLOPs cost model was inverted

`VisionTransformerBlock.forward` pools tokens to at most 8×8 before attending:

```python
needs_downsample = H > self.max_tokens_side or W > self.max_tokens_side
```

so attention runs on 64 tokens, not `H·W`. `compute_flops` used `n = h*w`:

| spatial | plain | resnet | `compute_flops` ViT | **true runtime ViT** |
|---|---|---|---|---|
| 32×32 | 42.66 M | 85.33 M | 239.08 M | **3.15 M** |
| 16×16 | 10.67 M | 21.33 M |  22.02 M | **3.15 M** |
| 8×8   |  2.67 M |  5.33 M |   3.15 M | **3.15 M** |

Two consequences:

1. The analytic model overstated ViT by **76×** at 32×32.
2. The proposal's justification — *"Conv ở tầng thấp tốn ít FLOPs hơn ViT"* — is
   **backwards for the two wide stages**. With the token cap in place ViT is the
   cheapest mode at 32×32 (13.5× cheaper than Plain) and at 16×16 (3.4× cheaper
   than ResNet). At 8×8 there is nothing left to pool, so Plain becomes the
   cheapest again and the ranking is Plain (2.67 M) < ViT (3.15 M) < ResNet (5.33 M).
   A `λ·Σ ModeWeight·FLOPs` term therefore pushes stages 1-2 toward ViT, not
   toward Conv-at-shallow-layers. No FLOPs objective says anything about *where*
   in the network a mode belongs — only about which mode is cheap.

`compute_flops` now derives `n` from the same `max_tokens_side` constant used by
`forward`, and adds the pooling/upsampling overhead it was omitting.

---

## 6. The proposed neutral prior

The proposal was `self.block_mode_prototypes.data.zero_()` with the stated
intent of a 33/33/33 split. At inference the block mode is
`one_hot(bm_logits.argmax(-1))`, and **argmax breaks ties to index 0**, so exact
zeros pin all 18 layers to PlainConv at task 0:

```
zero-init argmax per layer: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
zero-init eval argmax -> [18.0, 0.0, 0.0]
```

A small-noise variant (`N(0, 0.01)`) does break the tie. But it is measurably
much worse at this training budget:

| config | ep1 | ep2 | ep3 | ep4 |
|---|---|---|---|---|
| baseline `HEAD`, staged prior, arch-lr 2e-4 | 66.14 % | 78.68 % | **89.92 %** | 73.74 % |
| neutral prior, arch-lr 3e-3 | 9.39 % | 11.35 % | 11.35 % | 11.35 % |

(4 epochs × 60 steps, MNIST, no replay, `K=6`, rows 0-2 owned.)

With near-zero logits, `gumbel_softmax(bm_logits, tau=0.8)` is dominated by the
Gumbel noise: each layer picks a *different* branch at essentially every step,
so Plain / ResNet / ViT each receive roughly a third of the updates and the trunk
never co-adapts. The 81 % ResNet prior is not merely a bias — it is what makes
routing coherent enough for the weights to train at all at this budget.

**Decision:** `--arch-prior staged` is the default. `neutral` remains selectable,
and `N(0, 0.01)` is used for it so the argmax tie is at least broken rather than
collapsing. The 79-point regression is recorded here rather than hidden.

---

## 7. Continual storage: measured, not assumed

Per-task snapshot contents (`K=12`, `max_dim=48`, `max_layers=18`):

| component | params | fp32 |
|---|---|---|
| `vision_encoder` | 100,208 | 0.40 MB |
| `prototype_selector` | 201,217 | 0.80 MB |
| `stem` | 1,392 | 0.01 MB |
| `layers` (18 × HybridBlock) | 1,466,208 | 5.86 MB |
| `ln` | 1,536 | 0.01 MB |
| `structure_prototypes` + `block_mode_prototypes` | 11,232 | 0.04 MB |
| `proto_weights` (int8) | 92,280 | 0.09 MB |
| **total / task** | **1,781,793** | **≈ 7.2 MB** |

Total model: 1,874,073 parameters. For `T=3`: **≈ 21.6 MB**.

So the proposal's *"hàng chục MB/task"* overstates by ~3×. A per-task snapshot is
still the dominant term, but it is a 7 MB problem, not a 30 MB one.

### 3B already existed

The autonomous router was already implemented and reported:

* `train_continual_aml.py:273-282` — `dom_truth=None` → `d_pred = pred_argmax`
* `train_continual_aml.py:1017-1025` — runs it in parallel with the oracle pass
* result matrix `A_auto`, printed alongside `A` (oracle) and `A_live`

The open cost is the **ambiguous-sample fallback**: when `margin < route_tau`,
`evaluate` loops over *all* `T` snapshots and runs a full forward per snapshot
(dòng 305-352). That is the real inference multiplier, not depth skipping.

### Nothing was preventing forgetting

`grep -n "ewc\|EWC\|fisher\|orthogonal"` over `train_continual_aml.py` returned a
single hit — a comment. The only stability term was L2-SP on
`(vision_encoder, prototype_selector)`, and it shipped **disabled**:

```python
ap.add_argument('--anchor-lam', type=float, default=0.0, ...)
```

against

```
A_live:  A[1,1] 98.05%  →  A[2,1] 3.52%  →  A[3,1] 28.99%
```

---

## 8. Hidden bugs found while implementing

| # | Bug | How it surfaced |
|---|---|---|
| B1 | `skip_layer[i]` on `None` → `TypeError` | The new eval-only short-circuit crashed in `train()`; `i > 0 and None[i]` does not short-circuit |
| B2 | Per-sample skip scattered into `torch.empty_like` | Closed-gate samples received **uninitialised memory**; caught by comparing against a full-compute reference |
| B3 | Dispatch dtype mismatch under autocast | `empty_like(x)` is bf16 while branches return fp32 (LayerNorm/GELU) → `Index put requires the source and destination dtypes match` |
| B4 | Per-sample skip dtype mismatch under autocast | Same promotion issue, latent until B3's fix prompted a re-look |
| B5 | `FisherEWC.consolidate()` **replaced** the Fisher instead of folding it in | Penalty returned exactly 0 for every task after the first — EWC would have silently done nothing |
| B6 | `penalty()` guarded on the wrong dict | `_fisher` (reset per task) instead of `_fisher_all` — same silent no-op |
| B7 | FLOPs term swamped the objective | `0.01 × 638 MFLOP = 6.38` added to a ~2.3 cross-entropy; training loss read 8.63 |
| B8 | Stale `mgn` in site-packages shadowed the repo | Scripts run from subdirectories imported a copy with **no `mgn/blocks/` at all** |
| B9 | P2 thresholded the gate at `0.5` | Only `gate <= 0` is an exact identity. Domain routing supplies **soft** slot means in (0,1); a slot gate of `0.2` must blend as `0.2·h_cand + 0.8·h`, and short-circuiting it would silently change the routed model's output. Threshold lowered to `gate_skip_eps = 1e-6`. |

B5 and B6 were both written by the fix for this task and caught by the tests
written alongside it — which is the argument for keeping
`tests/test_dynamic_depth_routing.py` in CI. B9 was found by re-reading the two
sources of `layer_gates` after the tests passed, not by the tests themselves.

---

## 9. Environment

`import mgn` resolved to the repo only when the working directory was the repo
root. A stale, non-editable copy sat in site-packages alongside the editable
finder, and it had **no `mgn/blocks/` package at all**:

```
python -c "import mgn; print(mgn.__file__)"     ->  ...\New folder (8)\mgn\__init__.py   (repo root)
python scratch\foo.py                           ->  ...\site-packages\mgn\__init__.py    (stale)
```

The duplicate was removed. `pip install -e .` in CI keeps only the editable
finder, so CI was never affected — but local experiments were silently running
stale code whenever launched from a subdirectory.

---

## 10. Reproduction

```powershell
python scratch\diag_arch_grad.py        # arch gradient / step-budget diagnostic
python scratch\ab_arch.py --prior neutral --depth-cost 0 --steps 60
python scratch\ab_arch.py --prior staged  --depth-cost 0.02 --arch-lr 3e-3 --steps 60
python scratch\test_p2_p3.py            # P2/P3/FLOPs equivalence + gradient guards
python -m pytest tests\test_dynamic_depth_routing.py -q
```

Baseline comparison used a detached worktree at `7a9b920`:

```powershell
git worktree add $env:TEMP\aml_base HEAD
python $env:TEMP\aml_base\ab_base.py --steps 60 --freeze 1
```