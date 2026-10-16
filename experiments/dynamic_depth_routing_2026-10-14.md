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
| 8 | Full snapshot is **7.12 MB/task** at the default `--norm in`, not "tens of MB" | Overstated ~3× in the proposal |
| 9 | `A_live` collapses 98.05 → 3.52 → 28.99; `--anchor-lam` defaults to **0.0** | There was no working anti-forgetting mechanism at all |
| 10 | Stale non-editable `mgn` copy in site-packages shadowed the repo | Scripts run from subdirectories silently imported old code |
| 11 | Cost term ran unconditionally | +16 % inference latency, introduced by this work; found only by benchmarking against a baseline worktree |
| 12 | `struct_override` with a wrong-width gate vector | Opaque `IndexError` from inside the layer loop; now a named `ValueError` naming both widths |

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

## 2a. What the short-circuit actually buys

Once gates can close, the saving is real and monotone (RTX 3050, batch 64,
`max_dim=48`, bf16 autocast, `cudnn.benchmark=True`):

| layers closed | `n_executed` | latency | speed-up |
|---|---|---|---|
| 0 | 18 | 276.32 ms | 1.00× |
| 2 | 16 | 267.02 ms | 1.03× |
| 4 | 14 | 260.33 ms | 1.06× |
| 6 | 12 | 253.50 ms | 1.09× |
| 8 | 10 | 235.80 ms | 1.17× |
| 12 | 6 | 200.57 ms | 1.38× |
| 17 | 1 | 40.44 ms | **6.83×** |
| per-sample (half the batch closed on layers 6-11) | 18 | 266.39 ms | 1.04× |

Exactness was verified against the **old** always-evaluate loop using the *same*
gates, for every configuration above plus the per-sample case:

```
closed= 0  max|delta feature| = 0.00e+00
closed= 4  max|delta feature| = 0.00e+00
closed= 8  max|delta feature| = 0.00e+00
closed=12  max|delta feature| = 0.00e+00
closed=17  max|delta feature| = 0.00e+00
per-sample          max|delta feature| = 0.00e+00
```

Note that closing a layer legitimately changes the network's function, so the
reference must use the *same* gates. Comparing the short-circuited network
against an all-open one measures nothing.

The curve is strongly non-linear: the early layers dominate wall-clock, so
closing layers 6-17 (the 12 cheapest blocks) buys only 1.38×, while closing 17
layers buys 6.83×. Per-sample skipping is nearly free of benefit (1.04×) because
the block still runs for the open half of the batch and the gather/scatter
overhead eats what is left.

### Regression guard

Benchmarking the above exposed a regression **introduced by this work**: the
differentiable cost term runs `compute_flops` 18 × 3 = 54 times per forward, and
those small CUDA launches cost 275 → 319 ms (+16 %). It is now behind
`model.compute_cost` (default `False`), enabled by the trainer only when
`--flops-cost > 0`. Verified against a detached worktree at `7a9b920`:

| build | plain forward | with `struct_override` |
|---|---|---|
| baseline `7a9b920` | 277.60 ms | 276.47 ms |
| this branch | 275.09 ms | 278.20 ms |

No regression, and the 1.91× dispatch figure in §4.1 still holds.

---

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

## 2b. How much training a gate actually needs

To close, a logit must travel from `0` to `-2` (`sigmoid(δ + 2) = 0.5`). AdamW
travels ≈`lr` per step, so the required arch-trained step count is `2.0 / arch_lr`:

| `--arch-lr` | steps needed | reachable in the production schedule? |
|---|---|---|
| `2e-4` (previous default) | 10 000 | no — hence 0/216 gates ever closed |
| `3e-3` (new default) | 667 | yes: MNIST 5 ep × 234 = 1 170 steps; CIFAR-10 28 ep × 195 = 5 460 |
| `1.5e-2` | 134 | yes, comfortably |

**`--steps` caps batches per epoch**, so a short validation run cannot demonstrate
depth pruning no matter how large `--arch-lr` is. In the validation run below
(`--steps 50`, 3 epochs, 2 of them arch-trained = 100 arch steps at `1.5e-2`,
travel 1.5 against the 2.0 required) the gates stayed at `18.00/18` for all three
epochs. That is the expected arithmetic, not a failure of the mechanism — the
mechanism itself is verified directly in §2a by driving the gates to a target
retention and measuring latency and exactness.

## 2c. EWC storage is O(1) in the number of tasks

```
[*] EWC enabled (lambda=100.0): anchoring 1.77M shared weights
[ewc] consolidated Fisher over task 1: 3.42M values stored for all tasks so far
```

3.42 M values = 1.71 M Fisher + 1.71 M θ*, i.e. 13.7 MB fp32 (6.8 MB fp16) held
**once**, against 7.2 MB per task of full snapshots (21.6 MB at `T=3`). The
snapshot cost is `O(T)`; the EWC cost is `O(1)`. At `T=10` the comparison is
72 MB versus 13.7 MB.

Only the 152 of 398 shared parameters that actually receive a gradient are
recorded — routing is batch-consistent during training, so the two unselected
candidate branches of each `HybridBlock` are never touched. Those branches do not
drift either (AdamW skips parameters with no gradient), which is why recording
them with zero Fisher would be wrong rather than merely wasteful.

---

## 2d. End-to-end validation run

```powershell
python train_continual_aml.py --tasks mnist,cifar10,fashion --epochs 3,5,3 \
  --steps 50 --K 12 --freeze 1 --replay-per-class 0 --consolidate-epochs 0 \
  --arch-lr 1.5e-2 --depth-cost 0.5 --depth-target 0.85 --ewc-lambda 100
```

Zero replay, EWC λ=100, hinge depth target 0.85. `--steps 50` caps the schedule
hard (CIFAR-10 sees 12 800 of 50 000 images per epoch), so these numbers are a
regression check, not headline accuracy.

**$A_{\text{live}}$ — standard mutable weights, EWC anchoring the shared trunk**

| after \ eval on | MNIST | CIFAR-10 | Fashion |
|---|---|---|---|
| after MNIST | 67.50 % | — | — |
| after CIFAR-10 | 37.28 % | 15.31 % | — |
| after Fashion | 23.90 % | 12.81 % | 70.79 % |

$$\textbf{BWT} = -23.05\% \quad|\quad \textbf{Forgetting} = 23.05\%$$

**$A_{\text{auto}}$ — autonomous centroid router + snapshot retrieval**

| after \ eval on | MNIST | CIFAR-10 | Fashion |
|---|---|---|---|
| after MNIST | 52.06 % | — | — |
| after CIFAR-10 | 26.43 % | 12.30 % | — |
| after Fashion | 31.49 % | 13.19 % | 37.40 % |

$$\textbf{BWT} = -9.84\% \quad|\quad \textbf{Forgetting} = 9.84\%$$

**$A_{\text{oracle}}$ — task-id snapshot retrieval**

$$\textbf{BWT} = +0.00\% \quad|\quad \textbf{Forgetting} = 0.00\%$$

### What this does and does not show

* **EWC halves live forgetting.** The recorded baseline without any stability
  term is `BWT −46.50 %` / forgetting `46.50 %`; this run gives `−23.05 %`.
  *Caveat:* the two runs use different schedules (this one is `--steps 50`), so
  this is directional evidence, not a controlled A/B. A controlled comparison
  needs the same schedule with and without `--ewc-lambda`.
* **Structure now differentiates per domain.** Mean gates moved off the flat
  `0.88` into `0.77–0.86`, and the pairwise fingerprint divergence is non-zero
  (`gate_L1 = 0.234` MNIST-vs-Fashion, `0.463` MNIST-vs-CIFAR10) where before it
  was identically zero everywhere.
* **No layer actually pruned** (`active L: 18.00/18` throughout), which §2b
  predicts: 100 arch steps at `1.5e-2` gives travel 1.5 against the 2.0 needed.
  The pruning *mechanism* is verified separately in §2a.
* **The autonomous router is the weak link.** `$A_{\text{auto}}$` is below
  `$A_{\text{live}}$` in every off-diagonal cell — routing a sample to the wrong
  domain costs more than just using the live model. This is the highest-value
  target for future work, and it is a routing-accuracy problem, not a storage one.

---

### Open-set live fallback — implemented, NOT yet validated for accuracy

`--auto-live-thresh` (default `0`, disabled) evaluates a sample with the current
live weights when no snapshot reaches that softmax confidence. The motivation is
the measured weakness above: `$A_{\text{auto}}$` was below `$A_{\text{live}}$` in
every off-diagonal cell, so misrouting cost more than snapshot retrieval gained.

What we **did** verify:

```
A_live   = 54.30%
A_oracle = 63.09%   (invariant control)

 thresh |   A_auto |  vs live |  fallbk% | route |   sec
    0.00 |   63.09% |   +8.79 |     0.0% |  0.0% |  4.6
    0.30 |   63.09% |   +8.79 |     7.0% |  0.0% |  5.0
    0.50 |   63.09% |   +8.79 |    33.6% |  0.0% |  6.4
    0.70 |   63.09% |   +8.79 |    59.8% |  0.0% |  7.9
    0.90 |   63.09% |   +8.79 |    90.2% |  0.0% |  9.5

control: A_oracle with fallback on = 63.09% (must equal 63.09%) -> OK
```

* The fallback provably **does not leak into the task-oracle protocol** — `A_oracle`
  is bit-identical with the fallback enabled at every threshold. That was the main
  correctness risk, and it is closed (and now regression-tested).
* The fallback fraction responds to the threshold as intended (0 % → 90 %).

What we **could not** verify: any accuracy benefit. In a single-domain probe the
snapshot is built from the live weights, so live ≡ snapshot and the sweep is
degenerate by construction — every row is identical. Separately, `fp["route_acc"]`
is only meaningful when `dom_truth` is supplied (it accumulates hits only inside
the `if dom_truth is not None` branch), so the `0.0 %` above is an artefact of
that, not a router failure.

The flag therefore **ships disabled**. A proper test needs a two-domain setting
where live and snapshot weights genuinely differ, i.e. another full continual run,
which did not fit in the remaining budget. Two harness bugs were found and fixed
on the way: evaluating without the task's `pi_mask` let untrained prototype rows
vote (reported 12.5 % for a model that trains to 54 %), and the three-task
checkpoints had been overwritten by a later smoke run.

### 3A — shared backbone, no per-task 18-layer copy

`--shared-backbone 1` stores **no** per-task copy of `stem`, the 18 `layers`, or
`ln`. Per-task state drops from 7.2 MB to ≈1.2 MB (encoder + prototype selector +
structure heads + int8 prototypes), and the 5.86 MB trunk copy becomes a single
shared set of weights that EWC alone has to protect.

`evaluate` already restores the trunk conditionally (`if "layers" in snaps[d]`),
so this required no change to the retrieval path — only to what gets written.

The honest risk, stated up front: `A_live` decayed to `3.52 %` in the historical
record with no stability term at all, so removing the snapshot without EWC holding
the weights is exactly the configuration that collapses. The flag warns when it is
combined with `--ewc-lambda 0`:

```
[!] --shared-backbone without --ewc-lambda>0 has NOTHING holding the shared
    weights in place; A_live will collapse.
```

<results placeholder>

**Controlled A/B** — identical command, identical schedule, identical seed, the
only difference being `--shared-backbone`:

| | per-task 18-layer snapshot | `--shared-backbone` (3A) |
|---|---|---|
| per-task state | 7.2 MB | **≈1.2 MB** |
| trunk copies | 1 per task (5.86 MB each) | **1 shared, 0 copies** |
| $A_{\text{live}}$ BWT | −23.05 % | **−13.91 %** |
| $A_{\text{live}}$ forgetting | 23.05 % | **13.91 %** |
| $A_{\text{auto}}$ BWT | −9.84 % | −9.23 % |
| $A_{\text{oracle}}$ BWT | **+0.00 %** | −9.00 % |

Cell-by-cell, $A_{\text{live}}$ after CIFAR-10 on MNIST went **37.28 % → 65.81 %**,
and after Fashion **23.90 % → 48.78 %**.

**Interpretation.** The trade is exactly the one the proposal predicted, and it
cuts the opposite way from what "zero-forgetting" papers usually claim. Giving up
the per-task trunk copy:
* **halves live forgetting** (23.05 % → 13.91 %), because every task now optimises
  the same weights and the structure has to serve all of them rather than each
  task getting a private copy; and
* **destroys task-oracle exactness** (0.00 % → 9.00 %), which is unavoidable —
  with one shared trunk there is no second trunk to retrieve.

So 3A is *not* a strict improvement: it trades a headline "zero forgetting under
oracle retrieval" claim for lower average error and 6× less per-task storage.
Which side wins depends on whether the deployment protocol supplies a task id.

**Caveat, stated plainly.** These are single runs at a deliberately short
schedule (`--steps 50`, so CIFAR-10 sees a quarter of one epoch). Run-to-run
variance under `cudnn.benchmark` plus non-deterministic reductions is
substantial, and a 28-point swing in one cell is larger than we would trust from
one seed. **A multi-seed comparison is required before treating the 3A
improvement as real.** We report it as suggestive, and ship the flag default-off.

---

## 3. Why routing never changed

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

* `train_continual_aml.py:1006-1012` — `dom_truth=None` → `d_pred = pred_argmax`
* `train_continual_aml.py:1056-1064` — runs it in parallel with the oracle pass
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

### B19 — EWC does not work at production scale

The full run `--epochs 5,28,16 --ewc-lambda 100` refutes the short-schedule result:

| protocol | BWT | forgetting | §2d short-schedule claim |
|---|---|---|---|
| `A_live` | **−59.23 %** | **59.23 %** | was 23.05 % |
| `A_auto` | −48.07 % | 48.07 % | was 9.84 % |
| `A_oracle` | +0.00 % | 0.00 % | unchanged |

Every EWC figure in this document — §2c, §2d, and the shared-backbone A/B in
§"3A" — came from `--steps 50`, where CIFAR-10 saw a quarter of one epoch. At
5 460 steps the diagonal-Fisher anchor with λ=100 is simply too weak: the shared
trunk drifts regardless. The `--steps 50` numbers are marked superseded in
README rather than deleted, so the discrepancy stays visible.

**Conclusion:** EWC as implemented is not a working anti-forgetting mechanism
here. The earlier "halves live forgetting" claim does not survive scale.

### B21 — `--norm` never reached the encoder or the stem

`make_norm()` was introduced in B16 so that `--norm` would be honoured, and the 18
hybrid blocks do call it. But two modules were left as hardcoded
`nn.BatchNorm2d`:

- `hybrid_18.py` `vision_encoder` — the module the **domain router reads from**
- `hybrid_18.py` `stem`

Verified at runtime rather than by reading alone:

```
norm_type='in':
  vision_encoder: ['Conv2d', 'BatchNorm2d', ...]   ← not InstanceNorm2d
  stem         : ['Conv2d', 'BatchNorm2d', 'ReLU'] ← not InstanceNorm2d
  layers[0]    : HybridBlock                        ← correct
```

So `--norm in` bought nothing where it mattered most. BatchNorm is the only norm
in the set that carries running statistics, and in a continual supernet those are
rewritten by every subsequent task — the encoder the router queries under one
task's distribution was being compared against centroids captured under another.

Both now call `make_norm(norm_type, max_dim)` with `conv_has_bias(norm_type)` on
the conv, so a non-BatchNorm conv carries the shift that BatchNorm would have
supplied. Measured: `--norm in` yields `InstanceNorm2d` in both, `bn` yields
`BatchNorm2d`, `gn1` yields `GroupNorm`. Total parameters 1,871,385 (7.49 MB),
unchanged in magnitude.

Regression tests in `tests/test_domain_router.py::TestNormPropagation` assert the
norm type reaches both modules, that no running statistics survive under
`--norm in`, and that conv bias matches the norm choice.

### B22 — the centroid was blended with dead-encoder features

`train_task()` calls `update_task_centroid()` on every step, so by the end of a
task the running sum is mostly features from encoder states that no longer exist.
The trainer then zeroes the centroid and rebuilds it from a clean pass, but the
**shared covariance** kept the training-time contributions, because it is global
and cannot be rebuilt per task in isolation.

Measured, by reconstructing both variants from the same checkpoints:

```
clean centroid, covariance from the clean pass : routing 100.0 % / 100.0 %
blended centroid, covariance from mid-training  : routing 100.0 % /  82.0 %
```

MNIST alone fell to 64.6 %. A 3-epoch task leaves 65.8 % of its centroid mass
and a 6-epoch task 79.3 % describing encoders that no longer exist.

Fix: the two paths are now explicit. `update_task_centroid()` (training) leaves
the covariance alone; `finalize_task_features()` (the final pass) is the only
thing that feeds it.

### B23 — a centroid is stored in the space of the encoder that produced it

The router scored a query embedded with the **current** encoder against
centroids captured under **each task's** encoder. Measured drift on the real
MNIST → CIFAR checkpoints:

```
||mean_old - mean_new||      = 0.800
within-domain spread         = 0.336
drift / spread               = 2.38x
```

Routing accuracy at that point, by method:

| method | 2 tasks | 3 tasks |
|---|---|---|
| cosine, current encoder | 33.4 % | 33.4 % |
| shared Mahalanobis, current encoder | 33.3 % | 33.3 % |
| **per-domain** Mahalanobis, current encoder | 50.0 % | 34.2 % |
| **per-task embedding**, shared covariance | **100.0 %** | **99.1 %** |

No metric choice bridges a coordinate mismatch — which is why the ReLU and
metric fixes alone plateaued. Note that per-domain covariance costs 1 MB per task
and does **not** help. Embedding the query once per task needs no extra stored
state, because snapshot retrieval already loads each encoder per batch.

### B24 — `finalize_task` normalised the centroid under every metric

It L2-normalised unconditionally. Under Mahalanobis that moves the centroid off
the cloud it represents: a tight domain has ‖mean‖ near 1 (0.929 measured for
MNIST) while a broad one cancels itself down (0.695 for CIFAR-10), and rescaling
both to the sphere collapsed CIFAR-10 routing from 100.0 % to 4.0 %. Only the
cosine metric normalises now.

### Process note — three wrong turns, and what each cost

Recorded because the method is the part worth keeping.

1. **Measured on the wrong artifact.** An A/B on `after_fashion.pth` (a
   three-task checkpoint) was used to predict two-task behaviour. It did not
   transfer, and the change built on it was *worse* end-to-end.
2. **A wrong sign in a distance table.** `argmax` was used where `argmin` was
   required, which printed 0 % for a metric that was actually at 99.9 %.
3. **A silently overwritten checkpoint.** `aml_continual_after_mnist.pth` had
   been rewritten by a later two-task run, so a diagnostic "confirming" a fix
   was measuring an encoder that had trained on both domains.

Each cost a full twelve-minute run. A three-minute offline diagnostic would have
caught all three first. The rule adopted afterwards: **diagnose before training,
and never commit a change whose end-to-end number has not been seen.**

### B25 — bf16 autocast collapsed the router's input to nothing

The most expensive bug in the project, and the one that made four earlier
"fixes" look wrong.

`router_features()` ran inside the evaluation autocast context. The encoder ends
in `AdaptiveAvgPool2d((2, 2))`, which averages 32x32 down to 2x2: the values that
survive are small and nearly equal, and bf16 carries about three decimal digits.
Two genuinely different encoders therefore produced **bitwise identical**
features:

```
GPU fp32          maxdiff = 0.36782     cos = 0.915
GPU bf16 autocast maxdiff = 0.00000     cos = 1.000000
```

With every domain collapsed onto one point, the centroids became
indistinguishable and routing had no signal left. No metric change could have
recovered it.

**Why it survived four wrong diagnoses:** every offline measurement ran in fp32
on CPU and reported `cos(A,B) = 0.75` - i.e. "the encoder is weight-sensitive,
the router logic must be at fault". The bug only appears on the execution path
the trainer actually uses. See the process note above.

`router_features()` now pins fp32 via `torch.autocast(enabled=False)`, and
`TestRouterFeatures::test_bf16_autocast_cannot_collapse_two_encoders` asserts
that two encoders stay distinguishable under bf16.

### B26 — A_auto scored samples against prototype rows trained on another domain

`route_acc` (added this session) showed routing at **100 %** on the CIFAR-10
cell while `A_auto` was still 9.67 % against a 32.12 % oracle. The router was
no longer the suspect, so the fault was downstream.

The prototype selector is shared across tasks and each task only ever trained
its own rows:

```
MNIST -> rows [0:6)     CIFAR -> rows [6:12)
```

`A_auto` passed `pi_mask=None`, so a CIFAR image was scored against all 12 rows,
6 of which had never seen CIFAR. Measured on the same weights with routing
already correct:

```
pi_mask = the task's own rows (oracle) : 32.63 %
pi_mask = None (as shipped)             : 10.86 %
```

Passing `pi_mask_j` was also wrong, for a different reason: it is built from the
**true** task index, which hands the task identity straight to the selector.
The fix builds a per-sample mask from `d_pred` - the router's own decision - so
nothing leaks that the router had not already decided:

```python
pm = torch.zeros(B, K, dtype=torch.bool)
for d in range(T):
    pm[d_pred == d, lo_d:hi_d] = True
```

Routing is per image, so the mask has to be per image too; that required
`pi_mask` to accept `[B, K]` alongside `[K]` in `hybrid_18.py`. An ambiguous
sample retried under snapshot `d` uses **that** snapshot's rows, not the rows
its router argmax pointed at.

### B27 — the router was never the whole problem

A[2,2] across this session, each step exposing the next cause:

| step | A[2,2] | route_acc | cause still outstanding |
|---|---|---|---|
| session start | 8.01 % | - | ReLU tail, cosine, stale centroids |
| drop the ReLU tail | 22.00 % | - | metric |
| Mahalanobis | 16.37 % | - | metric (broke MNIST instead) |
| per-task embedding | 13.05 % | - | **bf16** |
| route_acc added | 9.67 % | **100 %** | **pi_mask** |
| per-sample pi_mask | **33.55 %** | 100 % | - |

Four independent causes, stacked. None was predictable from the one before it,
and the last two only became visible once the previous was removed.

### Process note — five wrong diagnoses, one root cause each

Recorded because the pattern, not the individual bugs, is the lesson.

1. A/B on a three-task checkpoint used to predict two-task behaviour. It did
   not transfer, and the change built on it was *worse* end-to-end.
2. `argmax` where `argmin` was required, printing 0 % for a metric that was
   actually at 99.9 %.
3. A checkpoint that had been silently overwritten by a later run, so a
   diagnostic "confirming" a fix measured an encoder trained on both domains.
4. A 1-epoch debug run read as an architecture defect: `cos(z0,z1) = 1.0` on
   two barely-diverged encoders is expected, not a bug.
5. **fp32/CPU measurements of code that runs bf16/GPU** (B25).

All five produced entirely plausible numbers, all five led to a confident claim,
and all five were wrong. The sixth measurement, on the same input and both
devices, found the cause in one shot.

The rule adopted: **measure on the execution path the trainer actually uses, in
the precision it actually uses.** Unit tests and offline reproductions are for
checking a hypothesis, never for confirming one. And `route_acc` now exists so
that "is this the router or something downstream" is answered by the log rather
than by a script written after the fact.

### B28 — the static extractor could not accept the model it was written for

`AML_Core_Code/extract_static_model.py` only handled the legacy
`ConvAML_CIFAR_TrueDynamic`. Handing it a `ConvAML_18Layer_Hybrid` produced

```
AttributeError: 'ConvAML_18Layer_Hybrid' object has no attribute 'ood_router'
```

Worse, the guard was the reason it was not caught earlier: it checked only for
`structure_prototypes` and `proto_weights`, and the hybrid has both. So a
completely unsupported model passed validation and failed deep inside the
extraction path. `extract_child` now dispatches on the actual class.

The two supernets are structurally different, not just differently named:

| | legacy TrueDynamic | 18-layer hybrid |
|---|---|---|
| routing | `ood_router` | `prototype_selector` |
| conv weights | inside `proto_weights` | real submodules (`block.plain/.resnet/.vit`) |
| block modes | 2 | 3 (Plain / ResNet / ViT) |
| trunk prefix | none | conv stem 3 -> max_dim |
| head norm | implicit | affine `LayerNorm` |
| prototype layout | `get_prototype_layout` | `[K, wo_size + num_classes]` |

Because there is no weighted sum to fold, the 2-conv `StaticLayer` does not
generalise; a new `StaticHybridBlock` keeps the `argmax` branch per layer, which
is what `HybridBlock.forward` already collapses to once the weights are one-hot.
Offsets are read off the model rather than from `get_prototype_layout` — the two
layouts describe different tensors, and that mismatch is how the drift happened.

Measured on the default config: **624,346 parameters / 2.50 MB**, against
1,874,073 / 7.50 MB for the supernet. Ten tests in
`tests/test_extract_static_model.py`.

### B29 — the paper embedded source that no longer exists

The appendix carried ~780 lines of `lstlisting` copied from v1 files that had
since been deleted (`export.py`, `test_export.py`, `test_conv_aml.py`), plus a
benchmark listing that still showed a retired `disable_vit` flag. The
`ExportedModel` name in the body text was also stale.

Replaced with listings taken from the files that ship, each with a version note
stating what changed and why it mattered — notably that the old `ExportedLayer`
carried one convolution where the AML block has two, so an extracted residual
branch would not have been a faithful replica. The two test sections now
describe the 134-test CI suite that replaced the standalone scripts. Also fixed
a dangling `Listing~\ref{lst:export}` and two `eq:block` references that point
at labels which do not exist; every `\ref` in the file now resolves.

### B20 — the autonomous router is now the binding constraint

With BatchNorm removed and a strong backbone, the remaining error concentrates in
one place:

```
A[2,1]  MNIST after CIFAR-10   auto  8.01 %   |  oracle 97.94 %
A[3,1]  MNIST after Fashion    auto 26.33 %   |  oracle 97.94 %
A[3,2]  CIFAR after Fashion     auto 43.19 %   |  oracle 67.91 %
```

`A_oracle` holds 97.94 % on MNIST through both later tasks, so **the stored
snapshot is fine**. The failure is entirely `dom_head` selecting the wrong
domain, after which the model is evaluated under a snapshot whose prototype rows
for the true domain were never trained. This is not a storage problem and not a
normalization problem — it is a routing-accuracy problem, and it is now the
single largest source of error in the system.

### Full-scale run summary (InstanceNorm, `--epochs 5,28,16`, seed 42)

| | MNIST | CIFAR-10 | Fashion |
|---|---|---|---|
| `A_oracle` | 97.94 % (stable) | 67.91 % | 92.11 % |
| `A_auto` | 97.94 % → 8.01 % → 26.33 % | 67.72 % → 43.19 % | 74.63 % |
| `A_live` | 98.09 % → 28.00 % → 16.23 % | 68.03 % → 31.43 % | 92.09 % |

InstanceNorm is confirmed at scale — three tasks, monotonic curves, and the three
eval protocols agree to within 0.3 pt on the diagonal (67.72 / 67.91 / 68.03),
where BatchNorm had them 29 pt apart. The normalization finding (B16) holds; the
EWC and 3A findings do not.

### B16 — BatchNorm running statistics were the real accuracy ceiling

Not an external review finding: this came out of the full production run, and it
invalidates most of the numbers above.

The run showed MNIST peaking at 40 % on CIFAR-10 and then sliding to ~21 % while
training loss kept falling — loss ↓ with test accuracy ↓ is a train/eval mismatch,
not overfitting. Evaluating the shipped MNIST checkpoint in both modes settled it:

```
eval()  — BN uses RUNNING stats :  73.81 %
train() — BN uses BATCH stats   :  97.04 %
```

**23.2 points on identical weights.** The network learned MNIST to 97 %; the
inference-time statistics destroyed it.

The mechanism is structural to this supernet. `HybridBlock` holds three candidate
branches, each with its own BN, and training routes batch-consistently — one branch
per layer per batch. A branch's running statistics are therefore updated *only
while that branch is routed to*. Raising `--arch-lr` to `3e-3` (needed to unblock
architecture search, §2.3) makes weights drift fast enough that stored statistics
stop describing them. The old default `2e-4` kept weights nearly frozen, which is
why the historic 92.83 % was reproducible — by being slow.

**And `bn` does not reproduce.** Two runs of the same config with the same seed 42
gave live accuracy **72.75 %** and **87.09 %**, a 14.3-point spread, with live and
routed accuracy *swapping order*. That irreproducibility matters more than the
lower mean.

Controlled A/B — MNIST, 5 epochs, seed 42, `--arch-lr 3e-3 --depth-cost 0.02`:

| `--norm` | per-epoch accuracy | A[1,1] | s/epoch | shape |
|---|---|---|---|---|
| `bn` | 88.86 → 84.40 → 84.75 → 83.93 → 87.09 | **84.75 %** | 100 | peaks, then decays |
| `gn1` | 85.28 → 93.06 → 94.57 → 95.87 → 96.38 | **96.26 %** | 116 | monotonic |
| `in` | 92.09 → 95.85 → 97.21 → 97.77 → 97.91 | **98.03 %** | 108 | monotonic |

InstanceNorm wins on accuracy, stability *and* wall-clock; the default changed from
`bn` to `in`. GroupNorm is second. Single-task measurement — the full 3-task
schedule has not been re-run, so 98.03 % is not yet a headline number.

**B17 — `--norm in` never worked.** `cnn_block.py` branched on
`if norm_type == "bn": BatchNorm else: GroupNorm`, and the trainer mapped
`args.norm != "bn"` to `"gn"`, so `--norm in` silently produced GroupNorm while
`choices=['bn','gn1','in']` advertised three options. Replaced with a
`make_norm()` factory that also rejects unknown values.

**B18 — `--help` was broken.** An argparse help string containing a literal `%`
(`"staged reaches 89.9%, neutral stalls at 11.4%"`) is treated as a format string;
`--help` died with `ValueError: unsupported format character '('`. Introduced by
this branch's own `--arch-prior` help text and unnoticed until the `--norm` help
was written. Both regression-tested now.

### B13–B15 — found in external review, all confirmed by measurement

**B13 (high) — the expected-FLOPs loop dropped mode 0's routing weight.**
`_c = _f if _c is None else _c + block_mode[:, _i, _m] * _f` seeds the
accumulator with mode 0's **raw** cost, so PlainConv entered with coefficient
`1.0` instead of `block_mode[:, _i, 0]`. Measured over-report at one-hot routing:

| routing → | reported | true | error |
|---|---|---|---|
| Plain | 16,739,935 | 16,739,935 | 0.0 % |
| ResNet | 48,238,116 | 31,498,182 | **+53.1 %** |
| ViT | 18,130,664 | 1,390,728 | **+1203.7 %** |

The cost term therefore penalised the *cheapest* modes hardest, inverting the
ranking it exists to express. Two corrections: the accumulator now starts from
zeros, and `layer_gates[:, _i]` is applied as a multiplier (B13b) — closing a
gate previously left the reported cost unchanged, so the complexity term could
not reward pruning.

*One correction to the review:* it claimed `∂expected_gflops/∂w_plain = 0`.
That is **false** — the unweighted term is a constant offset, so the gradient
w.r.t. the mode *logits* is shifted by a constant rather than annihilated
(measured `d/d logits = [-0.61, 0.00, -1.24]`). The ResNet-vs-ViT preference is
still learned correctly. The damage is the over-reporting above, not a dead
gradient path.

The bug was latent: `--flops-cost` defaults to `0`, so no shipped default
configuration exercised it.

**B14 (high) — task-identity leak into the autonomous protocol.**
The `A_auto` evaluation passed `pi_mask=pi_mask_j`, built from the *true* task
index `j`. Two consequences: the prototype selector was told which task's rows to
use (so the router's decision was pre-empted), and when the router predicted an
older domain `d < j`, `evaluate` had loaded `snaps[d]`, whose `proto_weights` rows
`[sj:ej)` were still untrained at the moment task `d` finished — so the model
classified against random head weights. This is a direct cause of the depressed
`A_auto` off-diagonals. `A_auto` now receives `pi_mask=None`; the oracle and live
protocols keep their task mask.

**B15 (medium) — README credited the wrong model with 12,759,873 parameters.**
That is the legacy `ConvAML_CIFAR_TrueDynamic` (`train.py`), which stores every
conv kernel inside `proto_weights`. `ConvAML_18Layer_Hybrid` has **1,871,385**
(7.49 MB fp32) at the default `--norm in` — the legacy model is **6.82× larger**.
All four tables corrected. The count is norm-dependent, because `conv_has_bias`
and the norm's own affine pair trade against each other, so it is quoted per
norm rather than as a single number:

| `norm_type` | parameters | fp32 size (MB, 10^6 B) |
|---|---|---|
| `bn` | 1,874,073 | 7.50 |
| `gn1` | 1,876,761 | 7.51 |
| **`in` (default)** | **1,871,385** | **7.49** |

B5 and B6 were both written by the fix for this task and caught by the tests
written alongside it — which is the argument for keeping
`tests/test_dynamic_depth_routing.py` in CI. B9 was found by re-reading the two
sources of `layer_gates` after the tests passed, not by the tests themselves.

### B30–B36 — found by re-reading the code against the math, then pinned by tests

**B30 (high) — EWC anchored every past task to the *most recent* task's weights.**
The across-task state was a running sum of Fisher `_fisher_all` plus a single
`_star` that `consolidate()` **overwrote** each task. After task 2 that scores
task 1's Fisher against task 2's `θ*`, i.e. it anchors the model to a point that
has already forgotten task 1. A second defect compounded it: the guard read
`if not self.enabled or not self._fisher_all or not self._star` while
`consolidate()` rebuilt `_star` from the parameters actually present in the
Fisher set, so the penalty could return exactly `0.0` for every task after the
first.

The exact expansion needs only two accumulators of the same size as the
`(F, θ*)` pair it replaces:

```
sum_t F^(t)(θ − θ*^(t))²  =  A·θ² − 2B·θ + c      A = Σ F^(t),  B = Σ F^(t)θ*^(t)
penalty()                 =  (λ/2)·Σ A(θ − B/A)²   completed square; c is θ-free
```

The completed square keeps the value non-negative and vanishing at the
**Fisher-weighted mean of every task's optimum**, which is where the true
quadratic is minimised. `state_bytes()` is unchanged at 2 vectors/parameter.
`tests/test_fisher_ewc.py` checks the gradient against a brute-force T-copy
reference, asserts the minimum is the weighted mean and not `θ*^(last)`, and
quantifies how much worse the old scheme was at the optimum. EWC remains
*disabled by default* (`--ewc-lambda 0`); this fixes the maths, not B19's verdict.

**B31 (medium, perf) — the fused backward was unreachable at every default config.**
`pipeline_backward` gated the fused mix kernels on `C_out <= 256 and 256 % C_out == 0`,
a v1 leftover; the v2 kernel walks channels sequentially and only parallelises
over spatial, so the divisor test is meaningless. At the shipped default
`--max-dim 48` it is `256 % 48 == 16`, so `use_mix` was **false on every default
run** and the backward silently fell back to ATen recompute. Lifted to
`C <= MB_MAX_C` (1024, the kernel's own cap) on both sides of the boundary —
the Python guard and the `TORCH_CHECK` in `mgn/csrc/bindings.cpp` had already
drifted, which is exactly the failure mode a source-level test can catch:
`tests/test_fused_kernel_guard.py` pins both guards to the `#define` in
`fused_pipeline.cu`. **The two guards must change together**: changing only the
Python one turns a silent fallback into a hard `RuntimeError`.

**B32 (medium) — the static child was a different function from its mother.**
`extract_child_hybrid18` compiled the structure argmax but dropped the **depth
gate** (every block ran at full strength, so `--depth-cost` had no effect on the
extracted model), carried the legacy `dropout_p = 0.2` where `hybrid_18.py` uses
`0.1`, and clamped the width at the legacy `24.0` instead of the mother's `16.0`.
Each is a constant that silently drifts the moment the mother changes.
`StaticHybridBlock` now takes `gate` / `is_first` / `gate_skip_eps`, compiles a
closed gate out entirely (`is_active = False` → returns `x` untouched, so it
costs nothing), and the constants are named and source-annotated. Four tests added.

**B33 (low) — class default `norm_type="bn"` contradicted the CLI default `in`.**
`ConvAML_18Layer_Hybrid()` built with no arguments produced BatchNorm while
`train_continual_aml.py --norm` defaulted to InstanceNorm, so any script that
constructed the model directly (tests, probes, benchmarks) silently measured a
different model from the trainer. Defaults aligned to `in`.

**B34 (low, tooling) — `python -m pytest` at the repo root collected `scratch/`.**
`scratch/test_p2_p3.py` calls `sys.exit()` at import time (it is a script, not a
test), which pytest reports as `INTERNALERROR … caught unexpected SystemExit` and
**zero tests run** — a green-looking failure that hides the entire suite. Fixed
with `testpaths = ["tests"]` in `pyproject.toml` rather than by deleting the
scratch script, which is still referenced by the reproduction commands below.

**B35 (medium) — "snapshot retrieval" in `benchmark_cl_comparison.py` was a partial swap.**
It restored `stem`, `layers`, `proto_weights` and the structure heads, but not
`vision_encoder`, `prototype_selector` or `ln` — the three blocks that carry the
domain signal — and then evaluated with `pi_mask=None`, so a snapshot was scored
against all K prototype rows, including rows never trained for that domain (the
same defect as B14, one layer down). Both fixed: the benchmark now swaps the
full snapshot and evaluates each task under its own prototype row range.

The same function had two more silent omissions, both of which made the measured
gap *smaller* than reality: `vision_encoder` and `prototype_selector` were not in
the optimiser (so the routing path never trained while the trunk did), and
`isolator.begin_task()` was called while `mask_grad()` / `restore()` never were,
so the isolation window was opened and then ignored.

**B36 (medium) — the AML-vs-HtS benchmark never trained its router.**
`dom_head = nn.Linear(1024, T)` was created, used at evaluation, and put in
**no optimizer and no loss term**, so autonomous routing there was a fixed random
projection. It now has its own `AdamW`, a `0.3`-weighted cross-entropy against
the current task id, and its own gradient-norm clip.

**B40 (high) — the router collapses at the third task: one covariance cannot span three encoders.**
The 2026-10-15 production run (epochs 5,28,16, seed 42, zero replay) reports
`route_acc` 100.00 / 93.96 / 100.00 through stage 2, and then **5.94 %** on
MNIST and **50.01 %** on CIFAR-10 after the third task. `A_auto` falls to
`A_avg` 50.31 % against `A_oracle` 86.73 %. 5.94 % is far *below* chance (33 %),
so it is systematic misrouting rather than noise — and `route_acc` is what makes
it visible at all: without it, "A_auto is bad" has six possible causes.

The cause is a mismatch that only appears once there are three tasks.
`score_per_task` embeds each query with the encoder of the candidate domain, so
the comparison for task *k* happens in *encoder_k's* feature space, while the
Mahalanobis metric used **one** covariance pooled over all tasks. Two pooled
spaces are still roughly compatible; three are not, and the pooled matrix ends
up dominated by whichever task contributed the most variance.

Measured on the run's **own checkpoints** — `aml_continual_after_<task>.pt` is
the model at the end of that task, i.e. exactly the encoder whose space the
stored centroid lives in — so no retraining is needed and the offline path
reproduces the run rather than approximating it:

| routing accuracy, 3 tasks | shared (shipped) | per-task covariance |
|---|---|---|
| MNIST | **6.07 %** (run: 5.94 %) | **95.10 %** |
| CIFAR-10 | **41.16 %** (run: 50.01 %) | **99.98 %** |
| Fashion | 98.78 % (run: 99.39 %) | 85.75 % |

Per-task alone trades Fashion away, so the shipped metric is the hierarchical
mixture `S_k = (1−α)·S_k + α·S_pooled`:

| α | MNIST | CIFAR-10 | Fashion | mean |
|---|---|---|---|---|
| 0.00 | 95.10 % | 99.98 % | 85.75 % | 93.61 % |
| **0.15** | **95.37 %** | **99.67 %** | **96.36 %** | **97.13 %** |
| 0.50 | 46.39 % | 90.48 % | 98.82 % | 78.56 % |
| 1.00 | 6.07 % | 41.16 % | 98.78 % | 48.67 % |

`--router-metric mahalanobis-task --router-cov-alpha 0.15` is the new default;
`mahalanobis` and `cosine` stay available. Storage grows from one `D×D` matrix
to `T×D×D` (3.0 MB at T=3, D=512) and the method is still zero-replay — these
are second moments, not images.

Two things this repeats from earlier in the same log. First, the honest way to
re-open a question is to persist the state that answers it: `--save-routing`
now writes the router buffers and the per-task snapshots, because for six
commits the router existed only in memory and every question about it required
retraining, which does not reproduce. Second, the "two tasks work" result that
this whole session chased was real and was never about three tasks; a fix that
reproduces on two of three sequences is not a fix.

**B39 (high, measurement) — the CUDA parity gate failed a correct kernel.**
`validate_ext_in.py` reports `IN forward` and `grad L=2,3,4` as FAIL
(5–17 % relative error), and CI runs it whenever the runner has a GPU — but
`ubuntu-latest` has none, so it has been failing unseen. The first instinct is
"the InstanceNorm kernel is wrong". Measurement says otherwise: the eager
reference convolves in bf16, so **it disagrees with itself** (bf16 vs fp32) by
more than the kernel's entire deviation from it, at every depth:

| L | fused vs reference | reference vs itself (bf16 vs fp32) |
|---|---|---|
| 1 | 0.042 | 0.061 |
| 2 | 0.084 | 0.107 |
| 3 | 0.116 | 0.177 |
| 4 | 0.146 | 0.279 |
| 18 (forward) | 0.147 | 0.169 |

The kernel is *closer to the fp32 answer than the reference is to itself*. The
fixed `1e-2` gradient tolerance and `5e-2` forward tolerance were unsatisfiable
by construction — they would also have passed a broken kernel that merely tracked
the reference's rounding. The gate is now `max(tol, 1.5 x measured noise floor)`
and prints the floor next to every number. A real bug still exceeds it; a
correct kernel no longer fails. The individual pieces check out exactly:
`_C.instance_norm_bwd` vs autograd through the op it differentiates is
`rel_err 0.000000`.

This is the same failure mode as the router investigation, in a different place:
a number that looked damning turned out to be a property of the measuring
instrument. The rule that caught it is the one already in this repo — compare
against the reference's own spread, not against a constant someone picked.

**B38 (high, honesty) — the default run was not the documented run.**
`--replay-per-class` defaulted to **400**, so a bare
`python train_continual_aml.py` stored 4 000 exemplars and ran a consolidation
pass over them, while every headline number in the README and the paper is a
**zero-replay** number obtained by passing `--replay-per-class 0` explicitly.
Nothing warned; the reproduction command and the claim simply described two
different methods. The default is now `0`, and `build_arg_parser()` was extracted
from `main()` so `tests/test_zero_replay_default.py` can assert the defaults
against the trainer's own parser — the defaults *are* part of the scientific
claim, so they are tested like one.

**B37 (perf) — `load_state_dict` was the evaluation bottleneck, not the GPU.**
`evaluate` swaps the encoder, selector, stem, 18 layers and the head LayerNorm
once per snapshot per batch. `load_state_dict` re-validates every key and shape
on each call, so the bookkeeping — not the 1.87 M parameter copy — dominated.
`apply_state()` copies into the live tensors directly and `state_targets()`
caches the destination dicts once per `evaluate` call instead of rebuilding
them per swap. CPU measurement of one full 5-module swap at `--max-dim 48`:

| | per full swap |
|---|---|
| `load_state_dict` | 8.32 ms |
| `apply_state` | 2.04 ms |
| speedup | **4.08×** |

The saving is per *call*, and `evaluate` makes `len(snaps)` of them per batch in
both the routing pass and the snapshot pass, so it closes the 4.4 s
"repeated `load_state_dict`" item from the not-yet-addressed list. Correctness
is pinned by `tests/test_state_swap.py`: value-for-value equality with
`load_state_dict`, writes into live storage (parameter identity and `data_ptr`
unchanged, so an optimizer holding a reference keeps training the right tensor),
dtype/device mismatch handling for the quantised-snapshot path, and — the one
that actually matters for the continual protocol — that **all weights are
bit-identical after `evaluate` returns**. A snapshot left loaded would turn the
"live weights" protocol into a second oracle run and quietly inflate every
number in the README.

---

## 9. Environment

`import mgn` resolved to the repo only when the working directory was the repo
root. A stale, non-editable copy sat in site-packages alongside the editable
finder, and it had **no `mgn/blocks/` package at all**:

```
python -c "import mgn; print(mgn.__file__)"     ->  ...\New folder (8)\mgn\__init__.py   (repo root)
python <a script in a subdirectory>            ->  ...\site-packages\mgn\__init__.py    (stale)
```

The duplicate was removed. `pip install -e .` in CI keeps only the editable
finder, so CI was never affected — but local experiments were silently running
stale code whenever launched from a subdirectory.

---

## 10. Reproduction

```powershell
python -m pytest tests -q                                  # the enforced guards
python -m pytest tests\test_dynamic_depth_routing.py -q     # depth gates, P2, P3
python -m pytest tests\test_cost_and_autoprotocol.py -q      # cost model, protocol
python -m pytest tests\test_router_cov_per_task.py -q       # per-task covariance
python -m pytest tests\test_auto_equals_oracle.py -q         # A_auto == A_oracle
python validate_ext_in.py                                   # CUDA parity (GPU only)
python probes\probe_router_calibration.py                   # router metric sweep
python probes\probe_router_agreement.py routing_state_after_cifar10.pt
python train_continual_aml.py --tasks mnist,cifar10,fashion --epochs 5,28,16 --replay-per-class 0
```

**The one-off scripts that produced the original measurements are not shipped.**
They lived in `scratch/`, which is git-ignored, so a fresh clone has no
`scratch\ab_arch.py`, no `scratch\diag_arch_grad.py` and no
`scratch\test_p2_p3.py`. Where a check was worth keeping it was rewritten as one
of the tests listed above; the rest were throwaway A/B drivers and are not
claimed as reproducible. The `--steps 60` A/Bs behind §2c can be re-run with the
shipped trainer:

```powershell
python train_continual_aml.py --tasks mnist --epochs 5 --steps 60 --freeze 1 \
       --arch-prior neutral --depth-cost 0.0
python train_continual_aml.py --tasks mnist --epochs 5 --steps 60 --freeze 1 \
       --arch-prior staged --depth-cost 0.02 --arch-lr 3e-3
```

Baseline comparison used a detached worktree at `7a9b920`:

```powershell
git worktree add $env:TEMP\aml_base HEAD
python $env:TEMP\aml_base\ab_base.py --steps 60 --freeze 1
```

**B41 (critical) — B40's fix was measured on a leak, and the real cause was calibration.**

B40 concluded that one pooled covariance was the problem and that per-task
matrices with `alpha = 0.15` would fix it (mean routing accuracy 97.13 % against
48.67 %). The end-to-end run with that default **made stage 2 worse, not
better**: `A[2,1]` fell from 93.96 % to 1.29 %, `route_hist [129, 9871]` — almost
every MNIST image routed to CIFAR-10. The prediction was wrong, and the reason
is the one this log keeps rediscovering: the probe and the trainer were not
measuring the same thing.

Two separate defects, found in that order.

**The probe was fitted on a different split than the trainer.** The trainer ran
the router's clean pass over `el_self`, the **TEST** split — i.e. over the very
images the router is later asked to route. That is a leak, and it is also why
the offline probe (which fitted on TRAIN) disagreed with every run it was
supposed to explain. `probes/probe_router_agreement.py` settles it by comparing
centroids against the ones the run stored:

| task | fit split | ‖μ − μ_stored‖ | relative |
|---|---|---|---|
| CIFAR-10 | test | 0.00007 | **8e-5** |
| CIFAR-10 | train | 0.05715 | 6.4e-2 |

So the probe's embedding path reproduces the trainer exactly; the statistics
differed only by the split. Removing the leak is not free: CIFAR-10's train and
test centroids are **6.4 %** apart, and with honest train-fitted statistics
*every* metric degrades — including the one that shipped:

| routing accuracy, train fit | shared | per-task | per-task + log-det (β=1) |
|---|---|---|---|
| MNIST, 2 tasks | 4.74 % | 3.49 % | 100.00 % |
| MNIST, 3 tasks | 6.74 % | 2.55 % | 100.00 % |
| Fashion, 3 tasks | 2.43 % | 0.00 % | 99.93 % |

The shipped 93.96 % at stage 2 was therefore partly the leak. That is the
uncomfortable finding and it is stated rather than buried.

**With honest statistics, the missing term is calibration, not the matrix.**
Per-task whitening alone is *worse* than the shared matrix, because each distance
is measured in a space whitened for a different task and the scores stop being
comparable — a domain with a large spread wins every query. Adding the
log-determinant of the task's own covariance turns the score into a (tempered)
Gaussian likelihood, which is comparable by construction. Sweeping both knobs
(`probes/probe_router_calibration.py`, train-fit statistics):

| log-det weight β | MNIST | CIFAR-10 | Fashion | mean (3 tasks) |
|---|---|---|---|---|
| 0.00 | 2.55 % | 100.00 % | 0.00 % | 34.18 % |
| **0.50** | **100.00 %** | **92.62 %** | **99.95 %** | **97.52 %** |
| 1.00 | 100.00 % | 72.86 % | 99.93 % | 90.93 % |

and on two tasks: 51.74 / **99.97** / 99.25 at β = 0 / 0.5 / 1. Mixing the
pooled matrix back in only hurts (α = 0 → 97.52, α = 0.5 → 74.95, α = 1 → 64.83),
so the shipped defaults are now `cov_alpha = 0`, `logdet_weight = 0.5`.

**End-to-end confirmation.** Same command, same schedule, same seed, one default
changed (`--router-metric mahalanobis-task --router-logdet-weight 0.5
--router-fit-split train`):

| cell | before (`route_acc`) | after (`route_acc`) |
|---|---|---|
| A[1,1] | 100.00 % | 100.00 % |
| A[2,1] | 93.96 % | **100.00 %** |
| A[2,2] | 100.00 % | 100.00 % |
| A[3,1] | 5.94 % | **89.02 %** |
| A[3,2] | 50.01 % | **95.00 %** |
| A[3,3] | 99.39 % | 99.97 % |
| `A_auto` `A_avg` | 50.31 % | **81.96 %** |
| `A_auto` forgetting | 54.43 % | **6.34 %** |
| `A_oracle` `A_avg` | 86.73 % | 86.19 % |

`A_auto` is now **4.23 points** below the oracle instead of 36.42, and every
stage-2 cell is exactly equal to it. The residual has a direction: after task 3,
10.98 % of MNIST and 5.00 % of CIFAR-10 are routed *forward* to Fashion, never
backward to an older domain. That is the next thing to work on.

The lesson is worth stating as a rule rather than an anecdote: **an offline
probe built on saved checkpoints is only as trustworthy as the data path it
shares with the trainer.** Two of the three probes built here were wrong, and
both were wrong in the direction of over-estimating the fix. `--save-routing`
exists precisely so that the next question is answered by loading what the run
stored and checking agreement first (8e-5), not by rebuilding the statistics and
hoping.

**B42 (medium) — a class default was missed by the B33 fix, and the one caller it
did reach trained a different model from the documented one.**

B33 aligned `norm_type` to `in` in `cnn_block.py` and `hybrid_18.py`. It missed
`hybrid_router.HybridBlock`, which kept `norm_type: str = "bn"`. The search was
"which files take a `norm_type` argument", not "which classes do" — and the class
that was missed is the one every `HybridBlock` consumer goes through.

The production continual runs were **not** affected: `hybrid_18.py:169` passes
`norm_type=norm_type` explicitly. The casualty was
**`train_hybrid_aml.py`**, which constructs `HybridBlock(in, in, num_heads)`
without it and hard-coded `nn.GroupNorm(1, hidden_dim)` in its stem. So the
single-task hybrid trainer built a **BatchNorm/GroupNorm** network while
`train_continual_aml.py --norm in` built an **InstanceNorm** one — two scripts
documented as training the same architecture, training different models, with
nothing in either output saying so. For a claim about a 13-point normalization
effect, that is not a cosmetic inconsistency.

Fixed in both places: `HybridBlock`'s default, and `train_hybrid_aml.py`, which
now takes `--norm {in,bn,gn}` (default `in`), routes it to `make_norm` in the stem
and to the blocks, and is pinned by
`test_train_hybrid_norm_matches_the_cli_default`. The regression guard that
matters is not per-block:

* `test_every_block_default_matches_the_cli_default` walks the four block classes
  and asserts the signature default is `in`. A new block with a different default
  fails the test.
* `test_class_default_actually_builds_instance_norm` builds each one and inspects
  the module, because a signature can agree with a caller that ignores it.

All three are in `tests/test_dynamic_depth_routing.py` (40 tests). No reported
number changes: `train_hybrid_aml.py` is not the source of any figure in the
README, the experiments write-up or the paper, and the continual runs pass the
norm explicitly.

**B43 (low, packaging) — the CUDA extensions could not be cross-compiled.**

`setup.py` gated the extension build on `torch.cuda.is_available()`, so a Docker
image build stage — which by design has no GPU — silently produced an image whose
`mgn.accel` imports fail at run time with a bare `ModuleNotFoundError` instead of
at build time. `FORCE_CUDA=1` now forces the build and prints that it did.

**B44 (low, reproducibility) — the docs pointed at scripts that do not exist.**

`REPRODUCIBILITY.md` and §10 below instructed the reader to run `scratch/ab_arch.py`,
`scratch/diag_arch_grad.py` and `scratch/test_p2_p3.py`. `scratch/` is git-ignored
and has never been shipped, so a fresh clone cannot run any of them. Both places
now list the tests that replaced the checks worth keeping, say plainly that the
original scripts are not shipped, and give the shipped-trainer commands for the
`--steps 60` A/Bs behind §2c. A reproduction command that fails on a clean clone
is worse than no command.

**B45 (medium, honesty) — the paper described a system the repository no longer
contains.**

Three separate claims in the paper were stale against its own later sections:

* a limitation bullet read *"The current system supports only PlainConv and
  ResNet blocks"* — contradicting §1 (tri-modal vocabulary), §6.4 (the tri-modal
  configuration under test) and §7 (the CNN+ResNet+ViT 18-layer hybrid);
* *"AML currently generates a single architecture for the entire batch"* —
  superseded; the executor takes a `[B, K]` per-sample `pi_mask` and routes per
  sample (B35 fixed the benchmark half of this);
* *"all experiments were conducted on CPU"* — superseded by the fused CUDA
  path and its parity gate (B31, B39).

Each is now retracted in place rather than quietly edited, and the space is filled
with the limitation that actually holds: the paper's model equations are the
**legacy** two-mode GroupNorm supernet, the continual experiments use the
**three-mode** InstanceNorm hybrid, and the executor section says so at the top
and spells out the three differences. Two new limitations replace the retracted
ones — the domain router's one-directional error and the single-seed caveat.
`tests/test_paper_latex.py` now fails on the retracted sentences, on a missing
`8{\times}8` token-pooling claim, and on the paper and the shipped normalisation
default disagreeing.

**B46 (high) — the ambiguous retry ran one domain's weights under another
domain's architecture, and it runs by default.**

`evaluate`'s ambiguous path (samples whose router margin is below `route_tau`)
loads snapshot `d`'s weights and restricts the sample to `d`'s prototype rows --
but built `struct_override` from `gates_b = g[d_pred]` / `block_b = b[d_pred]`,
i.e. the structure slot of the router's *argmax*, not of the domain being
retried. Domain `d`'s weights were therefore evaluated with domain `d_pred`'s
layer gates and block modes. `slots[d]` is produced by `collect_structure_slot`
from `d`'s own data with `d`'s own weights, so it describes `d` and nothing
else; the rows and the weights in that call belong to `d`, so the structure has
to as well.

This is not a corner case: `route_tau` defaults to `0.6` and `amb_topk` to `0`,
so every seen domain is retried for every ambiguous sample on every default
run. Nothing caught it because the existing end-to-end test builds a router
whose logits are +/-50, giving a margin of ~1.0, which is never ambiguous at
0.6 -- the branch was structurally untested.

Fixed by building the retry's override from `g[d]` / `b[d]` expanded over the
ambiguous sub-batch; `test_ambiguous_retry_uses_the_retried_domain_s_slot`
records the gates actually handed to the model and asserts each retry used its
own slot (verified to fail on the old code). In the same function, `out` and
`best_out` were `torch.empty`: a row nothing writes is returned as
uninitialised memory. Both are `torch.zeros` now. `best_out` only happened to be
safe because the first snapshot writes every row; `out`'s coverage depends on
`len(snaps) == len(slots)`, which nothing enforces.

**B47 (medium) — the per-sample candidate dispatch dropped the routing weight,
so a sample's output depended on its batch neighbours.**

`HybridBlock.forward` has two hard fast paths. The whole-batch one returns
`cand * mode_weights[:, chosen]`; the per-sample dispatch returned the branch
output unscaled. `all_hard` is `max > 0.99`, not `== 1.0`, so with weights like
`(0.001, 0.004, 0.995)` the same input produced `0.995*y` when its batch agreed
on a mode and `y` when it did not: a ~0.5 % amplitude change that is a function
of batching, not of the model. Measured max abs difference `2.16e-2` on outputs
of order `1e1`. Fixed by scaling each gathered row by its own weight, which
makes the two paths agree. All existing routing tests use exactly one-hot
weights, where `1.0*y == y` and the defect is invisible;
`test_per_sample_dispatch_is_batch_composition_invariant` uses 0.995.

**B48 (medium, measurement integrity) — A_live was handed the true task id
while the README called it the no-task-id baseline.**

`evaluate`'s `pi_mask` is the task's prototype-row range. `A_live` is evaluated
with `pi_mask_j` (`dom_head` is None, so no routing), i.e. it is *told* which
task's read-out to use; `A_auto` must infer it. Every "A_auto is below A_live, so
misrouting costs more than it gains" statement therefore compares a task-aware
baseline against a task-agnostic one. The number is not wrong -- it is the right
number for *task-aware* sequential fine-tuning -- the label was. README, the
paper's table and caption, and `evaluate`'s docstring now say task-aware
explicitly, and the paper gains a paragraph stating that the like-for-like
routing comparison is `A_auto` vs `A_oracle`.
Guarded by `test_live_baseline_is_described_as_task_aware`.

**B49 (medium, measurement coverage) — raising the channel cap made the fused
backward reachable at the production width, and nothing tested that width.**

The v1 guard `256 % C == 0` was a leftover; the v2 kernel walks channels
sequentially and needs only `C <= MB_MAX_C`. Relaxing it to `C <= 1024` made the
fused backward reachable at the shipped `--max-dim 48` for the first time (at 48,
`256 % 48 == 16`). But `validate_ext_in.py`'s `build_case` defaults to `D=64`,
and 64 divides 256: every case in the only numeric suite covering the kernel
satisfied the *old* constraint, so the suite could not have detected a
C-dependent bug even if one existed, and the production width appeared nowhere in
it. The kernel turned out to be correct; the gap was in the evidence, and it was
created by the same change that made the path live. Section `[5]` now gates
gradient parity at non-divisor widths (asserting `256 % D != 0`) and the forward
at the production shape. Measured on the RTX 3050:

```
D= 48 (256 % 48 =  16): fwd=0.00703  worst_grad=0.11178 (noise 0.16759, tol 0.25138)  OK
D= 40 (256 % 40 =  16): fwd=0.00551  worst_grad=0.09837 (noise 0.11463, tol 0.17194)  OK
D= 96 (256 % 96 =  64): fwd=0.00542  worst_grad=0.08898 (noise 0.10927, tol 0.16390)  OK
L=18 D=48 (production) : rel_err 0.18970 (reference noise 0.21123, tol 0.31684)  OK
```

**B50 (low, precision -- measured, not assumed) — `--snap-q8` defaults to 1 and
its quantisation step is global.**

Every stored snapshot's prototype matrix is int8 with **one** scale per matrix
(`scale = max|P| / 127`). On the real checkpoints the maximum is a single element
in a row that `ProtoIsolator` froze after task 0 (`proto_weights[3, 6718] =
+0.2257`, identical in all three snapshots), so later tasks are quantised at a
resolution set by task 0's largest weight; the median non-zero element's
worst-case relative error is 2.6 % and 1.4 % of elements quantise to zero.
`probes/probe_snapshot_quant.py` evaluates the stored matrices exactly as the
protocol does, then re-quantises the same values to 16 bits: **accuracy is
identical for every snapshot** (MNIST 98.59 % / 98.59 %; the Fashion artefact's
three snapshots 11.65/11.65, 4.31/4.31, 92.51/92.51 on 10 000 test images). So
the default is defensible, but it was an unstated lossy step inside the headline
protocol, and now it is measured rather than assumed.

**B51 (low) — `--max-dim` below 16 silently disabled width routing.**

`d_lengths = clamp(sigmoid(...) * max_dim, min=16.0)`, so when `max_dim < 16` the
clamp binds for every sample: `dim_masks` saturates near 1.0 (every channel
always active) while `dim_weights = d_lengths / max_dim` exceeds 1.0 and
over-reports effective width. Shrinking the model looked like it worked. A
`UserWarning` at construction now says so, and
`test_max_dim_below_the_width_floor_is_not_silent` asserts both the warning and
the saturation (`max_dim = 48`, the default, stays quiet).

**B52 (medium, honesty) — the paper's routing equation stated a coefficient the
code had stopped using and omitted one it needs.**

The equation carried `\alpha = 0.15, measured`. `--router-cov-alpha` defaults to
0, and the sweep in the trainer shows routing degrades *monotonically* in alpha
(97.52 / 85.50 / 74.95 / 64.83 % at 0 / 0.25 / 0.5 / 1.0) because the pooled
moment mixes three feature spaces (B40) -- so 0.15 was a wrong fix for a
misdiagnosis, presented as measured. The equation also omitted
`\beta \log|S_t|`, the term the paper's own ablation table shows is required
(34.18 % without it, 97.52 % at beta = 0.5). Both corrected in place and named as
withdrawn; the equation is now `\label{eq:router}` with the shipped
alpha = 0 and beta = 0.5.
`test_router_equation_matches_the_shipped_defaults` cross-checks the parser.

**B53 (low, docs) — section 3.1's command did not reproduce section 3.1's
table.**

The block passed `--ewc-lambda 100`; two lines below it the text says the shipped
default is `--ewc-lambda 0`, and EWC-on is the ablation the limitations section
reports as *increasing* forgetting. The block now carries the canonical default
command and names the EWC run as the ablation it is.

**B54 (medium-low, disclosed protocol asymmetry) — two of the three tasks are
trained without augmentation, because the speed flag also switches it off.**

`get_task_data` takes `aug="basic"` and `fast_cache=True`, and the cached path
for MNIST/FashionMNIST resizes and normalises once and then reuses the tensors
unchanged. `aug` is read only by the CIFAR-10 branch. So in the default
configuration of the production run:

* MNIST (task 1): `RandomCrop`/`RandomHorizontalFlip` never applied;
* CIFAR-10 (task 2): `RandomCrop(32, padding=4) + RandomHorizontalFlip` always;
* Fashion-MNIST (task 3): same as MNIST.

The `aug` argument therefore looks like it controls augmentation for the whole
loader and in fact controls it for one of three datasets, and the flag that turns
the augmentation off is documented as a *performance* option. The non-cached
MNIST/Fashion branches do apply `RandomCrop(32, padding=2)`, so
`fast_cache=False` changes the data the model sees, not only its speed -- which
means the MNIST and Fashion columns of any run are not comparable to a
non-cached reproduction of "the same command". This is not changed here: it
would move reported numbers, and the honest fix is a re-run with a single,
explicit augmentation policy. It is stated in `get_task_data`'s docstring and in
open item 10 below.

**B55 (medium, packaging) — the release artifacts had never been built, and both
were wrong, in opposite directions.**

The PyPI path was added without ever producing an artifact. Building both on the
committed tree: the wheel (`aml_framework-2.0.0-py3-none-any.whl`, 44 entries)
carried the repository's own kernel tests and benchmarks
(`mgn/tests/benchmark_pipeline.py`, `benchmark_speed.py`, `test_backward.py`,
`test_fused_conv.py`), while the sdist (89 members) carried **no `mgn/csrc` file
at all** -- and shipped the whole top-level `tests/` tree plus a stale
`aml_framework.egg-info/` left behind by an editable install. So the wheel is
larger than the library and the sdist cannot rebuild the library's CUDA kernels;
the documented `FORCE_CUDA=1 pip install --no-build-isolation
./aml_framework-2.0.0.tar.gz` dies on a missing `mgn/csrc/bindings.cpp`. Fixed
with an explicit `MANIFEST.in` (ship the csrc sources and the quickstarts; prune
`tests`, `data`, the paper, `build/`) and
`packages.find.exclude = ["mgn.tests*", "mgn.csrc*", "tests*"]` with
`namespaces = false`. After: wheel 42 entries, no tests; sdist 76 members, all 14
csrc files present, no tests / paper / checkpoints / datasets. The publish
workflow now asserts both shapes before any upload.

**B56 (medium, latent, packaging) — `include-package-data` defaults to true under
`pyproject.toml`, so writing the sdist manifest put the CUDA sources one flag
away from being inside a `py3-none-any` wheel.**

A `pyproject.toml`-configured setuptools defaults `include-package-data` to
**true**, and that copies every manifest-listed file *inside a package directory*
into the wheel. `MANIFEST.in` lists `mgn/csrc/*.cu|*.cpp|*.h|*.cuh`, all under
`mgn/`, so the default stopped being harmless the moment the manifest existed.
Measured on a throwaway copy with the flag left at its default and a fake
`mgn/accel/_C.cp312-win_amd64.pyd` planted in the tree: the wheel goes 42 -> 56
entries, gaining all 14 `mgn/csrc` sources as package data. The planted `.pyd`
does **not** enter, in either configuration, because `global-exclude *.pyd` keeps
it out of the manifest (the pre-fix 44-entry wheel was likewise binary-free). The
csrc payload is bloat, not breakage -- but the flag is pinned to `false`, guarded
by `tests/test_packaging.py`, and re-checked against the real artifact by the
publish workflow before the upload.

**B57 (low, packaging) — a plain `pip install .` skipped the CUDA extensions
silently.**

`pip install .` builds in an isolated environment with no torch, so `import torch`
fails in `setup.py` and both extensions are skipped without a word. A GPU user
doing the obvious thing gets a working-but-slower install and no indication that
`mgn.accel._C` exists or how to obtain it -- the same silent path the docstring
reserved for the Docker case, reached by the ordinary case. Measured here: the
checkout's `_C`/`_Cin` import, and the wheel install reports
`not built (ModuleNotFoundError) -- optional, using the PyTorch fallback`.
`setup.py` now prints one line naming what was skipped and the two commands that
build it, and accepts `AML_BUILD_CUDA=1` alongside `FORCE_CUDA=1`.

**B58 (low, metadata) — the license field was already deprecated, and pairing it
with a license classifier is now forbidden.**

`license = { file = "LICENSE" }` printed a deprecation banner on every build
("update your project ... by 2027-Feb-18"), and PEP 639 forbids combining it with
a `License ::` trove classifier. Replaced with the SPDX string `license = "MIT"`
plus `license-files = ["LICENSE"]`, with `setuptools>=77` as the build
requirement that makes both legal. `python -m build` is now warning-free and
`twine check --strict` passes on both artifacts.

**Verification of the release chain (measured, not asserted).** Artifacts built in
a clean virtualenv (Python 3.12, setuptools 84.0.0, build 1.6.1);
`python -m build` exits 0 (the "Exited with code 1" PowerShell reports is a
stderr/pipeline artifact, reproduced with a command that only writes to stderr).
The wheel was installed with `--no-deps` into a second venv that has torch but no
`aml-framework`, and `mgn` / `hts_lib` resolve to the wheel's `site-packages`
rather than to the checkout; the new `aml-selfcheck` console script exits 0 on the
GPU (kernels present) and on a CPU-only run (kernels absent, reported as
optional); `twine check --strict` PASSED on the wheel and the sdist. What is *not*
measured: nothing has been uploaded, so the OIDC exchange with a real index is
still untested (open item 11).

**B59 (report triage, part 1: Python) — an external report of ~23 defects was
read against the tree rather than accepted on its word. The real Python defects
are fixed and pinned by tests; the "critical" labels mostly do not survive the
check.**

Measured, not asserted: `ruff check .` clean; `pytest -q` **275 passed + 45
subtests** (was 256 + 45), 10 warnings (the `max_dim=8` fixtures from B51). Each
item below was reproduced against the current tree before it was touched, and
the reachability is stated because several are not on any documented path. The
remaining report items described code that had already been fixed before the
report was filed, or paths that do not exist in this tree; they were checked
against the source and are not re-listed.

- **The fused executor silently dropped the HtS low-rank adapters (silent wrong
  answer).** `ModelGenerator` folds the adapters into the generated weights, and
  `mgn/executor.py` re-derives those weights from `(prototypes, pi)` in the fused
  CUDA path, which has no adapter term. `TaskNetwork._fusable` intended to refuse
  such models but tested `hasattr(model, "hts_adapters")` on the `GeneratedModel`
  the generator returns, which never carries that attribute, so the check was
  always False. `GeneratedModel` now carries `has_hts_adapters`, the generator
  sets it, and `_fusable` refuses. Reachable through
  `experiments/benchmark_conv_aml.py --fused-executor` and library users of
  `TaskNetwork(fused=True)`; **not** the production trainer, which builds
  `AcceleratedCVNetwork`. Consequence, disclosed rather than hidden: every
  conv-mode generator attaches adapters, so that flag now falls back to eager.
  A fused kernel that computes the adapters is future work (open item 13); the
  parity tests delete the adapters so the kernel stays exercised.
- **`contrastive_architecture_loss` read a key conv models do not have.** It took
  its device from `weights["fc1.weight"]`; conv models expose `c1.weight`/
  `out.weight`, so it raised `KeyError`. The device now comes from
  `structure.layer_gates`. Only reached when `lambda_contrastive > 0` (default
  0.0), which is why no shipped run tripped it.
- **`HtSResidualBlock` built invalid GroupNorm group counts.** `min(8, out_d//4)`
  is 7 for `out_d=30` (`30 % 7 != 0` -> `num_channels must be divisible by
  num_groups`) and 0 for `out_d < 4`. Two helpers keep every already-working
  width bit-identical and pick a true divisor otherwise: `_norm_groups` for
  gn2/shortcut (24/48/96 -> 6/8/8, unchanged) and `_stem_groups` for gn1, which
  reproduces its historical constant `GroupNorm(4, C)`. A single shared helper
  preferring `min(cap, C//4)` would have quietly moved C=8/12 from 4 groups to
  2/3 and changed every HtS number; the regression test pins both.
- **`VisionTransformerBlock` did not reconcile `num_heads` with the channel
  count.** `num_heads` is a free argument and `--max-dim` a free CLI value, and
  nothing checked that the width divides by the head count; an indivisible pair
  failed deep inside `forward`. It is rejected at construction now.
- **`train_continual_aml.py`.** Four independent fixes, all on paths the
  documented command reaches only in their crash cases, so none of them moves the
  published numbers:
  * `ambiguous_candidate_mask` scatters into an `[B, n_domains]` tensor with
    STRUCTURE-SLOT indices, and `len(slots)` is not forced to equal
    `len(snaps)`; a top-k column `>= n_domains` was an out-of-bounds `scatter_`
    (a CUDA device-side assert). Invalid slots are `-inf`'d before the top-k and
    `k` is capped. With `--amb-topk 0` (the default, and the production command)
    the function returns all-ones exactly as before, so the fix is a strict
    no-op there.
  * `score_per_task`'s Mahalanobis branches looped `range(T_cur)` over
    `z_per_task`: an `IndexError` when `num_tasks` (a monotonic counter) exceeded
    the stored snapshots. Bounded like the cosine branch.
  * `evaluate` treated `snaps=[]` as "has snapshots": the per-snapshot loop did
    nothing, `outs_conf`/`best_out` stayed empty and `ref_o.shape[1]` raised
    `AttributeError` on `None`. An empty list now means no snapshots, and a batch
    whose every row routes to a domain without a snapshot falls back to the live
    weights.
  * `--auto-live-thresh`'s live fallback ran the LIVE weights under the rejected
    domain's gate/block-mode slot — architecture from one model, weights from
    another. It runs the live model under its own structure now. Off by default
    and absent from the production command, so it does not alter the documented
    numbers.
  * `_DATA_CACHE` was keyed on the dataset name alone, so calling `get_task_data`
    twice for the same name with different roots returned the first root's
    tensors. Keyed on `(name, abspath(data_root))`.
- **The static extractor was not a faithful replica of its mother.**
  `extract_child_truedynamic` read `pi` from `ood_router.get_pi`, but the mother
  routes with `prototype_selector`; `get_pi` additionally partitioned `K=12`
  rows against `rows_per_task=24`, so with one active task the row sums were 0.5
  and later tasks were zeroed. It uses the mother's own router now, and `get_pi`
  partitions the rows exactly (`sum_k pi_k = 1`, every active task's rows
  non-zero). `StaticLayer` hardcoded `nn.GroupNorm(1, C)` regardless of the
  mother's `norm_mode`, so a child extracted from an InstanceNorm mother
  normalised differently from the model it came from; it builds the matching op
  now (`_norm_layer`). Reachable through `quickstart_aml.py`, which extracts a
  `ConvAML_CIFAR_TrueDynamic` child. The paper's appendix listing is updated to
  the shipping source.
- **`fused_autograd._norm_bwd`** looked for `instance_norm_bwd` only in `_C`, so
  a `_Cin`-only build fell back to ATen even though `_Cin.instance_norm_bwd`
  existed and was never referenced. It dispatches lazily on either module now
  (the other half of the report's item is open item 14).
- **`mgn/trainer.py`.** The stability snapshot rebuilt a hand-listed
  `ModelStructure` and dropped `use_conv`/`in_channels`; it copies the whole
  structure now. The `torch.compile` dry run fed `VisionModelGenerator` a flat
  vector, so `--compile` silently disabled itself for a 4D generator.

Checked and **refuted, left unchanged**: the report's memory-buffer item.
`EpisodicMemoryBuffer` mixes `torch.randperm` with `random.shuffle`, but
`hts_lib.utils.set_seed` seeds `random`, `numpy` and `torch` together, and the
buffer is only reached through it (`experiments/benchmark_hts_baselines.py`
imports that `set_seed`), so exemplar selection already reproduced. Rewriting the
shuffle to the torch RNG would not have added reproducibility and would have
moved the HtS baseline numbers, so the file is untouched;
`tests/test_hts_core.py` now pins the actual contract.

**B60 (report triage, part 2: CUDA) — three latent memory/launch defects in the
native kernels, verified by rebuilding.**

Measured, not asserted: `python build_ext.py` -> `_C: OK (11 exports)`,
`_Cin: OK (4 exports)`; the full suite above runs against the rebuilt `.pyd`;
and `validate_ext_in.py` (the B39 forward/gradient parity gate) still ends
`done -- all gated checks passed` at every gated width (GN1 rel_err 0.00409 vs
tol 0.05000; L=18 D=48 production rel_err 0.18970 vs tol 0.31684), so the widened
indices and the contiguity fix do not move the fused numerics.

- `in/bindings_in.cpp:in_maxpool2x2_bwd` passed `data_ptr()` straight through
  from the incoming tensors, and `dh_out` is an upstream autograd gradient that
  can be a non-contiguous view (a sliced gradient, a transposed activation) while
  the launcher indexes it as flat `[B, C, H, W]`. The GN twin
  (`maxpool2x2_bwd`) already normalised with `.contiguous()`; this one did not,
  so a non-contiguous gradient silently indexed the wrong memory. Normalised to
  match.
- `bindings.cpp:mix_bwd_v2_b` had no input validation at all, while its sibling
  `mix_bwd_v2_a` checks contiguity/CUDA and the `MB_MAX_C` bound. A
  non-contiguous or CPU tensor reached the kernel through a raw `data_ptr`. The
  same `TORCH_CHECK`s are added.
- The four maxpool/norm kernels computed the per-IMAGE base as
  `int base = ((b * C + c) * H + h0) * W_` (or `(b * C + c) * spatial`). That
  scales with `B*C*H*W` — about 4x the flat output index the grid is sized for —
  and could overflow before the flat index did. Widened to `long` in
  `fused_pipeline.cu`, `fused_backward.cu`, `in/fused_backward_in.cu` and
  `in/fused_pipeline_in.cu`. No behaviour change at any reachable shape.
- `kernels/batched_hyper_linear.cu`'s three launchers used the implicit legacy
  default stream and never checked the launch, and `grid.z = batch` is capped at
  65535. On a non-default current stream the default stream's implicit
  serialisation does not apply, so the kernel could race the surrounding ops (and
  this is what breaks under CUDA graph capture); an oversize batch silently
  covered only the first 65535 elements. Now on `getCurrentCUDAStream()` with
  `cudaGetLastError()` checked and a loud throw past 65535.
- **Dead code.** `kernels/fused_chain_conv.cu`, `kernels/fused_post_conv.cu` and
  `kernels/fused_norm_relu_gate.cu` were compiled into `_C` but referenced by
  nothing: their exports do not appear in `bindings.cpp`,
  `fused_norm_relu_gate.cu` existed only to satisfy `fused_chain_conv.cu`, and
  even the latter's declared `tiled_conv_fwd_launcher` was never called. Removed
  from `setup.py` and `build_ext.py` and deleted (486 lines). Verified after the
  removal: the sdist still builds (78 members, the 11 remaining `mgn/csrc` files
  present, none of the three removed files present), and `publish.yml` looks for
  `mgn/csrc/bindings.cpp` by name rather than a count, so the release path is
  unaffected. (B55's "14 csrc files" was accurate when it was written; the count
  is 11 now.)

**B61 (report triage, part 3) — reported defects examined and deliberately NOT
changed, so that "not fixed" is not mistaken for "not read".**

Each would move training or evaluation numerics for no correctness gain, or is
unreachable. The report's severity labels are quoted, not endorsed.

- **`drop_last` when fitting the router** (the report's own estimate: 0.16 %).
  That is below the 0.26-point same-seed spread already documented (open item 4)
  and the fit-split question is open item 1. Changing it would move the routing
  calibration every published number sits on, for an effect smaller than the
  measurement noise.
- **`GradScaler` under bf16, and the missing clip.** Under bf16 autocast the
  scaler is a numerical no-op except for inf-skip; but `hts_lib`'s autocast
  defaults to **fp16**, where the scaler is required, so it stays. Adding the
  clip the report asks for in `hts_lib` would move its numbers without touching
  the AML ones; left as a deliberate asymmetry.
- **`attn_dim = out_c // 4`** (`mgn/executor.py`) is 0 for `out_c < 4`.
  Unreachable: `hybrid_18.py` rejects `max_dim < 16`, and the reachable half of
  the same class of bug (`num_heads` divisibility) is fixed in B59.
- **AdamW momentum on frozen rows.** The reported concern is stale momentum on
  weights that are "worn". `restore()` rewrites those weights bit-exactly each
  step, so the momentum has no practical effect; changing the optimiser would
  move the numbers.
- **`mix_bwd_v2_a/b` are not exported by `_Cin`.** Only the `_norm_bwd` dispatch
  half is fixed (B59). Exporting the mix backward from `_Cin` means adding
  `fused_backward.cu` to its source list — a build-shape change — so it is open
  item 14, not silently done.

**B62 (release, verified against a real index) — the trusted publisher works:
`aml-framework 2.0.0` is on TestPyPI and installs.**

`publish.yml` was dispatched with `target=testpypi` (run 38734470869 on this
commit): `verify` (suite) -> `build` -> `publish-testpypi` all passed and
`publish-pypi` was skipped as intended, so the OIDC exchange with a real index --
the one step B55--B58 could not test -- is now exercised. TestPyPI serves
`aml-framework 2.0.0` with `aml_framework-2.0.0-py3-none-any.whl` and
`aml_framework-2.0.0.tar.gz`. Installed from the index into a venv that has torch
but is not the checkout (`pip install --no-deps -i
https://test.pypi.org/simple/ aml-framework==2.0.0`): `mgn`/`hts_lib` resolve to
`site-packages`, `mgn.accel._C` is reported optional and absent, the supernet
forwards and backprops, `hts_lib` forwards, and `aml-selfcheck` exits 0. The
artifact gates and the console-script entry point are therefore proven against a
real index, not just locally.

What is still **not** exercised: the pypi.org publish itself -- a `v2.0.0` tag, or
`workflow_dispatch target=pypi` -- which uses the same workflow and needs the
like-named trusted publisher on pypi.org (open item 11).

**B63 (new capability) — a task registry, so a sequence is no longer locked to
10 classes and 32x32.**

The trainer used to hardcode three things: the task list (`mnist`, `fashion`,
`cifar10`), the class count (`num_classes=10`, baked into the flat prototype head
as `16*max_dim*num_classes + num_classes`), and a 32x32 input. That made
"CIFAR-10 -> CIFAR-100 -> Tiny ImageNet" impossible without a rewrite, and it
meant the documented `--tasks mnist,fashion,cifar10,svhn,cifar100` command in
`experiments/README.md` had never worked -- it raised `ValueError` on the first
unknown name (`svhn`), and the `--max-batches` flag in the same block does not
exist either (it is `--steps`).

`mgn/datasets.py` is now the single source of truth: `TASK_SPECS` gives each task
its class count, native input side and normalisation, and `--tasks` accepts any
of `mnist`, `fashion`, `cifar10`, `cifar100`, `tinyimagenet`. Three choices are
load-bearing:

* **One shared head, per-task class ranges.** The head is `max(n_classes)` wide
  (200 for CIFAR-10/100/Tiny ImageNet) and a task only ever supervises and reads
  back the prefix `[0, n)`. A per-task head width was rejected: it would change
  `get_prototype_layout`'s row width, and with it every `wo`/`bo` offset the
  fused CUDA kernels index. `class_range_ce` reduces to the old
  `F.cross_entropy(outputs, targets)` whenever every sample shares the full head
  width, and `evaluate(n_classes=None)` keeps the old whole-head argmax, so the
  10-class protocol is **bit-identical** -- asserted in `tests/test_task_registry.py`.
* **Native resolution per task.** The trunk does not depend on the input size:
  stride-1 convs with fixed pooling at layers 5/11, an `AdaptiveAvgPool2d((4,4))`
  classifier head and an `AdaptiveAvgPool2d((2,2))` encoder, and the ViT block
  pools tokens to <= 8x8 whatever the input. So a 32x32 and a 64x64 task share
  one trunk and one router, and Tiny ImageNet stays at its native 64x64 instead
  of being downsampled to 32.
* **`--data-root` may be an `os.pathsep` list.** Each task uses the first root
  that already holds it, which is what a Kaggle notebook needs (each dataset is
  mounted under its own `/kaggle/input/<slug>`). A single root is returned
  unchanged, so existing runs are unaffected.

`TinyImageNet` reads the directory release directly (torchvision has none),
tolerating `val`/`valid`, an optional `images/` sub-step, per-class folders or
`val_annotations.txt`, and numeric class-folder names (ordered numerically). It
was validated against the export actually in `data/tiny-imagenet`: 100,000 train
/ 10,000 val images, 200 classes, 500 per class. A Kaggle 2xT4 walkthrough is in
`notebooks/kaggle_continual_aml.ipynb`.

Evidence: `ruff` clean; `pytest -q` -> 296 passed; a real 3-task smoke run
(`--tasks cifar10,cifar100,tinyimagenet --steps 8`) on the local data. The
CIFAR-100 / Tiny ImageNet cells are **not** claimed as results: the README's and
the paper's numbers remain the MNIST/CIFAR-10/Fashion protocol.

---

**B64 (measurement) — the per-task snapshot is 81 % a copy of the 18-layer
trunk, while the framework already holds a ~0.5 KB/task structural
fingerprint.**

Read off the artifacts already on disk (`aml_continual_after_*.pth`,
`routing_state_after_*.pt`) by a CPU probe; no training was run.

Per-task snapshot, production 10-class configuration, key by key:

| key | MB |
|---|---|
| `enc` | 0.401 |
| `layers` | 5.854 |
| `ln` | 0.006 |
| `protos` | 0.092 |
| `sel` | 0.805 |
| `stem` | 0.005 |
| `struct_heads` | 0.045 |
| **total** | **7.209** |

So **81.2 %** (5.854 / 7.209) of the per-task snapshot is a full copy of the
18-layer trunk.

**Correction (2026-10-16): the router is a second O(T) term, and B64 read its
3-task total as a one-off cost.** The state_dict total of a three-task run is
4.209 MB, but that is not O(1): `TaskCentroidRouter.__init__` allocates
`centroids`, `task_counts`, `cov_task`, `zsum_task` and `cov_n_task` as
`(max_tasks, ...)` and `train_continual_aml.py` passes `max_tasks = T`
(`latent_dim` = 512 for `hybrid18`), so each holds `T` slots -- `cov_task` alone
is 512 x 512 x 4 B = **1.048576 MB per task**. Measured split by the probe
(`--dir <run>`, 10-class production run): once **1.051 MB** (`cov`, `zsum`,
`cov_n`, `num_tasks`), per task **1.053 MB** (`cov_task` plus the small vectors).

Total per-task state is therefore **8.263 MB**, not 7.209 MB, and the file sizes
quoted below are the *cumulative snapshot list* plus that state -- they grow
because `snaps[]` is saved inside the routing state, not because the buffers do:
`routing_state_after_mnist.pt` 11.536 MB (1 snapshot), `after_cifar10` 18.861 MB
(2), `after_fashion` 26.184 MB (3), i.e. +7.32 MB per task, which is the 7.209 MB
of snapshot tensors plus ~110 KB of pickle framing. The T = 10 figure moves with
it: 1.051 + 10 x 8.263 = **83.7 MB**, not 76.3 MB. Only `centroids` and the
per-task scalars are irreducible (2 KB/task); `cov_task` is O(T) *up to*
`max_tasks` and would be reallocated beyond it, so the 1.05 MB/task is a design
choice, not a requirement. The probe now computes this split rather than
asserting it.

**Class-count scaling (2026-10-16).** The probe reads whichever run wrote the
artefacts it is pointed at, and the repo root currently holds two of them -- a
3-task 10-class run overwrote `mnist`/`fashion`, a 3-task 200-wide-head run
overwrote `cifar10`/`cifar100`/`tinyimagenet` -- so its default output was
silently blending configurations. The probe now detects that (`stage` must equal
`len(snaps) - 1`) and skips the storage model rather than blending. From the
single-run artefacts: the class-independent part of the snapshot is 7.117 MB and
the prototype rows add 9.2 KB per class (`proto_weights` measures (12, 7690) at
ten classes and (12, 153800) at a 200-wide head; int8 in the snapshot gives
0.092 MB and 1.846 MB). The snapshot is therefore 7.209 MB at ten classes and
8.962 MB at 200, and the trunk copy falls from 81.2 % to 65.3 % of it -- the
cached-weights criticism gets *worse* as the class count grows, not better.

The counter-number sits in the same artifact. `slots[i]`, returned by
`collect_structure_slot`, is only `gate[max_layers]`, `bm_res[max_layers]` and
optionally `bm_mode[max_layers, 3]` -- about 90 float32, i.e. **~0.4 KB of
tensors**, measured at 0.001-0.002 MB per task in the saved state. The framework
therefore already stores a per-task *structural* fingerprint roughly four orders
of magnitude smaller than the weights it stores next to it.

Three consequences, stated plainly:

1. **The "parameter caching" criticism is factually correct for the default
   configuration.** `A_oracle` is documented as "Full 18-layer snapshot retrieval
   with task identity", so its zero forgetting is bought by caching weights, not
   by structure selection. The `--shared-backbone 1` mitigation (3A) removes the
   trunk copy and drops per-task state to 1.355 MB, but it is lossy and has only
   been measured at `--steps 50` on a single seed.
2. **The cost is O(T) twice over.** Storage is `1.051 MB + T * 8.263 MB` (83.7 MB
   at T = 10; 1.053 MB of that per-task term is the router, see the correction
   above), and evaluation is also linear: `--amb-topk 0` "loads ~1.5M params
   per domain per batch", and `--route-per-task 1` costs `len(snaps)` encoder
   forwards per batch.
3. **The question worth asking is not "how do we drop the snapshot" but "can a
   task be served from `slots[t]` alone (~0.5 KB) on a shared trunk held by an
   O(1) mechanism?"** That is a ~5000x cut and would retire the criticism
   entirely. Whether it survives is empirical and, at the full schedule, still
   unmeasured.

`REPRODUCIBILITY.md` (3A) and `experiments/README.md` already record the
7.2 MB -> 1.2 MB change; this entry adds the key-level breakdown that shows
*where* the bytes are, and the slot number that shows what the bytes are not
buying.

---

**B65 (measurement) - the proposed "0 KB per task" mechanism is ~5 MiB per task,
and its advantage over a random projection is the only claim that survives.**

A separate proposal was put forward to replace the per-task weight snapshot with a
per-task *subspace*: decompose each trunk weight matrix into a basis `U` and
residual coefficients `V`, and consolidate `U` with Orthogonal Gradient
Projection (OGP) so later tasks cannot move the directions task `t` relied on.
Reported cost: **0 KB per task**.

The original script asserted rather than measured its headline row. It set the
snapshot arm's accuracy equal to the freshly-trained accuracy and its forgetting
to exactly `0.0`, so the reference was true by construction; and its gradient hook
read `inp[0]` of the convolution while the convolution actually consumes
`res = F.gelu(self.gn1(x))`, so the projection it measured was not the tensor the
gradient flows through. `experiments/exp_basis_ogp_corrected.py` re-runs the same
idea with both defects fixed and a control arm added. Task-IL oracle, one seed per
arm, MNIST -> Fashion-MNIST, on the same small HtS model (`d1,d2,d3 = 24,48,96`,
5.60 MiB total).

| variant | per-task | fixed | MNIST | FASHION | forgetting |
| :-- | --: | --: | --: | --: | --: |
| naive (no mechanism) | 0 | 0 | 38.49 % | 89.46 % | 60.62 % |
| per-task snapshot | 5.621 MiB | 0 | 99.11 % | 89.95 % | **0.00 %** |
| ogp, original variant | 4.585 MiB | 0 | 97.23 % | 89.32 % | 1.88 % |
| ogp, "corrected" variant | 5.173 MiB | 0 | 92.68 % | 88.42 % | 6.43 % |
| **random projection (control)** | 5.173 MiB | 0 | 69.05 % | 88.86 % | 30.06 % |
| freeze the basis, store nothing | **0.000 MiB** | 0 | 94.54 % | 83.40 % | 4.57 % |
| ogp, corrected + kernel lock | 5.173 MiB | 0 | 97.84 % | 88.33 % | 1.27 % |

Four things follow, and the second one is the uncomfortable one:

1. **"0 KB" is not 0 KB.** The basis is built from task `t`'s own weights and is
   passed into `train_task` for every later task, so it is per-task state that has
   to be kept. The arms that genuinely store nothing per task are `naive` and
   `freeze_basis`.
2. **The mechanism does not work for the stated reason.** The variant the theory
   calls *incorrect* (4.585 MiB, 1.88 %) beats the variant the theory calls
   *correct* (5.173 MiB, 6.43 %). If the explanation were the binding constraint,
   the ordering would be the other way round. What the control does establish is
   that the gain is not an artifact of picking any subspace at all: a random
   projection of the same rank and the same byte cost gives 30.06 % forgetting,
   i.e. the subspace choice is doing real work. So the mechanism is *empirical*,
   not explained.
3. **The storage/retention trade is inverted.** Sweeping the retained-variance
   threshold on the corrected variant: 0.90 -> 1.951 MiB with 1.22 % forgetting,
   0.95 -> 3.321 MiB / 1.91 %, 0.98 -> 5.170 MiB / 5.37 %, 0.99 -> 6.298 MiB /
   4.44 %, 1.00 -> 8.365 MiB / 4.13 %. More basis buys *worse* retention. "Keep
   98 % of the variance" is the wrong objective for this mechanism.
4. **The constraint leaks.** The diagnostic `|dW . U| / |dW|`, i.e. how much of
   the update lies inside the protected subspace, is 0.83 for naive and 0.42 for
   the corrected variant -- identical to the basis-only figure, so the lock is not
   binding. Four gradient paths are never projected (`task_ctrl`, `gn1`/`gn2`,
   `shortcut`, the per-task stems and heads) and AdamW momentum plus weight decay
   act *after* the projection.

Two failures of the original protocol are worth recording because they are what
made the first table look stronger than it was. The reference "exact by
construction" row was unreachable in fact: the rank is clamped to
`min(k, U.shape[1] - 1)`, so one direction is always left out. And the naive
baseline is not reproducible at one seed per arm -- across three runs of the
identical configuration it lands at **34.37 %, 38.49 % and 53.27 %**, a
19-point spread. Any comparison against a ±19-point baseline needs the seeds
reported, which is why `experiments/exp_zero_per_task.py` now records them.

The verdict on this mechanism as a way to reach 0 bytes per task: it is a working
anchor (better than nothing, and better than a random subspace) but it is not a
free one, it is not explained, and it does not dominate freezing the basis, which
stores a literal zero and reaches 4.57 % forgetting. See B66 for the direct
0-bytes/task test.

---

**B66 (measurement) - 0 bytes per task is reachable, it is not free, and the
price is set by the size of the domain gap.**

B64 established that the per-task snapshot is 81.2 % a copy of the trunk and that
the structural fingerprint stored beside it is ~0.4 KB. `--snapshot-mode
structural` makes the weight-free per-task state a shipped option, so the
question "can a task be served without a per-task weight copy?" becomes a
measurement instead of an argument.

`experiments/exp_zero_per_task.py` is the test bed, and it is deliberately
adversarial to the *mechanism* rather than to the framework: one shared stem, one
shared 20-wide head, the two tasks occupying disjoint class ranges of that single
head, and cross-entropy restricted to the task's own range. The per-task
parameter count is therefore exactly **zero by construction** -- not zero by some
accounting convention -- and the mechanisms can differ only in what they do to the
shared 5.6 MiB trunk. Two task pairs are run, differing solely in the domain gap:

* **easy pair** MNIST -> Fashion-MNIST (same modality, same 28x28 resolution);
* **hard pair** MNIST -> CIFAR-10 collapsed to one channel (colour, 32x32, natural
  images, forced through the same stem, so the trunk has to re-represent the
  input distribution).

ACC below is the mean of the two tasks' final accuracies, i.e. the quantity a
deployment actually cares about; "forget" is the drop on task 1.

Easy pair (MNIST -> Fashion, 3 epochs/task):

| mechanism | per-task | fixed | MNIST | FASHION | forget | ACC |
| :-- | --: | --: | --: | --: | --: | --: |
| per-task snapshot | 5.621 MiB | 0 | 98.99 % | 90.21 % | 0.00 % | **94.60** |
| freeze trunk (leave norm/head) | **0** | **0** | 93.41 % | 84.59 % | 5.64 % | **89.00** |
| Fisher mask, 90 % frozen | 0 | 0.172 MiB | 88.93 % | 87.90 % | 10.12 % | 88.42 |
| EWC, int8 Fisher | 0 | 2.798 MiB | 84.74 % | 89.61 % | 14.31 % | 87.18 |
| Fisher mask, 50 % frozen | 0 | 0.172 MiB | 81.67 % | 89.95 % | 17.38 % | 85.81 |
| random mask, 90 % frozen | 0 | **4 B** | 81.68 % | 88.09 % | 17.37 % | 84.89 |
| random mask, 50 % frozen | 0 | 4 B | 70.29 % | 90.32 % | 28.76 % | 80.31 |
| EWC fp32, lambda 1000 | 0 | 11.191 MiB | 70.03 % | 89.56 % | 29.02 % | 79.80 |
| EWC fp32, lambda 100 | 0 | 11.191 MiB | 54.06 % | 90.24 % | 44.99 % | 72.15 |
| naive, nothing retained | 0 | 0 | 36.79 % | 90.36 % | 62.26 % | 63.58 |
| freeze everything | 0 | 0 | 99.05 % | 5.97 % | 0.00 % | 52.51 |

Hard pair (MNIST -> CIFAR-10 grayscale, 3 epochs/task):

| mechanism | per-task | fixed | MNIST | CIFAR-G | forget | ACC |
| :-- | --: | --: | --: | --: | --: | --: |
| per-task snapshot | 5.621 MiB | 0 | 99.06 % | 58.08 % | 0.00 % | **78.57** |
| **random mask, 90 % frozen** | **0** | **4 B** | 73.58 % | 43.78 % | 25.48 % | **58.68** |
| Fisher mask, 90 % frozen | 0 | 0.172 MiB | 73.83 % | 43.03 % | 25.23 % | 58.43 |
| freeze trunk (leave norm/head) | 0 | 0 | 80.64 % | 33.60 % | 18.42 % | 57.12 |
| Fisher mask, 50 % frozen | 0 | 0.172 MiB | 56.13 % | 55.26 % | 42.93 % | 55.70 |
| freeze everything | 0 | 0 | 99.06 % | 11.04 % | 0.00 % | 55.05 |
| EWC, int8 Fisher | 0 | 2.798 MiB | 43.53 % | 55.09 % | 55.53 % | 49.31 |
| EWC fp32, lambda 1000 | 0 | 11.191 MiB | 40.69 % | 56.00 % | 58.37 % | 48.34 |
| random mask, 50 % frozen | 0 | 4 B | 39.62 % | 56.13 % | 59.44 % | 47.88 |
| EWC fp32, lambda 100 | 0 | 11.191 MiB | 26.16 % | 58.88 % | 72.90 % | 42.52 |
| naive, nothing retained | 0 | 0 | 24.45 % | 59.57 % | 74.61 % | 42.01 |

Six results, in order of how much they change what the framework can claim:

1. **Zero bytes per task is reachable, and it beats paying for an anchor.** On
   both pairs the best 0-byte mechanism outscores EWC in fp32 (11.191 MiB fixed)
   and in int8 (2.798 MiB fixed) by 5-10 ACC points. The framework's own O(1)
   anti-forgetting mechanism is dominated by simply not moving the weights, which
   is the cheap-baseline conclusion of B19 arriving at the per-task budget as
   well.
2. **The price of dropping the per-task copy is set by the domain gap, and it is
   not small.** Easy pair: 5.6 points (94.60 -> 89.00). Hard pair: **19.9 points**
   (78.57 -> 58.68). So "0 MB per task" is a real configuration, not a fantasy,
   but on a cross-domain shift the cached weights are doing work that no free
   mechanism recovered here. The paper's per-task storage is a deliberate trade,
   not an oversight, and this is the measurement that says so.
3. **The Fisher bitmask does not earn its 180 KB.** Fisher-selected masks and
   random-selected masks land within 0.25 points of each other at the same
   fraction on the hard pair (58.43 vs 58.68), and within 3.5 points on the easy
   pair. A 4-byte seed therefore matches a 180 KB importance bitmask, which is
   the same "importance ranking is not the binding constraint" result B65 found
   for the OGP basis. Which of the two is nominally ahead is *not* a finding: B68
   measures up to 3.6 points of run-to-run spread on exactly these rows, so the
   equality is the claim and the direction is noise.
4. **Which 0-byte mechanism wins depends on the gap, and the ranking is not the
   one intuition suggests.** Freezing the whole trunk is the best 0-byte position
   on the easy pair (89.00, and by 5.6 points -- outside the spread B68 measures,
   so that one holds). On the hard pair the best position in *this* run is a 90 %
   random mask (58.68), with the 10 % left plastic doing the work of
   re-representing a genuinely new input distribution; but the frozen trunk is
   within 1.6 points of it and B68 puts the spread on these rows at 3.6, so on the
   hard pair the top three 0-byte mechanisms are not separable. On the hard pair a
   *full* freeze costs 11 points of new-task accuracy (33.60 vs naive's 59.57)
   while buying 56 points of retention.
5. **`freeze everything` is not a 0-byte strategy, it is a 1-task strategy.** It
   scores 99.0 % on task 1 and 6-11 % on task 2 on both pairs. It is in the table
   as the control that shows the retention above is retention of *something the
   trunk could do*, not of a frozen accident.
6. **The real-model gap is deliberately conservative.** The mechanism carried to
   the 18-layer hybrid is `freeze trunk`: 0 bytes by construction, the best
   zero-byte arm on the easy pair (89.00), and on the hard pair 57.12 against the
   random mask's 58.68 -- a 1.6-point gap that B68 shows is inside the run-to-run
   spread, so it is not a real concession. It is also the *safest* arm to carry: a
   frozen parameter is 0 bytes under any implementation, while a mask is 0 bytes
   only if the decoupled decay is masked along with the gradient (the correction
   above). The gap it measures is therefore an upper bound on the cost of
   0 bytes/task, and the bias runs in the safe direction.

Two protocol defects were found and fixed while producing the easy-pair table,
and both are recorded because they had made the freeze arm look better than it
is. First, the snapshot row originally evaluated both tasks *after* restoring the
snapshot, which reported the pre-task-2 Fashion accuracy for a task-2 snapshot and
printed 5.97 %; the correct order is task 2 before the restore and task 1 after,
giving 90.21 %. Second, `set_seed` was called once at process start rather than
per mechanism, so the mechanisms shared RNG order -- the same confound that made
the OGP baseline in B65 move 19 points. The **hard pair was run after both fixes**
and is the clean table; the easy pair predates the per-mechanism re-seeding and
carries a +/-1-2 point order confound, which `--seed` plus
`experiments/aggregate_seeds.py` now exists to remove.

Neither table is multi-seed yet, and both use 3 epochs per task, so they are not
comparable in absolute terms to the production schedule in the paper. What they
establish is the *shape*: a free mechanism is competitive with every storage-backed
anchor, a 4-byte mask is competitive with a 180 KB one, and the per-task weight
copy is worth 6 points on a same-modality shift and 20 on a cross-domain one.

**Correction (2026-10-16): what a mask row had to store.** The `fixed` column
above is the size of the *selection* a mask persists -- 4 bytes for a random
seed, 0.172 MiB for one bit per weight. That is the whole cost only if the frozen
elements are never written, and the code that produced the table did not achieve
that: `build_mask` kept a full clone of every masked tensor
(`values[n] = p.detach().clone()`) and `train_task` restored the frozen elements
from it after every `opt.step()`, because `AdamW(weight_decay=1e-4)` multiplies
*every* element and would otherwise move them. Those rows therefore *held*
`|selection| + p x |basis|` -- 4.97 MiB at p = 0.9, 2.76 MiB at p = 0.5, and
5.52 MiB as coded, since the clone copies the whole tensor -- and not the 4 B the
column prints. This is a property of the implementation, not of the mechanism:
masking the decoupled decay together with the gradient (a decay factor of exactly
1.0 on frozen elements, exact in IEEE-754) leaves those values unwritten, so the
selection really is all that has to be carried. `exp_zero_per_task.py` now does
that, and `--self-check` proves the invariant on CPU in 0.1 s: manual decay is
bit-identical to AdamW's own for unmasked parameters, the frozen values stay
bit-equal to their initial values, and an unmasked decay does move them.
**The correction changes the byte column, not the trajectory.** The honest
update and the restore-based one are element-for-element identical -- `x * 1.0` is
exact in IEEE-754 and AdamW's decay is a per-element multiply by a scalar -- so no
ACC can move. That argument is exact and machine-independent, and `--self-check`
proves it on CPU in 0.1 s. The *empirical* version of the check was attempted and
is recorded as B68; it cannot settle the ACC question, because this table does not
reproduce run to run. B68 also records a real defect the attempt found in the
first implementation of the honest path, and it corrects two over-reads here
(claims 3 and 4).
What changes is the reading of the table: the two `freeze` rows are 0 bytes under
*any* implementation (a parameter with `requires_grad=False` is never visited by
the optimizer at all), while a mask row is 4 B / 0.172 MiB only under the stricter
implementation -- a naive one silently pays a snapshot with extra steps, which is
exactly the distinction the section is about.

The corresponding measurement on the real 18-layer hybrid
(`--snapshot-mode structural --freeze-trunk 1`, 0.045 MB/task against the 7.209 MB
default) was run on 2026-10-16 and is B67, below.

---

**B67 (measurement) - the matched pair on the real 18-layer hybrid: dropping the
per-task weight copy costs 10.85 points of task-aware accuracy and gains 28.75
points of the same quantity, because the arm that stores nothing is the arm that
cannot drift.**

`%TEMP%\opencode\run_aml_freeze.ps1` ran the production command twice, changing
one flag. Both arms: `--tasks mnist,cifar10,fashion --epochs 3,10,6
--replay-per-class 0 --norm in --encoder-tail-act none --arch-lr 0.003
--depth-cost 0.02`, one seed, zero replay, RTX 3050 Laptop. Arm 1 is the shipped
default (`--snapshot-mode full`, 7.209 MB/task, 40.3 min); arm 2 is the
weight-free point on the storage curve (`--snapshot-mode structural
--freeze-trunk 1`, 0.045 MB/task, 37.2 min). Both exit 0. Read with
`experiments/parse_aml_matrix.py`.

| protocol | arm | mnist | cifar10 | fashion | A_avg | forget |
| :-- | :-- | --: | --: | --: | --: | --: |
| oracle | full, 7.209 MB/task | 97.58 | 55.32 | 90.52 | **81.14** | 0.00 |
| oracle | structural+freeze, 0.045 MB/task | 97.71 | 33.46 | 79.71 | **70.29** | -0.04 |
| live | full | 10.73 | 22.95 | 90.30 | **41.33** | 59.79 |
| live | structural+freeze | 97.73 | 32.37 | 80.13 | **70.08** | 0.08 |
| auto | full | 35.80 | 46.46 | 90.51 | 57.59 | 35.31 |
| auto | structural+freeze | 8.89 | 10.15 | 79.71 | 32.92 | 56.02 |

Six readings.

1. **The two protocols disagree about which arm is better, and that is the
   result.** `oracle` (task identity *and* a snapshot restore) prefers the default
   by 10.85 points; `live` (task identity at the read-out, current weights only,
   no restore) prefers the weight-free arm by 28.75. The 7.209 MB is worth 10.85
   points *together with the restore*; without it the stored weights are dead
   capital and arm 1 forgets 59.79 %. B66's test bed found the same asymmetry on a
   harder pair, where dropping the copy cost 19.9 points.
2. **What buys the retention is that nothing moves -- and the head is not what
   forgets.** `A_live` keeps its read-out rows per task (`pi_mask_j` splits the
   K = 12 prototype rows 4/4/4 by task index; `evaluate` documents that `A_live`
   is task-aware, unlike `A_auto`), and a task's rows are only ever trained by
   that task, so task 1's read-out is intact in both arms. Arm 2's mnist is 97.73
   because a frozen trunk leaves mnist features where task 1 put them; arm 1's is
   10.73 because the trunk re-represented the input for CIFAR-10 and a read-out
   tuned to the old features then reads noise. Forgetting here is representational
   drift, not head overwrite.
3. **The plasticity cost lands on the cross-domain task.** `oracle` cifar10 33.46
   against 55.32 (-21.86), fashion 79.71 against 90.52 (-10.81), mnist 97.71
   against 97.58 (+0.13). Same ordering as B66: the cheap mechanisms are within a
   point of the snapshot on the same-modality task and ~20 points behind it on the
   cross-domain one.
4. **The autonomous protocol is unavailable at this storage point, for a
   structural reason.** Stage-3 `route_acc` collapses to 0.00 % in arm 2
   (`route_hist [0,0,10000]` -- every query assigned to the most recent task)
   against 27.04 / 79.33 / 99.99 % in arm 1, and `auto` A_avg falls 24.67 points.
   A centroid lives in the space of the encoder that produced it
   (`route_per_task`), and `structural` mode stores no per-task encoders -- only
   the live one -- so every candidate is scored in a space it does not belong to.
   The cheapest mode that keeps the router working is `shared` (1.343 MB/task),
   which does keep `enc`. Note that this column is not a headline in *either* arm:
   arm 1's router is also poor (27.04 % on mnist, `route_hist [2704,0,7296]`), and
   `A_auto` is the identity-free protocol, which is bad in both -- that is the
   B46/B47 problem, not this entry's.
5. **Storage was read off the artefacts, not asserted.** Arm 1 printed
   `[snap] ... snapshot 7.21 MB` and wrote `routing_state_after_*.pt` of 11.536 /
   18.861 / 26.184 MB; arm 2 printed `0.045 MB per task` and wrote 4.261 / 4.308 /
   4.355 MB. The per-task difference is exactly the snapshot (~7.3 MB). The router
   is identical in both arms and contributes 1.053 MB/task either way (B64
   correction), so the honest ratio is 8.263 against 1.098 MB per task -- 7.5x,
   not the ~160x the snapshot lines alone suggest.
6. **The default arm's `live` forgetting is B19 seen from the other side.** 59.79 %
   with `--ewc-lambda 0`: the snapshot does not *prevent* forgetting, it repairs it
   when the protocol permits a restore.

Caveats, so the numbers are not over-read. The schedule is `3,10,6`, not the
published `5,28,16`: this is a *paired* experiment and its absolute values are not
comparable to the published table. The same 3-task command under the full schedule
gives `oracle` A_avg 86.73 (the older `full_production.log`), against 81.14 here,
so roughly 5.6 of the 10.85-point oracle gap is schedule rather than mechanism. One
seed. `--freeze-trunk` was chosen over the sandbox's slightly better random mask
(B66: 58.68 against 57.12 on the hard pair -- a gap B68 shows is inside the
run-to-run spread, so the two are not separable) because a frozen parameter is
0 bytes under *any* implementation while a mask is 0 bytes only if the decay is
masked too (B66 correction) -- so 10.85 is an upper bound on the cost of
0 bytes/task, and the bias runs in the safe direction.

---

**B68 (measurement) - the 0-byte table does not reproduce run to run, and the
attempt to reproduce it found a real defect in the corrected implementation.**

B66's correction said the honest update "does not move any ACC" and that the four
mask rows were therefore "re-run as a *check*". The check was run. It produced two
results, and neither is the one that sentence predicted.

**(a) The first implementation of the honest path was wrong, in a way that a
toy-scale proof missed.** The manual substitute for AdamW's decoupled decay
multiplied *every* trainable parameter, but `torch.optim` skips a parameter whose
`.grad is None` (`if p.grad is None: continue` in `_single_tensor_adamw`), so
grad-less parameters were decayed by `1 - lr*wd` per step where AdamW left them
untouched. Latent parameters are normal in this model: an expert the gate did not
select receives no gradient. Measured on a two-parameter toy (8 steps, one
grad-bearing and one grad-less parameter): the grad-less tensor moved by `1.9e-6`
while the AdamW reference did not move it at all, and the grad-bearing tensor was
bit-identical. Fixed by gating on `p.grad is not None`, which is exactly AdamW's
own condition. `--self-check` now has two checks for it (a grad-bearing parameter
matches AdamW with the guard; a grad-less one is not decayed), so this class of
mistake fails a 0.1 s CPU check instead of surviving into a table. The existing
`--self-check` did not catch it because the toy it used gave its single parameter
a gradient every step.

**(b) The table does not reproduce, so the ACC column cannot be used as the
check.** Same command, same seed (42), same epochs (3), hard pair, six rows, run
again with the fix in place:

| row | B66 (acc_m / acc_f) | re-run (acc_m / acc_f) | dACC |
| :-- | :-- | :-- | --: |
| naive | 24.45 / 59.57 | 30.78 / 59.87 | +3.32 |
| freeze_trunk | 80.64 / 33.60 | 82.87 / 33.40 | +1.02 |
| mask_random_p0.5 | 39.62 / 56.13 | 50.66 / 54.93 | +4.92 |
| mask_random_p0.9 | 73.58 / 43.78 | 72.63 / 44.13 | -0.30 |
| mask_fisher_p0.5 | 56.13 / 55.26 | 53.58 / 54.37 | -1.72 |
| mask_fisher_p0.9 | 73.83 / 43.03 | 80.71 / 43.34 | +3.60 |
| `acc_m_init` | 99.06 | 98.96 | -0.10 |

`acc_m_init` is one evaluation of the task-1 model with no task-2 training
involved, and it moves 0.10. Every retention number moves by 1-5 points. Open item
4 already records 0.26 points of variation on the production command
(`cudnn.benchmark = True` selects convolution algorithms by timing, so the result
depends on machine load); this table is worse by an order of magnitude on the
unprotected rows, and for a reason the earlier entry did not name: a mechanism
that does not hold the representation ends on a *small* task-1 accuracy, and a
small numerical difference is a large relative one there. `naive` is the clearest
case -- 24.45 % against 30.78 %.

Three consequences.

1. **B66's correction is a byte-column correction, and it must not be defended by
   reproducing the ACC column.** What justifies it is the code argument (`x * 1.0`
   is exact in IEEE-754; AdamW's decay is a per-element multiply by a scalar) and
   the CPU self-check, both exact and machine-independent. Reproducing the ACC
   column would need a determinism this run does not have.
2. **The ordering inside B66's 0-byte table is not resolved at the precision the
   table prints.** Published order: `mask_random_p0.9` 58.68 > `mask_fisher_p0.9`
   58.43 > `freeze_trunk` 57.12. Re-run order: `mask_fisher_p0.9` 62.03 >
   `mask_random_p0.9` 58.38 > `freeze_trunk` 58.14. The top three sit within
   0.25-1.6 points of each other in one run and reorder in the next, against a
   measured spread of up to 3.6 points on these rows. The honest statement is that
   the Fisher mask, the random mask and the frozen trunk are **not separable** on
   this pair at this schedule. B66 claims 3, 4 and 6 have been rewritten
   accordingly.
3. **What survives is the magnitude, not the order.** Both runs put every 0-byte
   mechanism 10-20 points below the snapshot on the hard pair, within a few points
   on the easy pair, and 10+ points above `naive` on the hard pair. That contrast
   is the result; the ranking below the snapshot is noise. B67's real-model pair is
   unaffected in kind -- it compares one mechanism with the default, not five
   mechanisms with each other -- but the 1.6-point mask-versus-freeze gap it cited
   as a concession is noise, and B67 now says so.

Two artifact notes, both of which limit what can be checked later.

* The re-run crashed *after* printing its table (`UnboundLocalError: snap_bytes`
  when `--methods` excludes the `snapshot` row), so the table above is transcribed
  from `%TEMP%\opencode\zero_hard_v3.log` and there is no JSON for it. The report
  tail now handles an absent snapshot row.
* `zero_hard.json` -- the file B66's hard table was read from -- records only
  `acc_m_init`, `snapshot_bytes` and `shared_weight_bytes`. It predates the
  revision that writes `seed` and `task2`, so it does not pin which code produced
  the published rows. That is why this entry reports a fresh measurement rather
  than a reproduction.

---

**B69 (measurement + fix) - where a production training step actually spends its
time: the 64x64 step is 5x slower per sample than it should be and the cause is
VRAM oversubscription, not a kernel; the trunk carries fp32 activations because a
fp32 routing weight promotes a bf16 one; and 36 device->host synchronisations per
forward were removable, bit-for-bit.**

Artifacts: `benchmarks/profile_hybrid18.py` (step timing and kernel/memory tables),
`benchmarks/ab_router_flag_hoist.py` (interleaved A/B), and
`benchmarks/capture_router_identity.py` (bitwise before/after capture). Host: the
same RTX 3050 6 GB Laptop GPU, `cuda`, bf16 autocast, `K=12`, `max_dim=48`,
`norm=in`, `cudnn.benchmark=False`.

**(a) A step-time benchmark of this model is not a benchmark until the RNG is
pinned.** The first batch sweep measured batch 16 *slower* than batch 64 (239 ms
against 200 ms at 32x32), which is impossible on its face. The cause is that the
forward samples its own architecture every call -- Gumbel over `pi_logits`, Gumbel
over the per-layer block-mode logits, and a `0.05 * randn_like` perturbation of the
depth gates -- so two steps of the same configuration execute different branches
and cost up to 2x each other. Re-seeding before each measured step removes it. A
second, subtler form survives that fix: `torch.randn(B, 3, H, W)` consumes a
batch-dependent number of RNG values, so the draws feeding the routing Gumbel
differ between batch sizes, and a cross-batch sweep is still a sweep over sampled
architectures. Compare variants *within* one batch size, or fix the architecture
with `struct_override`. Every number below is from the seeded harness.

**(b) The 32x32 production step, measured.** batch 256: **449.7 ms** total
(forward 162.1, backward 286.8, optimizer 0.7), peak **2706 MiB**.

**(c) The 64x64 cliff is VRAM oversubscription, not super-linear compute.** Per
sample, 64x64 is 3.6x the 32x32 cost for 4x the pixels -- i.e. *sub-linear*, and
this model should be *cheaper* per pixel at higher resolution because the ViT
branch pools to <= 8x8 tokens. What actually happens at batch 256:

| 64x64 batch | ms/step | ms/sample | peak MiB |
|---|---|---|---|
| 16 | 204 | 12.8 | 656 |
| 32 | 265 | 8.3 | 1253 |
| 64 | 442 | 6.9 | 2532 |
| 128 | 762 | 5.9 | 4893 |
| 256 | **7702** | **30.1** | **9733** |

Peak allocation is linear in batch at **41 MiB/sample**, and the card has **6144
MiB**. The cliff sits between batch 128 (4893 MiB, fits) and batch 256 (9733 MiB,
does not): per-sample cost jumps 5.9 -> 30.1 ms, and the earlier unseeded sweep
showed the identical cliff (6162 ms, 9808 MiB). Windows oversubscribes silently to
host memory over PCIe rather than raising OOM, so this does not look like a bug --
it looks like "64x64 is 15x slower than 32x32". **There is no kernel to write
here.** At 64x64 on this card, batch 128 is ~5x faster per epoch than batch 256.
The same arithmetic on the production command explains the standing anomaly in
open item 5: the fused kernels were never the reason.

**(d) The trunk runs fp32 activations -- and that hypothesis was then refuted by
measurement, which is the useful part.** A dtype probe on the real forward
(`%TEMP%\opencode\probe_layer_dtype.py`) reports `bfloat16 -> float32` for layer 0
and `float32 -> float32` for all 17 remaining layers. The mechanism is type
promotion: the candidates are bf16 under autocast, but the mode weight
(`mode_weights[:, chosen]`), the channel mask (`dim_mask`) and the depth gate
(`gate_c`) are all views of fp32 tensors, so `out * dim_mask` and
`gate_c * h_cand + (1 - gate_c) * h` produce fp32 and every layer's boundary tensor
is fp32. That much is measured fact. What I then assumed -- that 4 bytes/element
across the trunk means roughly double the activation memory and double the
bandwidth, "on exactly the resource that is spilling in (c)" -- is **wrong**, and
the correction matters more than the observation.

The fix was written rather than argued. `--trunk-dtype compute` cast each block's
output back to its input dtype (a one-line wrapper around the block's forward,
plus casting `dim_mask`/`gate_c` in the loop). Measured with both arms in one
process, interleaved, RNG re-seeded per step, identical weights across the arms --
i.e. a pure precision comparison:

| config | `auto` (fp32 trunk) | `compute` (bf16 trunk) | time | peak |
|---|---|---|---|---|
| 32x32 batch 256 | 470.37 ms / 2700 MiB | 471.96 ms / 2604 MiB | -0.3 % (inside noise) | **-96 MiB** |
| 64x64 batch 128 | 755.68 ms / 4891 MiB | 751.37 ms / 4699 MiB | +0.6 % (inside noise) | **-192 MiB** |
| 64x64 batch 256 | 9523 ms / 9736 MiB | 7598 ms / 9352 MiB | both arms spill | **-384 MiB**, still 3.2 GiB over the card |

So the fp32 promotion costs **2-4 % of peak memory and no time at all**, and it
does not bring 64x64 batch 256 under a 6144 MiB card. The flag has therefore been
**removed rather than shipped**: it is a numerical change (the mask multiply and
the depth blend round to bf16) that buys ~3 % of one resource and 0 % of the
thing the user asked to improve. 41 MiB/sample is being spent somewhere else, and
the fp32 `instance_norm` round trips are the better candidate than the blend --
`instance_norm` is on autocast's fp32 list, so each of the 1-2 norm calls per
layer allocates an fp32 copy of a full-size bf16 tensor *regardless* of the trunk
dtype, and the trunk dtype cannot touch them. That is not measured yet; it is the
next hypothesis, and it is written down as such rather than as a finding.

Correction note: the commit that shipped the flag hoist (`b216a12`) states, in its
message, "2x the memory and bandwidth, on the resource that spills". That sentence
is wrong, this table is the correction, and it stays wrong in the history -- so it
is corrected here and in open item 18 rather than quietly dropped.

**(e) The fix that was applied: hoist the per-layer routing flags.** Each
`HybridBlock.forward` read two host scalars -- `_flags.tolist()` for
`all_hard`/`same_mode` and `mode_idx[0].item()` for `chosen` -- i.e. **36
device->host synchronisations per forward**, each of which drains the launch queue.
The whole `block_mode` `[B, 18, 3]` tensor already exists before the layer loop, so
`_hoist_block_mode_flags` packs all 18 layers' three scalars into one `[3, 18]`
tensor and reads them with a single `.tolist()`. This is the same hoist that was
already applied to the skip/partial masks, for the same reason. The per-sample
depth-skip branch passes no flags on purpose: it slices `w_mode` to the active
sub-batch, so this layer's row 0 is not that slice's row 0.

Evidence, in the order it should be trusted:

* **Bitwise identity.** `benchmarks/capture_router_identity.py` captures the
  forward output, `pi`, `layer_soft`, `bm_logits`, the loss, all 345 parameter
  tensors and all 139 gradient tensors from one fixed seed (7, batch 64, 32x32,
  bf16, `loss=2.744635`, `n_executed=18.0`), and compares a later build against
  that capture with `torch.equal`. Pre-change capture against post-change build:
  **every recorded tensor bitwise identical**. The training-mode case is the
  strong one -- it pins that the hoist does not consume the global RNG differently,
  hence that the sampled architecture is unchanged.
* **11 new tests** in `tests/test_router_flag_hoist.py`: the batched predicate
  against the per-layer predicate (training-style one-hot, mixed per-sample,
  the `> 0.99` boundary at 0.995 and 0.985, soft `struct_override` rows), the block
  with and without `flags` (all three branches, mixed modes, soft mixture,
  train and eval), the full model with the old path restored by monkeypatch in
  both modes, and the gradients.
* **304 tests pass** (`python -m unittest discover tests`, 56 s).
* **Interleaved A/B**, both variants in one process with the RNG re-seeded every
  step, so clock drift hits both equally (a separate-process comparison had shown a
  spurious 24 % regression at batch 16):

| config | old (per-layer sync) | new (hoisted) | change |
|---|---|---|---|
| 32x32 batch 256 | 489.22 ms | 451.57 ms | **-7.70 %** |
| 32x32 batch 64 | 208.92 ms | 198.32 ms | **-5.07 %** |
| 64x64 batch 128 | 785.34 ms | 755.20 ms | **-3.84 %** |

Means over the same 15 paired reps agree in sign and rough size (-10.3 %, -3.8 %,
-3.4 %). The spread on this machine is 22-70 ms sd, so any single-run comparison
here is unresolved; the interleaved median *and* mean agreeing across three
configurations is what makes the 4-8 % credible. It also means the published
`--epochs 5,28,16` schedule would finish 4-8 % sooner, with a **bit-identical**
trajectory -- no number in any table changes.

**(f) Adjudication of the external performance/robustness report.** The report was
treated as claims to verify. Verified as *code*: the missing input validation in
`mgn/csrc/in/bindings_in.cpp` (242 lines, zero `TORCH_CHECK`, zero `is_cuda`, zero
`is_contiguous`) against `bindings.cpp` (4 `TORCH_CHECK`, 2 `is_cuda`, 12
`is_contiguous`, all four in `mix_bwd_v2_a/b`); the 32-bit overflow sites; the
FP64 accumulators and `<<<B, MB_THREADS>>>` launches in `fused_pipeline.cu`
(`__shared__ double s_part`, `double ab0/ab1/alg/ch_term`); the 5D `einsum` in
`hts_lib/models/basis.py:13-17`; `mgn/trainer.py:209`'s up-to-16 per-sample
generator calls; and `mgn/regularization.py:58-66`'s O(n^2) loop with a `.item()`
sync per pair. Also verified: `collect_structure_slot` collects the structure slot
with `pi_mask=None` (`train_continual_aml.py:1265`) while training and the oracle
lock the row range -- that is existing open item 1, not a new finding -- and the
`structural`-mode routing collapse, which is B67.

Refuted or corrected:

* **The whole "very high" performance section is off-path.** `mgn/blocks/*` imports
  nothing from `csrc/`, `_C` or `executor`; it is pure PyTorch. The continuation
  trainer builds `ConvAML_18Layer_Hybrid` and never touches the extension, whose
  consumers are `mgn/executor.py` and `train_cv_cifar_18.py`. The profiler's kernel
  table for a real 32x32 step contains `aten::*` and cuDNN only -- **no custom AML
  kernel appears**. So the dynamic-conv/shared-memory/tensor-core item, the FP64
  item, the repeated `pi.to(kBFloat16)` item and the WMMA `s_b` item currently
  optimise a path this project's training does not execute. The report also misses
  both of the effects that *are* on the path: the 5x VRAM cliff in (c) and the fp32
  trunk in (d).
* **"Launchers do not call `cudaGetLastError`"** is file-dependent, not general:
  `fused_pipeline.cu` has 5 such checks, `batched_hyper_linear.cu` 6,
  `in/fused_pipeline_in.cu` 1; the zero-check files are
  `custom_dynamic_ops.cu`, `fused_backward.cu`, `in/fused_backward_in.cu`.
* **"loads environment variables through `shell=True`" attributed to
  `dllsetup.py`** is wrong: that file contains no `shell=True`. The repo's only one
  is `build_ext.py:91`, a compiler invocation. The `write_atomic` patch itself is
  quoted verbatim and real; its claimed failure mode (`Bus error`) is a Linux
  symptom, and on Windows the likelier outcome of `unlink`ing a loaded `.pyd` is
  `PermissionError` -- so the patch replaces a tolerated `FileExistsError` with a
  crash, which is worth noting but is not the described bug.
* **`HtSBasisIsolator` protecting only `b1_priv..b3_priv`** is accurate as code
  (`hts_lib/training/trainer.py:24-29`), but it is the *design* of that isolation
  scheme, not a memory-safety bug, and it is the same phenomenon B67 measured on
  the AML side: what is left unfrozen drifts, and forgetting follows.
* **The `int base` oddity** is real and sharper than stated: in
  `mgn/csrc/in/fused_pipeline_in.cu:163` the base index is `int base = ((b * C + c)
  * H + h0) * W_` while its siblings at `fused_pipeline.cu:92`,
  `fused_backward.cu:54` and `in/fused_backward_in.cu:49` all use `long base`, so
  the `in` copy was missed when every other copy was widened. Also unmentioned by
  the report: `int total = B * C * H2 * W2` (the launch-geometry computation)
  appears in eight places across the same four files. Still open item 15.

None of the security items were changed. They are unreachable from the training
command this project runs, and "unreachable" is an argument from control flow,
which this log has been wrong about before -- so they stay open items, not closed
ones.

**(g) The lever that actually works, shipped opt-in: recompute the blocks'
activations in backward (`--grad-checkpoint`).** Everything in (c)/(d) points at
one resource, and (d) left it unnamed. So it was attributed before it was guessed
at again (`%TEMP%\opencode\probe_norm_memory.py`, 64x64 batch 128, forward peak /
step peak / ms/step):

| norm | forward peak | step peak | ms/step |
|---|---|---|---|
| `in` (default) | 4870.7 MiB | 4971.1 MiB | 1024.1 |
| `bn` | 4879.4 MiB | 4977.5 MiB | **764.4** |
| `gn1` | 6163.6 MiB | 6359.5 MiB | 2183.9 |

`in` and `bn` peak within 8 MiB of each other, so **the fp32 `instance_norm`
round trip that (d) proposed as the better candidate is not where the memory
goes** -- withdraw that guess. `gn1` is the outlier and is worse on both axes,
and `bn` is faster at equal memory but is unusable on accuracy (running stats go
stale across tasks, README §376). What is left is what the arithmetic always
said: **41 MiB/sample is the autograd graph** -- 18 layers x ~6 saved
full-resolution tensors x 0.375 MiB. The way to remove a graph is to not keep it.

`mgn/blocks/hybrid_18._call_block` therefore runs each block inside
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`: inputs are saved,
intermediate activations are dropped and recomputed in backward. Exposed as
`--grad-checkpoint {off,on,auto}` (default `off`) and as
`benchmarks/profile_hybrid18.py --grad-checkpoint 1`.

**Bitwise identity, which is the whole reason this is safe to flip mid-run:**

* One seeded train-mode forward+backward: output, routing distribution and every
  gradient `torch.equal`, on CPU (fp32, `gn1`) and on the production path (bf16
  autocast, `norm=in`, batch 8, 64x64).
* **Three chained training steps from a single seed: identical losses and an
  identical final `state_dict`.** This is the strong test, not the single step.
  The trainer seeds once and then draws a fresh architecture every step from the
  global RNG (Gumbel over `pi_logits`, Gumbel over the block-mode logits,
  `0.05 * randn_like` on the depth gates), and `checkpoint` internally forks and
  restores the RNG; if that restored the stream differently, step 2 would sample a
  *different architecture* and the run would diverge. It does not -- measured end
  state after a forward is identical too. Same result for two steps on the
  production bf16 path.
* `benchmarks/capture_router_identity.py --compare` still passes on the default
  path, so (e)'s guarantee is intact: with the flag off, nothing changed.
* 8 new tests (`tests/test_grad_checkpoint.py`), including that `checkpoint` is
  never entered under `no_grad` and never when the flag is off.

**The price, from the committed harness** (seeded, `--no-benchmark`, 6 GiB card):

| config | off | on | time | peak |
|---|---|---|---|---|
| 32x32 batch 256 | 448.1 ms / 2706 MiB | 523.9 ms / 982 MiB | +16.9 % | **-1724 MiB** |
| 64x64 batch 128 | 757.7 ms / 4905 MiB | 905.0 ms / 1826 MiB | +19.4 % | **-3080 MiB** |
| 64x64 batch 256 | 10525.6 ms / 9737 MiB (spilling) | **1858.3 ms / 3594 MiB** | **-82.3 % (5.7x)** | **-6143 MiB** |

The `on` column is stable at 1.79-1.86 s across runs; the `off` column at batch
256 is not (an interleaved run of the same arm measured 7245 ms, an early sweep
7702 ms, this run 10526 ms) because the spill itself is unstable. The honest
statement of the speedup is **4-5.7x, and the correct claim is directional: it
turns a step that does not fit into one that does.**

Consequently the default stays `off` -- it costs ~17 % exactly where the model
already fits, so making it unconditional would be a straight regression on 32x32.
`auto` runs the first step normally and switches it on if the peak since process
start exceeds 85 % of the card; because the switch cannot change a number, this is
safe by construction, and it fires correctly on all three configurations above
(2706 and 4905 MiB pass, 9737 MiB does not). What is *not* validated is the
heuristic itself -- it reads a peak since process start rather than the step's own,
and 85 % is a judgement call. Its practical use is Tiny ImageNet: at 64x64 on a
6 GiB card, **batch 128 (4893 MiB, fits) or batch 256 with `--grad-checkpoint on`
(3594 MiB, 5.7x faster and also fits)** are the two shapes that work, and neither
is the one a user would pick by default.

Correction to (d): its closing paragraph named the fp32 `instance_norm` round trip
as "the better candidate" for the 41 MiB/sample. The norm probe above refutes that
(`in` and `bn` allocate the same peak). (d)'s dtype observation stands; its guess
at where the memory went does not, and **41 MiB/sample is now identified as the
autograd graph, not left open.**

---

**B70 (audit) - a second external report, five claims adjudicated: one is new and
correct (the normalisation constants are per-task, so `A_auto` is measured with
privileged information), one is correct but understated (the channel mask does not
save compute and it COSTS 16 % of a step and 25 % of peak memory), and three are
true as code but their conclusions are refuted by measurements already in this
log.**

**(1) "The channel mask saves no runtime" - confirmed, and the report understates
the cost.** The code is exactly as quoted: `mgn/blocks/cnn_block.py:51-55` runs the
conv at full width and then multiplies, `:97-108` does it *twice* per `ResNetBlock`
(once after the first conv, once after the residual add), and
`mgn/blocks/vit_block.py:112-113` does the same. `compute_flops(..., active_dim=)`
(`cnn_block.py:62-67,118-124`) is the only place the width reduces anything:

| call | reported |
|---|---|
| `ResNetBlock.compute_flops((32,32))` | 85.33 MFLOP |
| `ResNetBlock.compute_flops((32,32), active_dim=16)` | **19.01 MFLOP** |

a 4.5x smaller number for the same work. New measurements:

* The **extracted** child does not narrow either: `StaticLayer.__init__` builds
  `nn.Conv2d(conv_w.shape[1], conv_w.shape[0], 3)` - every channel - and masks in
  `forward` (`AML_Core_Code/extract_static_model.py:54,72,77,79`). Measured on a
  real `StaticLayer`: `conv=(48,48,3,3)` and **10.91 ms/call at 48 channels against
  10.89 ms/call at 16**. So the report's "only the accounting changes" is right,
  and it survives into the "hard-pruned" deployment artifact.
* Keeping the mask **costs**. Interleaved A/B, one process, one seed, the same
  sampled architecture, on the production step (32x32, batch 256, `norm=in`, bf16,
  `K=12`, `max_dim=48`), 8 reps per arm:

| arm | ms | peak MiB |
|---|---|---|
| mask removed entirely | 384.8 | 2169 |
| mask as production runs it | **448.6** | **2706** |
| mask cast to the activation dtype (B69d's `--trunk-dtype compute`) | 442.3 | 2682 |
| mask detached (no architecture gradient through it) | 423.4 | 2677 |

  So the width mechanism costs **+64 ms (+16 %) and +537 MiB (+25 % of peak)** on
  the step where it is supposed to be free, and B69d's dtype fix cannot touch it:
  the per-layer dtype chain is **identical** in the masked and unmasked arms
  (`bfloat16->float32` at layer 0, `float32->float32` for 1-17), so this is not the
  promotion B69d was chasing. Detaching the mask's gradient path (the mask is
  derived from `d_lengths`, so it requires grad) recovers 25 of the 64 ms and none
  of the memory. The remaining ~39 ms and the 537 MiB are **not attributed** - see
  open item 22. Note the asymmetry with depth, where the same class of "ghost work"
  was found and actually fixed (`hybrid_18.py:410-422` branches around a closed
  gate); width never got that treatment.

**(2) "The encoder is a single point of failure" - true as code, already measured
in more detail here, and the conclusion is refuted.** `vision_encoder`
(`hybrid_18.py:168-177`) is one `Conv2d(3, 48, 3)`, a norm, a ReLU, then
`AdaptiveAvgPool2d((2, 2))` and `Linear(192, 512)`. Everything it feeds is as the
report says: `prototype_selector` (pi), the structure decode (depth, width, mode)
and the domain router - which reads this same encoder through `router_features`
(`train_continual_aml.py:87-120`). What the report presents as hidden is written
out in that function's own docstring, with the measurement: the 2x2 pool "averages
32x32 down to 2x2 and flattens the signal; the values that survive are small and
nearly equal", and under bf16 two *different* encoders produced **bitwise
identical** features (maxdiff 0.000 against 0.368 in fp32), which collapsed routing
to 0 %. What is refuted is the consequence: routing accuracy is **89.02 % at its
worst cell and 100 % at every stage-2 cell** (README section 3.1a), and the
misrouting that does exist has a shape a shallow encoder does not predict - 10.98 %
of MNIST and 5.00 % of CIFAR-10 are routed to Fashion after the third task and
nothing is ever routed backwards (open item 2). "The router cannot distinguish
MNIST from Fashion-MNIST" is not what is measured.

**(3) "Normalisation is per task, so a task-agnostic deployment has no correct
constant" - confirmed, and this is the one genuinely new finding of the five.**
MNIST `0.1307/0.3081`, Fashion `0.2860/0.3530`, CIFAR the channel tuple
(`mgn.datasets.TASK_SPECS`, read at `train_continual_aml.py:552` in `get_task_data`), applied inside each task's
own loader. Every reported number - `A_auto` included, whose entire claim is that
the task identity is *not* known - is therefore produced with the ground-truth
task's constants, because the evaluation loader is built per task. A real
deployment sees a raw image, must choose one constant, and the wrong choice shifts
every activation before the first convolution - which is precisely the signal the
router reads. The autonomous number is measured under privileged information that
the autonomous setting does not provide. That gap was not disclosed anywhere, and
it is now open item 23.

**(4) "`StaticChildModel` cannot serve continual learning" - true, and it is not
claimed to.** `extract_child_hybrid18` averages the mode over the batch and over
prototypes weighted by pi (`AML_Core_Code/extract_static_model.py:298,317`) and
compiles one child; a static child is one point in structure space, so T tasks mean
T children. That is inherent to extraction and the code's own docstring says so
(`StaticHybridBlock`, `:85-105`). The README presents the extractor as a
**deployment compiler** (section 2: "zero Gumbel overhead at inference"), not as
the continual mechanism - which is the snapshot path, and it stores 0.045 MB per
task in `structural` mode (B66/B67) rather than T full models. Two corrections
nonetheless. (a) "Hard-prunes" is true only for **depth and mode**
(`is_active = is_first or gate > gate_skip_eps`, `:115`; the argmax branch is
compiled away), **not** for width - measured in (1), the child keeps
`conv=(48,48,3,3)` and multiplies by a constant mask - and that README line is
corrected in this commit because it overstates what the compiler does. (b) The
report's "batch-mean destroys the per-task shape-shifting" is accurate about
extraction and orthogonal to continual learning, where the per-task structure is
carried by the stored slot, not by a re-extraction.

**(5) "Batch-consistent Gumbel starves the other branches" - true per step,
refuted as a trajectory.** `hybrid_18.py:340-345` means the block-mode logits over
the batch and draws ONE hard one-hot per layer, so `all_hard`/`same_mode` hold and
`HybridBlock.forward:74-86` executes a single branch: the other two receive no
weight gradient **in that step**. The comment claims "a 3x speedup while preserving
full straight-through gradients", which is precise and the report misreads it - the
straight-through path preserves gradients to the **architecture parameters**
(`bm_logits`), not to the branches that were not executed. Refuted, by measurement:

* Over a run the mode distribution is **12.9 % Plain / 59.0 % ResNet / 28.1 % ViT**
  (measured on the shipped checkpoints, section 2), and it is identical to the
  initialisation. The net is not "a ResNet-18".
* 24 real production steps: **every** branch parameter group accumulates a non-zero
  gradient in 22-24 of the 24 steps (plain 374,112 params, resnet 748,224, vit
  341,280, shared 407,769). Starvation is proportional to the sampling probability,
  not absolute.

What the report is reaching for is already in this log, stated more strongly and
measured: the mode distribution is **indistinguishable from the initialisation**
(stage 1 18.4/81.1/0.6 against 18.1/81.3/0.5), i.e. the *mode search does not
move* - "the structure is the initialisation" (section 2). And the measured cause is
the initialisation, not branch starvation: `--arch-prior staged` 89.9 % against
`neutral` 11.4 %.

None of the five produced a code change. Two are now open items (22: the mask's
cost is only partly attributed; 23: `A_auto` is measured with privileged
normalisation), and one README line was corrected because the report showed it to
be an overstatement.

---

**B71 (implementation) - `--input-norm`: open item 23 made measurable.**

The gap B70 named: every loader in `train_continual_aml.py` normalises its images
with the **ground-truth task's** own mean/std, because `get_task_data` is called
once per task. `A_oracle` and `A_live` are told the task, so that is legitimate.
`A_auto` is the protocol whose entire content is that the task identity is *not*
available, and it is scored on images prepared with it. The README now says so
next to the table, and the number is an upper bound rather than a measurement.

Implemented, opt-in, default untouched:

  * `mgn/datasets.py`: `N_TRAIN` and `pooled_input_stats(tasks)` -- one
    `(mean, std)` for a sequence, weighted by training-set size, rebuilt from the
    per-task moments with `E[x^2] = Var[x] + E[x]^2`. For
    `mnist,cifar10,fashion` it is `mean = (0.2916, 0.2889, 0.2784)`,
    `std = (0.3323, 0.3302, 0.3246)`.
  * `train_continual_aml.py`: `_RenormDataset` (recovers the raw image with
    `x * std + mean` and applies the requested rule), `resolve_input_norm`,
    `check_input_norm_args`, the `--input-norm {per-task,pooled,imagewise,none}`
    flag, and the descriptor threaded through `train_task` and the three other
    `get_task_data` call sites.
  * The design choice worth recording: it wraps the **dataset**, not the
    transform, so all five data branches -- including the in-memory MNIST/Fashion
    cache -- go through one path, and the cache keeps storing **per-task** tensors
    so it stays valid whichever rule the caller asks for. Tested.

Two properties worth stating, because they are why this is an experiment rather
than a formality:

  * **No single constant can match a domain.** The pooled std (0.332) is larger
    than every per-task std (MNIST 0.3081, CIFAR 0.2023, Fashion 0.3530), because
    pooling distributions with different means adds between-domain variance. A
    "pooled" implementation that averaged the *means* and reused a std would put
    every image off its own distribution by a different amount and would have
    looked like a bug in the model rather than in the normalisation. The test
    pins the second moment against an independent hand computation.
  * **`--input-norm != per-task` with `--replay-per-class > 0` is refused**, not
    mixed: `build_replay_buffer()` stores its exemplars with the per-task
    constants, so the replay stream and the live stream would be normalised
    differently and the run would measure neither protocol.

Guarantee that nothing published moves: `inorm=None` (the default) makes `_wrap`
return the dataset unchanged, the flag defaults to `per-task` and is asserted by
`tests/test_input_norm.py`, and that file's release-gate test checks the cached
dataset comes back **unwrapped** on the default path. 19 new tests, 334 total.

**The measurement is complete.** A paired run at the production schedule
(`mnist,cifar10,fashion`, `--epochs 5,28,16`, zero replay, `--norm in`,
`--arch-lr 0.003`, `--depth-cost 0.02`), same commit `8c16a65` and same seed,
compared per-task against `--input-norm pooled`:

| Protocol   | A_auto A_avg | BWT       | Forgetting   |
|------------|-------------|-----------|--------------|
| per-task   | 66.05 %     | −29.77 %  | 29.77 %      |
| pooled     | 40.46 %     | −69.23 %  | 69.23 %      |
| Delta      | −25.59 pts  | −39.46 pts| +39.46 pts   |

`A_oracle` is unchanged (85.90 % → 86.61 %), confirming the router — not the model —
is affected. `A_auto` measured under per-task normalisation is therefore an **upper
bound** on what a deployment without task labels sees. `--input-norm {pooled,`
imagewise,none}` remains opt-in and refused with `--replay-per-class > 0`. Open item
23 closed.

---

## B72 — Audit of the code path under the paired run

Done *after* run B (pooled) completed, so run B's code was never disturbed. Commit `32ca0aa` on branch `code-audit-fixes` (off `8c16a65`), merged into `main`.

1. **Single source of truth for input-normalisation constants.** `mgn.datasets.TASK_SPECS` defines `mean`/`std` per task; `get_task_data` reads them from there instead of holding 10 literal copies across 4 branches that could drift. Tests in `tests/test_data_pipeline_contract.py` pin the historical constants. Verified: equivalence `ALL EQUAL` between old and new paths.
2. **`resolve_epoch_list()`** replaces a bare `assert` at the epoch-list parse site — better error message, no behaviour change.
3. **Per-step host syncs removed from the replay path.** The replay buffer rebuilt `task_id → prototype_row` lookup tables inside every training step, paying one `torch.unique(...).tolist()` sync plus one `int(t_)` sync PER EXEMPLAR (~48 replayed task ids). Replaced with `counts_tab`/`row_task` precomputed per task. Verified: tensor values identical.
4. **`worst_before → best_before`** in checkpoint retention: `worst_before` retained the *worst* checkpoint to compare against — a no-op. Fixed to retain best-before, restore next-epoch. Behaviour-neutral in the healthy path; covered by `tests/test_state_swap.py` bit-identity gate.
5. **`zip(strict=True)`** in `_hoist_block_mode_flags` (`hybrid_18.py:53`) — catches length mismatches that would silently truncate to the shorter iterable.
6. **`collect_structure_slot` `pi_mask` audit**: the mask is applied to the full (un-masked) tensor, not a sparse selection — correct per design. Left as-is (changing it alters numerics and requires a measured GPU run).

Verifications on `main` after merge: `pytest` **367 passed**, `unittest` **350 OK**, `ruff check .` clean except the 4 F401s in untracked `experiments/exp_verify_basis_ogp.py` (user instruction: leave unchanged).

---

## 11. Open items, stated as open

Written down so that "not done" is not later mistaken for "not noticed".

1. **The structure slot is collected without a prototype-row mask.**
   `collect_structure_slot` runs `model(x, tau=0.1)` with `pi_mask=None`, so the
   stored `gate` / `bm_mode` describe the structure implied by **all K**
   prototype rows, while the oracle protocol evaluates the same task under
   `pi_mask_j` (its own rows only). The slot is therefore slightly mismatched
   with the weights it is meant to describe. This is left as-is: changing it
   mid-experiment would invalidate the production numbers being measured, and the
   correct comparison is a full re-run, not a patch.
2. **All stage-3 misrouting goes forward.** 10.98 % of MNIST and 5.00 % of
   CIFAR-10 are routed to Fashion after the third task, and nothing is ever
   routed backwards. The newest encoder's feature space is the one the router
   finds most accommodating, which points at an asymmetry between "just
   trained" and "long since trained" spaces rather than at the metric. The
   obvious first experiment — re-fitting the most recent centroid when a new
   task arrives, or down-weighting it — has not been tried.
3. **EWC has not been re-measured since B30.** The "EWC does not help at
   production scale" verdict was produced by the broken accumulator. The maths is
   now right and tested; the *measurement* is stale. Until it is re-run the
   honest statement is "that implementation did not help".
4. **Everything is single-seed, and the run is not bit-reproducible even at a
   fixed seed.** `torch.backends.cudnn.benchmark = True` lets cuDNN pick
   convolution algorithms by *timing*, so the choice depends on machine load.
   Measured: two runs of the identical production command at seed 42 gave
   `A[1,1]` = 98.65 % and 98.91 %. That is a 0.26-point spread on the same seed,
   which is small compared with the effects under study but large enough that a
   single run cannot support a claim finer than about half a point. BatchNorm
   was already shown not to reproduce (two runs 14.3 points apart). **B68 measures
   a worse case in the sandbox: 1-5 points on retention rows and up to 3.6 on the
   0-byte ranking, because an unprotected mechanism ends on a small task-1
   accuracy where a small numerical difference is a large relative one.** Any
   table whose rows are within a few points of each other must say which spread it
   was measured against, and B66's did not.
5. **CUDA shared-memory tiling (point 5) not started.** The fused mix kernels are
   elementwise and already bandwidth-bound; the remaining candidates are the
   grouped-convolution path in `fused_dynamic_conv2d` and the two-convolution
   residual chain. No attempt has been made, so no claim is made.
6. **Benchmark scripts are not part of CI.** `benchmarks/` and `probes/` are run
   manually; the import smoke test guarantees they at least *import*, but their
   numerics remain unverified by CI.
7. **The 2026-10-15 production numbers predate B46 and B47, and have not been
   re-measured.** `A_auto` 81.96 % / 6.34 % was produced while the ambiguous
   retry used the router's *predicted* domain's structure slot (B46) and while
   the per-sample dispatch dropped the routing weight (B47). Both touch the
   evaluation path; neither is expected to move the headline (the retry only
   fires below a 0.6 margin and routing accuracy is 89--100 %), but "expected to
   be small" is not a measurement, and the affected direction is not known. The
   numbers stand as the pre-fix measurement and a re-run is the way to close it.
8. **A_live is task-aware, and the task-agnostic live number has never been
   measured (B48).** The honest way to size the task-identity component of the
   `A_auto`/`A_live` gap is to log a second live variant evaluated with
   `pi_mask=None`; that variant has not been run, so the gap is still reported as
   two effects together.
9. **The corrected ambiguous retry has only a synthetic test.** The branch was
   wrong for the entire production run, and the fix is pinned by a unit test that
   forces the branch on a CPU fixture. Nothing exercises it on real checkpoints
   at scale, so a second defect in the same block would still show up first as a
   routing number rather than as a test failure.
10. **The three tasks in the production sequence are not trained identically
    (B54).** With the default `fast_cache=True`, MNIST and Fashion-MNIST are
    trained with no augmentation while CIFAR-10 gets crop+flip, and `aug` is
    consulted only by CIFAR-10. The confound is disclosed in each place it
    matters but not removed; removing it means re-running the sequence under one
    explicit policy, which would change the MNIST and Fashion columns.
11. **TestPyPI is done; the real PyPI publish is the only release step left.**
    The trusted publisher works end to end (B62): `aml-framework 2.0.0` is on
    test.pypi.org and installs into a clean venv with `aml-selfcheck` exiting 0.
    What remains is the pypi.org publish -- a `v2.0.0` tag pushed to `main`, or
    `workflow_dispatch target=pypi` -- which is immutable once it lands, so it is
    left to the owner. It needs the like-named trusted publisher on pypi.org
    (owner, repository, workflow `publish.yml`, environment `pypi`) and the
    `v2.0.0` tag must match `pyproject.toml`, which it does. Until that runs,
    "run it on another machine" means installing from TestPyPI, not from PyPI.
    **Held, on the owner's instruction (2026-10-15):** no `v*` tag exists locally
    or on the remote and no `pypi` dispatch has been made, so nothing in this
    repository will publish to pypi.org by itself. `aml-framework` is still
    absent from pypi.org -- the JSON endpoint 404s -- i.e. the trusted publisher
    is configured but no release has been created yet, so the first upload will
    be a fresh, irreversible creation.
12. **There are no CUDA wheels, by design, and that is a real cost.** A user on a
    CUDA host must compile the sdist themselves, and nothing exercises that path
    end-to-end in CI: `validate_ext_in.py` runs only when the runner already has a
    GPU, and the release workflow never builds the extensions at all.
13. **The fused executor has no HtS-adapter term, so `--fused-executor` is now a
    no-op for the default generator (B59).** Refusing to run a fused kernel that
    computes a different function from the model is correct, but it means the
    speedup only exists for models without adapters, which is nothing in this
    repo. The fix is a fused kernel that folds `A(pi) @ B` into the re-mixed
    weights; it has not been written. The parity tests delete the adapters so the
    kernel itself stays under test.
14. **`mix_bwd_v2_a/b` are still not exported by `_Cin` (B59/B61).** The Python
    `_norm_bwd` dispatch now finds `_Cin.instance_norm_bwd`, but the mix backward
    kernels live in `fused_backward.cu`, which is not in `_Cin`'s source list.
    Adding it means rebuilding `_Cin` with the `mb_*` buffer guards it currently
    lacks; not attempted, so a `_Cin`-only build still falls back for the mix
    half.
15. **Most CUDA kernels still index with 32-bit `int` (B60).** B60 widened the
    per-image bases, which are the offsets that overflow first, but the flat
    indices (`int total = B * C * spatial`, `int idx = blockIdx.x * blockDim.x +
    threadIdx.x`) remain: past ~2^31 elements the grid arithmetic overflows and
    the kernel mis-indexes or writes zeros instead of failing. Unreachable at
    every shape this project trains on (32x32 inputs, `C <= 1024`, `B <= 256`),
    so it is a documented limit rather than a fixed bug.
16. **The B59 Python fixes are argued to be no-ops for the production command by
    reachability, not re-measured.** `--amb-topk` defaults to 0, the
    auto-live-thresh fallback is off, and the production run never passes
    `snaps=[]`, so none of the touched branches fire; but that is an argument
    from control flow, and "the branch cannot fire" has been wrong before in this
    log. A full re-run is the way to close it (with open item 7).
17. ~~**The weight-free configuration has not been measured on the real hybrid.**~~
    **Closed 2026-10-16 -- see B67.** It has now been run as a matched pair at
    `--epochs 3,10,6`, one seed: the weight-free arm costs 10.85 points of
    task-aware accuracy (oracle A_avg 70.29 against 81.14) and gains 28.75 on the
    no-restore protocol (70.08 against 41.33), with the plasticity loss
    concentrated on the cross-domain task (CIFAR-10 33.46 against 55.32). What
    remains open from it is the *reason* the autonomous protocol collapses to
    0.00 % routing accuracy in that arm -- `structural` stores no per-task
    encoders, so each centroid is scored in a space it does not belong to -- and
    whether a router fitted on the live encoder would recover it.
18. ~~**The trunk carries fp32 activations and that is worth ~2x.**~~ **Measured
    2026-10-16, refuted as a lever (B69d).** The dtype observation is true: fp32
    routing factors promote the bf16 candidates, so layers 1-17 see fp32
    boundary tensors. But the fix was written and measured before being believed,
    and it buys **2-4 % of peak memory and 0 % of time**, and does not bring
    64x64 batch 256 under a 6 GiB card (9352 MiB against 6144). The flag was
    removed rather than shipped, because it is a numerical change with no
    measurable benefit. What accounts for 41 MiB/sample is now identified: **the
    autograd graph**, after the `instance_norm` fp32-round-trip hypothesis was
    measured and refuted as well (`in` 4971 MiB against `bn` 4977 MiB at 64x64
    batch 128, `gn1` worse on both axes). See B69g, whose fix removes the graph
    exactly where removing it pays.
19. **A step-time benchmark of this model must pin the RNG (B69a).** The forward
    samples its architecture per call, and the input tensor's own `randn` consumes
    a batch-dependent number of RNG values, so a cross-batch sweep measures a
    sweep over sampled architectures. Two sweeps in this entry were affected. The
    new `benchmarks/profile_hybrid18.py --seed` re-seeds per measured step; that
    removes the repetition noise but *not* the cross-batch confound, which needs
    `struct_override` or a within-batch-size comparison. Any absolute ms/step
    figure for this model is only meaningful with the seed and batch recorded next
    to it.
20. **`evaluate()`'s per-batch snapshot swapping was not re-measured (B69f).**
    The external report's item 5 (copy `vision_encoder` weights per task per batch,
    reload all 18 layers for every domain present in a batch, and for *every*
    domain when a single sample falls below `--amb-topk 0`'s routing margin)
    describes the documented design, and its O(T) term is real, but that cost has
    never been separated from the accuracy numbers it produces. The suggested fix
    -- one grouped convolution over the `T` stacked encoders, `groups=T` -- is
    plausible and is a genuine optimisation of the eval path rather than of
    training. Not attempted.
21. **`--grad-checkpoint auto`'s threshold is a judgement call, and its identity
    guarantee is verified for two or three steps, not for a whole schedule
    (B69g).** The latch reads `max_memory_allocated` since process start rather
    than the first step's own peak, so it overestimates and biases towards
    switching checkpointing on -- the safe direction, but it costs ~17 % of step
    time if a later phase would have fitted anyway, and it has never been
    exercised on a schedule long enough for that to show up. Separately, bitwise
    identity is established for one step (CPU and GPU), three chained steps (CPU)
    and two chained steps (production bf16); a whole `--epochs 5,28,16` run has
    not been shown bit-identical under the flag, and a whole run is what the
    production tables come from. Neither is a correctness risk -- the tests bound
    it -- but neither is a measurement, so this is written down rather than
    claimed.
22. **The channel mask costs ~16 % of a step and ~25 % of peak memory, saves no
    compute, and only part of that cost is attributed (B70).** Measured on the
    production step, interleaved and seeded: 384.8 ms / 2169 MiB with the mask
    removed entirely, 448.6 ms / 2706 MiB as production runs it, 442.3 ms / 2682
    MiB with the mask cast to the activation dtype, 423.4 ms / 2677 MiB with the
    mask's gradient path detached. So the dtype fix (B69d) cannot touch it, and
    ~25 of the 64 ms is the architecture gradient through the mask. **Not
    attributed: the remaining ~39 ms and essentially all of the 537 MiB.** The
    candidates, none tested: one extra full-size fp32 tensor per layer saved for
    backward; the mask applied twice per `ResNetBlock` where once would do
    (`cnn_block.py:101,108`); `cand * mode_weights[:, chosen]`
    (`hybrid_router.py:86`), a multiply by exactly 1.0 under the one-hot fast
    path; and the mask being re-applied to the next layer's input. Any fix changes
    the function the model computes, so it needs a full run rather than a patch --
    which is why it is written down and not attempted here.
23. **CLOSED (B71).** Every reported number, `A_auto` included, is produced with
    per-task normalisation (the unchanged default). A paired production-schedule run
    compared `per-task` against `--input-norm pooled` (one task-agnostic constant
    `mean=(0.2916, 0.2889, 0.2784)`, `std=(0.3323, 0.3302, 0.3246)`, from
    `mgn.datasets.pooled_input_stats`). Result: `A_auto` drops from **66.05 %** to
    **40.46 %** (—25.59 pts), forgetting rises from **29.77 %** to **69.23 %**, while
    `A_oracle` is unchanged (**85.90 %** vs **86.61 %**). The oracle invariance
    confirms the router — not the model — is what suffers under task-agnostic
    normalisation, so `A_auto` measured under per-task normalisation is an upper
    bound on what a deployment without task labels sees. `--input-norm {pooled,imagewise,none}`
    remains opt-in and is refused with `--replay-per-class > 0`.