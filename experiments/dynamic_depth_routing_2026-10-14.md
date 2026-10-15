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
   was already shown not to reproduce (two runs 14.3 points apart).
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