"""Where does the router break at stage 3?

The production run (epochs 5,28,16, seed 42) reports:

    cell            auto    oracle   route_acc
    A[1,1] mnist    98.65   98.65    100.00
    A[2,1] mnist    93.47   98.65     93.96
    A[2,2] cifar    69.22   69.22    100.00
    A[3,1] mnist    16.09   98.65      5.94   <-- collapse
    A[3,2] cifar    42.93   69.22     50.01   <-- collapse
    A[3,3] fashion  91.91   92.32     99.39

5.94% is far *below* chance (33%), so this is systematic, not noise: the router is
confidently sending MNIST to the wrong domain.

Hypothesis: the Mahalanobis metric uses ONE covariance pooled over all seen
tasks, but `score_per_task` compares each query in the feature space of the
encoder that embedded it. Those are DIFFERENT spaces. With two tasks the pooled
second moment is still dominated by whichever task has the larger feature norm;
with three it tips over and the precision matrix stops discriminating.

This probe rebuilds the exact router state from the run's own per-task
checkpoints (aml_continual_after_<task>.pt is the model at the end of that task,
which is precisely the encoder whose features the stored centroid lives in) and
compares three scoring rules on the real features:

    shared   one covariance pooled over all tasks          (shipped)
    per-task one covariance per task                       (zero-replay; T x D x D)
    cosine   scale-free nearest direction

Run:  python probes/probe_router_stage3.py
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mgn.blocks.hybrid_18 import ConvAML_18Layer_Hybrid
from train_continual_aml import get_task_data, router_features

TASKS = ["mnist", "cifar10", "fashion"]
CKPT = "aml_continual_after_{}.pth"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT = 512
SHRINK = 0.01


def load_encoder(task):
    m = ConvAML_18Layer_Hybrid(K=12, max_layers=18, max_dim=48, num_classes=10,
                               norm_type="in", encoder_tail_act="none").to(DEV)
    sd = torch.load(os.path.join(ROOT, CKPT.format(task)), map_location=DEV)
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def embed(model, loader, max_batches=None):
    zs = []
    for i, (x, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(DEV, non_blocking=True)
        with torch.autocast("cuda", enabled=DEV.type == "cuda",
                            dtype=torch.bfloat16):
            zs.append(router_features(model, x).float())
    return torch.cat(zs, 0)


def precision_from(S, shrink=SHRINK):
    lam = shrink * torch.diagonal(S).mean().clamp(min=1e-12)
    return torch.linalg.pinv(S + lam * torch.eye(S.shape[0], device=S.device,
                                                 dtype=S.dtype))


def hierarchical_precision(S_task, S_pooled, alpha):
    """(1-alpha) * own precision structure + alpha * pooled structure.

    alpha = 0 is per-task, alpha = 1 is the shipped shared matrix. The pooled
    matrix carries directions that the task's own covariance happens not to
    excite (that is where Fashion loses accuracy under a purely per-task
    metric), while the per-task matrix carries the scale of the space the
    centroid actually lives in. Shrinking one towards the other keeps both.
    """
    S = (1.0 - alpha) * S_task + alpha * S_pooled
    return precision_from(S)


def scores(z_per_task, cents, precs, metric, shrink=SHRINK, logdets=None):
    """[B, T] logits, one column per domain."""
    T = len(cents)
    cols = []
    for k in range(T):
        d = z_per_task[k] - cents[k].unsqueeze(0)
        if metric == "cosine":
            cols.append(F.normalize(z_per_task[k], dim=-1)
                        @ F.normalize(cents[k], dim=-1))
        elif metric == "shared":
            cols.append(-((d @ precs * d).sum(-1)))
        elif metric == "per-task":
            cols.append(-((d @ precs[k] * d).sum(-1)))
        elif metric == "per-task-nll":
            # Gaussian negative log-likelihood under the task's OWN covariance.
            # The + log|S_k| term is what makes this a likelihood rather than a
            # bare distance: without it a task whose covariance is large in the
            # directions that matter wins every query.
            cols.append(-((d @ precs[k] * d).sum(-1))
                        - logdets[k])
        else:
            raise ValueError(metric)
    return torch.stack(cols, 1)


def main():
    torch.manual_seed(0)
    encoders = [load_encoder(t) for t in TASKS]
    loaders = {t: get_task_data(t, os.path.join(ROOT, "data")) for t in TASKS}
    tr = {t: loaders[t][0] for t in TASKS}
    te = {t: loaders[t][1] for t in TASKS}

    print("building per-task statistics from the run's own checkpoints ...")
    cents, second, first, cnt = [], [], [], []
    for k, t in enumerate(TASKS):
        z = embed(encoders[k], tr[t], max_batches=40)
        cents.append(z.mean(0))
        second.append(z.t() @ z / z.shape[0])
        first.append(z.mean(0))
        cnt.append(z.shape[0])
        print(f"  {t}: n={z.shape[0]}  ||mean||={z.mean(0).norm():.4f}  "
              f"trace(E[zz^T])={torch.diagonal(second[-1]).sum():.2f}")

    pooled = sum(second) / len(second)
    mu_all = sum(first) / len(first)
    centred_pooled = pooled - torch.outer(mu_all, mu_all)
    prec_shared = precision_from(centred_pooled)
    centred = [s - torch.outer(m, m) for s, m in zip(second, first)]
    prec_task = [precision_from(S) for S in centred]
    # log|S_k| for the per-task Gaussian, estimated with the same estimator the
    # precision uses so the two are consistent.
    logdets = []
    for S in centred:
        lam = SHRINK * torch.diagonal(S).mean().clamp(min=1e-12)
        Sr = S + lam * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
        sign, logdet = torch.linalg.slogdet(Sr)
        logdets.append(logdet)
    print(f"  pooled trace {torch.diagonal(pooled).sum():.2f}  "
          f"(per-task traces {[round(float(torch.diagonal(s).sum()), 2) for s in second]})")
    print(f"  log|S_k| = {[round(float(v), 2) for v in logdets]}")

    print("\nrouting accuracy on each test set (query embedded by EVERY encoder):")
    metrics = (("shared", prec_shared, None),
               ("per-task", prec_task, None),
               ("per-task-nll", prec_task, logdets),
               ("cosine", None, None))
    print(f"{'test set':>10} | " + " | ".join(f"{m:>13}" for m, _, _ in metrics))
    zte_cache = {}
    for j, t in enumerate(TASKS):
        zte = [embed(encoders[k], te[t], max_batches=40) for k in range(len(TASKS))]
        zte_cache[t] = zte
        row = []
        for metric, precs, ld in metrics:
            s = scores(zte, cents, precs, metric, logdets=ld)
            acc = (s.argmax(-1) == j).float().mean().item() * 100
            row.append(f"{acc:12.2f}%")
        print(f"{t:>10} | " + " | ".join(row))

    print("\nhierarchical shrinkage sweep: precision_k = (1-a) * S_k + a * S_pooled")
    print(f"{'alpha':>6} | " + " | ".join(f"{t:>9}" for t in TASKS) + " |    mean")
    for alpha in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0):
        precs = [hierarchical_precision(S, centred_pooled, alpha) for S in centred]
        accs = []
        for j, t in enumerate(TASKS):
            s = scores(zte_cache[t], cents, precs, "per-task")
            accs.append((s.argmax(-1) == j).float().mean().item() * 100)
        print(f"{alpha:>6.2f} | " + " | ".join(f"{a:8.2f}%" for a in accs)
              + f" | {sum(accs) / len(accs):7.2f}%")


if __name__ == "__main__":
    main()