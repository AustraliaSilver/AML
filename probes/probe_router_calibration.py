"""Why does a per-task covariance help offline and collapse in the trainer?

The offline diagnosis (probe_router_stage3.py, run on the OLD checkpoints)
predicted that per-task covariances would route MNIST at 95.10%. The end-to-end
run with that metric routes MNIST at **1.29%** after task 2 and sends almost
every MNIST image to CIFAR-10 (`route_hist [129, 9871]`). The prediction was
wrong, so the cause has to be found on the run's own artefacts, not argued.

The candidate mechanism is calibration. With one shared precision every
candidate distance lives on the same scale, so they are comparable. With one
precision per task, task k's distance is measured after whitening by S_k, and a
domain whose own features spread widely gets small distances for *everyone*'s
queries: it wins by default. Dividing by task k's own within-task scatter puts
every score in units of "within-task sigmas", which should restore
comparability.

This sweeps the scoring rules on the CURRENT run's checkpoints, in both the
2-task regime (after CIFAR-10) and the 3-task regime (after Fashion):

    shared      one pooled precision                       (shipped before B40)
    per-task    each candidate whitened by its own S_k     (B40's proposal)
    calibrated  per-task distance / that task's own scatter
    nll         per-task distance + log|S_k|               (Gaussian NLL)
    cosine      scale-free direction match

Run:  python probes/probe_router_calibration.py
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mgn.blocks.hybrid_18 import ConvAML_18Layer_Hybrid
from train_continual_aml import get_task_data, router_features

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHRINK = 0.01
MAXB = 40          # batches per loader, same cap as the offline probe


def ckpt(task):
    return os.path.join(ROOT, f"aml_continual_after_{task}.pth")


def load_encoder(task):
    m = ConvAML_18Layer_Hybrid(K=12, max_layers=18, max_dim=48, num_classes=10,
                               norm_type="in", encoder_tail_act="none").to(DEV)
    m.load_state_dict(torch.load(ckpt(task), map_location=DEV))
    m.eval()
    return m


@torch.no_grad()
def embed(model, loader, max_batches=MAXB):
    zs = []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(DEV, non_blocking=True)
        with torch.autocast("cuda", enabled=DEV.type == "cuda",
                            dtype=torch.bfloat16):
            zs.append(router_features(model, x).float())
    return torch.cat(zs, 0)


def precision(S):
    lam = SHRINK * torch.diagonal(S).mean().clamp(min=1e-12)
    return torch.linalg.pinv(S + lam * torch.eye(S.shape[0], device=S.device,
                                                 dtype=S.dtype))


def centred(S, mu):
    return S - torch.outer(mu, mu)


def build(tasks, fit_split="train"):
    encs, tr, te = {}, {}, {}
    for t in tasks:
        encs[t] = load_encoder(t)
        tl, el = get_task_data(t, os.path.join(ROOT, "data"))
        tr[t] = tl if fit_split == "train" else el
        te[t] = el
    cents, secs = {}, {}
    for t in tasks:
        z = embed(encs[t], tr[t])
        cents[t] = z.mean(0)
        secs[t] = z.t() @ z / z.shape[0]
        print(f"  {t}: n={z.shape[0]} ||mu||={z.mean(0).norm():.4f}")
    return encs, tr, te, cents, secs


def score_all(zte, tasks, cents, precs, metric, scales=None, logdets=None):
    cols = []
    for k in tasks:
        d = zte[k] - cents[k].unsqueeze(0)
        if metric == "cosine":
            cols.append(F.normalize(zte[k], dim=-1) @ F.normalize(cents[k], dim=-1))
            continue
        P = precs if metric == "shared" else precs[k]
        d2 = (d @ P * d).sum(-1)
        if metric in ("shared", "per-task"):
            cols.append(-d2)
        elif metric == "calibrated":
            cols.append(-d2 / scales[k])
        elif metric == "nll":
            cols.append(-d2 - logdets[k])
    return torch.stack(cols, 1)


def evaluate(tasks, regime, encs, tr, te, cents, secs):
    T = len(tasks)
    pooled = sum(secs[t] for t in tasks) / T
    mu_pool = sum(cents[t] for t in tasks) / T
    prec_shared = precision(centred(pooled, mu_pool))
    prec_task = {t: precision(centred(secs[t], cents[t])) for t in tasks}

    # within-task scatter under that task's own precision
    scales = {}
    logdets = {}
    for t in tasks:
        z = embed(encs[t], tr[t])
        d = z - cents[t].unsqueeze(0)
        scales[t] = ((d @ prec_task[t] * d).sum(-1)).mean().clamp(min=1e-9)
        S = centred(secs[t], cents[t])
        lam = SHRINK * torch.diagonal(S).mean().clamp(min=1e-12)
        _, logdets[t] = torch.linalg.slogdet(
            S + lam * torch.eye(S.shape[0], device=S.device, dtype=S.dtype))

    metrics = {
        "shared": (prec_shared, None, None),
        "per-task": (prec_task, None, None),
        "calibrated": (prec_task, scales, None),
        "nll": (prec_task, None, logdets),
    }
    print(f"\n=== {regime}: candidates = {tasks} ===")
    print(f"{'test set':>10} | " + " | ".join(f"{m:>10}" for m in metrics) +
          " | " + f"{'cosine':>10}")
    zte_all = {}
    for j, t in enumerate(tasks):
        zte = {k: embed(encs[k], te[t]) for k in tasks}
        zte_all[t] = zte
        row = []
        for m, (precs, sc, ld) in metrics.items():
            s = score_all(zte, tasks, cents, precs, m, sc, ld)
            row.append((s.argmax(-1) == j).float().mean().item() * 100)
        s = score_all(zte, tasks, cents, None, "cosine")
        row.append((s.argmax(-1) == j).float().mean().item() * 100)
        print(f"{t:>10} | " + " | ".join(f"{v:9.2f}%" for v in row))

    # NLL sweep: how much pooled structure to mix in, and how strongly the
    # log-determinant term is weighted.
    print("  NLL sweep (S_k = (1-a) S_k + a S_pooled, log-det weight beta):")
    print(f"    {'alpha':>5} {'beta':>5} | " + " | ".join(f"{t:>8}" for t in tasks)
          + " |    mean")
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        for beta in (0.0, 0.5, 1.0):
            precs = {t: precision((1 - alpha) * centred(secs[t], cents[t])
                                  + alpha * centred(pooled, mu_pool)) for t in tasks}
            accs = []
            for j, t in enumerate(tasks):
                s = score_all(zte_all[t], tasks, cents, precs, "nll", None,
                              {k: beta * logdets[k] for k in tasks})
                accs.append((s.argmax(-1) == j).float().mean().item() * 100)
            print(f"    {alpha:5.2f} {beta:5.2f} | "
                  + " | ".join(f"{a:7.2f}%" for a in accs)
                  + f" | {sum(accs) / len(accs):7.2f}%")


def main():
    tasks3 = [t for t in ("mnist", "cifar10", "fashion")
              if os.path.isfile(ckpt(t))]
    tasks2 = tasks3[:2]
    print(f"checkpoints found for: {tasks3}")
    print("\n############ fitting statistics on the TEST split (what the trainer "
          "did until this commit) ############")
    encs, tr, te, cents, secs = build(tasks2, fit_split="test")
    evaluate(tasks2, "2-task regime, TEST fit", encs, tr, te, cents, secs)

    print("\n############ fitting statistics on the TRAIN split (the new default) "
          "############")
    encs, tr, te, cents, secs = build(tasks2, fit_split="train")
    evaluate(tasks2, "2-task regime, TRAIN fit", encs, tr, te, cents, secs)

    if len(tasks3) >= 3:
        print("\n############ 3-task regime, TRAIN fit ############")
        encs, tr, te, cents, secs = build(tasks3, fit_split="train")
        evaluate(tasks3, "3-task regime, TRAIN fit", encs, tr, te, cents, secs)


if __name__ == "__main__":
    main()