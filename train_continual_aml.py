"""
Continual / multi-domain benchmark for the AML supernet.

Sequence: MNIST -> CIFAR-10 -> Fashion-MNIST (all mapped to 10 classes,
3x32x32 inputs so the shared supernet consumes them directly).

Architecture:
  - 18-Layer Hybrid Supernet (mgn/blocks/hybrid_18.py) supporting Plain CNN,
    ResNet, and Vision Transformer (ViT) in a 3-Stage CoAtNet/MobileViT hierarchy.
  - Full-State Supernet Snapshots preserving stem, 18 layers, layer norm, and prototypes.
  - Zero-Replay TaskCentroidRouter tracking pristine domain feature centroids.

Reports:
  - Three distinct Continual Learning evaluation matrices:
      * A_oracle[i, j]: Task-IL (Full 18-layer snapshot retrieval with task identity) -> Zero Forgetting (BWT = +0.00%)
      * A_auto[i, j]  : Domain-IL / Class-IL (Autonomous routing via centroid cosine distance)
      * A_live[i, j]  : Standard mutable network weights (comparable to ER/EWC/DER literature)
  - Backward Transfer (BWT) and Average Forgetting metrics across all three modes.
  - Per-domain structure fingerprints (mean layer gates, mode probabilities, pi entropy)
    and pairwise L1 divergence between domain fingerprints.
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from typing import Dict

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "AML_Core_Code"))

from train import ConvAML_CIFAR_TrueDynamic, set_seed
from mgn.blocks import ConvAML_18Layer_Hybrid


def _prevent_sleep():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

_DATA_CACHE = {}


def _base_gray(mean, std):
    return [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),
    ]



# ----------------------------------------------------------------------
# Zero-Replay Task Centroid Router
# ----------------------------------------------------------------------
def router_features(model, x: torch.Tensor) -> torch.Tensor:
    """Latent used by the domain router.

    Two properties are non-negotiable here, and both were violated at the cost
    of the whole router.

    1. Computed from the encoder WITHOUT its final activation. A terminal ReLU
       forces every latent into the non-negative orthant: measured 91.6 % of
       latents were exactly 0 and 0 % were negative, which pins pairwise cosine
       similarity near 1 and leaves a margin of only ~0.005 between the correct
       and the nearest wrong domain.

    2. Always in fp32, never in reduced precision. The encoder ends in
       AdaptiveAvgPool2d((2, 2)), which averages 32x32 down to 2x2 and flattens
       the signal; the values that survive are small and nearly equal. Under
       bf16 autocast they round to the same value, so two different encoders
       produced BITWISE IDENTICAL features: maxdiff 0.000, cos 1.000000. Measured
       on the real MNIST and CIFAR-10 checkpoints, fp32 gives maxdiff 0.368 and
       bf16 gives 0.000. That single difference collapsed routing to 0 %, and no
       amount of metric tuning could have recovered it - there was no signal
       left to separate the domains.

    Falls back to the full encoder if the tail is not an activation we know how
    to strip, so this stays safe against architecture changes.
    """
    enc = model.vision_encoder
    tail = enc[-1] if len(enc) else None
    with torch.autocast(device_type=x.device.type, enabled=False):
        xi = x.float()
        if isinstance(tail, (nn.ReLU, nn.GELU, nn.Tanh, nn.LeakyReLU, nn.ELU, nn.SiLU)):
            z = enc[:-1](xi)
        else:
            z = enc(xi)
    return F.normalize(z.float().flatten(1), dim=1)


class TaskCentroidRouter(nn.Module):
    """Zero-Replay OOD / Domain Router based on running normalized feature centroids.

    Stores only T task centroids (T x D floats) without storing any exemplars/replay.
    Immune to catastrophic forgetting because past centroids are never overwritten.

    metric="mahalanobis" (default) additionally maintains ONE shared covariance
    over all seen tasks and scores with a Mahalanobis distance instead of raw
    cosine. Measured on the 3-domain production checkpoint: cosine 75.9%,
    Mahalanobis 99.7%, a supervised linear probe 92.3%. The features were always
    separable - the cosine metric was throwing that away by treating every
    direction as equally informative. Still zero-replay: one (D x D) matrix, no
    exemplars, no gradient through past tasks.

    The centroid must be the TRUE MEAN under this metric, not a unit vector.
    A tightly clustered domain has ||mean|| near 1 while a broadly spread one
    cancels itself down (measured: 0.929 for MNIST, 0.695 for CIFAR-10);
    rescaling both to the sphere collapses CIFAR-10 routing from 100.0% to 4.0%.
    """
    def __init__(self, latent_dim=1024, max_tasks=10, temperature=0.1,
                 metric="mahalanobis-task", shrink=0.01, cov_alpha=0.0,
                 logdet_weight=0.5):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_tasks = max_tasks
        # temperature scales the logits; the Mahalanobis distance is unbounded,
        # so it is divided by a much larger scale to stay in a sane range
        self.temperature = temperature
        if metric not in ("cosine", "mahalanobis", "mahalanobis-task"):
            raise ValueError(
                f"metric must be 'cosine', 'mahalanobis' or "
                f"'mahalanobis-task', got {metric!r}")
        self.metric = metric
        self.shrink = shrink
        # How much of the POOLED second moment is mixed into each task's own.
        # 0 = purely per-task, 1 = the old shared matrix. Measured on the
        # production checkpoints with statistics fitted on the TRAIN split
        # (probes/probe_router_calibration.py), mean routing accuracy over the
        # three domains at logdet_weight = 0.5:
        #     alpha   0.00    0.25    0.50    1.00 (the old shared matrix)
        #             97.52   85.50   74.95   64.83
        # so no pooling at all; the flag stays because the sweep is the
        # evidence that keeps it at 0.
        self.cov_alpha = cov_alpha
        # Weight on the log-determinant term of the per-task Gaussian NLL.
        # Without it a domain whose own features spread widely wins every query;
        # with full weight it over-corrects. Measured at alpha = 0:
        #     weight   0.00    0.50    1.00
        #     3 tasks  34.18   97.52   90.93
        #     2 tasks  51.74   99.97   99.25
        self.logdet_weight = logdet_weight
        self.register_buffer("centroids", torch.zeros(max_tasks, latent_dim))
        self.register_buffer("task_counts", torch.zeros(max_tasks, dtype=torch.long))
        self.register_buffer("num_tasks", torch.zeros(1, dtype=torch.long))
        # Shared second-moment accumulator for the Mahalanobis metric.
        # cov/zsum/cov_n hold the CLEAN pass only: the trainer feeds these from
        # the final pass over each domain, never from the per-step training
        # accumulation, because mid-training features come from encoder states
        # that no longer exist by the time the task ends.
        self.register_buffer("cov", torch.zeros(latent_dim, latent_dim))
        self.register_buffer("cov_n", torch.zeros(1, dtype=torch.long))
        self.register_buffer("zsum", torch.zeros(latent_dim))
        # Per-task second moments for metric='mahalanobis-task'. The pooled
        # matrix above is measured to be the wrong metric as soon as a third
        # task arrives (B40): it mixes three feature spaces that only share a
        # dimension, so the directions it emphasises belong to whichever task
        # contributed the most variance. One matrix per task costs
        # T x D x D floats (3.0 MB at T=3, D=512) and is still zero-replay:
        # statistics only, no exemplars.
        self.register_buffer("cov_task", torch.zeros(max_tasks, latent_dim, latent_dim))
        self.register_buffer("zsum_task", torch.zeros(max_tasks, latent_dim))
        self.register_buffer("cov_n_task", torch.zeros(max_tasks, dtype=torch.long))
        self._prec_cache = None
        self._prec_cache_n = -1
        self._prec_task_cache = None
        self._prec_task_cache_n = None

    @torch.no_grad()
    def update_task_centroid(self, task_idx: int, z_features: torch.Tensor):
        """Accumulate a batch into the running centroid, for TRAINING only.

        Called on every step, so by the end of a 6-epoch task most of what it
        gathered came from encoder states that no longer exist - and it
        deliberately does NOT touch the shared covariance, because that is
        global and cannot be corrected later without keeping every intermediate
        feature around.

        finalize_task_features() is the authoritative pass: it rebuilds the
        centroid from the final encoder and is the only thing that feeds the
        covariance. Measured: letting this blend into the final centroid put
        routing at 82.0% overall and 64.6% on the first domain, against 100.0%
        when the clean pass wins.
        """
        self._accumulate(task_idx, z_features, into_cov=False)

    @torch.no_grad()
    def finalize_task_features(self, task_idx: int, z_features: torch.Tensor):
        """Accumulate a batch into the centroid AND the shared covariance.

        The authoritative path. The trainer runs it once per domain with the
        encoder in its final state, which is exactly the encoder the router is
        queried with later, so both the centroid and the covariance describe a
        space that really exists.
        """
        self._accumulate(task_idx, z_features, into_cov=True)

    @torch.no_grad()
    def _accumulate(self, task_idx: int, z_features: torch.Tensor,
                    into_cov: bool):
        z = z_features.float().to(self.centroids.device).flatten(1)
        # Store the RAW sum. A Mahalanobis score depends on where the centroid
        # actually sits, not just on its direction - see finalize_task.
        self.centroids[task_idx] += z.sum(dim=0)
        self.task_counts[task_idx] += z.shape[0]
        if self.metric in ("mahalanobis", "mahalanobis-task") and into_cov:
            # accumulate E[zz^T] and the running feature sum over every seen
            # task; the global mean is subtracted when the precision is built
            self.cov += z.t() @ z
            self.zsum += z.sum(0)
            self.cov_n += z.shape[0]
            if self.metric == "mahalanobis-task":
                # ... and the same two quantities kept per task, so each domain
                # can be scored in the space its own centroid was measured in.
                self.cov_task[task_idx] += z.t() @ z
                self.zsum_task[task_idx] += z.sum(0)
                self.cov_n_task[task_idx] += z.shape[0]
        if task_idx + 1 > self.num_tasks.item():
            self.num_tasks[0] = task_idx + 1

    @torch.no_grad()
    def finalize_task(self, task_idx: int):
        """Turn the accumulated sum into the centroid the metric expects.

        Only 'cosine' puts the centroid on the unit sphere. 'mahalanobis' must
        keep the true mean: measured on the CIFAR-10 checkpoint, normalising
        collapsed routing from 100.0% to 4.0%, because a broadly spread domain
        has a small ||mean|| (0.695 vs 0.929 for MNIST) and rescaling it to the
        sphere moves the centroid away from the cloud it represents.
        """
        if self.task_counts[task_idx] > 0:
            mean = self.centroids[task_idx] / self.task_counts[task_idx]
            if self.metric == "cosine":
                mean = F.normalize(mean.unsqueeze(0), dim=-1).squeeze(0)
            self.centroids[task_idx] = mean

    def _precision(self) -> torch.Tensor:
        """Inverse of the shrunk shared covariance, cached per task epoch.

        pinv on a 512x512 matrix costs more than the rest of the router, and it
        only changes when new features are folded in, so memoise it on cov_n.
        """
        if self.cov_n.item() == 0:
            return torch.eye(self.latent_dim, device=self.centroids.device,
                             dtype=self.centroids.dtype)
        if self._prec_cache is not None and self._prec_cache_n == self.cov_n.item():
            return self._prec_cache
        n = float(self.cov_n.item())
        mu = self.zsum / n
        # scatter around the global mean: E[zz^T] - mu mu^T
        S = self.cov / n - torch.outer(mu, mu)
        lam = self.shrink * torch.diagonal(S).mean().clamp(min=1e-12)
        S = S + lam * torch.eye(self.latent_dim, device=S.device, dtype=S.dtype)
        self._prec_cache = torch.linalg.pinv(S)
        self._prec_cache_n = self.cov_n.item()
        return self._prec_cache

    def _precisions(self, n_tasks: int):
        """Per task: the precision matrix AND log|S_k|, cached together.

        Cached on (cov_n, cov_n_task[:n]) so the pinv is paid once per finalize,
        not once per batch. Falls back to the shared matrix for any task whose
        clean pass has not run yet, so a partially built router still scores.
        """
        key = (self.cov_n.item(), tuple(self.cov_n_task[:n_tasks].tolist()))
        if self._prec_task_cache is not None and self._prec_task_cache_n == key:
            return self._prec_task_cache
        D = self.latent_dim
        eye = torch.eye(D, device=self.centroids.device, dtype=self.cov.dtype)
        out = []
        logdets = []
        n_all = max(float(self.cov_n.item()), 1.0)
        mu_all = self.zsum / n_all
        pooled = self.cov / n_all - torch.outer(mu_all, mu_all)
        for k in range(n_tasks):
            n_k = int(self.cov_n_task[k].item())
            if n_k == 0:
                out.append(self._precision())
                logdets.append(0.0)
                continue
            mu = self.zsum_task[k] / n_k
            S = self.cov_task[k] / n_k - torch.outer(mu, mu)
            S = (1.0 - self.cov_alpha) * S + self.cov_alpha * pooled
            lam = self.shrink * torch.diagonal(S).mean().clamp(min=1e-12)
            Sr = S + lam * eye
            out.append(torch.linalg.pinv(Sr))
            _, logdet = torch.linalg.slogdet(Sr)
            logdets.append(float(logdet))
        self._prec_task_cache = (out, logdets)
        self._prec_task_cache_n = key
        return self._prec_task_cache

    def score_per_task(self, z_per_task: list) -> torch.Tensor:
        """Domain logits when the query is embedded by EACH task's own encoder.

        Training moves the encoder, so the centroid stored at task k lives in
        encoder_k's feature space while a query encoded by the current encoder
        lands somewhere else. Measured on the real MNIST -> CIFAR checkpoints,
        ||mean_old - mean_new|| = 0.800 against a within-domain spread of 0.336,
        i.e. the drift is 2.38x the spread - far enough that no choice of metric
        can bridge it (cosine, shared-covariance Mahalanobis and per-domain
        Mahalanobis all landed at 34-52% once a real second task was trained).

        Embedding the same x through each task's encoder puts every comparison
        back inside one space. It needs no extra stored state, and the encoders
        are already resident: snapshot retrieval loads them per batch anyway.
        Measured routing accuracy: 100.0% after 2 tasks, 99.1% after 3.

        z_per_task[k] must be router_features(model_with_encoder_k, x).
        """
        T_cur = max(int(self.num_tasks.item()), 1)
        proto = self.centroids[:T_cur]
        if self.metric == "cosine":
            cols = [F.normalize(z.float().flatten(1), dim=-1)
                    @ F.normalize(proto[k].unsqueeze(0), dim=-1).squeeze(0)
                    for k, z in enumerate(z_per_task[:T_cur])]
            return torch.stack(cols, dim=1) / self.temperature

        if self.metric == "mahalanobis-task":
            P, logdets = self._precisions(T_cur)
            cols = []
            # Iterate the z's that actually exist (same bound as the cosine
            # branch above). num_tasks can exceed len(z_per_task) -- it is a
            # monotonic task counter, z_per_task is one entry per stored
            # snapshot -- and indexing z_per_task[k] for k in range(T_cur)
            # raised IndexError.
            for k, z in enumerate(z_per_task[:T_cur]):
                d = z.float().flatten(1) - proto[k].unsqueeze(0)
                # Per-task Gaussian NLL, tempered: the log|S_k| term stops a
                # broadly spread domain from winning every query, and halving
                # its weight stops it from over-correcting.
                cols.append((d @ P[k] * d).sum(-1) + self.logdet_weight * logdets[k])
        else:
            P = self._precision()
            cols = []
            for k, z in enumerate(z_per_task[:T_cur]):
                d = z.float().flatten(1) - proto[k].unsqueeze(0)
                cols.append((d @ P * d).sum(-1))
        d2 = torch.stack(cols, dim=1)
        scale = d2.mean().clamp(min=1e-6)
        return (-d2 / scale) / max(self.temperature, 1e-3) * 10.0

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Domain logits; cosine similarity, or negated Mahalanobis distance."""
        T_cur = max(int(self.num_tasks.item()), 1)
        z = z.float().flatten(1)
        proto = self.centroids[:T_cur]

        if self.metric == "cosine":
            return torch.matmul(F.normalize(z, dim=-1),
                                F.normalize(proto, dim=-1).t()) / self.temperature

        if self.metric == "mahalanobis-task":
            P, logdets = self._precisions(T_cur)
            d2 = torch.stack([
                ((z - proto[k].unsqueeze(0)) @ P[k] * (z - proto[k].unsqueeze(0))).sum(-1)
                + self.logdet_weight * logdets[k]
                for k in range(T_cur)], dim=1)
        else:
            # Mahalanobis: d^2(x, c_k) = (x - c_k)^T S^-1 (x - c_k)
            P = self._precision()
            diff = z.unsqueeze(1) - proto.unsqueeze(0)          # [B, T, D]
            d2 = (diff @ P * diff).sum(-1)                      # [B, T]
        # d2 is unbounded, so normalise by the mean centroid separation to
        # land in a comparable range; argmax is unaffected by a positive
        # scale, but the softmax margin used for the open-set fallback is not
        scale = d2.mean().clamp(min=1e-6)
        return (-d2 / scale) / max(self.temperature, 1e-3) * 10.0


def get_task_data(name, data_root, workers=2, aug="basic", fast_cache=True, batch_size=256):
    """Ultra-fast data loader pipeline: In-memory pre-processed tensors for MNIST
    and FashionMNIST eliminate PIL resize bottlenecks on Windows, running epochs in ~30s.

    That speed has a cost, and the cost is silent, so it is stated here: the
    cached path applies **no augmentation**. `fast_cache` defaults to True, and
    `aug` is consulted by the CIFAR-10 branch only (`AutoAugment` when
    aug == "auto"). So in the default configuration MNIST and FashionMNIST are
    resized and normalised once and then reused unchanged, while CIFAR-10 always
    gets `RandomCrop(32, padding=4) + RandomHorizontalFlip`. The non-cached
    MNIST/Fashion branches apply `RandomCrop(32, padding=2)` and are not the
    default. The three tasks in one reported sequence are therefore not treated
    identically, and reproducing with `fast_cache=False` will not match the
    MNIST/Fashion columns. Disclosed as B54 in the audit log.
    """
    cache_key = (name, os.path.abspath(data_root))
    if fast_cache and cache_key in _DATA_CACHE:
        tr, te = _DATA_CACHE[cache_key]
        tl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                         num_workers=0, pin_memory=False, drop_last=True)
        el = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False,
                                         num_workers=0, pin_memory=False)
        return tl, el

    if name == "mnist":
        mean, std = (0.1307,) * 3, (0.3081,) * 3
        if fast_cache:
            raw_tr = torchvision.datasets.MNIST(data_root, train=True, download=True)
            raw_te = torchvision.datasets.MNIST(data_root, train=False, download=True)
            x_tr = F.interpolate(raw_tr.data.unsqueeze(1).float() / 255.0, size=(32, 32), mode='bilinear').repeat(1, 3, 1, 1)
            x_tr = (x_tr - 0.1307) / 0.3081
            x_te = F.interpolate(raw_te.data.unsqueeze(1).float() / 255.0, size=(32, 32), mode='bilinear').repeat(1, 3, 1, 1)
            x_te = (x_te - 0.1307) / 0.3081
            tr = torch.utils.data.TensorDataset(x_tr, raw_tr.targets)
            te = torch.utils.data.TensorDataset(x_te, raw_te.targets)
            _DATA_CACHE[cache_key] = (tr, te)
            tl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                             num_workers=0, pin_memory=False, drop_last=True)
            el = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False,
                                             num_workers=0, pin_memory=False)
            return tl, el
        else:
            tf_train = transforms.Compose(
                _base_gray(mean, std) + [
                    transforms.RandomCrop(32, padding=2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            tf_test = transforms.Compose(_base_gray(mean, std) + [
                transforms.ToTensor(), transforms.Normalize(mean, std)])
            tr = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=tf_train)
            te = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=tf_test)
    elif name == "fashion":
        mean, std = (0.2860,) * 3, (0.3530,) * 3
        if fast_cache:
            raw_tr = torchvision.datasets.FashionMNIST(data_root, train=True, download=True)
            raw_te = torchvision.datasets.FashionMNIST(data_root, train=False, download=True)
            x_tr = F.interpolate(raw_tr.data.unsqueeze(1).float() / 255.0, size=(32, 32), mode='bilinear').repeat(1, 3, 1, 1)
            x_tr = (x_tr - 0.2860) / 0.3530
            x_te = F.interpolate(raw_te.data.unsqueeze(1).float() / 255.0, size=(32, 32), mode='bilinear').repeat(1, 3, 1, 1)
            x_te = (x_te - 0.2860) / 0.3530
            tr = torch.utils.data.TensorDataset(x_tr, raw_tr.targets)
            te = torch.utils.data.TensorDataset(x_te, raw_te.targets)
            _DATA_CACHE[cache_key] = (tr, te)
            tl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                             num_workers=0, pin_memory=False, drop_last=True)
            el = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False,
                                             num_workers=0, pin_memory=False)
            return tl, el
        else:
            tf_train = transforms.Compose(
                _base_gray(mean, std) + [
                    transforms.RandomCrop(32, padding=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            tf_test = transforms.Compose(_base_gray(mean, std) + [
                transforms.ToTensor(), transforms.Normalize(mean, std)])
            tr = torchvision.datasets.FashionMNIST(data_root, train=True, download=True, transform=tf_train)
            te = torchvision.datasets.FashionMNIST(data_root, train=False, download=True, transform=tf_test)
    elif name == "cifar10":
        aug_ops = [transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip()]
        if aug == "auto":
            aug_ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
        tf_train = transforms.Compose(aug_ops + [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        tr = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=tf_train)
        te = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=tf_test)
    else:
        raise ValueError(name)

    tl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                     num_workers=workers, pin_memory=True,
                                     persistent_workers=workers > 0, drop_last=True)
    el = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False,
                                     num_workers=workers, pin_memory=True,
                                     persistent_workers=workers > 0)
    return tl, el



# ----------------------------------------------------------------------
# Eval: returns acc plus structure fingerprint
# ----------------------------------------------------------------------
def snapshot_bytes(snap: dict) -> int:
    """Bytes actually held by one snapshot, including the int8 scale.

    The headline storage claim ("+7.5 MB per task") is about the whole
    snapshot, not the prototype matrix. Counting only `protos` understated it
    by two orders of magnitude and printed as "0 MB".
    """

    def walk(v):
        if torch.is_tensor(v):
            return v.numel() * v.element_size()
        if isinstance(v, dict):
            return sum(walk(x) for x in v.values())
        if isinstance(v, (list, tuple)):
            return sum(walk(x) for x in v)
        return 0

    return walk(snap)


def state_targets(module: nn.Module) -> dict:
    """Live parameter/buffer tensors of `module`, keyed like its state_dict.

    `state_dict()` already returns references to the live tensors (detached but
    sharing storage), so writing into this dict writes into the module. Caching
    it once per evaluate() call avoids rebuilding the dict on every swap.
    """
    return module.state_dict()


@torch.no_grad()
def apply_state(targets: dict, src: dict) -> None:
    """Copy `src` into `targets` in place, without load_state_dict's bookkeeping.

    `load_state_dict` re-validates every key and shape on every call. Inside
    `evaluate` that call is made len(snaps) times per batch for the encoder, the
    selector, the stem, 18 layers and the head LayerNorm -- it dominated the
    measured 4.4 s evaluation cost. The keys here are ours (both sides come from
    `state_dict()` of the same module), so the validation cannot fail; it is
    pure overhead. The shape/dtype branch keeps a CPU-side snapshot usable
    against CUDA parameters, which is what the int8-quantised path relies on.
    """
    for k, v in src.items():
        t = targets[k]
        if t.shape == v.shape and t.dtype == v.dtype:
            t.copy_(v)
        else:
            t.copy_(v.to(device=t.device, dtype=t.dtype))


def ambiguous_candidate_mask(pd: torch.Tensor, amb_mask: torch.Tensor,
                             n_domains: int, amb_topk: int) -> torch.Tensor:
    """Which seen domains are worth re-trying for the ambiguous samples.

    amb_topk = 0 keeps the original behaviour: try every seen domain, which
    costs a full load_state_dict of enc + sel + stem + 18 layers + ln per domain
    per batch (~1.5M parameters copied T times) for a subset that usually has a
    clear runner-up. amb_topk = k restricts the retry to the k most likely
    domains of each ambiguous sample.

    Returns a bool tensor of shape [n_domains].
    """
    if amb_topk <= 0 or not bool(amb_mask.any()):
        return torch.ones(n_domains, dtype=torch.bool, device=pd.device)
    # pd is over STRUCTURE SLOTS; the destination is over stored SNAPSHOTS, and
    # nothing enforces len(slots) == len(snaps). A top-k index >= n_domains made
    # scatter_ write past the [B, n_domains] tensor (CUDA device-side assert /
    # "index out of bounds"), so mask the slots that have no snapshot to -inf
    # before the top-k and cap k to n_domains: every returned index is then a
    # real snapshot.
    k = min(amb_topk, n_domains)
    if k <= 0:
        return torch.zeros(n_domains, dtype=torch.bool, device=pd.device)
    pdc = pd if pd.shape[1] <= n_domains else pd.clone()
    if pd.shape[1] > n_domains:
        pdc[:, n_domains:] = float("-inf")
    topk_d = pdc.topk(k, dim=-1).indices                      # [B, k]
    allow = torch.zeros(pd.shape[0], n_domains, dtype=torch.bool, device=pd.device)
    allow.scatter_(1, topk_d, True)
    return allow[amb_mask].any(dim=0)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=None, dom_head=None, slots=None,
             snaps=None, route_tau=None, dom_truth=None, pi_mask=None, amb_topk=0,
             auto_live_thresh=0.0, route_per_task=True, route_truth=None,
             proto_row_ranges=None):
    """If slots+dom_head given, each batch is routed to its predicted domain's
    stored structure slot. If snaps given as well, the full functional state of
    that domain (vision_encoder, prototype_selector, ENTIRE prototype matrix,
    optionally int8-quantized) is swapped in before the forward pass, making
    old-domain inference immune to shared-weight drift (probe: pair-swap
    80.01% -> +protos 99.42%). Samples whose router margin < route_tau are
    evaluated under ALL seen snapshots and the highest-confidence prediction
    wins (open-set / ambiguous-input fallback). dom_truth (optional): true
    domain id of the loader, for router diagnostics (fp["route_acc"]).
    amb_topk: restrict the ambiguous-sample retry to the k most likely domains.
    0 (default) keeps the original behaviour of retrying under every seen
    snapshot, which costs a full load_state_dict of enc + sel + stem + 18
    layers + ln per domain per batch.
    auto_live_thresh: open-set fallback. If no snapshot reaches this softmax
    confidence for a sample, evaluate it with the current live weights instead.
    0 disables it. Motivated by A_auto scoring below A_live in every
    off-diagonal cell, i.e. misrouting costing more than it gained.
    route_per_task: embed the routing query with each task's own encoder
    instead of the current one, because a centroid is stored in the space of
    the encoder that produced it. Costs len(snaps) encoder forwards per batch,
    and those encoders are already loaded for snapshot retrieval anyway.

    route_truth: label used ONLY to report fp["route_acc"]. It never influences
    the prediction - dom_truth does that, which pins the domain and turns the
    run into an oracle. Without this, A_auto below A_oracle cannot be
    attributed to the router versus the snapshot, and diagnosing it offline is
    slow and, repeatedly, wrong.
    proto_row_ranges: [(lo, hi), ...] per seen task, the prototype rows that task
    owns. When given, each sample may only use the rows of the task the ROUTER
    chose. This is derived from the router's own decision, never from the true
    task id, so it leaks nothing.
    pi_mask: prototype rows the model may read out from, applied to the
    whole batch. When dom_head is None (no routing) this is how the caller
    tells the model which task's rows to use, i.e. it hands over the true
    task id at the read-out. A_live is measured that way and is therefore
    TASK-AWARE, unlike A_auto; see the protocol note in README 3.1."""
    model.eval()
    correct, total = 0, 0
    gate_soft_sum = torch.zeros(model.max_layers, device=device)
    bm_res_sum = torch.zeros(model.max_layers, device=device)
    pi_ent_sum, n = 0.0, 0
    gate_list = []
    route_hits = route_n = 0
    n_live_fallback = 0
    route_hist = torch.zeros(max(len(slots) if slots else 1, 1), device=device)
    use_route = dom_head is not None and slots is not None
    # `snaps=[]` is not None, so the old `snaps is not None` test enabled the
    # snapshot path over an empty list: the per-snapshot loop did nothing and
    # `ref_o` stayed None -> AttributeError on ref_o.shape[1]. An empty list
    # means "no snapshots", which is exactly the snaps=None behaviour.
    use_snaps = use_route and snaps is not None and len(snaps) > 0
    cur_stem = cur_layers = cur_ln = None
    # Cached live-tensor targets for the in-place snapshot swap. Rebuilt per
    # evaluate() call, reused for every batch and every snapshot.
    dst_enc = state_targets(model.vision_encoder)
    dst_sel = state_targets(model.prototype_selector)
    dst_stem = state_targets(model.stem) if hasattr(model, "stem") else None
    dst_layers = state_targets(model.layers) if hasattr(model, "layers") else None
    dst_ln = state_targets(model.ln) if hasattr(model, "ln") else None
    dst_sdelta = state_targets(model.structure_delta) if hasattr(model, "structure_delta") else None
    dst_bhead = state_targets(model.block_mode_head) if hasattr(model, "block_mode_head") else None
    if use_snaps:
        cur_enc = {k: v.detach().clone() for k, v in dst_enc.items()}
        cur_sel = {k: v.detach().clone() for k, v in dst_sel.items()}
        cur_protos = model.proto_weights.detach().clone()
        if dst_layers is not None:
            cur_stem = {k: v.detach().clone() for k, v in dst_stem.items()}
            cur_layers = {k: v.detach().clone() for k, v in dst_layers.items()}
            if dst_ln is not None:
                cur_ln = {k: v.detach().clone() for k, v in dst_ln.items()}
        # Structure parameters (structure_prototypes / block_mode_prototypes or legacy)
        _sheads = snaps[0].get("struct_heads") if snaps else None
        if _sheads is not None:
            if hasattr(model, "structure_prototypes"):
                cur_sprotos = model.structure_prototypes.detach().clone()
                cur_bmprotos = model.block_mode_prototypes.detach().clone()
            elif hasattr(model, "structure_delta"):
                cur_sdelta = {k: v.detach().clone() for k, v in
                              model.structure_delta.state_dict().items()}
                cur_bhead = {k: v.detach().clone() for k, v in
                             model.block_mode_head.state_dict().items()}
        deq_protos = []
        for d in range(len(snaps)):
            pr = snaps[d]["protos"]
            if isinstance(pr, dict):
                deq_protos.append(pr["q"].to(device).float() * pr["scale"])
            else:
                deq_protos.append(pr.to(device))
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        override = None
        amb = None
        if use_route:
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp_dtype is not None,
                                                     dtype=amp_dtype or torch.bfloat16):
                if route_per_task and len(snaps) > 0 and hasattr(dom_head, "score_per_task"):
                    # Embed the query with EACH task's own encoder. The centroid
                    # for task k was built in encoder_k's space; encoding x with
                    # the current encoder compares two different spaces, and the
                    # drift between them is larger than the within-domain spread.
                    zk = []
                    for d in range(len(snaps)):
                        apply_state(dst_enc, snaps[d]["enc"])
                        zk.append(router_features(model, x))
                    apply_state(dst_enc, cur_enc)
                    dom_logits = dom_head.score_per_task(zk)[:, :len(slots)]
                else:
                    z = router_features(model, x)
                    # Restrict router to the domains seen so far
                    dom_logits = dom_head(z)[:, :len(slots)]      # [B,T]
            pd = dom_logits.float().softmax(-1)
            top2 = pd.topk(min(2, pd.shape[1]), dim=-1).values
            margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else torch.zeros_like(top2[:, 0]))
            pred_argmax = dom_logits.argmax(-1)
            route_n += pred_argmax.numel()
            # route_truth scores the router WITHOUT letting it steer the
            # prediction. dom_truth pins d_pred to the oracle, so it measures
            # the snapshot rather than the router - which is why the autonomous
            # protocol had no routing telemetry at all.
            if route_truth is not None:
                route_hits += int((pred_argmax == route_truth).sum().item())
            if dom_truth is not None:
                # oracle: the domain is given, so the router is bypassed and
                # scoring it here would just double-count route_truth
                d_pred = torch.full((x.shape[0],), dom_truth, dtype=torch.long, device=device)
                amb = None
            else:
                d_pred = pred_argmax
                if use_snaps and route_tau is not None:
                    amb = margin < route_tau                           # ambiguous -> try several domains
            route_hist += torch.bincount(pred_argmax.clamp(max=route_hist.numel() - 1),
                                         minlength=route_hist.numel())
            # Which snapshots are worth loading at all.
            #
            # Confident samples need only their predicted domain. For ambiguous
            # ones the original code retried EVERY seen domain, which means a
            # full load_state_dict of enc + sel + stem + 18 layers + ln per
            # snapshot per batch -- ~1.5M parameters copied T times for a subset
            # that usually has a clear runner-up. amb_topk > 0 restricts that
            # retry to the k most likely domains; 0 preserves the original
            # try-everything behaviour.
            need_dom = ambiguous_candidate_mask(
                pd,
                amb if amb is not None else torch.zeros_like(d_pred, dtype=torch.bool),
                len(snaps), amb_topk)
            g = torch.stack([slots[d]["gate"].to(device) for d in range(len(slots))])   # [T,L]
            if "bm_mode" in slots[0]:
                b = torch.stack([slots[d]["bm_mode"].to(device) for d in range(len(slots))]) # [T,L,3]
            else:
                b = torch.stack([slots[d]["bm_res"].to(device) for d in range(len(slots))]) # [T,L]
                b = torch.stack([1.0 - b, b], dim=-1)                                     # [T,L,3]
            # Structure slot of the domain the ROUTER chose, for the confident
            # samples. The ambiguous retry below must NOT reuse this: it loads
            # snapshot d's weights, so it needs d's slot (see there).
            override = (g[d_pred], b[d_pred])

            # Per-sample prototype mask, built from the ROUTER's own choice.
            # The prototype selector is shared across tasks, and each task only
            # ever trained its own rows. Scoring a sample against rows that were
            # trained on a different domain is what cost A[2,2] everything:
            # measured 32.63 % with the correct rows allowed and 10.86 % with
            # all of them. This uses d_pred, never the true task id, so it
            # leaks nothing the router did not already decide.
            if proto_row_ranges is not None:
                pm = torch.zeros(x.shape[0], model.K, dtype=torch.bool, device=device)
                for d2 in range(len(proto_row_ranges)):
                    lo, hi = proto_row_ranges[d2]
                    pm[d_pred == d2, lo:hi] = True
            else:
                pm = None   # fall back to pi_mask (which may be [K]) below

        if use_snaps:
            amb_t = amb if amb is not None else torch.zeros_like(d_pred, dtype=torch.bool)
            has_amb = bool(amb_t.any())
            outs_conf, pi_conf = [], []
            ls_full = torch.zeros(x.shape[0], model.max_layers, device=device)
            bm_wsum = torch.zeros(model.max_layers, device=device)
            wtot = 0
            best_conf = best_out = best_pi = best_ls = None
            if has_amb:
                best_conf = torch.full((int(amb_t.sum()),), -1.0, device=device)
            ac = dict(enabled=amp_dtype is not None, dtype=amp_dtype or torch.bfloat16)
            for d in range(len(snaps)):
                idx_c = (~amb_t) & (d_pred == d)
                # Load a snapshot only if this domain actually has work: either
                # confident samples routed to it, or (with amb_topk) it is one of
                # the retry candidates for an ambiguous sample.
                want_amb = has_amb and bool(need_dom[d])
                if not idx_c.any() and not want_amb:
                    continue
                apply_state(dst_enc, snaps[d]["enc"])
                apply_state(dst_sel, snaps[d]["sel"])
                model.proto_weights.data.copy_(deq_protos[d])
                if "layers" in snaps[d] and dst_layers is not None:
                    apply_state(dst_stem, snaps[d]["stem"])
                    apply_state(dst_layers, snaps[d]["layers"])
                    if "ln" in snaps[d] and dst_ln is not None:
                        apply_state(dst_ln, snaps[d]["ln"])
                if _sheads is not None and "struct_heads" in snaps[d]:
                    sh = snaps[d]["struct_heads"]
                    if hasattr(model, "structure_prototypes"):
                        model.structure_prototypes.data.copy_(sh["sprotos"])
                        model.block_mode_prototypes.data.copy_(sh["bmprotos"])
                    elif dst_sdelta is not None:
                        apply_state(dst_sdelta, sh["delta"])
                        apply_state(dst_bhead, sh["block"])
                if idx_c.any():
                    with torch.amp.autocast('cuda', **ac):
                        _pm = pm[idx_c] if pm is not None else pi_mask
                        o, st, p, dg = model(x[idx_c], tau=0.1,
                                             struct_override=(override[0][idx_c], override[1][idx_c]),
                                             pi_mask=_pm)
                    outs_conf.append((idx_c, o))
                    pi_conf.append((idx_c, p.float()))
                    ls_full[idx_c] = dg["layer_soft"].float()
                    st_bm = st.block_mode.float()
                    bm_wsum += (st_bm[:, :, 1] if st_bm.shape[-1] >= 2 else st_bm[:, :, 0]).mean(0) * idx_c.sum().item()
                    wtot += idx_c.sum().item()
                if want_amb:
                    with torch.amp.autocast('cuda', **ac):
                        # An ambiguous sample is being retried under THIS
                        # snapshot, so the rows it may use are this snapshot's
                        # rows - not the ones its router argmax pointed at.
                        _pma = pi_mask
                        if proto_row_ranges is not None and d < len(proto_row_ranges):
                            _lo, _hi = proto_row_ranges[d]
                            _pma = torch.zeros(int(amb_t.sum()), model.K,
                                               dtype=torch.bool, device=device)
                            _pma[:, _lo:_hi] = True
                        # ... and for the same reason its STRUCTURE slot must be
                        # this snapshot's. The weights loaded above are d's and
                        # the rows just selected are d's, so taking the gates and
                        # block modes from d_pred (as this used to) evaluated
                        # domain d's weights under another domain's architecture:
                        # `slots[d]` was collected from d's own data and weights,
                        # so it describes d and nothing else. Live by default:
                        # route_tau = 0.6 and amb_topk = 0 retry every seen
                        # domain, so this path runs on every default run.
                        _n_amb = int(amb_t.sum())
                        _ov = (g[d].unsqueeze(0).expand(_n_amb, -1),
                               b[d].unsqueeze(0).expand(_n_amb, -1, -1))
                        o2, _, p2, dg2 = model(x[amb_t], tau=0.1,
                                               struct_override=_ov,
                                               pi_mask=_pma)
                    conf2 = o2.float().softmax(-1).max(-1).values
                    upd = conf2 > best_conf
                    if best_out is None:
                        # zeros, not empty: a row whose confidence never
                        # beats the initial -1 would otherwise be read back as
                        # uninitialised memory rather than as "not better".
                        best_out = torch.zeros(amb_t.sum().item(), o2.shape[1],
                                               device=device, dtype=torch.float32)
                        best_pi = torch.zeros(amb_t.sum().item(), p2.shape[1], device=device)
                        best_ls = torch.zeros(amb_t.sum().item(), model.max_layers, device=device)
                    if upd.any():
                        best_out[upd] = o2[upd].float()
                        best_pi[upd] = p2[upd].float()
                        best_ls[upd] = dg2["layer_soft"].float()[upd]
                        best_conf[upd] = conf2[upd]
            apply_state(dst_enc, cur_enc)
            apply_state(dst_sel, cur_sel)
            model.proto_weights.data.copy_(cur_protos)
            if cur_layers is not None:
                apply_state(dst_stem, cur_stem)
                apply_state(dst_layers, cur_layers)
                if cur_ln is not None:
                    apply_state(dst_ln, cur_ln)
            if _sheads is not None:
                if hasattr(model, "structure_prototypes"):
                    model.structure_prototypes.data.copy_(cur_sprotos)
                    model.block_mode_prototypes.data.copy_(cur_bmprotos)
                elif hasattr(model, "structure_delta"):
                    apply_state(dst_sdelta, cur_sdelta)
                    apply_state(dst_bhead, cur_bhead)
            if not outs_conf and best_out is None:
                # No snapshot produced any output for this batch: `snaps` was
                # non-empty but every row routed to a domain with no snapshot
                # (d_pred >= len(snaps)) and none was retried. Falling through
                # would read ref_o.shape[1] on None (AttributeError). Score
                # with the live weights -- what use_snaps=False does, and the
                # same fallback the open-set branch below takes.
                with torch.amp.autocast('cuda', **ac):
                    out_deg, st_deg, pi_deg, dg_deg = model(x, tau=0.1, pi_mask=pi_mask)
                out_deg = out_deg.float()
                pi_deg = pi_deg.float()
                ls_deg = dg_deg["layer_soft"].float()
                gate_soft_sum += ls_deg.mean(0)
                gate_list.append(ls_deg)
                bmf_deg = st_deg.block_mode.float()
                bm_res_sum += (bmf_deg[:, :, 1] if bmf_deg.shape[-1] >= 2
                               else bmf_deg[:, :, 0]).mean(0)
                correct += (out_deg.argmax(-1) == y).sum().item()
                total += y.numel()
                pi_ent_sum += -(pi_deg * torch.log(pi_deg.clamp(min=1e-9))).sum(-1).sum().item()
                n += x.shape[0]
                continue
            ref_o = outs_conf[0][1] if outs_conf else best_out
            # zeros, not empty: every row below is written in the normal case,
            # but the normal case depends on len(snaps) == len(slots), which
            # nothing enforces. A shorter `snaps` leaves the rows whose d_pred
            # has no snapshot reading uninitialised memory instead of failing.
            # (Same defect class as the scatter destination fixed in
            # hybrid_18.forward - "gate == 0 makes the blend the identity, it
            # does not write anything".)
            out = torch.zeros(x.shape[0], ref_o.shape[1], device=x.device, dtype=ref_o.dtype)
            pi = torch.zeros(x.shape[0], (pi_conf[0][1] if pi_conf else best_pi).shape[1],
                             device=device)
            for m_, o_ in outs_conf:
                out[m_] = o_
            for m_, p_ in pi_conf:
                pi[m_] = p_
            if best_out is not None:
                out[amb_t] = best_out.to(out.dtype)
                pi[amb_t] = best_pi
                ls_full[amb_t] = best_ls

            # Open-set fallback: if NO snapshot produced a confident prediction for
            # a sample, it is likely out-of-distribution for every seen domain and
            # the current (live) weights are a better bet than a confidently wrong
            # snapshot. Measured motivation: A_auto fell below A_live in every
            # off-diagonal cell, i.e. misrouting was costing more than the
            # snapshot protocol gained.
            if auto_live_thresh > 0:
                conf_all = out.float().softmax(-1).max(-1).values
                lowc = conf_all < auto_live_thresh
                if bool(lowc.any()):
                    with torch.amp.autocast('cuda', **ac):
                        # Run the LIVE model under its OWN structure. This used
                        # to pass g[d_pred]/b[d_pred], forcing the current
                        # weights to execute the rejected domain's gate and
                        # block-mode slot: architecture from one model, weights
                        # from another, so the result was neither A_live nor
                        # that domain's prediction. (Same defect class the
                        # ambiguous retry was fixed for in B46.)
                        o_live, st_l, p_l, dg_l = model(x[lowc], tau=0.1,
                                                        pi_mask=pi_mask)
                    out[lowc] = o_live
                    pi[lowc] = p_l.float()
                    ls_full[lowc] = dg_l["layer_soft"].float()
                    n_live_fallback += int(lowc.sum().item())

            gate_soft_sum += ls_full.mean(0)
            bm_res_sum += bm_wsum / max(wtot, 1)
            gate_list.append(ls_full)
            correct += (out.argmax(-1) == y).sum().item()
            total += y.numel()
            pi_ent_sum += -(pi * torch.log(pi.clamp(min=1e-9))).sum(-1).sum().item()
            n += x.shape[0]
        else:
            with torch.amp.autocast('cuda', enabled=amp_dtype is not None, dtype=amp_dtype or torch.bfloat16):
                out, struct, pi, diag = model(x, tau=0.1, struct_override=override, pi_mask=pi_mask)
            correct += (out.argmax(-1) == y).sum().item()
            total += y.numel()
            ls = diag["layer_soft"].float()
            gate_soft_sum += ls.mean(0)
            gate_list.append(ls)
            bmf = struct.block_mode.float()
            if bmf.shape[-1] >= 2:
                bm_res_sum += bmf[:, :, 1].mean(0)
            else:
                bm_res_sum += bmf[:, :, 0].mean(0)
            pif = pi.float()
            pi_ent_sum += -(pif * torch.log(pif.clamp(min=1e-9))).sum(-1).sum().item()
            n += pif.shape[0]
    fp = {
        "gate": (gate_soft_sum / len(loader)).cpu(),
        "bm_res": (bm_res_sum / len(loader)).cpu(),
        "pi_H": pi_ent_sum / n,
        "gate_tensor": torch.cat(gate_list, 0),
        "route_acc": (route_hits / route_n) if route_n else None,
        "route_hist": route_hist.cpu(),
        "live_fallback_frac": (n_live_fallback / total) if total else 0.0,
    }
    return correct / total, fp


@torch.no_grad()
def collect_structure_slot(model, loader, device, amp_dtype=None):
    """Mean soft structure of the CURRENT model on one domain -> its slot."""
    model.eval()
    gate_sum = torch.zeros(model.max_layers, device=device)
    bm_mode_list = []
    nb = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=amp_dtype is not None,
                                dtype=amp_dtype or torch.bfloat16):
            _, _, _, diag = model(x, tau=0.1)
        gate_sum += diag["layer_soft"].float().mean(0)
        p = torch.softmax(diag["bm_logits"].float(), dim=-1)
        bm_mode_list.append(p.mean(0))
        nb += 1
    bm_mode_mean = torch.stack(bm_mode_list).mean(0)  # [L, C_modes]
    res = {
        "gate": (gate_sum / max(nb, 1)).cpu(),
        "bm_res": (bm_mode_mean[:, 1] if bm_mode_mean.shape[-1] >= 2 else bm_mode_mean[:, 0]).cpu(),
    }
    if bm_mode_mean.shape[-1] == 3:
        res["bm_mode"] = bm_mode_mean.cpu()
        res["bm_plain"] = bm_mode_mean[:, 0].cpu()
        res["bm_vit"] = bm_mode_mean[:, 2].cpu()
    return res



# ----------------------------------------------------------------------
# Episodic memory replay buffer (balanced per class, canonical images)
# ----------------------------------------------------------------------
def build_replay_buffer(dataset, per_class=100, num_classes=10):
    """Balanced exemplar set stored with canonical normalization.
    Vectorized with direct tensor operations for instant execution (<0.02s)."""
    ds_name = type(dataset).__name__
    targets = torch.as_tensor(dataset.targets)
    chosen = []
    for c in range(num_classes):
        cls_idx = (targets == c).nonzero(as_tuple=True)[0]
        sel = cls_idx[torch.randperm(len(cls_idx))[:per_class]]
        chosen.append(sel)
    idx = torch.cat(chosen)

    if ds_name == "CIFAR10":
        # dataset.data is numpy [N, 32, 32, 3] uint8
        data_np = dataset.data[idx.numpy()]
        data_t = torch.from_numpy(data_np).permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
        xs = (data_t - mean) / std
    elif ds_name in ("MNIST", "FashionMNIST"):
        mean_val = 0.1307 if ds_name == "MNIST" else 0.2860
        std_val = 0.3081 if ds_name == "MNIST" else 0.3530
        raw = dataset.data[idx].float().unsqueeze(1) / 255.0
        data_t = F.interpolate(raw, size=(32, 32), mode='bilinear').repeat(1, 3, 1, 1)
        xs = (data_t - mean_val) / std_val
    else:
        raise ValueError(ds_name)

    ys = targets[idx]
    return xs, ys


class ReplayBuffer:
    def __init__(self):
        self.x = None  # [N, C, H, W] on CPU
        self.y = None
        self.t = None  # task id per exemplar

    def extend(self, xs, ys, ts):
        if self.x is None:
            self.x, self.y, self.t = xs, ys, ts
        else:
            self.x = torch.cat([self.x, xs])
            self.y = torch.cat([self.y, ys])
            self.t = torch.cat([self.t, ts])

    def __len__(self):
        return 0 if self.x is None else self.x.shape[0]

    def sample(self, n, device):
        idx = torch.randint(0, self.x.shape[0], (n,))
        return (self.x[idx].to(device, non_blocking=True),
                self.y[idx].to(device, non_blocking=True),
                self.t[idx].to(device, non_blocking=True))

    def loader(self, batch_size=256):
        ds = torch.utils.data.TensorDataset(self.x, self.y, self.t)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                           num_workers=0)


class ProtoIsolator:
    """
    Per-task ownership of prototype rows with THREE zones during task i:
      [0, s)     : past-task rows  -> FROZEN (snapshot-restored every step)
      [s, e)     : current rows    -> trainable
      [e, K)     : future rows     -> FROZEN too!
    Protecting FUTURE rows matters as much as past ones: if the router
    learns to lean on rows that a LATER task re-initializes, that routing
    dependence itself becomes forgetting. Masking their gradients forces
    each task's routing to live entirely on stable rows.
    """

    def __init__(self, model):
        self.model = model
        self.p = model.proto_weights
        self.K = self.p.shape[0]
        self.frozen_hi = 0          # rows [0, frozen_hi) frozen (past)
        self.future_lo = self.K     # rows [future_lo, K) frozen (future)
        self.snapshot = None

        self.mask = torch.ones_like(self.p)
        def hook(grad):
            return grad * self.mask
        self.p.register_hook(hook)

        self.arch_items = []
        for name in ('block_mode_prototypes', 'structure_prototypes'):
            if hasattr(model, name):
                param = getattr(model, name)
                mask = torch.ones_like(param)
                def make_hook(m):
                    return lambda grad: grad * m
                param.register_hook(make_hook(mask))
                self.arch_items.append({'param': param, 'mask': mask, 'snap': None})

    def begin_task(self, task_idx, num_tasks, reinit=True):
        s = round(task_idx * self.K / num_tasks)
        e = round((task_idx + 1) * self.K / num_tasks)
        self.frozen_hi = s
        self.future_lo = e
        self.snapshot = self.p.data.clone()
        for item in self.arch_items:
            item['snap'] = item['param'].data.clone()
            item['mask'].fill_(1.0)
            if self.frozen_hi > 0: item['mask'][:self.frozen_hi] = 0.0
            if self.future_lo < self.K: item['mask'][self.future_lo:] = 0.0

        if task_idx > 0 and reinit and e > s:
            if hasattr(self.model, 'exec_'):
                import math as _math
                from mgn.layout import get_prototype_layout, get_second_conv_layout
                max_layers = getattr(self.model, 'max_layers', 18)
                max_dim = getattr(self.model, 'max_dim', 64)
                _, offsets, _, wo_offset, bo_offset = get_prototype_layout(max_layers, max_dim, 10)
                offsets2 = get_second_conv_layout(max_layers, max_dim, 10)
                with torch.no_grad():
                    for i in range(max_layers):
                        w_start, w_end, b_start, b_end, in_c = offsets[i]
                        fan_in = in_c * 3 * 3
                        bound = _math.sqrt(6.0 / fan_in)
                        self.p.data[s:e, w_start:w_end].uniform_(-bound, bound)
                        self.p.data[s:e, b_start:b_end].zero_()
                    for (w2_start, w2_end, b2_start, b2_end) in offsets2:
                        fan_in2 = max_dim * 3 * 3
                        bound2 = _math.sqrt(6.0 / fan_in2)
                        self.p.data[s:e, w2_start:w2_end].uniform_(-bound2, bound2)
                        self.p.data[s:e, b2_start:b2_end].zero_()
                    fan_in_head = 16 * max_dim
                    bound_head = _math.sqrt(6.0 / fan_in_head)
                    self.p.data[s:e, wo_offset:bo_offset].uniform_(-bound_head, bound_head)
                    self.p.data[s:e, bo_offset:].zero_()
            else:
                import math as _math
                head_dim = getattr(self.model, 'head_dim', 768)
                bound = _math.sqrt(6.0 / head_dim)
                with torch.no_grad():
                    if hasattr(self.model, 'wo_size'):
                        self.p.data[s:e, :self.model.wo_size].uniform_(-bound, bound)
                        self.p.data[s:e, self.model.wo_size:].zero_()
                    else:
                        self.p.data[s:e].uniform_(-bound, bound)
        
        self.mask.fill_(1.0)
        if self.frozen_hi > 0: self.mask[:self.frozen_hi] = 0.0
        if self.future_lo < self.K: self.mask[self.future_lo:] = 0.0
        return s, e

    @torch.no_grad()
    def mask_grad(self):
        # Handled automatically by backward hooks
        pass

    @torch.no_grad()
    def restore(self):
        if self.snapshot is None:
            return
        # Tensorized restore: capturable by CUDA Graphs
        self.p.data.copy_(torch.where(self.mask > 0, self.p.data, self.snapshot))
        for item in self.arch_items:
            if item['snap'] is not None:
                item['param'].data.copy_(torch.where(item['mask'] > 0, item['param'].data, item['snap']))


class FisherEWC:
    """Elastic Weight Consolidation over the shared trunk.

    Motivation: the previous stability mechanism was L2-SP anchored on
    (vision_encoder, prototype_selector) only, and shipped DISABLED
    (--anchor-lam defaults to 0.0), so A_live collapsed across tasks
    (98.05% -> 3.52% -> 28.99%). The alternative of storing a full 18-layer
    snapshot per task is what this class exists to make unnecessary: EWC needs
    one diagonal Fisher vector (one scalar per weight) plus the parameter value
    at the end of each task, i.e. ~1.47M floats, shared across all tasks.

        L_total = CE + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

    F is accumulated online as an *unbiased running mean of squared gradients*
    over post-task batches, which is the standard diagonal-Fisher estimate and
    needs no separate backward pass.
    """

    def __init__(self, model, ewc_lambda: float = 0.0):
        self.model = model
        self.lambda_ = ewc_lambda
        self.enabled = ewc_lambda > 0.0
        self._fisher: Dict[str, torch.Tensor] = {}   # within-task accumulator
        # Across tasks, EWC needs sum_t F^(t) and sum_t F^(t) * theta*^(t).
        # Storing a running mean of theta* alongside a running sum of F is NOT
        # equivalent: task 1's Fisher would be spent pulling theta towards
        # theta*^(2), which already forgot task 1. The exact expansion
        #   sum_t F^(t)(theta - theta*^(t))^2
        #     = (sum_t F^(t)) * theta^2 - 2 (sum_t F^(t) theta*^(t)) * theta + c
        # needs only the two accumulators below, and c is theta-independent so it
        # never enters the gradient. Same memory as the old (F, theta*) pair.
        self._fisher_all: Dict[str, torch.Tensor] = {}   # A = sum_t F^(t)
        self._star_all: Dict[str, torch.Tensor] = {}      # B = sum_t F^(t) theta*^(t)
        self._n = 0

    @staticmethod
    def _shared_trunk(model) -> Dict[str, torch.nn.Parameter]:
        """Parameters shared by every task and therefore subject to drift."""
        out = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # architecture prototypes are handled by ProtoIsolator, prototypes
            # by per-task rows, and the domain router lives outside the model.
            if any(k in n for k in ("structure_prototypes", "block_mode_prototypes",
                                    "mode_prototypes", "proto_weights")):
                continue
            out[n] = p
        return out

    def penalty(self) -> torch.Tensor:
        # Guard on _fisher_all (the across-task accumulator), NOT _fisher (the
        # within-task one, which consolidate() resets). Guarding on the latter
        # made the penalty identically zero for every task after the first.
        if not self.enabled or not self._fisher_all:
            return torch.zeros((), device=next(self.model.parameters()).device)
        total = None
        params = dict(self.model.named_parameters())
        for n, a in self._fisher_all.items():
            p = params.get(n)
            b = self._star_all.get(n)
            if p is None or b is None:
                continue
            # Completed square: A(theta - B/A)^2
            #   = A theta^2 - 2 B theta + B^2/A
            # has exactly the gradient of the true multi-task quadratic
            # (2A theta - 2B) while staying non-negative and vanishing at the
            # Fisher-weighted mean B/A. Writing it as A theta^2 - 2 B theta
            # instead would report a large NEGATIVE number at the optimum,
            # because the theta-independent term is dropped.
            mu = b / a.clamp(min=1e-12)
            d = a * (p - mu) ** 2
            total = d.sum() if total is None else total + d.sum()
        if total is None:
            dev = next(self.model.parameters()).device
            return torch.zeros((), device=dev)
        return 0.5 * self.lambda_ * total

    @torch.no_grad()
    def observe_batch(self) -> None:
        """Accumulate squared gradients from the gradients already on the params.

        Must be called after GradScaler.unscale_() and BEFORE
        clip_grad_norm_(), otherwise the statistics carry the loss scale (or a
        norm rescale) and the Fisher is off by a constant factor.
        Reading .grad rather than calling autograd.grad again avoids an extra
        backward pass per step and keeps the main graph untouched.
        """
        if not self.enabled:
            return
        for n, p in self._shared_trunk(self.model).items():
            g = p.grad
            if g is None:
                continue
            sq = g.detach().float() ** 2
            prev = self._fisher.get(n)
            self._fisher[n] = sq if prev is None else prev + sq
        self._n += 1

    @torch.no_grad()
    def consolidate(self) -> None:
        """Close the task window: average this task's Fisher and fold it into
        the two across-task accumulators.

        A += F^(t) and B += F^(t) * theta*^(t), where theta*^(t) is this task's
        own weights at the moment it finished. Pairing the running sum of F with
        a single re-pinned theta* was the defect: after task 2 it scored task
        1's Fisher against task 2's weights, i.e. it anchored the model to a
        point that had already forgotten task 1.
        """
        if not self.enabled or self._n == 0:
            return
        inv = 1.0 / float(self._n)
        params = dict(self.model.named_parameters())
        for n, f in self._fisher.items():
            f.mul_(inv)
            p = params.get(n)
            if p is None:
                continue
            prev_a = self._fisher_all.get(n)
            self._fisher_all[n] = f.clone() if prev_a is None else prev_a + f
            prev_b = self._star_all.get(n)
            contrib = f * p.detach().float()
            self._star_all[n] = contrib if prev_b is None else prev_b + contrib
        self._fisher = {}
        self._n = 0

    def state_bytes(self) -> int:
        return sum(v.numel() for v in self._fisher_all.values()) + \
            sum(v.numel() for v in self._star_all.values())


def consolidate(model, replay: ReplayBuffer, device, amp_dtype, epochs=1, lr=1e-4,
                isolator: ProtoIsolator = None, dom_head=None):
    """Low-LR pass over the pure replay set after each task: rebalances the
    shared prototypes across ALL seen domains (weights only, arch frozen).
    Also keeps the domain router calibrated on all seen tasks."""
    if len(replay) == 0:
        return
    model.train()
    arch_names = ("structure_prototypes", "block_mode_prototypes", "mode_prototypes")
    arch_params = [p for n_, p in model.named_parameters() if any(k in n_ for k in arch_names)]
    other_params = [p for n_, p in model.named_parameters() if not any(k in n_ for k in arch_names) and p.requires_grad]
    opt = optim.AdamW(other_params, lr=lr, weight_decay=1e-4)
    opt_arch = optim.AdamW(arch_params, lr=lr * 0.2, weight_decay=1e-4)
    opt_dom = None
    if dom_head is not None and not isinstance(dom_head, TaskCentroidRouter):
        opt_dom = optim.AdamW(dom_head.parameters(), lr=1e-3, weight_decay=0.0)
    ce = lambda lg, tg: F.cross_entropy(lg.float(), tg, label_smoothing=0.1)

    for _ in range(epochs):
        for xb, yb, tb in replay.loader(batch_size=64):
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            opt.zero_grad(set_to_none=True)
            opt_arch.zero_grad(set_to_none=True)
            if opt_dom is not None:
                opt_dom.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp_dtype is not None,
                                    dtype=amp_dtype or torch.bfloat16):
                out, _, _, diag = model(xb, tau=0.8)
                loss = ce(out, yb)
                if opt_dom is not None and "z" in diag:
                    z_norm = F.normalize(diag["z"].float(), dim=1)
                    dom_logits = dom_head(z_norm)
                    valid = tb.max().item() + 1
                    loss = loss + F.cross_entropy(dom_logits[:, :valid], tb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if isolator is not None:
                isolator.mask_grad()
            torch.nn.utils.clip_grad_norm_(other_params, 5.0)
            torch.nn.utils.clip_grad_norm_(arch_params, 5.0)
            opt.step()
            opt_arch.step()
            if opt_dom is not None:
                torch.nn.utils.clip_grad_norm_(dom_head.parameters(), 5.0)
                opt_dom.step()
            if isolator is not None:
                isolator.restore()


# ----------------------------------------------------------------------
# Train one task (bi-level, short warmup freeze, replay against forgetting)
# ----------------------------------------------------------------------
def train_task(model, name, epochs, freeze_epochs, device, amp_dtype,
               scaler=None, w_pi_final=0.0, log_prefix="", replay: ReplayBuffer = None,
               isolator: ProtoIsolator = None, dom_head=None, task_idx=0,
               anchor=None, anchor_lam=1e-3, aug="basic", workers=2, steps=0,
               arch_lr=3e-3, depth_cost=0.0, flops_cost=0.0, depth_target=0.0, ewc=None,
               n_tasks: int = 1):
    trainloader, _ = get_task_data(name, os.path.join(_ROOT, "data"), workers=workers, aug=aug)
    _, testloader = get_task_data(name, os.path.join(_ROOT, "data"), workers=workers)

    arch_names = ("structure_prototypes", "block_mode_prototypes", "mode_prototypes")
    arch_params = [p for n_, p in model.named_parameters() if any(k in n_ for k in arch_names)]
    other_params = [p for n_, p in model.named_parameters() if not any(k in n_ for k in arch_names) and p.requires_grad]

    opt = optim.AdamW(other_params, lr=1e-3, weight_decay=1e-4)
    # arch_lr: AdamW's step size is ~lr regardless of gradient magnitude, so the
    # architecture logits can travel at most lr * n_steps. At the previous 2e-4 a
    # whole 585-step task could move structure_prototypes by ~0.12, while flipping
    # sigmoid(delta + 2.0) past the 0.5 gate needs |delta| ~ 2.0. The depth gates
    # were therefore mathematically unable to close, and the FLOPs short-circuit
    # had nothing to short-circuit. 3e-3 gives ~1.75 of travel per task.
    opt_arch = optim.AdamW(arch_params, lr=arch_lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    sch_arch = optim.lr_scheduler.CosineAnnealingLR(opt_arch, T_max=epochs)

    anchor_pairs = None
    if anchor is not None:
        anchor_pairs = [(p, a) for p, a in anchor]

    ce = lambda lg, tg: F.cross_entropy(lg.float(), tg, label_smoothing=0.1)

    # The differentiable cost term costs 54 extra CUDA launches per forward, so
    # it is switched on only when something actually consumes it.
    if flops_cost > 0 and hasattr(model, 'compute_cost'):
        model.compute_cost = True

    task_pi_mask = None
    if isolator is not None and hasattr(model, 'K'):
        task_pi_mask = torch.zeros(model.K, device=device, dtype=torch.bool)
        task_pi_mask[isolator.frozen_hi:isolator.future_lo] = True

    for epoch in range(epochs):
        model.train()
        freeze = epoch < freeze_epochs
        if freeze:
            for p in arch_params:
                p.requires_grad_(False)
        elif epoch == freeze_epochs:
            for p in arch_params:
                p.requires_grad_(True)
            print(f"  {log_prefix}[arch] unfroze", flush=True)

        w_pi = 0.0 if epoch < epochs // 2 else w_pi_final
        t0, tot_loss = time.time(), 0.0
        n_batches = len(trainloader) if steps == 0 else min(steps, len(trainloader))
        for step, (new_x, new_y) in enumerate(trainloader):
            if steps > 0 and step >= steps:
                break

            new_x = new_x.to(device, non_blocking=True)
            new_y = new_y.to(device, non_blocking=True)

            # Joint Single-Pass: Forward & backward over full batch (256) in ONE pass,
            # updating both weights and architecture simultaneously (4x speedup over split bi-level).
            inputs, targets = new_x, new_y

            # Replay mixing: exemplars are appended to batch to prevent forgetting
            re_t = None
            if replay is not None and len(replay) > 0:
                n_re = min(48, inputs.size(0), len(replay))
                re_x, re_y, re_t = replay.sample(n_re, device)
                inputs = torch.cat([inputs, re_x])
                targets = torch.cat([targets, re_y])

            opt.zero_grad(set_to_none=True)
            opt_arch.zero_grad(set_to_none=True)

            # Per-sample prototype mask. Replayed exemplars belong to an EARLIER
            # task, whose prototype rows were trained on that task's domain, so
            # forcing them through the CURRENT task's rows would replay them
            # against the wrong head - the replay would actively teach the wrong
            # association. The 1-D task_pi_mask covers only the current task, so
            # it is correct only when there is no replay at all.
            if re_t is not None and hasattr(model, "K"):
                rows = torch.zeros(inputs.size(0), model.K, dtype=torch.bool,
                                   device=device)
                src = torch.full((inputs.size(0),), task_idx,
                                 device=device, dtype=torch.long)
                src[inputs.size(0) - re_t.numel():] = re_t
                for t_ in src.unique().tolist():
                    lo = round(t_ * model.K / n_tasks)
                    hi = round((t_ + 1) * model.K / n_tasks)
                    rows[src == t_, lo:hi] = True
                batch_pi_mask = rows
            else:
                batch_pi_mask = task_pi_mask

            with torch.amp.autocast('cuda', enabled=amp_dtype is not None, dtype=amp_dtype or torch.bfloat16):
                outputs, struct, pi, diag = model(inputs, tau=0.8, pi_mask=batch_pi_mask)
                loss = ce(outputs, targets)
                if w_pi > 0:
                    pif = pi.float()
                    ent_pi = -(pif * torch.log(pif.clamp(min=1e-9))).sum(-1).mean()
                    usage = pif.mean(0)
                    ent_usage = -(usage * torch.log(usage.clamp(min=1e-9))).sum()
                    loss = loss + w_pi * (ent_pi - ent_usage)

                # Domain-router supervision: current samples -> task_idx,
                # replay samples carry their original task id.
                if dom_head is not None and diag.get("z", None) is not None:
                    if isinstance(dom_head, TaskCentroidRouter):
                        # Centroid router accumulates running z representations without gradient degradation
                        dom_head.update_task_centroid(task_idx, router_features(model, inputs).detach())
                    else:
                        task_lbl = torch.full((inputs.size(0),), task_idx,
                                              device=device, dtype=torch.long)
                        if re_t is not None:
                            task_lbl[inputs.size(0) - re_t.numel():] = re_t
                        z_norm = F.normalize(diag["z"].float(), dim=1)
                        dom_logits = dom_head(z_norm)
                        loss = loss + 0.3 * F.cross_entropy(dom_logits[:, :task_idx + 1], task_lbl)

                # Dynamic-depth sparsity: pushes gates closed. With depth_target>0
                # this is a hinge, so it can prune *at most* (1-target) of the
                # capacity and can never collapse the trunk to zero gates no
                # matter how large the coefficient is.
                if depth_cost > 0 and "layer_soft" in diag:
                    ls = diag["layer_soft"].float()
                    if depth_target > 0:
                        pen = torch.relu(ls.mean() - depth_target)
                    else:
                        pen = ls.mean()
                    loss = loss + depth_cost * pen

                # Complexity regularisation on the chosen candidate block per
                # layer. The model reports this in GFLOP/sample, so an
                # O(0.01-1) coefficient is a gentle nudge rather than something
                # that swamps the cross-entropy.
                if flops_cost > 0 and "expected_gflops" in diag:
                    loss = loss + flops_cost * diag["expected_gflops"].float()

                # L2-SP anchor on shared encoder/router
                if anchor_pairs is not None:
                    l2sp = sum((p - a).square().sum() for p, a in anchor_pairs)
                    loss = loss + anchor_lam * l2sp

                # Elastic Weight Consolidation on the shared trunk: quadratic
                # anchor weighted by the diagonal Fisher of the tasks seen so far.
                if ewc is not None and ewc.enabled and ewc._fisher_all:
                    loss = loss + ewc.penalty()

            if not torch.isfinite(loss):
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if not freeze:
                    scaler.unscale_(opt_arch)
                if ewc is not None:
                    ewc.observe_batch()
                if isolator is not None:
                    isolator.mask_grad()
                torch.nn.utils.clip_grad_norm_(other_params, 5.0)
                if not freeze:
                    torch.nn.utils.clip_grad_norm_(arch_params, 5.0)
                scaler.step(opt)
                if not freeze:
                    scaler.step(opt_arch)
                scaler.update()
            else:
                loss.backward()
                if ewc is not None:
                    ewc.observe_batch()
                if isolator is not None:
                    isolator.mask_grad()
                torch.nn.utils.clip_grad_norm_(other_params, 5.0)
                if not freeze:
                    torch.nn.utils.clip_grad_norm_(arch_params, 5.0)
                opt.step()
                if not freeze:
                    opt_arch.step()
            if isolator is not None:
                isolator.restore()

            # Detached GPU-side sum (fp32: loss itself is bf16 under
            # autocast); synced once at epoch end. Log-only change.
            tot_loss = tot_loss + loss.detach().float()
            if (step + 1) % 100 == 0 or (step + 1) == n_batches:
                print(f"    [{name}] step {step+1}/{n_batches} | {time.time()-t0:.1f}s | loss {loss.item():.4f}", flush=True)

        sch.step()
        # The architecture optimizer does not step while the structure is
        # frozen; stepping its scheduler anyway would silently decay the LR
        # for epochs that never trained (and trips PyTorch's
        # scheduler-before-optimizer warning).
        if not freeze:
            sch_arch.step()
        acc, _ = evaluate(model, testloader, device, amp_dtype, pi_mask=task_pi_mask)
        avg_loss = (tot_loss / n_batches).item()
        mode_str = ""
        if hasattr(model, 'block_mode_prototypes') and model.block_mode_prototypes.shape[-1] == 3:
            with torch.no_grad():
                p_modes = torch.softmax(model.block_mode_prototypes, dim=-1).mean(dim=(0, 1))
                mode_str = f" | [Plain: {p_modes[0]*100:.0f}%, ResNet: {p_modes[1]*100:.0f}%, ViT: {p_modes[2]*100:.0f}%]"
        # Depth + cost telemetry: the whole point of P2 is that this number drops.
        with torch.no_grad():
            _ls = diag["layer_soft"].float()
            depth_str = f" | [active L: {( _ls > 0.5).float().sum(1).mean():.2f}/18"
            if "expected_mflops" in diag:
                depth_str += f", {diag['expected_mflops'].float():.1f} MFLOP/s"
            depth_str += "]"
        print(f"  {log_prefix}ep {epoch+1}/{epochs} | {time.time()-t0:5.1f}s | "
              f"loss {avg_loss:.4f} | {name} acc {acc*100:.2f}%{mode_str}{depth_str}"
              f"{' [frozen]' if freeze else ''}", flush=True)


# ----------------------------------------------------------------------
# Main continual loop
# ----------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """The trainer's CLI, as its own function.

    Extracted from main() so the defaults can be asserted by a test without
    launching a training run. That matters here because the defaults *are* the
    scientific claim: `--replay-per-class 0` and `--norm in` are what every
    README number was measured with, and a default that drifts from the
    documented command silently changes which method was measured.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=str, default='mnist,cifar10,fashion')
    ap.add_argument('--epochs', type=str, default='5,28,16', help='per-task epochs')
    ap.add_argument('--steps', type=int, default=0, help='cap train batches per epoch (0 = full pass)')

    ap.add_argument('--model-type', type=str, default='hybrid18', choices=['hybrid18', 'legacy'],
                    help='Model architecture: hybrid18 (18-layer CNN+ResNet+ViT) or legacy')
    ap.add_argument('--K', type=int, default=12,
                    help='prototype count; split across tasks (4 rows/task at K=12, T=3)')
    ap.add_argument('--max-dim', type=int, default=48,
                    help='maximum channel dimension for prototype layout (default: 48)')
    ap.add_argument('--encoder-tail-act', type=str, default='none',
                    choices=['relu', 'tanh', 'none', 'gelu'],
                    help="activation on the LAST layer of the vision encoder. "
                         "DEFAULT CHANGED relu -> none. A terminal ReLU forces "
                         "every latent into the non-negative orthant, so "
                         "pairwise cosine is pinned near 1 and the domain router "
                         "cannot separate tasks: measured 91.6 pct of latents "
                         "were exactly 0 and 0 pct were negative before it, "
                         "giving centroid cosines of 0.92/0.90/0.81 and "
                         "router accuracy of only 68/77/79 pct on its OWN "
                         "encoder, which collapsed A_auto to 8 pct while "
                         "A_oracle held 98 pct. 'none' keeps signed features.")
    ap.add_argument('--norm', type=str, default='in', choices=['bn', 'gn1', 'in'],
                    help="post-conv normalization. DEFAULT CHANGED bn -> in. "
                         "'bn' is the only option carrying running statistics, "
                         "and in this supernet each of the three candidate "
                         "branches keeps its own BN whose stats update only "
                         "when that branch is routed to; as the architecture "
                         "parameters move, those stats go stale and inference "
                         "collapses. bn also failed to reproduce - two runs of "
                         "the SAME config differed by 14.3 points. "
                         "MNIST, 5 epochs, seed 42, --arch-lr 3e-3 "
                         "--depth-cost 0.02, per-epoch acc then A[1,1]: "
                         "bn 88.86/84.40/84.75/83.93/87.09 -> 84.75 (100s/ep); "
                         "gn1 85.28/93.06/94.57/95.87/96.38 -> 96.26 (116s/ep); "
                         "in 92.09/95.85/97.21/97.77/97.91 -> 98.03 (108s/ep). "
                         "InstanceNorm wins on accuracy, monotonicity AND speed; "
                         "bn is kept only for comparison. Single-task "
                         "measurement - re-check on the full 3-task schedule "
                         "before quoting as final.")
    ap.add_argument('--workers', type=int, default=2, help='DataLoader workers')
    ap.add_argument('--freeze', type=int, default=2)
    ap.add_argument('--replay-per-class', type=int, default=0,
                    help='exemplars per class stored in the replay buffer (0 = off). '
                         'DEFAULT CHANGED 400 -> 0. Every headline result in the '
                         'README and the paper is a ZERO-REPLAY result; shipping '
                         'a default that silently stores 400 exemplars per class '
                         'meant the documented command and a bare '
                         '`python train_continual_aml.py` measured two different '
                         'methods.')
    ap.add_argument('--consolidate-epochs', type=int, default=2)
    ap.add_argument('--route-tau', type=float, default=0.6,
                    help='router margin below which a sample is evaluated under ALL '
                         'domain snapshots and the most confident prediction wins')
    ap.add_argument('--snap-q8', type=int, default=1,
                    help='store prototype snapshots int8-quantized (1) or fp32 (0)')
    ap.add_argument('--save-routing', type=int, default=1,
                    help='persist the router buffers and per-task snapshots next '
                         'to the model checkpoints (routing_state_after_*.pt). '
                         'Without this the router cannot be re-examined after '
                         'the run, because its centroids and covariance exist '
                         'only in memory.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--aug', type=str, default='basic', choices=['basic', 'auto'],
                    help='CIFAR train augmentation: basic (crop+flip) or auto (+AutoAugment)')
    ap.add_argument('--router-type', type=str, default='centroid', choices=['centroid', 'linear'],
                    help='Domain router: centroid (zero-replay OOD prototype) or linear')
    ap.add_argument('--anchor-lam', type=float, default=0.0,
                    help='L2-SP encoder anchoring strength; >0 slows feature '
                         'drift BUT collapses per-domain specialization '
                         '(fingerprints converged to near-identical in v5)')
    ap.add_argument('--arch-prior', type=str, default='staged',
                    choices=['neutral', 'staged'],
                    help='block_mode_prototypes init. staged (default) = the '
                         'CoAtNet-style 3-stage prior. neutral = small symmetric '
                         'noise, no hand-set hierarchy. Measured on MNIST at a '
                         '4x60-step budget: staged reaches 89.9 pct, neutral stalls '
                         'at 11.4 pct. Reason: with near-zero logits, '
                         'gumbel_softmax(tau=0.8) is noise-dominated, so every '
                         'layer switches branch almost every step and the trunk '
                         'never co-adapts. Exact zeros are worse still - argmax '
                         'breaks the tie to index 0 and pins all 18 layers to '
                         'PlainConv. See experiments/dynamic_depth_routing_*.md')
    ap.add_argument('--arch-lr', type=float, default=3e-3,
                    help='LR for the architecture prototypes. AdamW travels ~lr '
                         'per step, so this is the per-task budget for how far the '
                         'gates can move; 2e-4 could not reach the 0.5 gate.')
    ap.add_argument('--depth-cost', type=float, default=0.02,
                    help='sparsity penalty on the depth gates. >0 is what makes '
                         'layers actually prune so the P2 short-circuit can fire.')
    ap.add_argument('--depth-target', type=float, default=0.0,
                    help='if >0 the depth penalty becomes a hinge '
                         'relu(mean-gate - target): prunes at most (1-target) and '
                         'can never collapse the trunk whatever --depth-cost is.')
    ap.add_argument('--flops-cost', type=float, default=0.0,
                    help='complexity regularisation weight, GFLOP/sample. 0 = off. '
                         'With the 8x8 token cap, ViT is the cheapest mode at 32x32 '
                         'and 16x16, so this pushes THOSE STAGES toward ViT - the '
                         'opposite of "Conv at shallow layers". At 8x8 Plain is '
                         'cheapest again.')
    ap.add_argument('--ewc-lambda', type=float, default=0.0,
                    help='Elastic Weight Consolidation on the shared trunk. >0 '
                         'anchors earlier tasks without a snapshot per task.')
    ap.add_argument('--amb-topk', type=int, default=0,
                    help='restrict the ambiguous-sample retry (router margin < '
                         '--route-tau) to the k most likely domains. 0 = retry '
                         'under every seen snapshot, which loads ~1.5M params per '
                         'domain per batch. k=2 is the usual choice.')
    ap.add_argument('--shared-backbone', type=int, default=0,
                    help='P4-3A: share one backbone across all tasks - store NO '
                         'per-task copy of stem / 18 layers / ln (5.9 MB per task) '
                         'and rely on EWC alone to retain earlier domains. '
                         'Per-task state drops to ~1.2 MB, O(1) in depth.')
    ap.add_argument('--auto-live-thresh', type=float, default=0.0,
                    help='open-set fallback: if no snapshot reaches this softmax '
                         'confidence for a sample, evaluate it with the current '
                         'live weights. 0 = disabled. A_auto was measured below '
                         'A_live in every off-diagonal cell, so misrouting cost '
                         'more than snapshot retrieval gained.')
    ap.add_argument('--route-per-task', type=int, default=1,
                    help='embed the routing query with EACH task\'s own encoder '
                         '(1) or with the current encoder (0). A centroid is '
                         'stored in the space of the encoder that produced it, '
                         'and the drift between that and the current encoder '
                         'measured 2.38x the within-domain spread, so the '
                         'single-encoder router scored 34-52%% once two real '
                         'tasks were trained. Costs len(snaps) encoder forwards '
                         'per batch; those encoders are already loaded for '
                         'snapshot retrieval.')
    ap.add_argument('--router-metric', type=str, default='mahalanobis-task',
                    choices=['mahalanobis-task', 'mahalanobis', 'cosine'],
                    help='domain routing metric. mahalanobis-task scores each '
                         'candidate in the feature space its own centroid was '
                         'measured in, mixing in cov_alpha of the pooled second '
                         'moment (DEFAULT CHANGED mahalanobis -> '
                         'mahalanobis-task). The old shared matrix mixes three '
                         'feature spaces that share only a dimension: measured '
                         'on the production checkpoints it routes MNIST 6.07%% '
                         'and CIFAR 41.16%%, against 95.37%% / 99.67%% here. '
                         "'mahalanobis' keeps the single shared matrix and "
                         "'cosine' the original nearest-centroid rule.")
    ap.add_argument('--router-shrink', type=float, default=0.01,
                    help='shrinkage added to the diagonal of the shared '
                         'covariance, as a fraction of its mean diagonal. '
                         'Stabilises the inverse; 0.01 measured best.')
    ap.add_argument('--router-fit-split', type=str, default='train',
                    choices=['train', 'test'],
                    help='split the router\'s centroid and second moment are '
                         'estimated on. DEFAULT CHANGED test -> train. The clean '
                         'pass used to run on the TEST split, i.e. on the very '
                         'images the router is later asked to route: a leak, and '
                         'the reason offline analysis of the router kept '
                         'disagreeing with the trainer. Still zero-replay - no '
                         'image is stored, only second moments.')
    ap.add_argument('--router-cov-alpha', type=float, default=0.0,
                    help="fraction of the POOLED second moment mixed into each "
                         "task's own. DEFAULT 0 = per-task only. Measured mean "
                         "routing accuracy (train-fit statistics, 3 domains): "
                         "97.52 / 85.50 / 74.95 / 64.83 at alpha = 0 / 0.25 / "
                         "0.5 / 1.0.")
    ap.add_argument('--router-logdet-weight', type=float, default=0.5,
                    help="weight on the log|S_k| term of the per-task Gaussian "
                         "NLL. 0 = bare whitened distance, 1 = full NLL. "
                         "Measured (alpha = 0, mean over the 3 domains): "
                         "34.18 / 97.52 / 90.93 at 0 / 0.5 / 1.0; on two "
                         "domains 51.74 / 99.97 / 99.25. The term is what stops "
                         "a broadly spread domain from winning every query.")
    return ap


def main():
    _prevent_sleep()
    ap = build_arg_parser()
    args = ap.parse_args()
    use_replay = args.replay_per_class > 0
    shared_backbone = bool(args.shared_backbone)
    if shared_backbone and args.ewc_lambda <= 0:
        print("[!] --shared-backbone without --ewc-lambda>0 has NOTHING holding "
              "the shared weights in place; A_live will collapse. "
              "This is the intended experiment, not a bug.")

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else (
        torch.float16 if device.type == 'cuda' else None)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    tasks = args.tasks.split(',')
    ep_list = [int(e) for e in args.epochs.split(',')]
    assert len(ep_list) == len(tasks)

    if args.model_type == 'hybrid18':
        print("[*] Initializing 18-Layer Hybrid AML Supernet (Plain CNN + ResNet + ViT)...")
        model = ConvAML_18Layer_Hybrid(
            K=args.K, max_dim=args.max_dim, max_layers=18, num_classes=10,
            norm_type=args.norm,
            encoder_tail_act=args.encoder_tail_act,
            arch_prior=args.arch_prior
        ).to(device)
    else:
        print(f"[*] Initializing Legacy ConvAML_CIFAR_TrueDynamic (norm={args.norm})...")
        model = ConvAML_CIFAR_TrueDynamic(K=args.K, max_dim=args.max_dim, max_layers=18,
                                          norm_mode=args.norm).to(device)

    T = len(tasks)
    A = [[None] * T for _ in range(T)]        # method protocol (snapshot retrieval + task oracle)
    A_auto = [[None] * T for _ in range(T)]   # autonomous protocol (pure centroid router OOD prediction)
    A_live = [[None] * T for _ in range(T)]   # standard CL (current weights only)
    fingerprints = {}
    replay = ReplayBuffer()
    isolator = ProtoIsolator(model)
    latent_dim = getattr(model, 'latent_dim', 1024)
    if args.router_type == 'centroid':
        print(f"[*] Initializing Zero-Replay TaskCentroidRouter (latent_dim={latent_dim}, max_tasks={T})...")
        dom_head = TaskCentroidRouter(latent_dim=latent_dim, max_tasks=T, temperature=0.1,
                                     metric=args.router_metric,
                                     shrink=args.router_shrink,
                                     cov_alpha=args.router_cov_alpha,
                                     logdet_weight=args.router_logdet_weight).to(device)
    else:
        print(f"[*] Initializing Linear Domain Router (latent_dim={latent_dim}, T={T})...")
        dom_head = torch.nn.Linear(latent_dim, T).to(device)
    slots = [None] * T          # per-domain structure slots
    dom_snaps = [None] * T      # per-domain frozen (encoder, selector) snapshots
    anchor_snap = None          # L2-SP snapshot of shared encoder/router
    ewc = FisherEWC(model, args.ewc_lambda)
    if ewc.enabled:
        n_shared = sum(p.numel() for p in ewc._shared_trunk(model).values())
        print(f"[*] EWC enabled (lambda={args.ewc_lambda}): anchoring {n_shared/1e6:.2f}M "
              f"shared weights - one Fisher vector for ALL tasks, no per-task "
              f"18-layer snapshot needed")

    for i, name in enumerate(tasks):
        print(f"\n===== TASK {i+1}/{T}: {name} =====", flush=True)
        # reinit=False: future prototype rows keep their ORIGINAL kaiming
        # values forever. Earlier tasks' routing may lean on these rows;
        # re-randomizing them per task silently corrupts that routing
        # (measured -0.9pt MNIST in v9). Untrained rows stay equally fresh
        # for the new task either way.
        s, e = isolator.begin_task(i, T, reinit=False)
        print(f"  [proto-isolation] rows [{s}:{e}) owned; rows [0:{s}) frozen", flush=True)
        print(f"  [replay] buffer size: {len(replay)} exemplars", flush=True)
        tr_loader, _ = get_task_data(name, os.path.join(_ROOT, "data"), workers=args.workers)
        del tr_loader
        train_task(model, name, ep_list[i],
                   freeze_epochs=(args.freeze if i == 0 else min(args.freeze, 1)),
                   device=device, amp_dtype=amp_dtype,
                   scaler=scaler,
                   w_pi_final=0.01, log_prefix=f"[{name}] ",
                   replay=(replay if (use_replay and i > 0) else None),
                   isolator=isolator,
                   dom_head=dom_head,
                   task_idx=i,
                   anchor=(anchor_snap if args.anchor_lam > 0 else None),
                   anchor_lam=args.anchor_lam,
                   arch_lr=args.arch_lr,
                   depth_cost=args.depth_cost,
                   flops_cost=args.flops_cost,
                   depth_target=args.depth_target,
                   ewc=ewc,
                   aug=args.aug,
                   workers=args.workers,
                   steps=args.steps,
                   n_tasks=T)

        # Freeze this domain's structure into its slot & pristine router centroid
        tl_fit, el_self = get_task_data(name, os.path.join(_ROOT, "data"))
        # The router's centroid and second moment are estimated on the TRAIN
        # split by default. They used to be estimated on `el_self`, the TEST
        # split - i.e. on the very images the router is later asked to route.
        # No exemplars are stored either way, so the method is still
        # zero-replay, but a statistic fitted on the test inputs is not a
        # result about test-time behaviour: it is a leak, and it is also why
        # offline analysis of the router kept disagreeing with the trainer.
        router_fit = el_self if args.router_fit_split == 'test' else tl_fit
        print(f"  [centroid-router] fitting statistics on the "
              f"{args.router_fit_split.upper()} split "
              f"({len(router_fit.dataset)} images)")

        # Close the EWC task window: average the Fisher over this task and pin
        # theta*, so the NEXT task is quadratically anchored to this solution.
        if ewc.enabled:
            ewc.consolidate()
            print(f"  [ewc] consolidated Fisher over task {i+1}: "
                  f"{ewc.state_bytes()/1e6:.2f}M values stored for all tasks so far")

        # Finalize domain centroid router for this task
        if isinstance(dom_head, TaskCentroidRouter):
            # Rebuild the centroid from scratch over the clean eval split, with
            # the encoder in its FINAL state.
            #
            # This has to override what training accumulated. train_task() calls
            # update_task_centroid on every step, so the running sum mixes
            # features from encoder states that no longer exist - by the end of
            # a 6-epoch task that is 79% of the samples. The snapshot the router
            # is queried with is the final one, so a blended centroid describes
            # an encoder that never existed: measured routing fell from 100.0%
            # to 82.0% overall and to 64.6% on the first domain.
            #
            # The shared covariance is fed here and ONLY here: it is global, so
            # mid-training contributions could not be removed afterwards without
            # retaining every intermediate feature. The training-step path
            # (update_task_centroid) therefore leaves it alone.
            dom_head.centroids[i].zero_()
            dom_head.task_counts[i].zero_()
            model.eval()
            with torch.no_grad():
                for cx, _ in router_fit:
                    cx = cx.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda', enabled=amp_dtype is not None,
                                            dtype=amp_dtype or torch.bfloat16):
                        # same projection the router will query with at eval
                        cz = router_features(model, cx)
                    dom_head.finalize_task_features(i, cz)
            dom_head.finalize_task(i)
            print(f"  [centroid-router] finalized pristine centroid for task {i+1} ({name})")

        # Store balanced exemplars (with task id) for future tasks
        if use_replay:
            data_root = os.path.join(_ROOT, "data")
            ds_name = {"mnist": "MNIST", "fashion": "FashionMNIST", "cifar10": "CIFAR10"}[name]
            ds = getattr(torchvision.datasets, ds_name)(data_root, train=True, download=False)
            xs, ys = build_replay_buffer(ds, per_class=args.replay_per_class)
            replay.extend(xs, ys, torch.full((xs.shape[0],), i, dtype=torch.long))
            print(f"  [replay] added {len(xs)} exemplars -> total {len(replay)}")

        # Consolidation pass: rebalance shared components over all seen domains
        if use_replay and i > 0:
            t0c = time.time()
            consolidate(model, replay, device, amp_dtype,
                        epochs=args.consolidate_epochs, isolator=isolator,
                        dom_head=dom_head)
            print(f"  [consolidate] {args.consolidate_epochs} passes over "
                  f"{len(replay)} exemplars ({time.time()-t0c:.0f}s)")

        slots[i] = collect_structure_slot(model, router_fit, device, amp_dtype)
        print(f"  [slot] stored structure slot for '{name}'")

        dom_snaps[i] = {
            "enc": {k: v.detach().clone() for k, v in model.vision_encoder.state_dict().items()},
            "sel": {k: v.detach().clone() for k, v in model.prototype_selector.state_dict().items()},
        }
        if hasattr(model, "layers") and not shared_backbone:
            dom_snaps[i]["stem"] = {k: v.detach().clone() for k, v in model.stem.state_dict().items()}
            dom_snaps[i]["layers"] = {k: v.detach().clone() for k, v in model.layers.state_dict().items()}
            if hasattr(model, "ln"):
                dom_snaps[i]["ln"] = {k: v.detach().clone() for k, v in model.ln.state_dict().items()}
        if shared_backbone and hasattr(model, "layers"):
            n_l = sum(p.numel() for p in model.layers.parameters())
            print(f"  [shared-backbone] 3A: no per-task copy of stem + 18 layers "
                  f"+ ln ({n_l/1e6:.2f}M params, "
                  f"{n_l*4/1e6:.2f} MB) - EWC is solely responsible for "
                  f"retaining earlier domains")
        if hasattr(model, "structure_prototypes"):
            dom_snaps[i]["struct_heads"] = {
                "sprotos": model.structure_prototypes.detach().clone(),
                "bmprotos": model.block_mode_prototypes.detach().clone(),
            }
        elif hasattr(model, "structure_delta"):
            dom_snaps[i]["struct_heads"] = {
                "delta": {k: v.detach().clone() for k, v in model.structure_delta.state_dict().items()},
                "block": {k: v.detach().clone() for k, v in model.block_mode_head.state_dict().items()},
            }
        pw = model.proto_weights.detach()
        if args.snap_q8:
            scale = pw.abs().max().clamp(min=1e-12) / 127.0
            dom_snaps[i]["protos"] = {"q": torch.round(pw / scale).to(torch.int8).cpu(),
                                      "scale": scale.cpu()}
            tag = "enc+sel+struct+prototypes" if shared_backbone else \
                  "enc+sel+stem+layers+ln+struct+prototypes"
            # Report the WHOLE snapshot, not just the int8 prototype block: in
            # the 18-layer hybrid the prototype library is 92 k elements (~90 KB),
            # so printing its size in whole MB rounds to "0 MB" and reads as if
            # nothing had been stored, while the snapshot is dominated by the
            # fp32 encoder + trunk (7.49 MB at the default --norm in).
            proto_kb = dom_snaps[i]["protos"]["q"].numel() / 1024
            total_mb = snapshot_bytes(dom_snaps[i]) / 1e6
            print(f"  [snap] stored {tag}(int8, prototypes {proto_kb:.0f} KB; "
                  f"snapshot {total_mb:.2f} MB) for '{name}'")
        else:
            dom_snaps[i]["protos"] = pw.to("cpu", torch.float32).clone()
            print(f"  [snap] stored enc+sel+struct+prototypes(fp32) for '{name}'")

        # Persist the routing state. Without this the router's centroids,
        # covariance and the snapshots that define their feature spaces exist
        # only in memory, so a routing result cannot be re-examined after the
        # run: every offline attempt has to retrain, and retraining does not
        # reproduce. This is the artefact that makes "which metric" a question
        # that can be answered from a measurement instead of a guess.
        if args.save_routing and isinstance(dom_head, TaskCentroidRouter):
            torch.save(
                {"router": dom_head.state_dict(), "snaps": dom_snaps[:i + 1],
                 "slots": slots[:i + 1], "task": name, "stage": i,
                 "args": vars(args)},
                os.path.join(_ROOT, f"routing_state_after_{name}.pt"))
            print(f"  [routing] saved routing_state_after_{name}.pt "
                  f"({snapshot_bytes({'protos': dom_snaps[i]['protos']}) / 1e6:.2f} MB "
                  f"of prototypes + router buffers)")

        # Snapshot the shared encoder/router for next-task anchoring
        anchor_names = ("vision_encoder", "prototype_selector")
        anchor_snap = [(p, p.detach().clone()) for n_, p in model.named_parameters()
                       if any(k in n_ for k in anchor_names)]

        # Evaluate on ALL seen tasks through THREE protocols:
        #   A[i][j]      - Task-IL (snapshot retrieval with task identity)
        #   A_auto[i][j] - Domain-IL / Class-IL (autonomous centroid router prediction)
        #   A_live[i][j] - Standard CL (current mutable live weights, no snapshot)
        for j in range(i + 1):
            _, el = get_task_data(tasks[j], os.path.join(_ROOT, "data"))
            use_route = all(slots[k] is not None for k in range(i + 1))
            pi_mask_j = None
            if hasattr(model, 'K'):
                pi_mask_j = torch.zeros(model.K, device=device, dtype=torch.bool)
                sj = round(j * model.K / len(tasks))
                ej = round((j + 1) * model.K / len(tasks))
                pi_mask_j[sj:ej] = True

            # Task-IL (Oracle routing)
            acc_oracle, fp = evaluate(model, el, device, amp_dtype,
                                      dom_head=(dom_head if use_route else None),
                                      slots=(slots[:i + 1] if use_route else None),
                                      snaps=(dom_snaps[:i + 1] if use_route else None),
                                      route_tau=args.route_tau, dom_truth=j,
                                      pi_mask=pi_mask_j, amb_topk=args.amb_topk)

            # Domain-IL (Autonomous routing via Centroid / dom_head without truth)
            acc_auto = acc_oracle
            if use_route:
                # A_auto runs WITHOUT the true task id. Handing it pi_mask_j
                # leaked that id straight to the prototype selector.
                #
                # But pi_mask=None is not the fix either: the selector is shared
                # and each task only ever trained its OWN rows, so scoring a
                # sample against every task's rows means scoring it against rows
                # trained on a different domain. Measured on the CIFAR-10 cell
                # with correct routing: 32.63 % when the right rows are allowed,
                # 10.86 % when all 12 are.
                #
                # proto_row_ranges closes that gap using d_pred - the ROUTER's
                # own decision - so each sample may use the rows of the task the
                # router assigned it. No true task id is involved.
                ranges = [(round(k * model.K / len(tasks)),
                           round((k + 1) * model.K / len(tasks)))
                          for k in range(i + 1)]
                acc_auto, fp_auto = evaluate(
                    model, el, device, amp_dtype,
                    dom_head=dom_head,
                    slots=slots[:i + 1],
                    snaps=dom_snaps[:i + 1],
                    route_tau=args.route_tau, dom_truth=None,
                    pi_mask=None, amb_topk=args.amb_topk,
                    auto_live_thresh=args.auto_live_thresh,
                    route_per_task=args.route_per_task, route_truth=j,
                    proto_row_ranges=ranges)

            # Live mutable supernet
            acc_live, _ = evaluate(model, el, device, amp_dtype, pi_mask=pi_mask_j)

            A[i][j] = acc_oracle
            A_auto[i][j] = acc_auto
            A_live[i][j] = acc_live
            if j == i:
                fingerprints[name] = fp
            tag = "routed" if use_route else "live"
            lf = ""
            if use_route and args.auto_live_thresh > 0:
                lf = f" | live-fallback {fp_auto['live_fallback_frac']*100:.0f}%"
            # route_acc separates "the router chose the wrong domain" from "the
            # router chose right and the snapshot still scored badly". Without
            # this, A_auto below A_oracle is ambiguous and has to be diagnosed
            # offline, which is slow and, repeatedly, wrong.
            ra = fp_auto.get("route_acc") if fp_auto else None
            rt = f" | route_acc {ra*100:.2f}%" if ra is not None else ""
            # Where the misrouted samples actually went. route_acc alone says
            # "6% wrong"; the histogram says "6% went to CIFAR", which is the
            # difference between an overlap problem and a collapsed router.
            rh = fp_auto.get("route_hist") if fp_auto else None
            rhs = f" | route_hist [{','.join(str(int(v)) for v in rh)}]" if rh is not None else ""
            print(f"  A[{i+1},{j+1}] ({tasks[j]}, {tag}): auto: {acc_auto*100:.2f}% | oracle: {acc_oracle*100:.2f}% | live: {acc_live*100:.2f}%{lf}{rt}{rhs}", flush=True)

        torch.save(model.state_dict(), os.path.join(_ROOT, f"aml_continual_after_{name}.pth"))

    # ---- CL metrics ----
    def print_matrix(M, title):
        print(f"\n===== {title} =====")
        header = "after \\ eval on | " + " | ".join(f"{t:>8s}" for t in tasks)
        print(header)
        for i in range(T):
            row = " | ".join(f"{M[i][j]*100:8.2f}" if M[i][j] is not None else f"{'—':>8s}"
                             for j in range(T))
            print(f"{tasks[i]:>17s} | {row}")

    def cl_metrics(M):
        bwt = sum(M[T-1][j] - M[j][j] for j in range(T - 1)) / (T - 1)
        forg = []
        for j in range(T - 1):
            worst_before = max(M[t][j] for t in range(j, T - 1))
            forg.append(worst_before - M[T-1][j])
        # A_avg: mean of the final row - the single number the README tables
        # quote, so it is printed here instead of being recomputed by hand from
        # the matrix (and mis-transcribed, which is what the old tables did).
        a_avg = sum(M[T-1][j] for j in range(T)) / T
        return bwt, sum(forg) / len(forg), a_avg

    print_matrix(A_auto, "ACCURACY MATRIX A[i,j] - AUTONOMOUS ROUTER (Domain-IL / Class-IL)")
    print_matrix(A, "ACCURACY MATRIX A[i,j] - TASK-ORACLE SNAPSHOT RETRIEVAL (Task-IL)")
    print_matrix(A_live, "ACCURACY MATRIX A[i,j] - LIVE WEIGHTS (Standard Continual Learning)")

    if T > 1:
        bwt_l, forg_l, avg_l = cl_metrics(A_live)
        bwt_a, forg_a, avg_a = cl_metrics(A_auto)
        bwt_m, forg_m, avg_m = cl_metrics(A)
        print(f"\n[live]       A_avg: {avg_l*100:6.2f}% | BWT: {bwt_l*100:+.2f}% | Forgetting: {forg_l*100:.2f}%")
        print(f"[auto-route] A_avg: {avg_a*100:6.2f}% | BWT: {bwt_a*100:+.2f}% | Forgetting: {forg_a*100:.2f}%")
        print(f"[oracle]     A_avg: {avg_m*100:6.2f}% | BWT: {bwt_m*100:+.2f}% | Forgetting: {forg_m*100:.2f}%")
        print("  (method column uses test-time router+snapshot retrieval; report")
        print("   the LIVE column when comparing against standard CL literature)")

    # ---- Structure divergence across domains ----
    print("\n===== PER-DOMAIN STRUCTURE FINGERPRINTS =====")
    for name, fp in fingerprints.items():
        print(f"\n[{name}] pi_H={fp['pi_H']:.3f}")
        print("  gate : " + " ".join(f"{v:.2f}" for v in fp["gate"].tolist()))
        print("  bmRes: " + " ".join(f"{v:.2f}" for v in fp["bm_res"].tolist()))
        if fp.get("bm_plain") is not None and fp.get("bm_vit") is not None:
            print("  bmPln: " + " ".join(f"{v:.2f}" for v in fp["bm_plain"].tolist()))
            print("  bmViT: " + " ".join(f"{v:.2f}" for v in fp["bm_vit"].tolist()))

    names = list(fingerprints.keys())
    if len(names) >= 2:
        print("\nPairwise L1 divergence of gate fingerprints:")
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                fa, fb = fingerprints[names[a]]["gate"], fingerprints[names[b]]["gate"]
                d = (fa - fb).abs().sum().item()
                d_bm = (fingerprints[names[a]]["bm_res"] - fingerprints[names[b]]["bm_res"]).abs().sum().item()
                # per-sample gate divergence between domains
                ga = fingerprints[names[a]]["gate_tensor"]
                gb = fingerprints[names[b]]["gate_tensor"]
                nmin = min(ga.shape[0], gb.shape[0])
                d_sample = (ga[:nmin].mean(0) - gb[:nmin].mean(0)).abs().sum().item()
                print(f"  {names[a]} vs {names[b]}: gate_L1={d:.3f} | bmRes_L1={d_bm:.3f} | mean_gate_L1={d_sample:.3f}")
    print("\n(Interpretation: large L1 => architecture genuinely differs per domain;" 
          " ~0 => static network despite dynamic machinery)")


if __name__ == "__main__":
    main()
