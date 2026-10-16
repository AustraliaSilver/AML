"""What does the PUBLISHED AML continual protocol actually store per task?

The reference row of ``experiments/exp_verify_basis_ogp.py`` is
``copy.deepcopy(model)`` of a small HtS model (5.60 MiB).  The real AML protocol
stores something else entirely (``train_continual_aml.py``, the snapshot
construction in the training loop): ``enc`` + ``sel`` + ``stem`` + 18 ``layers``
+ ``ln`` + structure heads + int8 prototypes, one set per task.

This walks the artefacts of one run and reports the key-level byte breakdown, so
the "parameter caching" review question can be answered from a measurement
rather than from the docstring.  No training is run.

    python experiments/snapshot_storage_probe.py [--dir DIR]

``DIR`` defaults to the repo root, whose artefacts belong to whatever run wrote
them last: the breakdown below therefore depends on that run's class count
(``protos`` is int8 and grows with the shared head) and on its task count.  Pass
the directory of a known run to measure that run.

Two numbers matter for the storage claim and they are computed here, not
asserted:

* the per-task snapshot (one set of trunk/encoder/prototype/selector tensors);
* the per-task share of the ROUTER state.  ``cov_task``, ``centroids``,
  ``zsum_task`` and friends are allocated as ``(max_tasks, ...)`` and
  ``train_continual_aml.py`` passes ``max_tasks=T``, so they are **per task**
  (~1.05 MB each for ``latent_dim=512``), while ``cov``/``zsum``/``cov_n`` are
  genuinely once.  B64 read the router state_dict total of a 3-task run
  (4.209 MB) as an O(1) cost; it is 1.05 MB once plus 3 x 1.05 MB.
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_continual_aml import snapshot_bytes  # noqa: E402

TASKS = ["mnist", "fashion", "cifar10", "cifar100", "tinyimagenet"]

# Router buffers that hold ONE slot per task because they are allocated with
# max_tasks = T (TaskCentroidRouter.__init__, train_continual_aml.py:174-194).
ROUTER_PER_TASK = ("centroids", "task_counts", "cov_task", "zsum_task", "cov_n_task")
ROUTER_ONCE = ("cov", "cov_n", "zsum", "num_tasks")


def mb(n):
    return n / 1e6


def split_router(router):
    """Return (once_bytes, per_task_bytes, n_slots) for a router state_dict."""
    once = per_task = 0
    slots = 0
    for key, val in router.items():
        if not isinstance(val, torch.Tensor):
            once += sys.getsizeof(val)
            continue
        b = val.numel() * val.element_size()
        if key in ROUTER_PER_TASK and val.dim() > 0:
            slots = max(slots, val.shape[0])
            per_task += b // val.shape[0]
        else:
            once += b
    return once, per_task, slots


def report_routing(directory):
    print("=" * 96)
    print(f" routing_state_after_<task>.pt  in {directory}")
    print(" (router state + cumulative per-task snapshots + slots)")
    print("=" * 96)
    rows = []
    for name in TASKS:
        path = os.path.join(directory, f"routing_state_after_{name}.pt")
        if not os.path.exists(path):
            continue
        blob = torch.load(path, map_location="cpu", weights_only=False)
        snaps = blob.get("snaps", [])
        slots = blob.get("slots", [])
        router = blob.get("router", {})
        file_mb = mb(os.path.getsize(path))
        router_b = snapshot_bytes(router)
        slot_b = snapshot_bytes(slots)
        per_snap = [snapshot_bytes(s) for s in snaps]
        once_b, per_task_b, n_slots = split_router(router)

        print(f"\n[{name}]  stage={blob.get('stage')}  file={file_mb:.3f} MB  "
              f"tasks_stored={len(snaps)}")
        print(f"  router state_dict total     : {mb(router_b):8.3f} MB  "
              f"= once {mb(once_b):.3f} + per-task {mb(per_task_b):.3f} "
              f"x {n_slots} slots")
        print(f"  structure slots             : {mb(slot_b):8.3f} MB")
        for i, (s, b) in enumerate(zip(snaps, per_snap)):
            parts = {k: snapshot_bytes(v) for k, v in s.items()}
            detail = ", ".join(f"{k}={mb(v):.3f}" for k, v in sorted(parts.items()))
            print(f"  snapshot[{i}]                : {mb(b):8.3f} MB   ({detail})")
        if per_snap:
            print(f"  -> per-new-task snapshot cost (last) : {mb(per_snap[-1]):.3f} MB")
        accounted = router_b + slot_b + sum(per_snap)
        print(f"  accounted {mb(accounted):.3f} MB of {file_mb:.3f} MB on disk "
              f"(rest = pickle/framing overhead)")
        rows.append(dict(name=name, n=len(snaps), file_mb=file_mb,
                         snap_mb=mb(per_snap[-1]) if per_snap else 0.0,
                         once_mb=mb(once_b), per_task_mb=mb(per_task_b),
                         slot_mb=mb(slot_b), slots=n_slots))
    return rows


def report_storage_model(rows):
    if not rows:
        return
    last = max(rows, key=lambda r: r["n"])
    snap = last["snap_mb"]
    per_task = snap + last["per_task_mb"] + last["slot_mb"]
    once = last["once_mb"]
    print("\n" + "=" * 96)
    print(" STORAGE MODEL  (per task = snapshot + router per-task share + slot)")
    print("=" * 96)
    print(f"  per-task snapshot      : {snap:8.3f} MB")
    print(f"  router, per task       : {last['per_task_mb']:8.3f} MB  "
          f"({', '.join(ROUTER_PER_TASK)})")
    print(f"  structure slot         : {last['slot_mb']:8.3f} MB")
    print(f"  -> per task            : {per_task:8.3f} MB")
    print(f"  router, once           : {once:8.3f} MB  ({', '.join(ROUTER_ONCE)})")
    print(f"\n  totals:  T=3 {once + 3 * per_task:8.2f} MB   "
          f"T=10 {once + 10 * per_task:8.2f} MB   "
          f"T=20 {once + 20 * per_task:8.2f} MB")
    print("  (treating the router's per-task share as O(T): it is allocated with "
          "max_tasks=T,\n   so it is O(T) up to T and would have to be reallocated "
          "beyond it.)")


def report_checkpoints(directory):
    print("\n" + "=" * 96)
    print(" aml_continual_after_<task>.pth  (the model checkpoint written per task)")
    print("=" * 96)
    for name in TASKS:
        path = os.path.join(directory, f"aml_continual_after_{name}.pth")
        if not os.path.exists(path):
            continue
        file_mb = mb(os.path.getsize(path))
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict):
            top = {k: snapshot_bytes(v) for k, v in blob.items()}
            detail = ", ".join(f"{k}={mb(v):.2f}" for k, v in
                               sorted(top.items(), key=lambda kv: -kv[1])[:6])
            print(f"[{name}]  file={file_mb:8.3f} MB   {len(blob)} keys")
            print(f"          {detail}")
        else:
            print(f"[{name}]  file={file_mb:8.3f} MB   (not a dict: {type(blob)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=ROOT,
                    help="directory holding the run's artefacts (default: repo root)")
    ap.add_argument("--checkpoints", action="store_true",
                    help="also list the per-task model checkpoints")
    args = ap.parse_args()
    rows = report_routing(args.dir)
    report_storage_model(rows)
    if args.checkpoints:
        report_checkpoints(args.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
