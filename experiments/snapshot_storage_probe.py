"""What does the PUBLISHED AML continual protocol actually store per task?

The reference row of ``experiments/exp_verify_basis_ogp.py`` is
``copy.deepcopy(model)`` of a small HtS model (5.60 MiB).  The real AML protocol
stores something else entirely (``train_continual_aml.py``, the snapshot
construction in the training loop): ``enc`` + ``sel`` + ``stem`` + 18 ``layers``
+ ``ln`` + structure heads + int8 prototypes, one set per task.

This walks the artefacts already on disk in the repo root and reports the
key-level byte breakdown, so the "parameter caching" review question can be
answered from a measurement rather than from the docstring.  No training is run.

    python experiments/snapshot_storage_probe.py

Expected output is recorded as B64 in
``experiments/dynamic_depth_routing_2026-10-14.md``: 7.209 MB per task, of which
5.854 MB (81.2 %) is a copy of the 18-layer trunk, against a structure slot of
0.001-0.002 MB.
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_continual_aml import snapshot_bytes  # noqa: E402

TASKS = ["mnist", "fashion", "cifar10", "cifar100", "tinyimagenet"]


def mb(n):
    return n / 1e6


def report_routing():
    print("=" * 96)
    print(" routing_state_after_<task>.pt  (router + cumulative per-task snapshots + slots)")
    print("=" * 96)
    rows = []
    for name in TASKS:
        path = os.path.join(ROOT, f"routing_state_after_{name}.pt")
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

        print(f"\n[{name}]  stage={blob.get('stage')}  file={file_mb:.3f} MB  "
              f"tasks_stored={len(snaps)}")
        print(f"  router buffers (state_dict) : {mb(router_b):8.3f} MB")
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
        rows.append((name, len(snaps), file_mb,
                     mb(per_snap[-1]) if per_snap else 0.0,
                     mb(router_b), mb(slot_b)))
    return rows


def report_checkpoints():
    print("\n" + "=" * 96)
    print(" aml_continual_after_<task>.pth  (the model checkpoint written per task)")
    print("=" * 96)
    for name in TASKS:
        path = os.path.join(ROOT, f"aml_continual_after_{name}.pth")
        if not os.path.exists(path):
            continue
        file_mb = mb(os.path.getsize(path))
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict):
            top = {k: snapshot_bytes(v) for k, v in blob.items()}
            detail = ", ".join(f"{k}={mb(v):.2f}" for k, v in
                               sorted(top.items(), key=lambda kv: -kv[1])[:6])
            print(f"[{name}]  file={file_mb:8.3f} MB   keys={sorted(blob.keys())}")
            print(f"          {detail}")
        else:
            print(f"[{name}]  file={file_mb:8.3f} MB   (not a dict: {type(blob)})")


if __name__ == "__main__":
    report_routing()
    report_checkpoints()
