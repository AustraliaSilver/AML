"""Do the offline centroids match the ones the trainer actually stored?

The probe predicts routing numbers that contradict the run's own `route_acc`,
so the first thing to check is not the metric but the measurement: if the
centroids this script computes differ from the ones inside
`routing_state_after_*.pt`, the probe is not embedding the way the trainer
does, and every conclusion drawn from it is void.

Run:  python probes/probe_router_agreement.py routing_state_after_cifar10.pt
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mgn.blocks.hybrid_18 import ConvAML_18Layer_Hybrid
from train_continual_aml import get_task_data, router_features

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASKS = ["mnist", "cifar10", "fashion"]


def load_encoder(task):
    m = ConvAML_18Layer_Hybrid(K=12, max_layers=18, max_dim=48, num_classes=10,
                               norm_type="in", encoder_tail_act="none").to(DEV)
    m.load_state_dict(torch.load(
        os.path.join(ROOT, f"aml_continual_after_{task}.pth"), map_location=DEV))
    m.eval()
    return m


@torch.no_grad()
def embed(model, loader, batch, limit=None):
    zs = []
    for i, (x, _) in enumerate(loader):
        if limit and i >= limit:
            break
        x = x.to(DEV, non_blocking=True)
        with torch.autocast("cuda", enabled=DEV.type == "cuda",
                            dtype=torch.bfloat16):
            zs.append(router_features(model, x).float())
        del x
    return torch.cat(zs, 0)


def main(path):
    blob = torch.load(os.path.join(ROOT, path), map_location="cpu",
                      weights_only=False)
    sd = blob["router"]
    stored = sd["centroids"][:int(sd["num_tasks"].item())].to(DEV)
    n_stored = sd["task_counts"][:int(sd["num_tasks"].item())]
    print(f"{path}: stored {stored.shape[0]} centroids after {blob['task']}")

    for k in range(stored.shape[0]):
        task = TASKS[k]
        enc = load_encoder(task)
        for split, loader, batch in (
                ("test", get_task_data(task, os.path.join(ROOT, "data"))[1], 512),
                ("train", get_task_data(task, os.path.join(ROOT, "data"))[0], 256)):
            z = embed(enc, loader, batch)
            mu = z.mean(0)
            d = (mu - stored[k]).norm().item()
            rel = d / max(stored[k].norm().item(), 1e-9)
            print(f"  task {k} ({task}) fit={split:5s} n={z.shape[0]:6d} "
                  f"(stored n={int(n_stored[k])}) "
                  f"||mu-mu_stored||={d:.5f}  rel={rel:.5f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "routing_state_after_cifar10.pt")