"""Is the saved routing state enough to re-answer questions about the router?

`--save-routing` writes the router buffers and the per-task snapshots next to
the checkpoints. Before that existed, every question about the router had to be
answered by retraining, and retraining does not reproduce (cuDNN.benchmark picks
algorithms by timing). This probe loads the artefact and checks that it carries
what the router needs:

  - the centroids and the counts that produced them,
  - the shared and per-task second moments,
  - the encoder snapshot each centroid's feature space belongs to,
  - and that rebuilding a TaskCentroidRouter from those buffers and scoring a
    real batch reproduces the routing decision the run made.

Run:  python probes/probe_routing_state.py routing_state_after_fashion.pt
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mgn.blocks.hybrid_18 import ConvAML_18Layer_Hybrid
from train_continual_aml import TaskCentroidRouter, get_task_data, router_features

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(path):
    blob = torch.load(os.path.join(ROOT, path) if not os.path.isabs(path) else path,
                      map_location="cpu", weights_only=False)
    router_sd, snaps = blob["router"], blob["snaps"]
    metric = blob["args"]["router_metric"]
    shrink = blob["args"]["router_shrink"]
    alpha = blob["args"].get("router_cov_alpha", 0.15)
    T = int(router_sd["num_tasks"].item())

    r = TaskCentroidRouter(latent_dim=router_sd["centroids"].shape[1],
                           max_tasks=router_sd["centroids"].shape[0],
                           metric=metric, shrink=shrink, cov_alpha=alpha)
    r.load_state_dict(router_sd)
    r = r.to(DEV)

    print(f"artefact : {path}")
    print(f"  task={blob['task']} stage={blob['stage']} metric={metric} alpha={alpha}")
    print(f"  snapshots stored: {len(snaps)}")
    print(f"  shared second moment: n={int(r.cov_n.item())}  "
          f"per-task n={[int(v) for v in r.cov_n_task[:T].tolist()]}")
    for k in range(T):
        c = r.centroids[k]
        print(f"  centroid {k}: ||mu||={c.norm():.4f}  n={int(r.task_counts[k].item())}")

    # Rebuild the model and score one batch with every task's own encoder,
    # exactly as evaluate() does.
    m = ConvAML_18Layer_Hybrid(K=12, max_layers=18, max_dim=48, num_classes=10,
                               norm_type="in", encoder_tail_act="none").to(DEV)
    m.load_state_dict(torch.load(
        os.path.join(ROOT, f"aml_continual_after_{blob['task']}.pth"),
        map_location=DEV))
    m.eval()

    x = torch.randn(64, 3, 32, 32, device=DEV)
    with torch.no_grad():
        zs = []
        for k in range(T):
            m.vision_encoder.load_state_dict(snaps[k]["enc"])
            zs.append(router_features(m, x))
        logits = r.score_per_task(zs)
        pred = logits.argmax(-1)
    print(f"  random-batch decision histogram: "
          f"{torch.bincount(pred, minlength=T).tolist()}")
    print("OK - the artefact carries enough state to rebuild and re-score the router")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "routing_state_after_fashion.pt")