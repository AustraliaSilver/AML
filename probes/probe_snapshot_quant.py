"""Does the default int8 snapshot quantisation change any prediction?

`--snap-q8` defaults to 1, so every stored snapshot's prototype matrix is
int8 with ONE global scale over the whole matrix:

    scale = max|P| / 127        q = round(P / scale)        P_hat = q * scale

That is the matrix the snapshot-retrieval protocols (A_oracle, A_auto) actually
evaluate with, so any error it introduces is inside the headline numbers. Two
things make the loss worth quantifying rather than assuming:

1. The step is global, and on the real production checkpoints the maximum is a
   single element in a row FROZEN by ProtoIsolator after task 0
   (proto_weights[3, 6718], +0.2257, identical in all three snapshots). So every
   later task is quantised at a resolution set by task 0's largest weight.
2. The matrix is mostly small entries: median |P| is 0.0338 against a step of
   1.78e-3, so the median element's worst-case relative error is ~2.6 %, and
   1.4 % of elements quantise to exactly zero.

The measurement: evaluate with the int8 matrix exactly as the protocol does, and
again with the SAME matrix re-quantised to 16 bits (a strictly finer grid on the
identical values). If the two disagree on zero predictions, the int8 step is not
what limits this protocol and the default is defensible; if they disagree, the
stored artefact is the weaker of the two and that belongs in the write-up.

Usage:  python probes/probe_snapshot_quant.py [routing_state_after_mnist.pt]
"""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from mgn.blocks.hybrid_18 import ConvAML_18Layer_Hybrid          # noqa: E402
from train_continual_aml import get_task_data                    # noqa: E402

TASKS = {"mnist": "mnist", "cifar10": "cifar10", "fashion": "fashion"}


def quantise(P, bits):
    """Per-matrix symmetric quantisation, the same rule --snap-q8 uses."""
    levels = float(2 ** (bits - 1) - 1)
    scale = P.abs().max().clamp(min=1e-12) / levels
    if bits >= 16:                       # torch has no int16; keep the values
        return torch.round(P / scale) * scale
    return torch.round(P / scale).to(torch.int8).float() * scale


def load_into(model, snap, protos):
    with torch.no_grad():
        model.vision_encoder.load_state_dict(snap["enc"])
        model.prototype_selector.load_state_dict(snap["sel"])
        model.stem.load_state_dict(snap["stem"])
        model.layers.load_state_dict(snap["layers"])
        model.ln.load_state_dict(snap["ln"])
        model.structure_prototypes.copy_(snap["struct_heads"]["sprotos"])
        model.block_mode_prototypes.copy_(snap["struct_heads"]["bmprotos"])
        model.proto_weights.copy_(protos)


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out, _, _, _ = model(x, tau=0.1)
        correct += int((out.argmax(-1) == y).sum())
        total += int(y.numel())
    return correct / max(total, 1), total


def main(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(os.path.join(_ROOT, path), map_location="cpu", weights_only=False)
    a = d["args"]
    task = d["task"]
    print(f"artefact {path}: saved after '{task}' (stage {d['stage']}), "
          f"norm={a['norm']}, max_dim={a['max_dim']}, snap_q8={a['snap_q8']}")
    if not a.get("snap_q8"):
        print("this artefact was stored in fp32; nothing to measure")
        return

    model = ConvAML_18Layer_Hybrid(K=a["K"], max_dim=a["max_dim"], max_layers=18,
                                   num_classes=10, norm_type=a["norm"],
                                   encoder_tail_act=a["encoder_tail_act"]).to(device)

    _, el = get_task_data(task, os.path.join(_ROOT, "data"), fast_cache=True)
    print(f"evaluating on the {task} test split ({len(el.dataset)} images)")

    for i, snap in enumerate(d["snaps"]):
        pr = snap["protos"]
        P8 = (pr["q"].to(device).float() * pr["scale"].to(device))   # as stored
        P16 = quantise(P8, 16)                                       # finer grid

        load_into(model, snap, P8)
        acc8, n = accuracy(model, el, device)
        load_into(model, snap, P16)
        acc16, _ = accuracy(model, el, device)

        print(f"  snap{i}: int8 {acc8*100:.2f} %   int16 {acc16*100:.2f} %   "
              f"delta {(acc16-acc8)*100:+.2f} pt  (n={n})")

    # error statistics of the matrix that is actually stored
    pr = d["snaps"][0]["protos"]
    P8 = (pr["q"].float() * pr["scale"]).flatten()
    nz = P8[P8 != 0].abs()
    step = float(pr["scale"])
    print(f"  matrix: step={step:.3e}  max|P|={float(P8.abs().max()):.4f}  "
          f"quantised-to-zero={100*float((P8 == 0).float().mean()):.1f} %  "
          f"median worst-case rel err of a non-zero element="
          f"{100*float((step/2/nz).median()):.1f} %")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "routing_state_after_mnist.pt")
