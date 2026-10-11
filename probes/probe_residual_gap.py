"""probe_residual_gap.py — Pinpoint the remaining ~0.7pt MNIST loss in v10.

Checks:
1. Are MNIST-owned proto rows [0:16) bit-identical between the after-task-1
   checkpoint and the final checkpoint?  (isolation integrity)
2. How much softmax mass does MNIST routing put OUTSIDE its own rows?
   (leakage onto foreign rows whose weights later tasks trained)
"""
import os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "AML_Core_Code"))

from train import ConvAML_CIFAR_TrueDynamic, set_seed
from train_continual_aml import get_task_data

set_seed(42)


def run():
    device = torch.device("cuda")
    t1 = ConvAML_CIFAR_TrueDynamic(K=48, norm_mode="in").to(device)
    t1.load_state_dict(torch.load(os.path.join(_ROOT, "aml_continual_after_mnist.pth"),
                                  map_location=device, weights_only=True))
    fin = ConvAML_CIFAR_TrueDynamic(K=48, norm_mode="in").to(device)
    fin.load_state_dict(torch.load(os.path.join(_ROOT, "aml_continual_after_fashion.pth"),
                                   map_location=device, weights_only=True))

    p_t1 = t1.proto_weights.data
    p_fin = fin.proto_weights.data
    print("[1] MNIST rows [0:16) identical:",
          torch.equal(p_t1[:16], p_fin[:16]),
          f"| max abs diff {(p_t1[:16]-p_fin[:16]).abs().max().item():.3e}")
    print("    CIFAR rows [16:32) changed:", not torch.equal(p_t1[16:32], p_fin[16:32]),
          "| mean abs diff", f"{(p_t1[16:32]-p_fin[16:32]).abs().mean().item():.4f}")

    _, te_m = get_task_data("mnist", os.path.join(_ROOT, "data"))
    for tag, mdl in [("t1 ", t1), ("fin", fin)]:
        mdl.eval()
        mass_own, mass_tot, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, _ in te_m:
                x = x.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, _, pi, _ = mdl(x, tau=0.1)
                pif = pi.float()
                mass_own += pif[:, :16].sum().item()
                mass_tot += pif.sum().item()
                n += pif.shape[0]
        print(f"[2] {tag}: MNIST pi mass on OWN rows [0:16): {mass_own/mass_tot*100:.2f}% "
              f"(foreign {100-mass_own/mass_tot*100:.2f}%)")


if __name__ == "__main__":
    run()
