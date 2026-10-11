"""probe_v14_mech.py — Verify the v14 evaluate() upgrades against v13 numbers.

1. fp32 snapshot + forced routing  -> must match v13 (99.46 / 77.99 / 92.14)
2. int8-quantized proto snapshots  -> must match fp32 within ~0.05pt
3. broken router (uniform logits)  -> margin-fallback must rescue accuracy
"""
import os, sys
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "AML_Core_Code"))

from train import ConvAML_CIFAR_TrueDynamic
from train_continual_aml import get_task_data, collect_structure_slot, evaluate

device = torch.device("cuda")
amp = torch.bfloat16


class ForceDom(nn.Module):
    def __init__(self, n, cls):
        super().__init__(); self.n = n; self.cls = cls
    def forward(self, z):
        e = torch.zeros(z.shape[0], self.n, device=z.device)
        e[:, self.cls] = 10.0
        return e


def build_snaps(models, q8):
    snaps = []
    for m in models:
        s = {"enc": {k: v.clone() for k, v in m.vision_encoder.state_dict().items()},
             "sel": {k: v.clone() for k, v in m.prototype_selector.state_dict().items()}}
        pw = m.proto_weights.detach()
        if q8:
            scale = pw.abs().max().clamp(min=1e-12) / 127.0
            s["protos"] = {"q": torch.round(pw / scale).to(torch.int8).cpu(),
                           "scale": scale.cpu()}
        else:
            s["protos"] = pw.cpu().clone()
        snaps.append(s)
    return snaps


def run():
    ms, tes, slots = [], [], []
    names = ["mnist", "cifar10", "fashion"]
    for nm in names:
        m = ConvAML_CIFAR_TrueDynamic(K=48, norm_mode="in").to(device)
        m.load_state_dict(torch.load(os.path.join(_ROOT, f"aml_continual_after_{nm}.pth"),
                                     map_location=device, weights_only=True))
        ms.append(m)
        _, te = get_task_data(nm, os.path.join(_ROOT, "data"))
        tes.append(te)
        slots.append(collect_structure_slot(m, te, device, amp))

    ref = {"mnist": 99.46, "cifar10": 77.99, "fashion": 92.14}
    print(f"{'eval':>9s} | {'fp32':>7s} | {'int8':>7s} | {'v13':>6s}")
    for j, nm in enumerate(names):
        head = ForceDom(j + 1, cls=j).to(device)
        snaps_f = build_snaps(ms[:j + 1], q8=False)
        a_f, _ = evaluate(ms[j], tes[j], device, amp, dom_head=head,
                          slots=slots[:j + 1], snaps=snaps_f, route_tau=None)
        snaps_q = build_snaps(ms[:j + 1], q8=True)
        a_q, _ = evaluate(ms[j], tes[j], device, amp, dom_head=head,
                          slots=slots[:j + 1], snaps=snaps_q, route_tau=None)
        print(f"{nm:>9s} | {a_f*100:7.2f} | {a_q*100:7.2f} | {ref[nm]:6.2f}")

    # broken-router fallback test on MNIST under final model
    class GarbageRouter(nn.Module):
        def forward(self, z):
            return torch.randn(z.shape[0], 3, device=z.device)
    snaps3 = build_snaps(ms, q8=True)
    a_nb, _ = evaluate(ms[2], tes[0], device, amp, dom_head=GarbageRouter().to(device),
                       slots=slots, snaps=snaps3, route_tau=None)
    a_fb, _ = evaluate(ms[2], tes[0], device, amp, dom_head=GarbageRouter().to(device),
                       slots=slots, snaps=snaps3, route_tau=0.95)
    print(f"\nMNIST w/ RANDOM router : no-fallback {a_nb*100:.2f}% -> fallback(tau=.95) {a_fb*100:.2f}%")


if __name__ == "__main__":
    run()
