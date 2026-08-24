"""probe_route_confirm.py — Confirm that the v12 MNIST drop (80.01%) is caused
by dom_head misrouting, not by the snapshot mechanism itself.

Evaluates the after-CIFAR checkpoint on MNIST with the MNIST snapshot pair,
FORCING domain 0 (no learned router involved). If this recovers ~99%, the
snapshot swap is innocent and routing is the sole culprit.
"""
import os, sys
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "AML_Core_Code"))

from train import ConvAML_CIFAR_TrueDynamic
from train_continual_aml import get_task_data, collect_structure_slot, evaluate

device = torch.device("cuda")
amp = torch.bfloat16


class ForceDom0(nn.Module):
    def forward(self, z):
        return torch.zeros(z.shape[0], 2, device=z.device)


def run():
    t1 = ConvAML_CIFAR_TrueDynamic(K=48, norm_mode="in").to(device)
    t1.load_state_dict(torch.load(os.path.join(_ROOT, "aml_continual_after_mnist.pth"),
                                  map_location=device, weights_only=True))
    t2 = ConvAML_CIFAR_TrueDynamic(K=48, norm_mode="in").to(device)
    t2.load_state_dict(torch.load(os.path.join(_ROOT, "aml_continual_after_cifar10.pth"),
                                  map_location=device, weights_only=True))

    _, te_m = get_task_data("mnist", os.path.join(_ROOT, "data"))
    _, te_c = get_task_data("cifar10", os.path.join(_ROOT, "data"))
    with torch.no_grad():
        slot0 = collect_structure_slot(t1, te_m, device, amp)
        slot1 = collect_structure_slot(t2, te_c, device, amp)

    snaps = [
        {"enc": {k: v.clone() for k, v in t1.vision_encoder.state_dict().items()},
         "sel": {k: v.clone() for k, v in t1.prototype_selector.state_dict().items()},
         "protos": t1.proto_weights.detach().cpu().clone()},
        {"enc": {k: v.clone() for k, v in t2.vision_encoder.state_dict().items()},
         "sel": {k: v.clone() for k, v in t2.prototype_selector.state_dict().items()},
         "protos": t2.proto_weights.detach().cpu().clone()},
    ]

    acc_forced, _ = evaluate(t2, te_m, device, amp,
                             dom_head=ForceDom0().to(device),
                             slots=[slot0, slot1], snaps=snaps)
    print("MNIST (after-cifar model, FORCED domain-0 snapshot): %.2f%%" % (acc_forced * 100))
    print("(v12 live-routed gave 80.01%% -> gap = misrouted batches)")


if __name__ == "__main__":
    run()
