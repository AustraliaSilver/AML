"""probe_drift_fixed.py — Corrected mixed-swap probe (the previous version
matched substrings 'sel'/'enc' against '+t1 both', which matches nothing,
so the combined-swap row silently duplicated the baseline).

Swaps after-task-1 versions of (a) prototype_selector, (b) vision_encoder,
(c) BOTH into the FINAL model and measures MNIST recovery vs the
after-task-1 reference. Answers: is old-task forgetting fully explained by
drift in the (encoder, selector) pair?
"""
import os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "AML_Core_Code"))

from train import ConvAML_CIFAR_TrueDynamic
from train_continual_aml import get_task_data, collect_structure_slot

device = torch.device("cuda")
amp = torch.bfloat16


def acc_eval(model, loader, override=None):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", dtype=amp):
                out, _, _, _ = model(x, tau=0.1,
                                     struct_override=(override(x.shape[0]) if override else None))
            correct += (out.argmax(-1) == y).sum().item()
            total += y.numel()
    return correct / total


def run():
    fin = ConvAML_CIFAR_TrueDynamic(K=48, norm_mode="in").to(device)
    fin.load_state_dict(torch.load(os.path.join(_ROOT, "aml_continual_after_fashion.pth"),
                                   map_location=device, weights_only=True))
    t1 = ConvAML_CIFAR_TrueDynamic(K=48, norm_mode="in").to(device)
    t1.load_state_dict(torch.load(os.path.join(_ROOT, "aml_continual_after_mnist.pth"),
                                  map_location=device, weights_only=True))

    _, te_m = get_task_data("mnist", os.path.join(_ROOT, "data"))
    with torch.no_grad():
        slot_m = collect_structure_slot(fin, te_m, device, amp)

    def ov(B):
        return (slot_m["gate"].unsqueeze(0).expand(B, -1),
                torch.stack([1.0 - slot_m["bm_res"], slot_m["bm_res"]], -1).unsqueeze(0).expand(B, -1, -1))

    fin_sel = {k: v.clone() for k, v in fin.prototype_selector.state_dict().items()}
    fin_enc = {k: v.clone() for k, v in fin.vision_encoder.state_dict().items()}
    t1_sel = {k: v.clone() for k, v in t1.prototype_selector.state_dict().items()}
    t1_enc = {k: v.clone() for k, v in t1.vision_encoder.state_dict().items()}

    def swap(use_t1_sel, use_t1_enc):
        fin.prototype_selector.load_state_dict(t1_sel if use_t1_sel else fin_sel)
        fin.vision_encoder.load_state_dict(t1_enc if use_t1_enc else fin_enc)

    print("reference (after-task-1 model): %.2f%%" % (acc_eval(t1, te_m) * 100))
    for sel, enc, label in [(False, False, "fin (baseline)"),
                            (True,  False, "+t1 selector"),
                            (False, True,  "+t1 encoder"),
                            (True,  True,  "+t1 BOTH")]:
        swap(sel, enc)
        print("%16s: %.2f%%" % (label, acc_eval(fin, te_m, ov) * 100))


if __name__ == "__main__":
    run()
