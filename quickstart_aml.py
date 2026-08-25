"""
AML Unified Quickstart Demonstration
====================================
Demonstrates the full Architecture Mode Learning (AML) system:
1. Meta-Generator generating dynamic Depth, Width, and Hybrid Blocks (Plain / ResNet).
2. End-to-End Supernet Execution (ConvAML_CIFAR_TrueDynamic) with Prototype Library
   and Soft Dynamic Routing.
3. Static Subnet Extraction (compiling the generated model into a standalone
   PyTorch nn.Module with zero Gumbel overhead).
4. Continual Hypernetwork Lifelong Learning (hts_lib Dual-Residual Basis).

Runs on CPU or GPU. The fused CUDA kernels are used automatically when
available; otherwise the executor falls back to native PyTorch ops.
"""

import os
import sys

import torch

# The static extractor and the prototype layout live in AML_Core_Code.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AML_Core_Code'))

from train import ConvAML_CIFAR_TrueDynamic
from extract_static_model import extract_child, StaticChildModel
from mgn.constraints import compute_aml_efficiency
from hts_lib.models.hypernetwork import HtSContinualModel


def demo_aml_unified():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("==========================================================================")
    print(f"ARCHITECTURE MODE LEARNING (AML) UNIFIED FRAMEWORK DEMO ({device})")
    print("==========================================================================\n")

    # -------------------------------------------------------------------------
    # PART 1: AML SUPERNET (mother network synthesises structure + weights)
    # -------------------------------------------------------------------------
    print("[1] AML SUPERNET (ConvAML_CIFAR_TrueDynamic: 18 layers, K=32 prototypes)")
    aml_model = ConvAML_CIFAR_TrueDynamic(K=32, max_dim=64, max_layers=18).to(device)
    aml_model.eval()

    # Input batch (4 CIFAR images: 4 x 3 x 32 x 32)
    x = torch.randn(4, 3, 32, 32, device=device)

    # Mother network executes a forward pass
    with torch.no_grad():
        logits, structure, pi, _ = aml_model(x)

    active_layers = structure.layer_gates[0].detach().cpu().numpy().round(2).tolist()
    width_fracs = structure.dim_weights[0, :, 0].detach().cpu().numpy().round(2).tolist()
    print(f"  * Mother Network Input          : {list(x.shape)}")
    print(f"  * Generated Active Layers (L)   : {active_layers}")
    print(f"  * Generated Width Fractions (D) : {width_fracs}")
    print(f"  * Prototype Routing pi          : {list(pi.shape)} "
          f"(max weight {pi.max().item():.3f})")
    print(f"  * Output Logits                 : {list(logits.shape)}")

    # -------------------------------------------------------------------------
    # PART 2: C^infinity SMOOTH EFFICIENCY METRIC
    # -------------------------------------------------------------------------
    print("\n[2] C-INFINITY SMOOTH EFFICIENCY BARRIER (E_AML)")
    n_params = sum(p.numel() for p in aml_model.parameters())
    eff = compute_aml_efficiency(accuracy=0.85, p_allowance=float(n_params),
                                 q_used=float(n_params))
    print(f"  * Supernet Parameters           : {n_params:,}")
    print(f"  * E_AML (acc=0.85, Q=P_allow)   : {eff:.6f}")

    # -------------------------------------------------------------------------
    # PART 3: STATIC MODEL EXTRACTION (hard pruning into a standalone subnet)
    # -------------------------------------------------------------------------
    print("\n[3] STATIC MODEL EXTRACTION (compiling subnet into StaticChildModel)")
    static_subnet = extract_child(aml_model, x).to(device)
    static_subnet.eval()
    with torch.no_grad():
        out_static = static_subnet(x)
    child_params = sum(p.numel() for p in static_subnet.parameters())
    print(f"  * Extracted Subnet Layers       : {len(static_subnet.features)} layers")
    print(f"  * Extracted Subnet Parameters   : {child_params:,}")
    print(f"  * Static Subnet Inference Output: {list(out_static.shape)} "
          f"(runs standalone, 0% Gumbel overhead)")

    # -------------------------------------------------------------------------
    # PART 4: CONTINUAL HYPERNETWORK (dual-residual lifelong specialisation)
    # -------------------------------------------------------------------------
    print("\n[4] HtS CONTINUAL HYPERNETWORK (multi-domain lifelong learning)")
    hts_model = HtSContinualModel(d1=24, d2=48, d3=96, K=8, num_classes=10).to(device)
    hts_stats = hts_model.count_parameters_breakdown(task="cifar")

    with torch.no_grad():
        out_cifar, alpha_c = hts_model(x, task="cifar")

    print(f"  * Active Child Execution Params : {hts_stats['child_active_params']:,} (< 10k params)")
    print(f"  * Total Trainable Hypernetwork  : {hts_stats['total_hypernetwork_params']:,}")
    print(f"  * HtS Output Logits             : {list(out_cifar.shape)} "
          f"| Dynamic Alpha: {list(alpha_c.shape)}")

    print("\n==========================================================================")
    print("SUCCESS: AML Framework unified across Meta-Generation & Continual Learning")
    print("==========================================================================\n")


if __name__ == '__main__':
    demo_aml_unified()
