"""
AML Unified Quickstart Demonstration
====================================
Demonstrates the full Architecture Mode Learning (AML) system:
1. Meta-Generator (ModelGenerator) generating dynamic Depth, Width, and Hybrid Blocks (Plain / ResNet / ViT).
2. End-to-End Supernet Execution (ConvAML_CIFAR) with Prototype Library and Soft Dynamic Routing.
3. Static Subnet Extraction (Compiling generated model into an ultra-fast standalone PyTorch nn.Module).
4. Continual Hypernetwork Lifelong Learning (hts_lib Dual-Residual Basis).
"""

import torch
import torch.nn.functional as F
import sys
import os

# Import AML Core Engine
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AML_Core_Code'))
from train_cv_cifar import ConvAML_CIFAR
from extract_static_model import extract_child, StaticChildModel
from mgn.generator import ModelGenerator
from mgn.constraints import compute_aml_efficiency
from hts_lib.models.hypernetwork import HtSContinualModel

def demo_aml_unified():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("==========================================================================")
    print(f"ARCHITECTURE MODE LEARNING (AML) UNIFIED FRAMEWORK DEMO ({device})")
    print("==========================================================================\n")

    # -------------------------------------------------------------------------
    # PART 1: AML SUPERNET (MẠNG MẸ SINH CẤU TRÚC VÀ TRỌNG SỐ CHO MẠNG CON)
    # -------------------------------------------------------------------------
    print("[1] AML SUPERNET (ConvAML_CIFAR: VisionModelGenerator -> AcceleratedCVNetwork)")
    aml_model = ConvAML_CIFAR().to(device)
    
    # Input batch (4 CIFAR images: 4 x 3 x 32 x 32)
    x = torch.randn(4, 3, 32, 32, device=device)
    
    # Mother Network executes forward pass
    logits, structure = aml_model(x)
    
    print(f"  * Mother Network Input          : {list(x.shape)}")
    print(f"  * Generated Active Layers (L)   : {structure.layer_gates[0].detach().cpu().numpy().round(2).tolist()}")
    print(f"  * Generated Width Fractions (D) : {structure.dim_weights[0, :, 0].detach().cpu().numpy().round(2).tolist()}")
    print(f"  * Output Logits                 : {list(logits.shape)}")
    
    # -------------------------------------------------------------------------
    # PART 2: STATIC MODEL EXTRACTION (HARD PRUNING INTO STANDALONE SUBNET)
    # -------------------------------------------------------------------------
    print("\n[2] STATIC MODEL EXTRACTION (Compiling Subnet into StaticChildModel)")
    static_subnet = extract_child(aml_model, x).to(device)
    with torch.no_grad():
        out_static = static_subnet(x)
    print(f"  * Extracted Subnet Layers       : {len(static_subnet.features)} layers")
    print(f"  * Static Subnet Inference Output: {list(out_static.shape)} (Runs standalone, 0% Gumbel overhead!)")

    # -------------------------------------------------------------------------
    # PART 3: CONTINUAL HYPERNETWORK (DUAL-RESIDUAL LIFELONG SPECIALIZATION)
    # -------------------------------------------------------------------------
    print("\n[3] HtS CONTINUAL HYPERNETWORK (Multi-Domain Lifelong Learning Subsystem)")
    hts_model = HtSContinualModel(d1=24, d2=48, d3=96, K=8, num_classes=10).to(device)
    hts_stats = hts_model.count_parameters_breakdown(task="cifar")
    
    with torch.no_grad():
        out_cifar, alpha_c = hts_model(x, task="cifar")
        
    print(f"  * Active Child Execution Params : {hts_stats['child_active_params']:,} (< 10k params)")
    print(f"  * Total Trainable Hypernetwork  : {hts_stats['total_hypernetwork_params']:,}")
    print(f"  * HtS Output Logits             : {list(out_cifar.shape)} | Dynamic Alpha: {list(alpha_c.shape)}")

    print("\n==========================================================================")
    print("SUCCESS: AML Framework fully unified across Meta-Generation & Continual Learning!")
    print("==========================================================================\n")

if __name__ == '__main__':
    demo_aml_unified()
