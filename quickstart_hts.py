"""
HtS Library Quickstart Demonstration
====================================
Demo of how to use the hts_lib library to initialize models, load data,
and inspect parameter statistics.
"""

import torch
from hts_lib import HtSContinualModel, ContinualEvaluator, get_continual_dataloaders

def demo_quickstart():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=================================================================")
    print(f"HtS LIBRARY (Hypernetwork to Shared-Prototype) - QUICKSTART ({device})")
    print("=================================================================\n")

    # 1. Initialize model
    print("[1] Initialize HtSContinualModel (K=8 Experts, D=24->48->96)...")
    model = HtSContinualModel(d1=24, d2=48, d3=96, K=8, num_classes=10).to(device)

    # 2. Detailed parameter statistics
    stats = model.count_parameters_breakdown(task="cifar")
    print("\n------------------------------------------------------------")
    print("PARAMETER STATISTICS:")
    print("------------------------------------------------------------")
    print(f"  * Active Child Parameters : {stats['child_active_params']:>10,}")
    print(f"  * Shared Base + K=8 Basis : {stats['basis_params']:>10,}")
    print(f"  * Router & Controller     : {stats['router_params']:>10,}")
    print(f"  * Total Model             : {stats['total_hypernetwork_params']:>10,}")
    print("------------------------------------------------------------\n")

    # 3. Test forward synthesis on-the-fly
    print("[2] Test Forward On-The-Fly Synthesis:")
    x_dummy_cifar = torch.randn(4, 3, 32, 32, device=device)
    x_dummy_mnist = torch.randn(4, 1, 28, 28, device=device)

    with torch.no_grad():
        out_c, alpha_c = model(x_dummy_cifar, task="cifar")
        out_m, alpha_m = model(x_dummy_mnist, task="mnist")

    print(f"  * CIFAR Input  : {list(x_dummy_cifar.shape)} -> Output Logits: {list(out_c.shape)} | Alpha Shape: {list(alpha_c.shape)}")
    print(f"  * MNIST Input  : {list(x_dummy_mnist.shape)} -> Output Logits: {list(out_m.shape)} | Alpha Shape: {list(alpha_m.shape)}")
    print(f"  * Sample 0 CIFAR Alpha (Stage 3): {alpha_c[0, 7].cpu().numpy().round(2).tolist()}")
    print(f"  * Sample 0 MNIST Alpha (Stage 3): {alpha_m[0, 7].cpu().numpy().round(2).tolist()}")

    print("\n[SUCCESS] hts_lib is working correctly!")
    print("=================================================================\n")

if __name__ == '__main__':
    demo_quickstart()
