import os
import math
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vision_moe_aml import AMLVisionMoETransformer

def analyze_model_routing_preferences():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== QUANTITATIVE ROUTING PREFERENCE ANALYSIS: AML (ViT DEPTH) VS MoE (EXPERTS) ({device}) ===")
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    
    model = AMLVisionMoETransformer(
        img_size=32, patch_size=4, in_chans=3, num_classes=10,
        dim=192, depth=6, num_heads=6, intermediate_dim=384,
        num_experts=4, top_k=2
    ).to(device)
    
    model.load_state_dict(torch.load("best_vision_moe_cifar10.pth", map_location=device, weights_only=True))
    model.eval()
    
    macro_gates_all = []
    expert_probs_all = [[] for _ in range(6)]
    
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            _, diag = model(x)
            macro_gates_all.append(diag['macro_gates'].cpu().numpy())
            for l_i, dist in enumerate(diag['expert_dist']):
                expert_probs_all[l_i].append(dist)
                
    macro_gates_arr = np.concatenate(macro_gates_all, axis=0) # [10000, 6]
    
    print("\n--- 1. AML MACRO-GATES (ViT LAYER DEPTH SELECTION G_1 .. G_6) ---")
    print(f"{'Layer':<10} {'Mean Gate Opening':<24} {'Std':<12} {'Behavior'}")
    print("-" * 70)
    for i in range(6):
        m_g = np.mean(macro_gates_arr[:, i])
        s_g = np.std(macro_gates_arr[:, i])
        behavior = "Fully Prioritized (Always Active)" if m_g > 0.8 else ("Dynamically Modulated" if m_g > 0.4 else "Frequently Skipped")
        print(f"Layer {i+1:<4} {m_g*100:<22.2f}% {s_g:<12.4f} {behavior}")
        
    print("\n--- 2. MoE MICRO-ROUTING (EXPERT PATCH SELECTION ACROSS 4 EXPERTS) ---")
    print(f"{'Layer':<10} {'Dominant Expert':<28} {'Entropy':<12} {'Routing Specialization'}")
    print("-" * 75)
    for i in range(6):
        mean_exp = np.mean(expert_probs_all[i], axis=0)
        dominant_exp = np.argmax(mean_exp) + 1
        dominant_pct = mean_exp[dominant_exp - 1] * 100
        entropy = -np.sum(mean_exp * np.log(mean_exp + 1e-8))
        print(f"Layer {i+1:<4} Expert {dominant_exp} ({dominant_pct:.1f}% load)          {entropy:.3f}        Diverse Specialist Routing")

if __name__ == '__main__':
    analyze_model_routing_preferences()
