import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pure_vision_aml_benchmark import AMLVisionArchitectureSearchModel

def extract_and_print_vision_aml_report():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_set  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader  = DataLoader(test_set, batch_size=128, shuffle=False)
    
    model = AMLVisionArchitectureSearchModel(dim=192, depth=6, K=16).to(device)
    model.load_state_dict(torch.load("best_vision_aml_arch_search.pth", map_location=device, weights_only=True))
    model.eval()
    
    mode_probs_all = []
    d_lengths_all = []
    gates_all = []
    
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            with torch.amp.autocast('cuda'):
                _, diag = model(x, tau=0.3)
            mode_probs_all.append(diag['block_mode_probs'].cpu().numpy())
            d_lengths_all.append(diag['d_lengths'].cpu().numpy())
            gates_all.append(diag['layer_gates'].cpu().numpy())
            
    mode_arr = np.concatenate(mode_probs_all, axis=0) # [10000, 6, 4]
    d_arr = np.concatenate(d_lengths_all, axis=0)     # [10000, 6, 1]
    g_arr = np.concatenate(gates_all, axis=0)         # [10000, 6]
    
    mean_modes = np.mean(mode_arr, axis=0) * 100
    mean_d = np.mean(d_arr, axis=0).squeeze(-1)
    mean_g = np.mean(g_arr, axis=0) * 100
    
    print("\n" + "="*95)
    print("BAO CAO TU DONG CHON KIEN TRUC MANG CON CUA VISION AML (10,000 ANH TEST)")
    print("="*95)
    print(f"{'Tang':<8} {'Cong Sau G':<12} {'Chieu D(x)':<12} {'CNN (%)':<12} {'ViT (%)':<12} {'MoE (%)':<12} {'ResNet (%)':<12} {'Kien Truc Thang'}")
    print("-" * 95)
    arch_names = ["CNN", "ViT", "MoE", "ResNet"]
    for i in range(6):
        c, v, m, r = mean_modes[i]
        dom = arch_names[np.argmax(mean_modes[i])]
        print(f"Layer {i+1:<3} {mean_g[i]:<10.1f}% {mean_d[i]:<11.1f} {c:<11.2f}% {v:<11.2f}% {m:<11.2f}% {r:<11.2f}% {dom}")
        
    print("-" * 95)
    tot_mode = np.mean(mean_modes, axis=0)
    print(f"{'TOTAL':<8} {np.mean(mean_g):<10.1f}% {np.mean(mean_d):<11.1f} {tot_mode[0]:<11.2f}% {tot_mode[1]:<11.2f}% {tot_mode[2]:<11.2f}% {tot_mode[3]:<11.2f}% {arch_names[np.argmax(tot_mode)]} (DOMINANT)")
    print("="*95)

if __name__ == '__main__':
    extract_and_print_vision_aml_report()
