"""
SCRIPT MEASURING DYNAMIC DIVERSITY:
Tính toán ma trận Cosine Similarity giữa các trọng số W_child(x_i) và W_child(x_j):
1. Intra-task Diversity (Sự đa dạng giữa các ảnh trong cùng 1 task)
2. Inter-task Diversity (Sự phân tách giữa ảnh MNIST vs ảnh CIFAR-10)
3. Router Entropy & Variance của hệ số alpha(x)
"""

import torch
import torch.nn.functional as F
import numpy as np
from hts_multistage_cifar import HtSMultiStageModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def measure_dynamic_diversity():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== DO DAC DYNAMIC DIVERSITY CUA HtS ROUTER & WEIGHT GENERATOR ({device}) ===\n")
    
    # Load model đã huấn luyện
    model = HtSMultiStageModel(d1=24, d2=48, d3=96, K=4, num_classes=10).to(device)
    
    # Tải tập test MNIST và CIFAR-10
    tf_m = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tf_c = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    m_loader = DataLoader(datasets.MNIST('./data', False, download=True, transform=tf_m), batch_size=100, shuffle=False)
    c_loader = DataLoader(datasets.CIFAR10('./data', False, download=True, transform=tf_c), batch_size=100, shuffle=False)
    
    x_mnist, y_mnist = next(iter(m_loader))
    x_cifar, y_cifar = next(iter(c_loader))
    
    x_mnist = x_mnist.to(device)
    x_cifar = x_cifar.to(device)
    
    model.eval()
    with torch.no_grad():
        # Trích xuất hệ số alphas cho MNIST và CIFAR-10
        h_m = model.stem_mnist(x_mnist)
        z_m = model.task_ctrl(h_m)
        alphas_m = F.softmax(model.alpha_gen(z_m).view(100, 8, model.K), dim=-1) # [100, 8, 4]
        
        h_c = model.stem_cifar(x_cifar)
        z_c = model.task_ctrl(h_c)
        alphas_c = F.softmax(model.alpha_gen(z_c).view(100, 8, model.K), dim=-1) # [100, 8, 4]
        
        # Sinh trọng số W_child đầy đủ cho từng sample của Stage 1 Block 1 (Kernel 24x24x3x3)
        # [100, 24, 24, 3, 3] -> flatten thành vector [100, 5184]
        w_mnist_b1 = model.basis_sys.get_kernel('b1', alphas_m[:, 0]).flatten(1)
        w_cifar_b1 = model.basis_sys.get_kernel('b1', alphas_c[:, 0]).flatten(1)
        
        # Sinh trọng số W_child của Stage 3 Block 8 (Kernel 96x96x3x3)
        # [100, 96, 96, 3, 3] -> flatten thành vector [100, 82944]
        w_mnist_b8 = model.basis_sys.get_kernel('b3', alphas_m[:, 7]).flatten(1)
        w_cifar_b8 = model.basis_sys.get_kernel('b3', alphas_c[:, 7]).flatten(1)

    # =========================================================================
    # 1. TÍNH COSINE SIMILARITY PAIRWISE
    # =========================================================================
    def compute_pairwise_cosine(w1, w2):
        w1_norm = F.normalize(w1, p=2, dim=-1)
        w2_norm = F.normalize(w2, p=2, dim=-1)
        return torch.mm(w1_norm, w2_norm.t()) # [N1, N2]

    # Block 1:
    cos_m_m_b1 = compute_pairwise_cosine(w_mnist_b1, w_mnist_b1)
    cos_c_c_b1 = compute_pairwise_cosine(w_cifar_b1, w_cifar_b1)
    cos_m_c_b1 = compute_pairwise_cosine(w_mnist_b1, w_cifar_b1)
    
    # Block 8:
    cos_m_m_b8 = compute_pairwise_cosine(w_mnist_b8, w_mnist_b8)
    cos_c_c_b8 = compute_pairwise_cosine(w_cifar_b8, w_cifar_b8)
    cos_m_c_b8 = compute_pairwise_cosine(w_mnist_b8, w_cifar_b8)

    # Lấy giá trị trung bình ngoài đường chéo (off-diagonal) cho intra-task
    def off_diagonal_mean(matrix):
        N = matrix.shape[0]
        mask = ~torch.eye(N, dtype=torch.bool, device=matrix.device)
        return matrix[mask].mean().item(), matrix[mask].std().item(), matrix[mask].min().item(), matrix[mask].max().item()

    print("="*65)
    print("KET QUA DO DAC DYNAMIC DIVERSITY (COSINE SIMILARITY)")
    print("="*65)
    
    print("\n1. TANG SAU CUA MANG (STAGE 3 - BLOCK 8: D=96):")
    m_mean, m_std, m_min, m_max = off_diagonal_mean(cos_m_m_b8)
    c_mean, c_std, c_min, c_max = off_diagonal_mean(cos_c_c_b8)
    mc_mean = cos_m_c_b8.mean().item()
    mc_min, mc_max = cos_m_c_b8.min().item(), cos_m_c_b8.max().item()
    
    print(f"  * Intra-MNIST cos(W(x_i), W(x_j)) : {m_mean:.4f}  (Std: {m_std:.4f}, Min: {m_min:.4f}, Max: {m_max:.4f})")
    print(f"  * Intra-CIFAR cos(W(x_i), W(x_j)) : {c_mean:.4f}  (Std: {c_std:.4f}, Min: {c_min:.4f}, Max: {c_max:.4f})")
    print(f"  * Inter-Task cos(W_mnist, W_cifar): {mc_mean:.4f}  (Min: {mc_min:.4f}, Max: {mc_max:.4f})")
    
    print("\n2. TANG NONG CUA MANG (STAGE 1 - BLOCK 1: D=24):")
    m1_mean, m1_std, m1_min, m1_max = off_diagonal_mean(cos_m_m_b1)
    c1_mean, c1_std, c1_min, c1_max = off_diagonal_mean(cos_c_c_b1)
    mc1_mean = cos_m_c_b1.mean().item()
    
    print(f"  * Intra-MNIST cos(W(x_i), W(x_j)) : {m1_mean:.4f}  (Std: {m1_std:.4f})")
    print(f"  * Intra-CIFAR cos(W(x_i), W(x_j)) : {c1_mean:.4f}  (Std: {c1_std:.4f})")
    print(f"  * Inter-Task cos(W_mnist, W_cifar): {mc1_mean:.4f}")

    # =========================================================================
    # 2. PHAN TICH HE SO ALPHA ROUTER
    # =========================================================================
    print("\n3. PHAN TICH PHAN PHOI HE SO ROUTER ALPHA (K=4):")
    alpha_m_mean = alphas_m.mean(dim=0).cpu().numpy()
    alpha_c_mean = alphas_c.mean(dim=0).cpu().numpy()
    
    print("\n  He so Alpha trung binh cho 8 Blocks:")
    print(f"  {'Block':<8} | {'MNIST (K=4 Basis Coeffs)':<26} | {'CIFAR-10 (K=4 Basis Coeffs)':<26}")
    print("  " + "-"*62)
    for b in range(8):
        m_str = f"[{alpha_m_mean[b,0]:.2f}, {alpha_m_mean[b,1]:.2f}, {alpha_m_mean[b,2]:.2f}, {alpha_m_mean[b,3]:.2f}]"
        c_str = f"[{alpha_c_mean[b,0]:.2f}, {alpha_c_mean[b,1]:.2f}, {alpha_c_mean[b,2]:.2f}, {alpha_c_mean[b,3]:.2f}]"
        print(f"  Block {b+1:<2} | {m_str:<26} | {c_str:<26}")
    print("="*65)

if __name__ == '__main__':
    measure_dynamic_diversity()
