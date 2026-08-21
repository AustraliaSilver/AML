import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_layer_aware_diversity_loss(alphas, model, K=8):
    """
    Computes the 3-component regularization suite for hypernetwork stability:
    1. Per-Sample Entropy (Sparsity encouragement)
    2. Batch Usage Balancing (KL divergence to Uniform distribution)
    3. Layer-Aware Orthogonality (Stage 1: 0.1, Stage 2: 0.5, Stage 3: 1.0)
    
    Args:
        alphas: Tensor of routing coefficients [B, 8, K]
        model: HtSContinualModel instance
        K: Number of experts (default 8)
    Returns:
        loss_entropy, loss_usage, loss_ortho
    """
    eps = 1e-8
    
    # 1. Layer-weighted Entropy (Sparsity)
    entropy_per_block = -(alphas * torch.log(alphas + eps)).sum(dim=-1).mean(dim=0)
    w_blocks = torch.tensor([0.1, 0.1, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0], device=alphas.device)
    loss_entropy = (entropy_per_block * w_blocks).mean()
    
    # 2. Usage Balancing (KL divergence to uniform)
    mean_alpha = alphas.mean(dim=0)
    uniform = torch.full_like(mean_alpha, 1.0 / K)
    kl_per_block = (mean_alpha * torch.log((mean_alpha + eps) / uniform)).sum(dim=-1)
    loss_usage = (kl_per_block * w_blocks).mean()
    
    # 3. Layer-aware Orthogonality
    l_s1, l_s2, l_s3 = model.basis_sys.compute_layerwise_ortho_loss()
    loss_ortho = 0.1 * l_s1 + 0.5 * l_s2 + 1.0 * l_s3
    
    return loss_entropy, loss_usage, loss_ortho
