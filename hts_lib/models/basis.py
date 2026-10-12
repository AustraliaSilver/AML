import torch
import torch.nn as nn
import torch.nn.functional as F

class HtSLowRankAdapter(nn.Module):
    def __init__(self, K, D_out, D_in, rank=8):
        super().__init__()
        self.K = K
        self.rank = rank
        self.B = nn.Parameter(torch.randn(K, rank, D_in, 3, 3) * 0.01)
        self.A = nn.Parameter(torch.randn(K, D_out, rank, 1, 1) * 0.01)

    def forward(self, alpha):
        A_active = torch.einsum('bk, korxy -> borxy', alpha, self.A)
        B_active = torch.einsum('bk, krihw -> brihw', alpha, self.B)
        delta_W = torch.einsum('borxy, brihw -> boihw', A_active, B_active)
        return delta_W

    def compute_ortho_loss(self):
        b_flat = self.A.view(self.K, -1)
        b_norm = F.normalize(b_flat, p=2, dim=-1)
        gram = torch.mm(b_norm, b_norm.t())
        mask = ~torch.eye(self.K, dtype=torch.bool, device=self.A.device)
        return (gram[mask] ** 2).mean()

import torch.nn as nn

class HtSDualResidualBasis(nn.Module):
    """
    Dual-Residual Basis System for Multi-Stage Dynamic Weight Synthesis.
    
    Formula:
        W_l(x) = B_shared,l + sum_{i=1}^K alpha_{l,i}(x) * B_private,l,i
    """
    def __init__(self, K=8, d1=24, d2=48, d3=96):
        super().__init__()
        self.K = K
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        
        # 1. SHARED BASE TENSORS (Preserves Core Representations & High Base Accuracy)
        self.base1 = nn.Parameter(torch.randn(d1, d1, 3, 3) * 0.02)
        self.base1_2 = nn.Parameter(torch.randn(d2, d1, 3, 3) * 0.02)
        self.base2 = nn.Parameter(torch.randn(d2, d2, 3, 3) * 0.02)
        self.base2_3 = nn.Parameter(torch.randn(d3, d2, 3, 3) * 0.02)
        self.base3 = nn.Parameter(torch.randn(d3, d3, 3, 3) * 0.02)

        # 2. PRIVATE DYNAMIC RESIDUAL EXPERTS (K=8 Specialized Basis Kernels)
        self.b1_priv = nn.Parameter(torch.randn(K, d1, d1, 3, 3) * 0.01)
        self.b1_2_priv = nn.Parameter(torch.randn(K, d2, d1, 3, 3) * 0.01)
        self.b2_priv = nn.Parameter(torch.randn(K, d2, d2, 3, 3) * 0.01)
        self.b2_3_priv = nn.Parameter(torch.randn(K, d3, d2, 3, 3) * 0.01)
        self.b3_priv = nn.Parameter(torch.randn(K, d3, d3, 3, 3) * 0.01)

    def get_kernel(self, name, alpha, use_tc=False):
        """
        Synthesizes dynamic kernels on-the-fly for the current batch.
        Args:
            name: Stage identifier ('b1', 'b1_2', 'b2', 'b2_3', 'b3')
            alpha: Softmax/Gumbel routing weights [B, K]
            use_tc: If True and on CUDA, accelerate kernel synthesis via Tensor Core WMMA
        Returns:
            Kernel tensor [B, C_out, C_in, 3, 3]
        """
        B = alpha.shape[0]

        def _synthesize(b_priv, d_out, d_in):
            b_flat = b_priv.view(self.K, -1)
            if use_tc and alpha.is_cuda:
                try:
                    from mgn.accel import _C
                    if hasattr(_C, "batched_hyper_linear_tc"):
                        # [1, B, K] x [1, K, N] -> [1, B, N]
                        res = _C.batched_hyper_linear_tc(alpha.unsqueeze(0), b_flat.unsqueeze(0))
                        return res.squeeze(0).view(B, d_out, d_in, 3, 3)
                except Exception:
                    pass
            return torch.mm(alpha, b_flat).view(B, d_out, d_in, 3, 3)

        if name == 'b1':
            return self.base1.unsqueeze(0) + _synthesize(self.b1_priv, self.d1, self.d1)
        elif name == 'b1_2':
            return self.base1_2.unsqueeze(0) + _synthesize(self.b1_2_priv, self.d2, self.d1)
        elif name == 'b2':
            return self.base2.unsqueeze(0) + _synthesize(self.b2_priv, self.d2, self.d2)
        elif name == 'b2_3':
            return self.base2_3.unsqueeze(0) + _synthesize(self.b2_3_priv, self.d3, self.d2)
        elif name == 'b3':
            return self.base3.unsqueeze(0) + _synthesize(self.b3_priv, self.d3, self.d3)
        else:
            raise ValueError(f"Unknown kernel stage name: {name}")

    def compute_layerwise_ortho_loss(self):
        """
        Computes pairwise orthogonality loss among K private basis kernels for each stage.
        """
        def get_ortho(b_tensor):
            b_flat = b_tensor.view(self.K, -1)
            b_norm = F.normalize(b_flat, p=2, dim=-1)
            gram = torch.mm(b_norm, b_norm.t())
            mask = ~torch.eye(self.K, dtype=torch.bool, device=b_tensor.device)
            return (gram[mask] ** 2).mean()

        l_s1 = get_ortho(self.b1_priv)
        l_s2 = (get_ortho(self.b1_2_priv) + get_ortho(self.b2_priv)) * 0.5
        l_s3 = (get_ortho(self.b2_3_priv) + get_ortho(self.b3_priv)) * 0.5
        return l_s1, l_s2, l_s3
