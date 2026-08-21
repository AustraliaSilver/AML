"""Debug: find the C_out threshold where kernel breaks."""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mgn.accel.functions as aml_ops

torch.manual_seed(42)
device = "cuda"

B, C_in, H, W = 2, 3, 4, 4
K = 4
W_dim = 861642

for C_out in [4, 8, 16, 32, 64, 128, 256]:
    w_offset = 0
    per_oc_weight = C_in * 9
    b_offset = w_offset + C_out * per_oc_weight

    X = torch.randn(B, C_in, H, W, device=device)
    prototypes = torch.randn(K, W_dim, device=device)
    pi = torch.ones(B, K, device=device) / K

    try:
        Y = aml_ops.fused_dynamic_conv2d(X, prototypes, pi, w_offset, b_offset)
        nz = torch.count_nonzero(Y).item()
        total = Y.numel()
        print(f"C_out={C_out:4d}: shape={Y.shape}, mean={Y.mean():.6f}, non_zero={nz}/{total} ({100*nz/total:.1f}%)")
    except Exception as e:
        print(f"C_out={C_out:4d}: ERROR {e}")
