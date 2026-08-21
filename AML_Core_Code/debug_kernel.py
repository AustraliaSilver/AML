"""Debug: why CUDA kernel outputs all zeros."""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mgn.accel.functions as aml_ops

torch.manual_seed(42)
device = "cuda"

# Simple case first
B, C_in, C_out, H, W = 2, 3, 8, 4, 4
K = 4
w_offset = 0
per_oc_weight = C_in * 9  # 27
b_offset = w_offset + C_out * per_oc_weight  # 216
W_dim = 256

X = torch.randn(B, C_in, H, W, device=device)
prototypes = torch.randn(K, W_dim, device=device)
pi = torch.ones(B, K, device=device) / K

print(f"X: mean={X.mean():.4f}, std={X.std():.4f}")
print(f"prototypes: mean={prototypes.mean():.4f}, std={prototypes.std():.4f}")
print(f"pi: {pi[0]}")

Y = aml_ops.fused_dynamic_conv2d(X.float(), prototypes.float(), pi.float(), w_offset, b_offset)
print(f"\nKernel output: shape={Y.shape}, mean={Y.mean():.6f}, std={Y.std():.6f}")
print(f"Kernel all_zero={torch.all(Y == 0).item()}")
print(f"Kernel sample: {Y[0, 0, :2, :2]}")

# Manual computation for comparison
W_mix = torch.mm(pi, prototypes[:, w_offset:b_offset]).view(B, C_out, C_in, 3, 3)
b_mix = torch.mm(pi, prototypes[:, b_offset:b_offset+C_out])
print(f"\nW_mix stats: mean={W_mix.mean():.4f}, std={W_mix.std():.4f}")
print(f"b_mix stats: mean={b_mix.mean():.4f}")

import torch.nn.functional as F
unfold = F.unfold(X, kernel_size=3, padding=1)
Y_bmm = torch.bmm(W_mix.view(B*C_out, C_in*9), unfold).view(B, C_out, H, W) + b_mix.view(B, C_out, 1, 1)
print(f"BMM output:   shape={Y_bmm.shape}, mean={Y_bmm.mean():.6f}, std={Y_bmm.std():.6f}")

diff = (Y - Y_bmm).abs()
print(f"\nDiff: max={diff.max():.8f}, mean={diff.mean():.8f}")

# Even simpler: test with known weights
print("\n=== Minimal test: 1x1 conv ===")
X2 = torch.ones(1, 1, 3, 3, device=device)
proto2 = torch.zeros(1, 256, device=device)
# Set weight for output channel 0: all 1s for the 3x3 kernel
proto2[0, 0:9] = 1.0
pi2 = torch.ones(1, 1, device=device)
# For 1 channel, C_out=1, per_oc_weight=9, b_offset=9
Y2 = aml_ops.fused_dynamic_conv2d(X2, proto2, pi2, 0, 9)
print(f"1x1 test: input=ones, weight=ones, expected=output=9.0 per pixel")
print(f"Got: {Y2[0, 0]}")
print(f"Expected: all 9.0")
