"""Minimal kernel correctness test - no autograd Function, direct _C call."""
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mgn.accel._C as _C

torch.manual_seed(42)
device = "cuda"

# The simplest possible case
B, C_in, H, W = 1, 3, 3, 3
K = 2
C_out = 2
per_oc = C_in * 9  # 27
w_offset = 0
b_offset = C_out * per_oc  # 54
W_dim = 100  # small

X = torch.ones(B, C_in, H, W, device=device)
prototypes = torch.zeros(K, W_dim, device=device)
# Set weight for (oc=0, ic=0, kh=1, kw=1) = center of 3x3 kernel
# idx in proto = oc*per_oc + ic*9 + kh*3 + kw = 0*27 + 0*9 + 1*3 + 1 = 4
prototypes[0, 4] = 1.0
prototypes[1, 4] = 0.0
pi = torch.tensor([[0.5, 0.5]], device=device)

print(f"X ones: {X.mean():.1f}")
print(f"proto[0, 0:10] = {prototypes[0, :10]}")
print(f"proto[1, 0:10] = {prototypes[1, :10]}")
print(f"pi = {pi}")

# Expected: weight for (oc=0, ic=0, kh=1, kw=1) = 0.5*1.0 + 0.5*0.0 = 0.5
# conv(X_ones, center_weight=0.5) = 0.5 for all spatial positions
# Y should be [0.5, 0, 0, ...] for oc=0 (only center weight is non-zero)
Y = _C.fused_dynamic_conv2d_fwd(X, prototypes, pi, w_offset, b_offset)
torch.cuda.synchronize()
print(f"\nKernel output Y shape={Y.shape}")
print(f"Y[0, 0] (oc=0, should be 0.5):\n{Y[0, 0]}")
print(f"Y[0, 1] (oc=1, should be 0.0):\n{Y[0, 1]}")
print(f"all_zero={torch.all(Y==0).item()}")
print(f"mean={Y.mean():.6f}")

# Now test with large W_dim like the real model
print("\n=== Large W_dim test ===")
W_dim_big = 861642
prototypes_big = torch.zeros(K, W_dim_big, device=device)
prototypes_big[0, 4] = 1.0
Y2 = _C.fused_dynamic_conv2d_fwd(X, prototypes_big, pi, w_offset, b_offset)
torch.cuda.synchronize()
print(f"Y2[0, 0] (should be 0.5):\n{Y2[0, 0]}")
print(f"all_zero={torch.all(Y2==0).item()}")

# Test with C_out=128
print("\n=== C_out=128 test ===")
C_out_big = 128
b_off_big = C_out_big * per_oc  # 3456
prototypes_big2 = torch.zeros(K, W_dim_big, device=device)
prototypes_big2[0, 4] = 1.0
pi2 = torch.tensor([[0.5, 0.5]], device=device)
Y3 = _C.fused_dynamic_conv2d_fwd(X, prototypes_big2, pi2, w_offset, b_off_big)
torch.cuda.synchronize()
print(f"Y3 shape={Y3.shape}")
print(f"Y3[0, 0] (oc=0, should be 0.5):\n{Y3[0, 0]}")
print(f"all_zero={torch.all(Y3==0).item()}")
print(f"nonzero={torch.count_nonzero(Y3).item()}/{Y3.numel()}")
