"""Debug: check CUDA errors + isolate the zero issue."""
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mgn.accel.functions as aml_ops

torch.manual_seed(42)
device = "cuda"

B, C_in, H, W = 2, 3, 4, 4
K = 4
C_out = 4
w_offset = 0
per_oc_weight = C_in * 9
b_offset = w_offset + C_out * per_oc_weight
W_dim = 256

X = torch.randn(B, C_in, H, W, device=device)
prototypes = torch.randn(K, W_dim, device=device)
pi = torch.ones(B, K, device=device) / K

torch.cuda.synchronize()
Y = aml_ops.fused_dynamic_conv2d(X, prototypes, pi, w_offset, b_offset)
torch.cuda.synchronize()
print(f"CUDA error check passed")
print(f"Y shape={Y.shape}, nonzero={torch.count_nonzero(Y).item()}/{Y.numel()}")
print(f"Y sample:\n{Y[0, 0]}")

# Now test with pi = zeros
print("\n=== pi=zeros test ===")
pi_zero = torch.zeros(B, K, device=device)
Y_zero = aml_ops.fused_dynamic_conv2d(X, prototypes, pi_zero, w_offset, b_offset)
print(f"Y_zero mean={Y_zero.mean():.6f}, nonzero={torch.count_nonzero(Y_zero).item()}/{Y_zero.numel()}")

# Compare: pi uniform vs pi zero -> the difference should show the weight mixing effect
print("\n=== Check weight mixing ===")
# Manual: w_local[i] = sum_k pi[k] * proto[k, w_offset + oc*per_oc_weight + i]
oc = 0
for i in range(min(5, per_oc_weight)):
    manual = sum(pi[0, k].item() * prototypes[k, w_offset + oc * per_oc_weight + i].item() for k in range(K))
    print(f"  weight[{oc},{i}] = {manual:.6f}")

# Now test C_out=128 with small spatial
print("\n=== C_out=128, small spatial ===")
C_out_big = 128
b_offset_big = w_offset + C_out_big * per_oc_weight
X2 = torch.randn(B, C_in, 4, 4, device=device)
prototypes2 = torch.randn(K, W_dim, device=device)
pi2 = torch.ones(B, K, device=device) / K

Y2 = aml_ops.fused_dynamic_conv2d(X2, prototypes2, pi2, w_offset, b_offset_big)
torch.cuda.synchronize()
print(f"Y2 shape={Y2.shape}, nonzero={torch.count_nonzero(Y2).item()}/{Y2.numel()}, mean={Y2.mean():.6f}")

# Check specific indices
print(f"Y2[0, 0, 0, 0] = {Y2[0, 0, 0, 0].item():.6f}")
print(f"Y2[0, 1, 0, 0] = {Y2[0, 1, 0, 0].item():.6f}")

# Compare with bmm
W_mix = torch.mm(pi2, prototypes2[:, w_offset:b_offset_big]).view(B, C_out_big, C_in * 9)
b_mix = torch.mm(pi2, prototypes2[:, b_offset_big:b_offset_big + C_out_big])
unfold = F.unfold(X2, kernel_size=3, padding=1)
Y_bmm = torch.bmm(W_mix, unfold).view(B, C_out_big, 4, 4) + b_mix.view(B, C_out_big, 1, 1)
print(f"Y_bmm[0, 0, 0, 0] = {Y_bmm[0, 0, 0, 0].item():.6f}")
print(f"Y_bmm[0, 1, 0, 0] = {Y_bmm[0, 1, 0, 0].item():.6f}")

# Check weight mixing for oc=0
W_manual = torch.mm(pi2, prototypes2[:, w_offset:w_offset + C_out_big * per_oc_weight]).view(B, C_out_big, C_in, 3, 3)
print(f"\nW_manual[0, 0, :, 0, 0] = {W_manual[0, 0, :, 0, 0]}")
print(f"W_manual[0, 1, :, 0, 0] = {W_manual[0, 1, :, 0, 0]}")
print(f"Are first two oc weights different? {not torch.allclose(W_manual[0, 0], W_manual[0, 1])}")
