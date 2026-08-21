"""Debug: test with real model dimensions."""
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mgn.accel.functions as aml_ops

torch.manual_seed(42)
device = "cuda"

B, C_in, C_out, H, W = 4, 3, 128, 32, 32
K = 20
w_offset = 0
per_oc_weight = C_in * 9  # 27
b_offset = w_offset + C_out * per_oc_weight  # 3456
W_dim = 861642

X = torch.randn(B, C_in, H, W, device=device)
prototypes = torch.randn(K, W_dim, device=device)
pi = torch.ones(B, K, device=device) / K

print(f"X: {X.shape}, proto: {prototypes.shape}, pi: {pi.shape}")
print(f"w_offset={w_offset}, b_offset={b_offset}, per_oc_weight={per_oc_weight}")
print(f"C_out from binding: {(b_offset - w_offset) / per_oc_weight}")

Y = aml_ops.fused_dynamic_conv2d(X, prototypes, pi, w_offset, b_offset)
print(f"\nKernel output: shape={Y.shape}, mean={Y.mean():.6f}, std={Y.std():.6f}")
print(f"all_zero={torch.all(Y == 0).item()}")
print(f"non_zero_count={torch.count_nonzero(Y).item()} / {Y.numel()}")

# Also test layer 1 dimensions
print("\n=== Layer 1 test (C_in=128) ===")
C_in_1 = 128
C_out_1 = 128
w_off_1 = b_offset  # = 3584
per_oc_1 = C_in_1 * 9  # 1152
b_off_1 = w_off_1 + C_out_1 * per_oc_1  # 3584 + 147456 = 151040
print(f"w_off={w_off_1}, b_off={b_off_1}, C_out={(b_off_1-w_off_1)/per_oc_1}")

# Can't easily test layer 1 without the full model, but let's verify offset calculation
print(f"\nLayer 0 total offset: {b_offset + C_out} = {3456 + 128} = 3584")
print(f"Layer 1 w_off should be 3584 + attn_skip")

# Test with the bmm path to compare
print("\n=== BMM comparison ===")
W_mix = torch.mm(pi, prototypes[:, w_offset:b_offset])  # [B, C_out*C_in*9]
b_mix = torch.mm(pi, prototypes[:, b_offset:b_offset+C_out])
print(f"W_mix: {W_mix.shape}, mean={W_mix.mean():.6f}")
print(f"b_mix: {b_mix.shape}, mean={b_mix.mean():.6f}")

unfold = F.unfold(X, kernel_size=3, padding=1)  # [B, C_in*9, H*W]
print(f"unfold: {unfold.shape}")
Y_bmm = torch.bmm(W_mix.view(B, C_out, C_in*9), unfold).view(B, C_out, H, W) + b_mix.view(B, C_out, 1, 1)
print(f"BMM output: shape={Y_bmm.shape}, mean={Y_bmm.mean():.6f}, std={Y_bmm.std():.6f}")

diff = (Y - Y_bmm).abs()
print(f"\nForward diff: max={diff.max():.8f}, mean={diff.mean():.8f}")
if diff.mean() < 1e-4:
    print("MATCH!")
else:
    print("MISMATCH!")
    # Check a single element
    b, oc = 0, 0
    print(f"\nKernel[{b},{oc},:,:]:\n{Y[b,oc]}")
    print(f"BMM[{b},{oc},:,:]:\n{Y_bmm[b,oc]}")
