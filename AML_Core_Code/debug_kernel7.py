"""Minimal possible test case for the CUDA kernel - FIXED dimensions."""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mgn.accel._C as _C

device = "cuda"
torch.manual_seed(0)

# Test 1: ABSOLUTE MINIMUM
B, C_in, C_out, H, W = 1, 1, 1, 1, 1
K = 1
per_oc = C_in * 9  # 9
w_offset = 0
b_offset = per_oc  # 9
W_dim = 32  # enough

X = torch.ones(1, 1, 1, 1, device=device)
prototypes = torch.zeros(K, W_dim, device=device)
prototypes[0, 4] = 3.0  # center weight
prototypes[0, 9] = 7.0  # bias
pi = torch.ones(1, 1, device=device)

print("=== Test 1: B=1, C_in=1, C_out=1, H=1, W=1 ===")
print(f"prototypes shape: {prototypes.shape}")
print(f"pi shape: {pi.shape}")
print(f"w_offset={w_offset}, b_offset={b_offset}")
Y = _C.fused_dynamic_conv2d_fwd(X, prototypes, pi, w_offset, b_offset)
torch.cuda.synchronize()
print(f"Expected: 10.0, Got: {Y.item():.6f}")

# Test 2: B=1, C_in=1, C_out=1, H=3, W=3
print("\n=== Test 2: H=3, W=3 ===")
X2 = torch.ones(1, 1, 3, 3, device=device)
prototypes2 = torch.zeros(K, W_dim, device=device)
prototypes2[0, 4] = 1.0
Y2 = _C.fused_dynamic_conv2d_fwd(X2, prototypes2, pi, w_offset, b_offset)
torch.cuda.synchronize()
print(f"Expected: all 1.0, Got:\n{Y2[0,0]}")

# Test 3: B=2, C_in=3, C_out=8, H=4, W=4 (matches working debug_kernel3 C_out=8)
print("\n=== Test 3: B=2, C_in=3, C_out=8, H=4, W=4 ===")
B3, C_in3, C_out3, H3, W3 = 2, 3, 8, 4, 4
K3 = 4
per_oc3 = C_in3 * 9  # 27
w_off3 = 0
b_off3 = C_out3 * per_oc3  # 216
W_dim3 = 256
X3 = torch.randn(B3, C_in3, H3, W3, device=device)
prototypes3 = torch.randn(K3, W_dim3, device=device)
pi3 = torch.ones(B3, K3, device=device) / K3
Y3 = _C.fused_dynamic_conv2d_fwd(X3, prototypes3, pi3, w_off3, b_off3)
torch.cuda.synchronize()

# Compare with bmm
import torch.nn.functional as F
W_mix = torch.mm(pi3, prototypes3[:, w_off3:b_off3]).view(B3, C_out3, C_in3 * 9)
b_mix = torch.mm(pi3, prototypes3[:, b_off3:b_off3 + C_out3])
unfold = F.unfold(X3, kernel_size=3, padding=1)
Y3_bmm = torch.bmm(W_mix, unfold).view(B3, C_out3, H3, W3) + b_mix.view(B3, C_out3, 1, 1)

diff3 = (Y3 - Y3_bmm).abs()
print(f"Kernel mean={Y3.mean():.6f}, BMM mean={Y3_bmm.mean():.6f}")
print(f"Max diff: {diff3.max():.6f}, Mean diff: {diff3.mean():.6f}")
print(f"Kernel[0,0,:2,:2]:\n{Y3[0,0,:2,:2]}")
print(f"BMM[0,0,:2,:2]:\n{Y3_bmm[0,0,:2,:2]}")

# Test 4: B=1, C_in=3, C_out=1, H=3, W=3
print("\n=== Test 4: B=1, C_in=3, C_out=1, H=3, W=3 ===")
X4 = torch.ones(1, 3, 3, 3, device=device)
K4 = 1
per_oc4 = 27
w_off4 = 0
b_off4 = 27
W_dim4 = 32
prototypes4 = torch.zeros(K4, W_dim4, device=device)
prototypes4[0, 4] = 1.0   # ic=0, kh=1, kw=1
prototypes4[0, 13] = 2.0  # ic=1, kh=1, kw=1
prototypes4[0, 22] = 3.0  # ic=2, kh=1, kw=1
prototypes4[0, 27] = 5.0  # bias
pi4 = torch.ones(1, 1, device=device)
Y4 = _C.fused_dynamic_conv2d_fwd(X4, prototypes4, pi4, w_off4, b_off4)
torch.cuda.synchronize()
# Expected: 1+2+3+5 = 11.0 per pixel
print(f"Expected: all 11.0, Got:\n{Y4[0,0]}")
