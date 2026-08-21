"""Numerical comparison: custom CUDA kernel vs PyTorch bmm for dynamic conv2d."""
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mgn.accel.functions as aml_ops

torch.manual_seed(42)
device = "cuda"
DTYPE = torch.float32

B, C_in, C_out, H, W = 4, 3, 128, 8, 8
K_total = 20
w_offset = 0
per_oc_weight = C_in * 9  # 27
b_offset = w_offset + C_out * per_oc_weight  # 3456
W_dim = 861642

X = torch.randn(B, C_in, H, W, device=device, dtype=DTYPE, requires_grad=True)
prototypes = torch.randn(K_total, W_dim, device=device, dtype=DTYPE, requires_grad=True)
pi = torch.randn(B, K_total, device=device, dtype=DTYPE, requires_grad=True)
pi = torch.softmax(pi, dim=-1)

print("=== Forward comparison ===")
Y_kernel = aml_ops.fused_dynamic_conv2d(X, prototypes, pi, w_offset, b_offset)
print(f"Kernel output: mean={Y_kernel.mean():.6f}, std={Y_kernel.std():.6f}, all_zero={torch.all(Y_kernel == 0).item()}")

# BMM path (matching kernel: per-sample weight mixing + conv)
W_mix = torch.mm(pi, prototypes[:, w_offset:b_offset])  # [B, C_out * C_in * 9]
W_mix = W_mix.view(B, C_out, C_in * 9)
b_mix = torch.mm(pi, prototypes[:, b_offset:b_offset + C_out])  # [B, C_out]
unfold = F.unfold(X, kernel_size=3, padding=1)  # [B, C_in*9, H*W]
Y_bmm = torch.bmm(W_mix, unfold).view(B, C_out, H, W) + b_mix.view(B, C_out, 1, 1)
print(f"BMM output:    mean={Y_bmm.mean():.6f}, std={Y_bmm.std():.6f}")

diff = (Y_kernel - Y_bmm).abs()
print(f"Forward diff:  max={diff.max():.8f}, mean={diff.mean():.8f}, rel={diff.mean()/(Y_bmm.abs().mean()+1e-12):.8f}")

print("\n=== Backward comparison ===")
loss_k = Y_kernel.sum()
loss_k.backward()
gX_k = X.grad.clone()
gP_k = prototypes.grad.clone()
gpi_k = pi.grad.clone()
X.grad, prototypes.grad, pi.grad = None, None, None

loss_b = Y_bmm.sum()
loss_b.backward()
gX_b = X.grad.clone()
gP_b = prototypes.grad.clone()
gpi_b = pi.grad.clone()
X.grad, prototypes.grad, pi.grad = None, None, None

print(f"grad_X:  max_diff={(gX_k-gX_b).abs().max():.8f}, mean_diff={(gX_k-gX_b).abs().mean():.8f}")
print(f"grad_pi: max_diff={(gpi_k-gpi_b).abs().max():.8f}, mean_diff={(gpi_k-gpi_b).abs().mean():.8f}")

gP_k_slice = gP_k[:, w_offset:b_offset + C_out]
gP_b_slice = gP_b[:, w_offset:b_offset + C_out]
print(f"grad_proto[layer]: max_diff={(gP_k_slice-gP_b_slice).abs().max():.8f}, mean_diff={(gP_k_slice-gP_b_slice).abs().mean():.8f}")
print(f"grad_proto[outside] kernel: abs_mean={gP_k[:, :w_offset].abs().mean():.8f}, {gP_k[:, b_offset+C_out:].abs().mean():.8f}")
print(f"grad_proto[outside] bmm:    abs_mean={gP_b[:, :w_offset].abs().mean():.8f}, {gP_b[:, b_offset+C_out:].abs().mean():.8f}")

print(f"\ngrad_X:  kernel_mean={gX_k.abs().mean():.6f}, bmm_mean={gX_b.abs().mean():.6f}, ratio={gX_b.abs().mean()/(gX_k.abs().mean()+1e-12):.4f}")
print(f"grad_pi: kernel_mean={gpi_k.abs().mean():.6f}, bmm_mean={gpi_b.abs().mean():.6f}, ratio={gpi_b.abs().mean()/(gpi_k.abs().mean()+1e-12):.4f}")
print(f"grad_proto: kernel_mean={gP_k_slice.abs().mean():.6f}, bmm_mean={gP_b_slice.abs().mean():.6f}, ratio={gP_b_slice.abs().mean()/(gP_k_slice.abs().mean()+1e-12):.4f}")

print(f"\ngrad_Y from kernel: {(gX_k==0).all().item()}")
print(f"grad_Y from bmm: {(gX_b==0).all().item()}")
