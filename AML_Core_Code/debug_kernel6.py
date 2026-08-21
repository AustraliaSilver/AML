"""Test: is the bug in W_dim indexing?"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mgn.accel._C as _C

torch.manual_seed(42)
device = "cuda"

B, C_in, H, W = 1, 3, 3, 3
K = 2
C_out = 2
per_oc = C_in * 9  # 27
w_offset = 0
b_offset = C_out * per_oc  # 54
X = torch.ones(B, C_in, H, W, device=device)
pi = torch.tensor([[0.5, 0.5]], device=device)

for W_dim in [54, 100, 1000, 10000, 100000, 500000, 861642]:
    prototypes = torch.zeros(K, W_dim, device=device)
    prototypes[0, 4] = 1.0
    Y = _C.fused_dynamic_conv2d_fwd(X, prototypes, pi, w_offset, b_offset)
    torch.cuda.synchronize()
    val = Y[0, 0, 0, 0].item()
    all_same = torch.allclose(Y[0, 0], torch.full_like(Y[0, 0], 0.5), atol=1e-5)
    print(f"W_dim={W_dim:>8d}: Y[0,0,0,0]={val:.6f}, all_0.5={all_same}, nonzero={torch.count_nonzero(Y).item()}/{Y.numel()}")

# Also test: does K indexing overflow with large W_dim?
print("\n=== Proto indexing test ===")
for W_dim in [54, 100, 861642]:
    prototypes = torch.zeros(K, W_dim, device=device)
    # Set proto[1, 4] = 1.0 instead of proto[0, 4]
    prototypes[1, 4] = 1.0
    pi_one_hot = torch.tensor([[0.0, 1.0]], device=device)
    Y = _C.fused_dynamic_conv2d_fwd(X, prototypes, pi_one_hot, w_offset, b_offset)
    torch.cuda.synchronize()
    print(f"W_dim={W_dim:>8d}: proto[1,4]=1.0, pi=[0,1], Y[0,0,0,0]={Y[0,0,0,0].item():.6f}")
    
    # Check: does the kernel correctly access prototypes[k * W_dim + idx]?
    # For k=1, idx=4: access index = 1 * W_dim + 4
    expected_idx = 1 * W_dim + 4
    print(f"  Expected access index: {expected_idx}, proto value at that index: {prototypes.flatten()[expected_idx].item():.6f}")
