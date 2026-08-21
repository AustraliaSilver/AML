import torch
import triton
import triton.language as tl

@triton.jit
def fused_weight_mix_kernel(
    pi_ptr, proto_ptr, out_w_ptr,
    B, K, weight_dim,
    stride_pi_b, stride_pi_k,
    stride_proto_k, stride_proto_w,
    stride_out_b, stride_out_w,
    BLOCK_W: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_w = tl.program_id(1)

    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_offs < weight_dim

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load pi: [B, K] -> shape [BLOCK_K]
        pi_vals = tl.load(pi_ptr + pid_b * stride_pi_b + k_offs * stride_pi_k, mask=k_mask, other=0.0)
        
        # Vectorized outer loop over K
        for ki in range(BLOCK_K):
            if (k + ki) < K:
                pi_scalar = tl.load(pi_ptr + pid_b * stride_pi_b + (k + ki) * stride_pi_k)
                proto_vals = tl.load(proto_ptr + (k + ki) * stride_proto_k + w_offs * stride_proto_w, mask=w_mask, other=0.0)
                acc += pi_scalar * proto_vals

    tl.store(out_w_ptr + pid_b * stride_out_b + w_offs * stride_out_w, acc, mask=w_mask)


def triton_weight_mix(pi: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """
    Computes W_mix = pi @ prototypes
    pi: [B, K]
    prototypes: [K, weight_dim]
    Returns: [B, weight_dim]
    """
    B, K = pi.shape
    K_p, weight_dim = prototypes.shape
    assert K == K_p, "K dimension mismatch"

    out = torch.empty((B, weight_dim), device=pi.device, dtype=pi.dtype)
    BLOCK_W = 256
    BLOCK_K = 16
    grid = (B, triton.cdiv(weight_dim, BLOCK_W))

    fused_weight_mix_kernel[grid](
        pi, prototypes, out,
        B, K, weight_dim,
        pi.stride(0), pi.stride(1),
        prototypes.stride(0), prototypes.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_W=BLOCK_W,
        BLOCK_K=BLOCK_K
    )
    return out


@triton.jit
def fused_post_conv_kernel(
    h_conv_ptr, h_expand_ptr, out_ptr,
    dim_mask_ptr, gate_ptr, bm_ptr,
    B, C, H, W,
    is_first_layer: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_sp = tl.program_id(2)

    HW = H * W
    sp_offs = pid_sp * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    sp_mask = sp_offs < HW

    # Pointers
    base_idx = pid_b * (C * HW) + pid_c * HW + sp_offs
    
    # Load h_conv
    h_c = tl.load(h_conv_ptr + base_idx, mask=sp_mask, other=0.0)
    
    # Load gating and mask
    d_mask = tl.load(dim_mask_ptr + pid_b * C + pid_c)
    gate = tl.load(gate_ptr + pid_b)
    
    # Block mode logits: [B, 4] (0: plain, 1: resnet, 2: vit, 3: mlp)
    g_plain = tl.load(bm_ptr + pid_b * 4 + 0)
    g_resnet = tl.load(bm_ptr + pid_b * 4 + 1)

    # ReLU + Mask
    h_relu = tl.maximum(h_c, 0.0)
    h_plain = h_relu * d_mask

    if is_first_layer:
        h_expand = h_plain
    else:
        h_expand = tl.load(h_expand_ptr + base_idx, mask=sp_mask, other=0.0) * d_mask

    h_resnet = tl.maximum(h_expand + h_c, 0.0) * d_mask
    h_active = g_plain * h_plain + g_resnet * h_resnet

    if is_first_layer:
        h_out = h_active
    else:
        h_out = gate * h_active + (1.0 - gate) * h_expand

    tl.store(out_ptr + base_idx, h_out, mask=sp_mask)


def triton_fused_post_conv(h_conv, h_expand, dim_mask, gate, bm, is_first_layer=False):
    B, C, H, W = h_conv.shape
    out = torch.empty_like(h_conv)
    HW = H * W
    BLOCK_SPATIAL = 256
    grid = (B, C, triton.cdiv(HW, BLOCK_SPATIAL))

    fused_post_conv_kernel[grid](
        h_conv, h_expand if h_expand is not None else h_conv, out,
        dim_mask, gate, bm,
        B, C, H, W,
        is_first_layer=is_first_layer,
        BLOCK_SPATIAL=BLOCK_SPATIAL
    )
    return out
