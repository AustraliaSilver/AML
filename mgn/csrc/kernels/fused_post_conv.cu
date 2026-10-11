/*
 * DEPRECATED (v1 single-conv algebra):
 * Retained strictly for legacy build compatibility. The active v2 pipeline
 * uses fused_pipeline.cu (fused_mix_conv_kernel) and custom_dynamic_ops.cu.
 * DO NOT USE in active training or inference paths.
 *
 * fused_post_conv.cu
 */
#include "common.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define POST_BLOCK 256

__global__ void fused_post_conv_bf16(
    const __nv_bfloat16* __restrict__ h_conv,
    const __nv_bfloat16* __restrict__ h_prev,
    const float* __restrict__ dim_mask,   /* [B, C] */
    const float* __restrict__ gate,       /* [B] */
    const float* __restrict__ g_plain,    /* [B] */
    const float* __restrict__ g_resnet,   /* [B] */
    __nv_bfloat16* __restrict__ h_out,
    int B, int C, int H, int W_,
    int first_layer  /* 1 if first layer (no h_prev/gate) */
) {
    int total = B * C * H * W_;
    int spatial = H * W_;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        int b = idx / (C * spatial);
        int c = (idx / spatial) % C;
        int hw = idx % spatial;

        float hc = __bfloat162float(h_conv[idx]);

        /* Channel-wise normalization: mean-subtract then scale by 1/sqrt(var+eps) */
        /* For num_groups=1: mean = sum_all / (C*H*W), var = ... */
        /* Actually with num_groups=1 and no affine, it's just standardization per sample */
        /* But this is expensive for full spatial. Let's skip norm for now — it's just */
        /* a constant scale that can be folded into W. In practice the norm is applied */
        /* AFTER the conv but it's a simple per-channel operation. */
        /* Actually, F.group_norm(x, num_groups=1) with no affine = InstanceNorm(B,C,H,W) */
        /* This requires mean/var over spatial+channel dims. Too expensive for element-wise. */
        /* We'll handle norm separately or use a different approach. */

        /* For now, implement the post-conv ops assuming h_conv is already normalized */
        /* (we'll do norm as a separate pass if needed) */

        float dm = dim_mask[b * C + c];

        /* h_plain = relu(h_conv) * dim_mask */
        float h_plain_val = (hc > 0.0f ? hc : 0.0f) * dm;

        float h_prev_val = 0.0f;
        if (!first_layer)
            h_prev_val = __bfloat162float(h_prev[idx]);

        /* h_expand = (first_layer) ? h_plain : h_prev * dim_mask */
        float h_expand_val = first_layer ? h_plain_val : h_prev_val * dm;

        /* h_resnet = relu(h_expand + h_conv) * dim_mask */
        float h_resnet_val = (h_expand_val + hc > 0.0f ? h_expand_val + hc : 0.0f) * dm;

        /* h_active = g_plain * h_plain + g_resnet * h_resnet */
        float gp = g_plain[b];
        float gr = g_resnet[b];
        float h_active = gp * h_plain_val + gr * h_resnet_val;

        float result;
        if (first_layer) {
            result = h_active;
        } else {
            float g = gate[b];
            result = g * h_active + (1.0f - g) * h_expand_val;
        }

        h_out[idx] = __float2bfloat16(result);
    }
}

/* ================================================================
 * Host launcher
 * ================================================================ */
void fused_post_conv_launcher(
    const float* h_conv,       /* bf16 as float* */
    const float* h_prev,       /* bf16 as float* (may be NULL for first layer) */
    const float* dim_mask,     /* [B, C] f32 */
    const float* gate,         /* [B] f32 */
    const float* g_plain,      /* [B] f32 */
    const float* g_resnet,     /* [B] f32 */
    float* h_out,              /* bf16 as float* */
    int B, int C, int H, int W_,
    int first_layer
) {
    auto hc = reinterpret_cast<const __nv_bfloat16*>(h_conv);
    auto hp = first_layer ? nullptr : reinterpret_cast<const __nv_bfloat16*>(h_prev);
    auto ho = reinterpret_cast<__nv_bfloat16*>(h_out);

    int total = B * C * H * W_;
    int grid = (total + POST_BLOCK - 1) / POST_BLOCK;
    grid = min(grid, 65535);

    fused_post_conv_bf16<<<grid, POST_BLOCK>>>(
        hc, hp, dim_mask, gate, g_plain, g_resnet, ho,
        B, C, H, W_, first_layer);
}
