/*
 * fused_norm_relu_gate.cu
 *
 * Fuses GroupNorm(1 group) + ReLU + dim_mask + residual gating
 * into a single elementwise pass.
 *
 * Supports two modes (selected by block_mode per sample):
 *   mode 0 (Plain):      h_out = relu(norm(x)) * dim_mask
 *   mode 1 (ResNet):     h_out = relu(h_expand + norm(x)) * dim_mask
 *
 * Then applies layer gate:  h = gate * h_out + (1-gate) * h_expand
 *
 * Grid:  (B,)
 * Block: (N_THREADS) over spatial channels
 */

#include "common.h"
#include "cuda_utils.cuh"

#define N_THREADS 256

__global__ void fused_norm_relu_gate_kernel(
    float* __restrict__ X,            /* [B, C, H, W] — conv output, overwritten in-place */
    const float* __restrict__ h_prev, /* [B, C, H, W] — previous hidden state (NULL if layer 0) */
    const float* __restrict__ dim_mask, /* [B, D] per-layer dim mask */
    const float* __restrict__ block_mode, /* [B, 3] — plain/resnet/vit gates */
    const float* __restrict__ layer_gate, /* [B] — inter-layer residual gate */
    const int B,
    const int C,
    const int H,
    const int W,
    const int D,         /* max_dim — for dim_mask indexing */
    const int is_layer0  /* 1 if first layer, no h_prev */
) {
    const int b = blockIdx.x;
    if (b >= B) return;

    const int spatial = H * W;
    const int total   = C * spatial;
    const int tid     = threadIdx.x;

    /* ---- GroupNorm(1 group) statistics: mean and variance over C per spatial pos ---- */
    /* For each spatial position, compute mean and var across C channels. */
    /* This is memory-bound, so we do two passes. */

    extern __shared__ float smem[];
    /* Use shared memory for partial sums if C is large, otherwise register-based */

    const float g_plain  = block_mode[b * 3 + 0];
    const float g_resnet = block_mode[b * 3 + 1];
    const float l_gate   = layer_gate[b];

    /* Each thread handles spatial positions */
    for (int s = tid; s < spatial; s += N_THREADS) {
        /* ---- Compute norm statistics for this spatial position ---- */
        float mean = 0.0f;
        float var  = 0.0f;
        for (int c = 0; c < C; c++) {
            float val = X[b * C * spatial + c * spatial + s];
            mean += val;
        }
        mean /= (float)C;
        for (int c = 0; c < C; c++) {
            float diff = X[b * C * spatial + c * spatial + s] - mean;
            var += diff * diff;
        }
        var /= (float)C;
        float inv_std = rsqrtf(var + 1e-5f);

        /* ---- Norm + ReLU + dim_mask for plain path ---- */
        float h_expand_val = 0.0f;
        float h_plain_val  = 0.0f;

        if (!is_layer0 && h_prev != nullptr) {
            h_expand_val = h_prev[b * C * spatial + /* c=0 */ s] * dim_mask[b * D + 0];
            /* Approximate: use first channel for expand — correct when dim_mask is channel-independent */
            h_expand_val = 0.0f;
            for (int c = 0; c < C && c < D; c++) {
                float hv = h_prev[b * C * spatial + c * spatial + s];
                float dm = dim_mask[b * D + c];
                h_expand_val += hv * dm;
            }
            h_expand_val /= (float)C;  /* average pooled expand */
        }

        for (int c = 0; c < C; c++) {
            int idx = b * C * spatial + c * spatial + s;
            float raw = X[idx];
            float normed = (raw - mean) * inv_std;
            float dm = (c < D) ? dim_mask[b * D + c] : 0.0f;

            /* Plain path */
            float plain = fmaxf(normed, 0.0f) * dm;

            /* ResNet path */
            float prev_val = (!is_layer0 && h_prev != nullptr)
                             ? h_prev[idx] * dm : 0.0f;
            float resnet = fmaxf(prev_val + normed, 0.0f) * dm;

            /* Combine paths */
            float h_active = g_plain * plain + g_resnet * resnet;

            /* Inter-layer gate */
            float h_out;
            if (is_layer0) {
                h_out = h_active;
            } else {
                h_out = l_gate * h_active + (1.0f - l_gate) * prev_val;
            }

            X[idx] = h_out;
        }
    }
}

/* ================================================================
 * Host launcher
 * ================================================================ */
void fused_norm_relu_gate_launcher(
    float* X,
    const float* h_prev,
    const float* dim_mask,
    const float* block_mode,
    const float* layer_gate,
    int B, int C, int H, int W, int D,
    int is_layer0
) {
    dim3 grid(B);
    dim3 block(N_THREADS);
    int smem_size = 0;

    fused_norm_relu_gate_kernel<<<grid, block, smem_size>>>(
        X, h_prev, dim_mask, block_mode, layer_gate,
        B, C, H, W, D, is_layer0
    );
}
