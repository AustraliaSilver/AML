/*
 * fused_backward.cu — CUDA kernels for fused AML backward pipeline.
 *
 * Kernels:
 *   1. fused_post_conv_bwd_kernel  — backward through group_norm(1) + relu + dim_mask + resnet + gate
 *   2. maxpool2x2_bwd_kernel       — backward through 2x2 maxpool
 */
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdio>

#define FPB_THREADS 256

/* ================================================================
 * fused_post_conv_bwd_kernel
 *
 * Backward through the fused post-conv chain:
 *   normed = GN(h_conv, 1)
 *   hp = relu(normed) * dm
 *   he = (layer0) ? hp : h_prev * dm
 *   hr = relu(he + normed) * dm
 *   ha = g_p * hp + g_r * hr
 *   out = (layer0) ? ha : l_g * ha + (1-l_g) * he
 *
 * Also computes gradients w.r.t. dim_mask, block_mode, layer_gate.
 * ================================================================ */
/* Retained: maxpool2x2 backward ONLY. The v1 single-conv post-conv backward
 * and grouped_conv_bwd kernels were removed 2026-09-15: zero live callers,
 * retired v1 algebra (git history preserves them; see fused_mix_bwd_v2_* in
 * fused_pipeline.cu for the v2 backward). */

__global__ void maxpool2x2_bwd_kernel(
    const float* dh_out,
    const float* h_in,
    float* dh_in,
    int B, int C, int H, int W_
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H2 = H / 2, W2 = W_ / 2;
    int total = B * C * H2 * W2;
    if (idx >= total) return;

    int w2 = idx % W2;
    int tmp = idx / W2;
    int h2 = tmp % H2;
    int cb = tmp / H2;
    int b = cb / C;
    int c = cb % C;

    int h0 = h2 * 2, w0 = w2 * 2;
    int base = ((b * C + c) * H + h0) * W_;

    float v00 = h_in[base + w0];
    float v01 = h_in[base + w0 + 1];
    float v10 = h_in[base + W_ + w0];
    float v11 = h_in[base + W_ + w0 + 1];

    float mx = fmaxf(fmaxf(v00, v01), fmaxf(v10, v11));
    float dout = dh_out[idx];

    /* Non-overlapping windows → no race conditions */
    dh_in[base + w0]         = (v00 == mx) ? dout : 0.0f;
    dh_in[base + w0 + 1]     = (v01 == mx) ? dout : 0.0f;
    dh_in[base + W_ + w0]    = (v10 == mx) ? dout : 0.0f;
    dh_in[base + W_ + w0 + 1]= (v11 == mx) ? dout : 0.0f;
}

void maxpool2x2_bwd_launcher(
    const float* dh_out, const float* h_in, float* dh_in,
    int B, int C, int H, int W_
) {
    int H2 = H / 2, W2 = W_ / 2;
    int total = B * C * H2 * W2;
    int blocks = (total + 255) / 256;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    maxpool2x2_bwd_kernel<<<blocks, 256, 0, stream>>>(dh_out, h_in, dh_in, B, C, H, W_);
}
