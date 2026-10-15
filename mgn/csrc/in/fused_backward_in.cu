/*
 * fused_backward_in.cu — InstanceNorm variant of the fused AML backward.
 *
 * Mirrors fused_backward.cu with per-(sample, channel) normalization stats:
 *   normed = IN_c(h_conv)            (stats over spatial, per channel)
 *   hp = relu(normed) * dm
 *   he = (layer0) ? hp : h_prev * dm
 *   hr = relu(he + normed) * dm
 *   ha = g_p * hp + g_r * hr
 *   out = (layer0) ? ha : l_g * ha + (1-l_g) * he
 *
 * Grid is (B*C): one block owns one (sample, channel). Per-sample scalar
 * grads (block_mode / layer_gate) are partial sums here and are combined
 * across channels with atomicAdd into PRE-ZEROED buffers. dim_mask grads
 * use atomicAdd as in the original kernel. dh_conv uses the standard
 * instance-norm backward:
 *   dx = rstd * (dy - mean(dy) - xhat * mean(dy * xhat))
 */
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdio>

#define INB_THREADS 256

/* Retained: maxpool2x2 backward ONLY (see kernels/fused_backward.cu note).
 * The IN post-conv backward kernel/launcher were removed 2026-09-15. */

__global__ void in_maxpool2x2_bwd_kernel(
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
    /* 64-bit: per-image pixel offset, ~4x the flat `idx` bound (see
     * kernels/fused_backward.cu). */
    long base = (((long)b * C + c) * H + h0) * W_;

    float v00 = h_in[base + w0];
    float v01 = h_in[base + w0 + 1];
    float v10 = h_in[base + W_ + w0];
    float v11 = h_in[base + W_ + w0 + 1];

    float mx = fmaxf(fmaxf(v00, v01), fmaxf(v10, v11));
    float dout = dh_out[idx];

    dh_in[base + w0]           = (v00 == mx) ? dout : 0.0f;
    dh_in[base + w0 + 1]       = (v01 == mx) ? dout : 0.0f;
    dh_in[base + W_ + w0]      = (v10 == mx) ? dout : 0.0f;
    dh_in[base + W_ + w0 + 1]  = (v11 == mx) ? dout : 0.0f;
}

void in_maxpool2x2_bwd_launcher(
    const float* dh_out, const float* h_in, float* dh_in,
    int B, int C, int H, int W_
) {
    int H2 = H / 2, W2 = W_ / 2;
    int total = B * C * H2 * W2;
    int blocks = (total + 255) / 256;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    in_maxpool2x2_bwd_kernel<<<blocks, 256, 0, stream>>>(dh_out, h_in, dh_in, B, C, H, W_);
}
