/*
 * fused_pipeline_in.cu — InstanceNorm variant of the fused AML post-conv chain.
 *
 * Only change vs fused_pipeline.cu: normalization statistics are computed
 * PER (sample, channel) over the spatial extent (InstanceNorm semantics,
 * no affine) instead of over the whole C*H*W volume (GroupNorm-1 semantics).
 *
 * Grid is (B*C): one block owns exactly one (sample, channel) pair, so the
 * per-channel reduction is cheap and race-free. Scalar structure grads in the
 * backward counterpart accumulate across the C blocks via atomicAdd into
 * pre-zeroed buffers.
 */
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdio>

#define IN_THREADS 256

/* v2 block-mode mix (see kernels/fused_pipeline.cu for the contract):
 * purely elementwise, norm-agnostic (GN/IN run in the ATen driver). */
__global__ void in_fused_mix_conv_kernel(
    const float* __restrict__ hp,
    const float* __restrict__ n2,
    const float* __restrict__ h_prev,
    const float* __restrict__ dim_mask,
    const float* __restrict__ block_mode,
    const float* __restrict__ layer_gate,
    float* __restrict__ out,
    int B, int C, long spatial, int D, int is_layer0
) {
    long idx = (long)blockIdx.x * IN_THREADS + threadIdx.x;
    long total = (long)B * C * spatial;
    if (idx >= total) return;

    int b = (int)(idx / ((long)C * spatial));
    int c = (int)((idx / spatial) % C);

    float dm  = (c < D) ? dim_mask[b * D + c] : 0.0f;
    float g_p = block_mode[b * 2];
    float g_r = block_mode[b * 2 + 1];
    float l_g = layer_gate[b];

    float hpv = hp[idx];
    float he  = is_layer0 ? hpv : h_prev[idx] * dm;
    float hr  = fmaxf(he + n2[idx], 0.0f) * dm;
    float ha  = g_p * hpv + g_r * hr;
    out[idx]  = is_layer0 ? ha : l_g * ha + (1.0f - l_g) * he;
}

void in_fused_mix_conv_launcher(
    const float* hp, const float* n2, const float* h_prev,
    const float* dim_mask, const float* block_mode, const float* layer_gate,
    float* out, int B, int C, long spatial, int D, int is_layer0
) {
    long total = (long)B * C * spatial;
    int blocks = (int)((total + IN_THREADS - 1) / IN_THREADS);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    in_fused_mix_conv_kernel<<<blocks, IN_THREADS, 0, stream>>>(
        hp, n2, h_prev, dim_mask, block_mode, layer_gate,
        out, B, C, spatial, D, is_layer0
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "AML kernel launch error (in_fused_mix_conv): %s\n", cudaGetErrorString(err));
}

__global__ void in_fused_post_conv_kernel(
    float* __restrict__ h_conv,
    const float* __restrict__ h_prev,
    const float* __restrict__ dim_mask,
    const float* __restrict__ block_mode,
    const float* __restrict__ layer_gate,
    int B, int C, int spatial, int D, int is_layer0
) {
    int bc = blockIdx.x;
    int b = bc / C;
    int c = bc % C;
    if (b >= B) return;

    /* 64-bit: per-image channel offset (B*C*spatial can exceed INT_MAX). */
    long base = ((long)b * C + c) * spatial;
    float count = (float)spatial;
    float g_p = block_mode[b * 2];
    float g_r = block_mode[b * 2 + 1];
    float l_g = layer_gate[b];
    float dm  = (c < D) ? dim_mask[b * D + c] : 0.0f;

    /* ---- Per-channel statistics over the spatial extent (two-pass) ---- */
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < spatial; i += IN_THREADS)
        local_sum += h_conv[base + i];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
    __shared__ float s_red[8];
    __shared__ float s_mean, s_inv;
    if (threadIdx.x % 32 == 0)
        s_red[threadIdx.x / 32] = local_sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float a = (threadIdx.x < IN_THREADS / 32) ? s_red[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            a += __shfl_down_sync(0xffffffff, a, off);
        if (threadIdx.x == 0)
            s_mean = a / count;
    }
    __syncthreads();
    float mean = s_mean;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < spatial; i += IN_THREADS) {
        float d = h_conv[base + i] - mean;
        local_var += d * d;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_var += __shfl_down_sync(0xffffffff, local_var, off);
    if (threadIdx.x % 32 == 0)
        s_red[threadIdx.x / 32] = local_var;
    __syncthreads();
    if (threadIdx.x < 32) {
        float a = (threadIdx.x < IN_THREADS / 32) ? s_red[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            a += __shfl_down_sync(0xffffffff, a, off);
        if (threadIdx.x == 0)
            s_inv = rsqrtf(a / count + 1e-5f);
    }
    __syncthreads();
    float inv_std = s_inv;

    /* ---- Normalize + ReLU + dim_mask + mode/gate mixing (same math as GN1) ---- */
    for (int i = threadIdx.x; i < spatial; i += IN_THREADS) {
        long idx = base + i;
        float normed = (h_conv[idx] - mean) * inv_std;
        float hp = fmaxf(normed, 0.0f) * dm;
        float he = is_layer0 ? hp : h_prev[idx] * dm;
        float hr = fmaxf(he + normed, 0.0f) * dm;
        float ha = g_p * hp + g_r * hr;
        h_conv[idx] = is_layer0 ? ha : l_g * ha + (1.0f - l_g) * he;
    }
}

__global__ void in_maxpool2x2_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
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
    Y[idx] = fmaxf(fmaxf(X[base + w0], X[base + w0 + 1]),
                   fmaxf(X[base + W_ + w0], X[base + W_ + w0 + 1]));
}

void in_fused_post_conv_launcher(
    float* h_conv, const float* h_prev,
    const float* dim_mask, const float* block_mode, const float* layer_gate,
    int B, int C, int spatial, int D, int is_layer0
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    in_fused_post_conv_kernel<<<B * C, IN_THREADS, 0, stream>>>(
        h_conv, h_prev, dim_mask, block_mode, layer_gate,
        B, C, spatial, D, is_layer0
    );
}

void in_maxpool2x2_launcher(float* X, float* Y, int B, int C, int H, int W_) {
    int H2 = H / 2, W2 = W_ / 2;
    int total = B * C * H2 * W2;
    int blocks = (total + 255) / 256;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    in_maxpool2x2_kernel<<<blocks, 256, 0, stream>>>(X, Y, B, C, H, W_);
}
