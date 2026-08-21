/*
 * fused_pipeline.cu - CUDA kernels for the fused AML forward pipeline.
 */
#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define FPC_THREADS 256

__global__ void fused_post_conv_kernel(
    float* __restrict__ h_conv,
    const float* __restrict__ h_prev,
    const float* __restrict__ dim_mask,
    const float* __restrict__ block_mode,
    const float* __restrict__ layer_gate,
    int B, int C, int spatial, int D, int is_layer0
) {
    int b = blockIdx.x;
    if (b >= B) return;

    float g_p = block_mode[b * 2];
    float g_r = block_mode[b * 2 + 1];
    float l_g = layer_gate[b];

    int base = b * C * spatial;
    int total = C * spatial;
    float count = (float)total;

    __shared__ float smean, svar;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < total; i += FPC_THREADS)
        local_sum += h_conv[base + i];

    __shared__ float s_partial[FPC_THREADS];
    s_partial[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = FPC_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_partial[threadIdx.x] += s_partial[threadIdx.x + stride];
        __syncthreads();
    }
    float mean = s_partial[0] / count;
    if (threadIdx.x == 0) smean = mean;
    __syncthreads();
    mean = smean;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < total; i += FPC_THREADS) {
        float d = h_conv[base + i] - mean;
        local_var += d * d;
    }
    s_partial[threadIdx.x] = local_var;
    __syncthreads();
    for (int stride = FPC_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_partial[threadIdx.x] += s_partial[threadIdx.x + stride];
        __syncthreads();
    }
    float var = s_partial[0] / count;
    float inv_std = rsqrtf(var + 1e-5f);
    if (threadIdx.x == 0) svar = inv_std;
    __syncthreads();
    inv_std = svar;

    for (int i = threadIdx.x; i < total; i += FPC_THREADS) {
        int idx = base + i;
        int c = i / spatial;
        float normed = (h_conv[idx] - mean) * inv_std;
        float dm = (c < D) ? dim_mask[b * D + c] : 0.0f;
        float hp = fmaxf(normed, 0.0f) * dm;
        float he = is_layer0 ? hp : h_prev[idx] * dm;
        float hr = fmaxf(he + normed, 0.0f) * dm;
        float ha = g_p * hp + g_r * hr;
        h_conv[idx] = is_layer0 ? ha : l_g * ha + (1.0f - l_g) * he;
    }
}

__global__ void maxpool2x2_kernel(
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

void fused_post_conv_launcher(
    float* h_conv, const float* h_prev,
    const float* dim_mask, const float* block_mode, const float* layer_gate,
    int B, int C, int spatial, int D, int is_layer0
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    fused_post_conv_kernel<<<B, FPC_THREADS, 0, stream>>>(
        h_conv, h_prev, dim_mask, block_mode, layer_gate,
        B, C, spatial, D, is_layer0
    );
}

void maxpool2x2_launcher(float* X, float* Y, int B, int C, int H, int W_) {
    int H2 = H / 2, W2 = W_ / 2;
    int total = B * C * H2 * W2;
    int blocks = (total + 255) / 256;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    maxpool2x2_kernel<<<blocks, 256, 0, stream>>>(X, Y, B, C, H, W_);
}
