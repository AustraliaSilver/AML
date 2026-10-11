#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
#include "custom_dynamic_ops.h"
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

// Helper to convert inputs to float
template <typename T>
__device__ __forceinline__ float to_float_val(T v);

template <>
__device__ __forceinline__ float to_float_val<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ float to_float_val<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

/* ====================================================================
 * 1. Direct Dynamic 3x3 Conv2d Kernel (per-sample weights and biases)
 *    X: [B, C_in, H, W]
 *    W: [B, C_out, C_in, 3, 3]
 *    bias: [B, C_out]
 *    Y: [B, C_out, H, W]
 * ==================================================================== */
template <typename InT>
__global__ void dynamic_conv2d_3x3_kernel(
    const InT* __restrict__ X,
    const InT* __restrict__ W,
    const InT* __restrict__ bias,
    float* __restrict__ Y,
    int B, int C_in, int C_out, int H, int W_spatial
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = H * W_spatial;
    int total = B * C_out * spatial;
    if (idx >= total) return;

    int ow = idx % W_spatial;
    int tmp = idx / W_spatial;
    int oh = tmp % H;
    int oc = (tmp / H) % C_out;
    int b  = tmp / (H * C_out);

    float val = bias ? to_float_val(bias[b * C_out + oc]) : 0.0f;

    const InT* w_ptr = W + (size_t)(b * C_out + oc) * (C_in * 9);
    const InT* x_b   = X + (size_t)b * (C_in * spatial);

    int ih0 = oh - 1;
    int ih1 = oh;
    int ih2 = oh + 1;

    int iw0 = ow - 1;
    int iw1 = ow;
    int iw2 = ow + 1;

    bool h0_ok = (ih0 >= 0);
    bool h2_ok = (ih2 < H);

    bool w0_ok = (iw0 >= 0);
    bool w2_ok = (iw2 < W_spatial);

    #pragma unroll 4
    for (int ic = 0; ic < C_in; ++ic) {
        const InT* x_ic = x_b + (size_t)ic * spatial;
        const InT* w_ic = w_ptr + ic * 9;

        // Row 0
        if (h0_ok) {
            int r0 = ih0 * W_spatial;
            if (w0_ok) val += to_float_val(w_ic[0]) * to_float_val(x_ic[r0 + iw0]);
            val += to_float_val(w_ic[1]) * to_float_val(x_ic[r0 + iw1]);
            if (w2_ok) val += to_float_val(w_ic[2]) * to_float_val(x_ic[r0 + iw2]);
        }

        // Row 1
        {
            int r1 = ih1 * W_spatial;
            if (w0_ok) val += to_float_val(w_ic[3]) * to_float_val(x_ic[r1 + iw0]);
            val += to_float_val(w_ic[4]) * to_float_val(x_ic[r1 + iw1]);
            if (w2_ok) val += to_float_val(w_ic[5]) * to_float_val(x_ic[r1 + iw2]);
        }

        // Row 2
        if (h2_ok) {
            int r2 = ih2 * W_spatial;
            if (w0_ok) val += to_float_val(w_ic[6]) * to_float_val(x_ic[r2 + iw0]);
            val += to_float_val(w_ic[7]) * to_float_val(x_ic[r2 + iw1]);
            if (w2_ok) val += to_float_val(w_ic[8]) * to_float_val(x_ic[r2 + iw2]);
        }
    }

    Y[idx] = val;
}

void custom_dynamic_conv2d_bf16_launcher(
    const at::BFloat16* X,
    const at::BFloat16* W,
    const at::BFloat16* bias,
    float* Y,
    int B, int C_in, int C_out, int H, int W_spatial
) {
    int total = B * C_out * H * W_spatial;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    dynamic_conv2d_3x3_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(X),
        reinterpret_cast<const __nv_bfloat16*>(W),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        Y, B, C_in, C_out, H, W_spatial
    );
}

void custom_dynamic_conv2d_f32_launcher(
    const float* X,
    const float* W,
    const float* bias,
    float* Y,
    int B, int C_in, int C_out, int H, int W_spatial
) {
    int total = B * C_out * H * W_spatial;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    dynamic_conv2d_3x3_kernel<float><<<blocks, threads, 0, stream>>>(
        X, W, bias, Y, B, C_in, C_out, H, W_spatial
    );
}

/* ====================================================================
 * 2. GroupNorm (num_groups=1) Kernel
 *    Normalizes across all C * H * W elements per sample b.
 * ==================================================================== */
__global__ void custom_group_norm1_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
    int B, int C, int H, int W_spatial,
    float eps
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int M = C * H * W_spatial;
    const float* x_b = X + (size_t)b * M;
    float* y_b = Y + (size_t)b * M;

    __shared__ float s_mean;
    __shared__ float s_rstd;
    __shared__ float s_warp_sum[16];

    // Pass 1: compute mean
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        thread_sum += x_b[i];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        s_warp_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < (blockDim.x / 32)) ? s_warp_sum[lane] : 0.0f;
        for (int offset = 8; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffff, val, offset);
        }
        if (lane == 0) {
            s_mean = val / (float)M;
        }
    }
    __syncthreads();
    float mean = s_mean;

    // Pass 2: compute variance
    float thread_var = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        float diff = x_b[i] - mean;
        thread_var += diff * diff;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        thread_var += __shfl_down_sync(0xffffffff, thread_var, offset);
    }
    if (lane == 0) {
        s_warp_sum[warp_id] = thread_var;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < (blockDim.x / 32)) ? s_warp_sum[lane] : 0.0f;
        for (int offset = 8; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffff, val, offset);
        }
        if (lane == 0) {
            float var = val / (float)M;
            s_rstd = rsqrtf(var + eps);
        }
    }
    __syncthreads();
    float rstd = s_rstd;

    // Pass 3: normalize and write
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        y_b[i] = (x_b[i] - mean) * rstd;
    }
}

void custom_group_norm1_launcher(
    const float* X,
    float* Y,
    int B, int C, int H, int W_spatial,
    float eps
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    custom_group_norm1_kernel<<<B, 512, 0, stream>>>(
        X, Y, B, C, H, W_spatial, eps
    );
}

/* ====================================================================
 * 3. InstanceNorm Kernel
 *    Normalizes across H * W elements per (b, c) instance.
 * ==================================================================== */
__global__ void custom_instance_norm_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
    int total_instances, int M,
    float eps
) {
    int inst = blockIdx.x;
    if (inst >= total_instances) return;

    const float* x_inst = X + (size_t)inst * M;
    float* y_inst = Y + (size_t)inst * M;

    __shared__ float s_mean;
    __shared__ float s_rstd;
    __shared__ float s_warp_sum[8];

    // Pass 1: compute mean
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        thread_sum += x_inst[i];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        s_warp_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < 8) ? s_warp_sum[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            s_mean = val / (float)M;
        }
    }
    __syncthreads();
    float mean = s_mean;

    // Pass 2: compute variance
    float thread_var = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        float diff = x_inst[i] - mean;
        thread_var += diff * diff;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        thread_var += __shfl_down_sync(0xffffffff, thread_var, offset);
    }
    if (lane == 0) {
        s_warp_sum[warp_id] = thread_var;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < 8) ? s_warp_sum[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            float var = val / (float)M;
            s_rstd = rsqrtf(var + eps);
        }
    }
    __syncthreads();
    float rstd = s_rstd;

    // Pass 3: normalize and write
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        y_inst[i] = (x_inst[i] - mean) * rstd;
    }
}

void custom_instance_norm_launcher(
    const float* X,
    float* Y,
    int B, int C, int H, int W_spatial,
    float eps
) {
    int total_instances = B * C;
    int M = H * W_spatial;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    custom_instance_norm_kernel<<<total_instances, 256, 0, stream>>>(
        X, Y, total_instances, M, eps
    );
}

/* ====================================================================
 * 4. Fused ReLU + DimMask Kernel
 *    Y = ReLU(X) * dim_mask
 * ==================================================================== */
__global__ void custom_relu_dimmask_kernel(
    const float* __restrict__ X,
    const float* __restrict__ dim_mask,
    float* __restrict__ Y,
    int B, int C, int spatial, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * spatial;
    if (idx >= total) return;

    int tmp = idx / spatial;
    int c = tmp % C;
    int b = tmp / C;

    float dm = (c < D) ? dim_mask[b * D + c] : 0.0f;
    float val = X[idx];
    Y[idx] = (val > 0.0f ? val : 0.0f) * dm;
}

void custom_relu_dimmask_launcher(
    const float* X,
    const float* dim_mask,
    float* Y,
    int B, int C, int spatial, int D
) {
    int total = B * C * spatial;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    custom_relu_dimmask_kernel<<<blocks, threads, 0, stream>>>(
        X, dim_mask, Y, B, C, spatial, D
    );
}
