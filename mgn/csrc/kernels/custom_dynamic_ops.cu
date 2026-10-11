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
 * 1. Direct Dynamic 3x3 Conv2d Forward Kernel
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

        if (h0_ok) {
            int r0 = ih0 * W_spatial;
            if (w0_ok) val += to_float_val(w_ic[0]) * to_float_val(x_ic[r0 + iw0]);
            val += to_float_val(w_ic[1]) * to_float_val(x_ic[r0 + iw1]);
            if (w2_ok) val += to_float_val(w_ic[2]) * to_float_val(x_ic[r0 + iw2]);
        }
        {
            int r1 = ih1 * W_spatial;
            if (w0_ok) val += to_float_val(w_ic[3]) * to_float_val(x_ic[r1 + iw0]);
            val += to_float_val(w_ic[4]) * to_float_val(x_ic[r1 + iw1]);
            if (w2_ok) val += to_float_val(w_ic[5]) * to_float_val(x_ic[r1 + iw2]);
        }
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
 * 2. GroupNorm (num_groups=1) Forward Kernel
 * ==================================================================== */
__global__ void custom_group_norm1_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
    int B, int C, int H, int W_spatial,
    float eps
) {
    int b = blockIdx.x;
    if (b >= B) return;

    size_t M = (size_t)C * H * W_spatial;
    const float* x_b = X + (size_t)b * M;
    float* y_b = Y + (size_t)b * M;

    __shared__ float s_mean;
    __shared__ float s_rstd;
    __shared__ float s_warp_sum[16];

    float thread_sum = 0.0f;
    for (size_t i = threadIdx.x; i < M; i += blockDim.x) {
        thread_sum += x_b[i];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    if (lane == 0 && warp_id < 16) {
        s_warp_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < num_warps) ? s_warp_sum[lane] : 0.0f;
        for (int offset = 8; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffff, val, offset);
        }
        if (lane == 0) {
            s_mean = val / (float)M;
        }
    }
    __syncthreads();
    float mean = s_mean;

    float thread_var = 0.0f;
    for (size_t i = threadIdx.x; i < M; i += blockDim.x) {
        float diff = x_b[i] - mean;
        thread_var += diff * diff;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        thread_var += __shfl_down_sync(0xffffffff, thread_var, offset);
    }
    if (lane == 0 && warp_id < 16) {
        s_warp_sum[warp_id] = thread_var;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < num_warps) ? s_warp_sum[lane] : 0.0f;
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

    for (size_t i = threadIdx.x; i < M; i += blockDim.x) {
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
 * 3. InstanceNorm Forward Kernel
 * ==================================================================== */
__global__ void custom_instance_norm_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
    int total_instances, size_t M,
    float eps
) {
    int inst = blockIdx.x;
    if (inst >= total_instances) return;

    const float* x_inst = X + (size_t)inst * M;
    float* y_inst = Y + (size_t)inst * M;

    __shared__ float s_mean;
    __shared__ float s_rstd;
    __shared__ float s_warp_sum[8];

    float thread_sum = 0.0f;
    for (size_t i = threadIdx.x; i < M; i += blockDim.x) {
        thread_sum += x_inst[i];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    if (lane == 0 && warp_id < 8) {
        s_warp_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < num_warps) ? s_warp_sum[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            s_mean = val / (float)M;
        }
    }
    __syncthreads();
    float mean = s_mean;

    float thread_var = 0.0f;
    for (size_t i = threadIdx.x; i < M; i += blockDim.x) {
        float diff = x_inst[i] - mean;
        thread_var += diff * diff;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        thread_var += __shfl_down_sync(0xffffffff, thread_var, offset);
    }
    if (lane == 0 && warp_id < 8) {
        s_warp_sum[warp_id] = thread_var;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < num_warps) ? s_warp_sum[lane] : 0.0f;
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

/* ====================================================================
 * 5. BACKWARD: Dynamic Conv2D 3x3 Backward Kernels
 *    a) conv_bwd_d_input_kernel: computes d_in [B, C_in, H, W]
 *       Transpose conv of g_out with weights W (flipped kh, kw)
 *    b) conv_bwd_d_weight_kernel: computes d_w [B, C_out, C_in, 3, 3]
 *    c) conv_bwd_d_bias_kernel: computes d_b [B, C_out]
 * ==================================================================== */
template <typename InT>
__global__ void dynamic_conv2d_bwd_d_input_kernel(
    const InT* __restrict__ g_out,
    const InT* __restrict__ W,
    float* __restrict__ d_in,
    int B, int C_in, int C_out, int H, int W_spatial
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = H * W_spatial;
    int total = B * C_in * spatial;
    if (idx >= total) return;

    int iw = idx % W_spatial;
    int tmp = idx / W_spatial;
    int ih = tmp % H;
    int ic = (tmp / H) % C_in;
    int b  = tmp / (H * C_in);

    float sum = 0.0f;
    const InT* gout_b = g_out + (size_t)b * (C_out * spatial);

    for (int oc = 0; oc < C_out; ++oc) {
        const InT* w_oc_ic = W + ((size_t)(b * C_out + oc) * C_in + ic) * 9;
        const InT* gout_oc = gout_b + (size_t)oc * spatial;

        for (int kh = 0; kh < 3; ++kh) {
            int oh = ih - (kh - 1);
            if (oh < 0 || oh >= H) continue;
            int oh_stride = oh * W_spatial;

            for (int kw = 0; kw < 3; ++kw) {
                int ow = iw - (kw - 1);
                if (ow < 0 || ow >= W_spatial) continue;

                float g_val = to_float_val(gout_oc[oh_stride + ow]);
                float w_val = to_float_val(w_oc_ic[kh * 3 + kw]);
                sum += g_val * w_val;
            }
        }
    }

    d_in[idx] = sum;
}

template <typename InT>
__global__ void dynamic_conv2d_bwd_d_weight_kernel(
    const InT* __restrict__ g_out,
    const InT* __restrict__ x_in,
    float* __restrict__ d_w,
    int B, int C_in, int C_out, int H, int W_spatial
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = B * C_out * C_in * 9;
    if (idx >= total_weights) return;

    int k = idx % 9;
    int kh = k / 3;
    int kw = k % 3;
    int tmp = idx / 9;
    int ic = tmp % C_in;
    int oc = (tmp / C_in) % C_out;
    int b  = tmp / (C_in * C_out);

    int spatial = H * W_spatial;
    const InT* gout_oc = g_out + ((size_t)b * C_out + oc) * spatial;
    const InT* xin_ic  = x_in  + ((size_t)b * C_in + ic) * spatial;

    float sum = 0.0f;
    for (int oh = 0; oh < H; ++oh) {
        int ih = oh + kh - 1;
        if (ih < 0 || ih >= H) continue;
        int oh_stride = oh * W_spatial;
        int ih_stride = ih * W_spatial;

        for (int ow = 0; ow < W_spatial; ++ow) {
            int iw = ow + kw - 1;
            if (iw >= 0 && iw < W_spatial) {
                float g = to_float_val(gout_oc[oh_stride + ow]);
                float x = to_float_val(xin_ic[ih_stride + iw]);
                sum += g * x;
            }
        }
    }

    d_w[idx] = sum;
}

template <typename InT>
__global__ void dynamic_conv2d_bwd_d_bias_kernel(
    const InT* __restrict__ g_out,
    float* __restrict__ d_b,
    int B, int C_out, int H, int W_spatial
) {
    int idx = blockIdx.x;
    int total = B * C_out;
    if (idx >= total) return;

    int spatial = H * W_spatial;
    const InT* gout_ptr = g_out + (size_t)idx * spatial;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
        sum += to_float_val(gout_ptr[i]);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float s_warp[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        s_warp[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < 8) ? s_warp[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            d_b[idx] = val;
        }
    }
}

void custom_dynamic_conv2d_bwd_launcher(
    const at::BFloat16* g_out,
    const at::BFloat16* x_in,
    const at::BFloat16* w_mix,
    float* d_in,
    float* d_w,
    float* d_b,
    int B, int C_in, int C_out, int H, int W_spatial
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // 1. d_in
    int total_in = B * C_in * H * W_spatial;
    int blocks_in = (total_in + 255) / 256;
    dynamic_conv2d_bwd_d_input_kernel<__nv_bfloat16><<<blocks_in, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(g_out),
        reinterpret_cast<const __nv_bfloat16*>(w_mix),
        d_in, B, C_in, C_out, H, W_spatial
    );

    // 2. d_w
    int total_w = B * C_out * C_in * 9;
    int blocks_w = (total_w + 255) / 256;
    dynamic_conv2d_bwd_d_weight_kernel<__nv_bfloat16><<<blocks_w, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(g_out),
        reinterpret_cast<const __nv_bfloat16*>(x_in),
        d_w, B, C_in, C_out, H, W_spatial
    );

    // 3. d_b
    dynamic_conv2d_bwd_d_bias_kernel<__nv_bfloat16><<<B * C_out, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(g_out),
        d_b, B, C_out, H, W_spatial
    );
}

/* ====================================================================
 * 6. BACKWARD: GroupNorm (1 group) Backward Kernel
 *    Formula: dX = (rstd / M) * ( M * dY - sum(dY) - x_hat * sum(dY * x_hat) )
 * ==================================================================== */
__global__ void custom_group_norm1_bwd_kernel(
    const float* __restrict__ g_out,
    const float* __restrict__ raw,
    const float* __restrict__ mean_in,
    const float* __restrict__ rstd_in,
    float* __restrict__ d_in,
    int B, int C, int H, int W_spatial
) {
    int b = blockIdx.x;
    if (b >= B) return;

    size_t M = (size_t)C * H * W_spatial;
    const float* g_b = g_out + (size_t)b * M;
    const float* x_b = raw   + (size_t)b * M;
    float* din_b     = d_in  + (size_t)b * M;

    float mean = mean_in[b];
    float rstd = rstd_in[b];

    __shared__ float s_sum_dy;
    __shared__ float s_sum_dy_xhat;
    __shared__ float s_warp_dy[16];
    __shared__ float s_warp_dy_xhat[16];

    float thread_sum_dy = 0.0f;
    float thread_sum_dy_xhat = 0.0f;

    for (size_t i = threadIdx.x; i < M; i += blockDim.x) {
        float dy = g_b[i];
        float xhat = (x_b[i] - mean) * rstd;
        thread_sum_dy += dy;
        thread_sum_dy_xhat += dy * xhat;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum_dy      += __shfl_down_sync(0xffffffff, thread_sum_dy, offset);
        thread_sum_dy_xhat += __shfl_down_sync(0xffffffff, thread_sum_dy_xhat, offset);
    }

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    if (lane == 0 && warp_id < 16) {
        s_warp_dy[warp_id] = thread_sum_dy;
        s_warp_dy_xhat[warp_id] = thread_sum_dy_xhat;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v_dy = (lane < num_warps) ? s_warp_dy[lane] : 0.0f;
        float v_dy_xhat = (lane < num_warps) ? s_warp_dy_xhat[lane] : 0.0f;
        for (int offset = 8; offset > 0; offset /= 2) {
            v_dy      += __shfl_down_sync(0xffff, v_dy, offset);
            v_dy_xhat += __shfl_down_sync(0xffff, v_dy_xhat, offset);
        }
        if (lane == 0) {
            s_sum_dy = v_dy;
            s_sum_dy_xhat = v_dy_xhat;
        }
    }
    __syncthreads();

    float sum_dy = s_sum_dy;
    float sum_dy_xhat = s_sum_dy_xhat;
    float scale = rstd / (float)M;

    for (size_t i = threadIdx.x; i < M; i += blockDim.x) {
        float dy = g_b[i];
        float xhat = (x_b[i] - mean) * rstd;
        din_b[i] = scale * ((float)M * dy - sum_dy - xhat * sum_dy_xhat);
    }
}

void custom_group_norm1_bwd_launcher(
    const float* g_out,
    const float* raw,
    const float* mean,
    const float* rstd,
    float* d_in,
    int B, int C, int H, int W_spatial
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    custom_group_norm1_bwd_kernel<<<B, 512, 0, stream>>>(
        g_out, raw, mean, rstd, d_in, B, C, H, W_spatial
    );
}

/* ====================================================================
 * 7. BACKWARD: InstanceNorm Backward Kernel
 *    Formula: dX = (rstd / M) * ( M * dY - sum(dY) - x_hat * sum(dY * x_hat) )
 *    Per (b, c) instance
 * ==================================================================== */
__global__ void custom_instance_norm_bwd_kernel(
    const float* __restrict__ g_out,
    const float* __restrict__ raw,
    const float* __restrict__ mean_in,
    const float* __restrict__ rstd_in,
    float* __restrict__ d_in,
    int total_instances, size_t M
) {
    int inst = blockIdx.x;
    if (inst >= total_instances) return;

    const float* g_inst = g_out + (size_t)inst * M;
    const float* x_inst = raw   + (size_t)inst * M;
    float* din_inst     = d_in  + (size_t)inst * M;

    float mean = mean_in[inst];
    float rstd = rstd_in[inst];

    __shared__ float s_sum_dy;
    __shared__ float s_sum_dy_xhat;
    __shared__ float s_warp_dy[8];
    __shared__ float s_warp_dy_xhat[8];

    float thread_sum_dy = 0.0f;
    float thread_sum_dy_xhat = 0.0f;

    for (size_t i = threadIdx.x; i < M; i += blockDim.x) {
        float dy = g_inst[i];
        float xhat = (x_inst[i] - mean) * rstd;
        thread_sum_dy += dy;
        thread_sum_dy_xhat += dy * xhat;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum_dy      += __shfl_down_sync(0xffffffff, thread_sum_dy, offset);
        thread_sum_dy_xhat += __shfl_down_sync(0xffffffff, thread_sum_dy_xhat, offset);
    }

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    if (lane == 0 && warp_id < 8) {
        s_warp_dy[warp_id] = thread_sum_dy;
        s_warp_dy_xhat[warp_id] = thread_sum_dy_xhat;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v_dy = (lane < num_warps) ? s_warp_dy[lane] : 0.0f;
        float v_dy_xhat = (lane < num_warps) ? s_warp_dy_xhat[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset /= 2) {
            v_dy      += __shfl_down_sync(0xffffffff, v_dy, offset);
            v_dy_xhat += __shfl_down_sync(0xffffffff, v_dy_xhat, offset);
        }
        if (lane == 0) {
            s_sum_dy = v_dy;
            s_sum_dy_xhat = v_dy_xhat;
        }
    }
    __syncthreads();

    float sum_dy = s_sum_dy;
    float sum_dy_xhat = s_sum_dy_xhat;
    float scale = rstd / (float)M;

    for (size_t i = threadIdx.x; i < M; i += blockDim.x) {
        float dy = g_inst[i];
        float xhat = (x_inst[i] - mean) * rstd;
        din_inst[i] = scale * ((float)M * dy - sum_dy - xhat * sum_dy_xhat);
    }
}

void custom_instance_norm_bwd_launcher(
    const float* g_out,
    const float* raw,
    const float* mean,
    const float* rstd,
    float* d_in,
    int B, int C, int H, int W_spatial
) {
    int total_instances = B * C;
    int M = H * W_spatial;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    custom_instance_norm_bwd_kernel<<<total_instances, 256, 0, stream>>>(
        g_out, raw, mean, rstd, d_in, total_instances, M
    );
}
