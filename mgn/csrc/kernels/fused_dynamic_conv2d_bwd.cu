/*
 * fused_dynamic_conv2d_bwd.cu
 *
 * Backward for tiled 3x3 conv with pre-computed W.
 */
#include "common.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define BWD_NTHREADS 256
#define BTILE 16
#define BBLOCK (BTILE * BTILE)

__global__ void bwd_input_kernel(
    const __nv_bfloat16* __restrict__ grad_Y,
    const float* __restrict__ W,
    float* __restrict__ grad_X,
    const int C_in, const int C_out,
    const int H, const int W_
) {
    const int b  = blockIdx.x;
    const int ic = blockIdx.y;
    const int tile_idx = blockIdx.z;

    const int tiles_w = (W_ + BTILE - 1) / BTILE;
    const int ih_base = (tile_idx / tiles_w) * BTILE;
    const int iw_base = (tile_idx % tiles_w) * BTILE;

    const int ih_local = threadIdx.x / BTILE;
    const int iw_local = threadIdx.x % BTILE;
    const int ih = ih_base + ih_local;
    const int iw = iw_base + iw_local;
    const int spatial = H * W_;
    const int per_oc_w = C_in * 9;

    extern __shared__ float smem[];
    float* W_local = smem;
    float* gy_tile = smem + 9;

    float acc = 0.0f;

    for (int oc = 0; oc < C_out; oc++) {
        if (threadIdx.x < 9)
            W_local[threadIdx.x] = W[b * C_out * per_oc_w + oc * per_oc_w + ic * 9 + threadIdx.x];
        __syncthreads();

        const int tile_elements = (BTILE + 2) * (BTILE + 2);
        const int gy_base = b * C_out * spatial + oc * spatial;

        for (int i = threadIdx.x; i < tile_elements; i += BBLOCK) {
            int tih = i / (BTILE + 2);
            int tiw = i % (BTILE + 2);
            int rh = ih_base + tih - 1;
            int rw = iw_base + tiw - 1;
            if (rh >= 0 && rh < H && rw >= 0 && rw < W_)
                gy_tile[tih * (BTILE + 2) + tiw] =
                    __bfloat162float(grad_Y[gy_base + rh * W_ + rw]);
            else
                gy_tile[tih * (BTILE + 2) + tiw] = 0.0f;
        }
        __syncthreads();

        if (ih < H && iw < W_) {
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                int gy_r = ih_local + 2 - kh;
                #pragma unroll
                for (int kw = 0; kw < 3; kw++) {
                    int gy_c = iw_local + 2 - kw;
                    acc += gy_tile[gy_r * (BTILE + 2) + gy_c] * W_local[kh * 3 + kw];
                }
            }
        }
        __syncthreads();
    }

    if (ih < H && iw < W_)
        grad_X[b * C_in * spatial + ic * spatial + ih * W_ + iw] = acc;
}

/* ================================================================
 * Kernel 2: grad_W — each block (b, oc, ic) reduces over spatial
 * using shared memory (no atomicAdd).
 *
 * Grid: (B, C_out, C_in), Block: (BWD_NTHREADS)
 * Shared: 9 floats for reduction
 * ================================================================ */
__global__ void bwd_weight_kernel(
    const __nv_bfloat16* __restrict__ grad_Y,
    const __nv_bfloat16* __restrict__ X,
    float* __restrict__ grad_W,
    const int C_in, const int C_out,
    const int H, const int W_
) {
    const int b  = blockIdx.x;
    const int oc = blockIdx.y;
    const int ic = blockIdx.z;

    const int spatial = H * W_;
    const int per_oc_w = C_in * 9;
    const int w_base = b * C_out * per_oc_w + oc * per_oc_w + ic * 9;
    const int gy_offset = b * C_out * spatial + oc * spatial;
    const int x_offset = b * C_in * spatial + ic * spatial;

    extern __shared__ float smem[];
    float* partial = smem; /* [9] for 9 weight elements */

    /* Each thread accumulates partial sums for all 9 weight elements */
    float local_w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int idx = threadIdx.x; idx < spatial; idx += BWD_NTHREADS) {
        const int oh = idx / W_;
        const int ow = idx % W_;
        const float gy = __bfloat162float(grad_Y[gy_offset + oh * W_ + ow]);

        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            const int ih = oh - 1 + kh;
            if (ih < 0 || ih >= H) continue;
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                const int iw = ow - 1 + kw;
                if (iw < 0 || iw >= W_) continue;
                const float x_val = __bfloat162float(X[x_offset + ih * W_ + iw]);
                local_w[kh * 3 + kw] += gy * x_val;
            }
        }
    }

    /* Reduce across threads: store each thread's 9 values, then tree-reduce */
    /* Use smem[9 * BWD_NTHREADS] — but that's large. Instead, 9 sequential reductions. */
    for (int k = 0; k < 9; k++) {
        partial[threadIdx.x] = local_w[k];
        __syncthreads();

        /* Tree reduction */
        for (int stride = BWD_NTHREADS / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            grad_W[w_base + k] = partial[0];
        __syncthreads();
    }
}

/* ================================================================
 * Kernel 3: grad_bias
 * ================================================================ */
__global__ void bwd_bias_kernel(
    const __nv_bfloat16* __restrict__ grad_Y,
    float* __restrict__ grad_bias,
    const int C_out, const int H, const int W_
) {
    const int b  = blockIdx.x;
    const int oc = blockIdx.y;
    const int spatial = H * W_;
    const int gy_offset = b * C_out * spatial + oc * spatial;

    extern __shared__ float smem[];
    float* partial = smem;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < spatial; i += BWD_NTHREADS)
        sum += __bfloat162float(grad_Y[gy_offset + i]);
    partial[tid] = sum;
    __syncthreads();

    for (int stride = BWD_NTHREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        grad_bias[b * C_out + oc] = partial[0];
}

/* ================================================================
 * Host launcher
 * ================================================================ */
void tiled_conv_bwd_launcher(
    const float* grad_Y_f,
    const float* W,
    const float* X_f,
    float* grad_X,
    float* grad_W,
    float* grad_bias,
    int B, int C_in, int C_out, int H, int W_
) {
    auto gY = reinterpret_cast<const __nv_bfloat16*>(grad_Y_f);
    auto X_ = reinterpret_cast<const __nv_bfloat16*>(X_f);

    /* 1. grad_X */
    {
        int tiles_h = (H + BTILE - 1) / BTILE;
        int tiles_w = (W_ + BTILE - 1) / BTILE;
        int num_tiles = tiles_h * tiles_w;
        dim3 grid(B, C_in, num_tiles);
        dim3 block(BBLOCK);
        int smem = (9 + (BTILE + 2) * (BTILE + 2)) * sizeof(float);
        bwd_input_kernel<<<grid, block, smem>>>(
            gY, W, grad_X, C_in, C_out, H, W_);
    }

    /* 2. grad_W */
    {
        dim3 grid(B, C_out, C_in);
        dim3 block(BWD_NTHREADS);
        int smem = BWD_NTHREADS * sizeof(float);
        bwd_weight_kernel<<<grid, block, smem>>>(
            gY, X_, grad_W, C_in, C_out, H, W_);
    }

    /* 3. grad_bias */
    {
        dim3 grid(B, C_out);
        dim3 block(BWD_NTHREADS);
        int smem = BWD_NTHREADS * sizeof(float);
        bwd_bias_kernel<<<grid, block, smem>>>(
            gY, grad_bias, C_out, H, W_);
    }
}
