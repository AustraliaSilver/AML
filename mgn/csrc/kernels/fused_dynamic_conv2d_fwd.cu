/*
 * fused_dynamic_conv2d_fwd.cu
 *
 * Tiled 3x3 conv with shared-memory X caching.
 * W is pre-computed (via torch.mm pi@proto) and passed in.
 * For each output channel, we iterate over C_in input channels:
 *   load a (TILE_H+2)x(TILE_W+2) X tile into shared memory per channel,
 *   compute 3x3 conv from shared memory.
 *
 * Grid:  (B, C_out, num_spatial_tiles)
 * Block: (TILE_H * TILE_W)
 *
 * Eliminates the im2col intermediate entirely.
 */
#include "common.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define TILE_H 16
#define TILE_W 16
#define FWD_BLOCK (TILE_H * TILE_W)

/*
 * W:       [B, C_out, C_in * 9]  float32 (pre-computed weight mix)
 * X:       [B, C_in, H, W]       bf16
 * bias:    [B, C_out]             float32
 * Y:       [B, C_out, H, W]      bf16
 */
__global__ void tiled_conv_fwd_kernel(
    const float* __restrict__ W,
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ Y,
    const int C_in, const int C_out,
    const int H, const int W_
) {
    const int b  = blockIdx.x;
    const int oc = blockIdx.y;
    const int tile_idx = blockIdx.z;

    const int tiles_w = (W_ + TILE_W - 1) / TILE_W;
    const int oh_base = (tile_idx / tiles_w) * TILE_H;
    const int ow_base = (tile_idx % tiles_w) * TILE_W;

    const int oh_local = threadIdx.x / TILE_W;
    const int ow_local = threadIdx.x % TILE_W;
    const int oh = oh_base + oh_local;
    const int ow = ow_base + ow_local;
    const int spatial = H * W_;
    const int per_oc_w = C_in * 9;

    extern __shared__ float smem[];
    float* W_shared = smem;                          /* [per_oc_w] */
    float* X_shared = smem + per_oc_w;               /* [18 * 18] */

    /* Load W for this (b, oc) — reused across all spatial tiles */
    for (int i = threadIdx.x; i < per_oc_w; i += FWD_BLOCK)
        W_shared[i] = W[b * C_out * per_oc_w + oc * per_oc_w + i];
    __syncthreads();

    float acc = 0.0f;

    for (int ic = 0; ic < C_in; ic++) {
        /* Cooperatively load X[b, ic, oh_base-1..oh_base+TILE_H, ow_base-1..ow_base+TILE_W] */
        const int tile_elements = (TILE_H + 2) * (TILE_W + 2); /* 324 */
        const int x_base = b * C_in * spatial + ic * spatial;

        for (int i = threadIdx.x; i < tile_elements; i += FWD_BLOCK) {
            int tih = i / (TILE_W + 2);
            int tiw = i % (TILE_W + 2);
            int rh = oh_base + tih - 1;
            int rw = ow_base + tiw - 1;
            if (rh >= 0 && rh < H && rw >= 0 && rw < W_)
                X_shared[tih * (TILE_W + 2) + tiw] =
                    __bfloat162float(X[x_base + rh * W_ + rw]);
            else
                X_shared[tih * (TILE_W + 2) + tiw] = 0.0f;
        }
        __syncthreads();

        if (oh < H && ow < W_) {
            const float* w_ic = W_shared + ic * 9;
            #pragma unroll
            for (int kh = 0; kh < 3; kh++)
                #pragma unroll
                for (int kw = 0; kw < 3; kw++)
                    acc += w_ic[kh * 3 + kw] *
                           X_shared[(oh_local + kh) * (TILE_W + 2) + (ow_local + kw)];
        }
        __syncthreads();
    }

    if (oh < H && ow < W_)
        Y[b * C_out * spatial + oc * spatial + oh * W_ + ow] =
            __float2bfloat16(acc + bias[b * C_out + oc]);
}

/* ================================================================ */

void fused_dynamic_conv2d_fwd_launcher(
    const float* X_f, const float* prototypes_f, const float* pi_f, float* Y_f,
    int w_offset, int b_offset,
    int B, int C_in, int C_out, int H, int W_, int K, int W_dim
) {
    /* This launcher does the FULL fused path: weight_mix + conv.
     * We keep it for backward compatibility but the new tiled_conv_fwd
     * is called separately from Python. */
    /* Not used in the new pipeline — see tiled_conv_fwd below. */
}

/* ================================================================
 * New entry point: tiled conv forward (W pre-computed in Python)
 * ================================================================ */
void tiled_conv_fwd_launcher(
    const float* W,      /* [B, C_out, C_in*9] float32 */
    const float* X,      /* [B, C_in, H, W] bf16 (as raw bytes) */
    const float* bias,   /* [B, C_out] float32 */
    float* Y,            /* [B, C_out, H, W] bf16 (as raw bytes) */
    int B, int C_in, int C_out, int H, int W_
) {
    int tiles_h = (H + TILE_H - 1) / TILE_H;
    int tiles_w = (W_ + TILE_W - 1) / TILE_W;
    int num_tiles = tiles_h * tiles_w;

    dim3 grid(B, C_out, num_tiles);
    dim3 block(FWD_BLOCK);

    int smem = (C_in * 9 + (TILE_H + 2) * (TILE_W + 2)) * sizeof(float);

    tiled_conv_fwd_kernel<<<grid, block, smem>>>(
        W,
        reinterpret_cast<const __nv_bfloat16*>(X),
        bias,
        reinterpret_cast<__nv_bfloat16*>(Y),
        C_in, C_out, H, W_
    );
}
