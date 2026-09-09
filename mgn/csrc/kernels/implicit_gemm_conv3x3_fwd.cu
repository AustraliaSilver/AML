/*
 * implicit_gemm_conv3x3_fwd.cu
 *
 * Implicit GEMM forward for 3x3 conv with padding=1.
 * No im2col buffer needed — loads X elements on-the-fly with shared memory tiling.
 *
 * Y[b, oc, oh, ow] = sum_{ic, kh, kw} W[b, oc, ic*9+kh*3+kw] * X[b, ic, oh-1+kh, ow-1+kw]
 *
 * Tiling:
 *   TILE_OC = 16 (output channels per block)
 *   TILE_S  = 8  (spatial dimension per block, 2D: TILE_S × TILE_S)
 *   Block:  (TILE_OC × TILE_S × TILE_S) = 1024 threads
 *   Grid:   (B, ceil(C_out/TILE_OC), ceil(H/TILE_S) × ceil(W/TILE_S))
 *
 * Shared memory: X_tile[TILE_S+2][TILE_S+2] + W_tile[TILE_OC][9]
 *                = (10×10 + 16×9) × 4 = (100+144) × 4 = 976 bytes
 */
#include "common.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdio>

#define IG_TILE_OC 16
#define IG_TILE_S  8
#define IG_THREADS (IG_TILE_OC * IG_TILE_S * IG_TILE_S)  /* 1024 */

__global__ void implicit_gemm_conv3x3_fwd_kernel(
    const float* __restrict__ W,     /* [B, C_out, C_in*9] f32 */
    const __nv_bfloat16* __restrict__ X, /* [B, C_in, H, W] bf16 */
    const float* __restrict__ bias,  /* [B, C_out] f32 */
    __nv_bfloat16* __restrict__ Y,   /* [B, C_out, H, W] bf16 */
    int C_in, int C_out, int H, int W_
) {
    const int b  = blockIdx.x;
    const int oc_block = blockIdx.y;  /* which tile of output channels */
    const int sp_block = blockIdx.z;  /* which tile of spatial positions */

    const int tiles_w = (W_ + IG_TILE_S - 1) / IG_TILE_S;
    const int oh_base = (sp_block / tiles_w) * IG_TILE_S;
    const int ow_base = (sp_block % tiles_w) * IG_TILE_S;
    const int oc_base = oc_block * IG_TILE_OC;

    if (oc_base >= C_out) return;

    const int tid = threadIdx.x;
    const int oc_local = tid / (IG_TILE_S * IG_TILE_S);
    const int sp_idx   = tid % (IG_TILE_S * IG_TILE_S);
    const int oh_local = sp_idx / IG_TILE_S;
    const int ow_local = sp_idx % IG_TILE_S;

    extern __shared__ float smem[];
    float* X_tile = smem;                   /* [IG_TILE_S+2][IG_TILE_S+2] = [10][10] */
    float* W_tile = smem + (IG_TILE_S+2)*(IG_TILE_S+2);  /* [IG_TILE_OC][9] = [16][9] */

    float acc = 0.0f;

    for (int ic = 0; ic < C_in; ic++) {
        /* ---- Load X tile into shared memory (with padding) ---- */
        const int tile_elems = (IG_TILE_S + 2) * (IG_TILE_S + 2);  /* 100 */
        for (int i = tid; i < tile_elems; i += IG_THREADS) {
            int ti = i / (IG_TILE_S + 2);
            int tj = i % (IG_TILE_S + 2);
            int ih = oh_base + ti - 1;
            int iw = ow_base + tj - 1;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W_)
                X_tile[ti * (IG_TILE_S + 2) + tj] =
                    __bfloat162float(X[b * C_in * H * W_ + ic * H * W_ + ih * W_ + iw]);
            else
                X_tile[ti * (IG_TILE_S + 2) + tj] = 0.0f;
        }

        /* ---- Load W tile into shared memory ---- */
        const int w_elems = IG_TILE_OC * 9;  /* 144 */
        for (int i = tid; i < w_elems; i += IG_THREADS) {
            int oc = i / 9;
            int k  = i % 9;
            if (oc_base + oc < C_out)
                W_tile[i] = W[b * C_out * C_in * 9 + (oc_base + oc) * C_in * 9 + ic * 9 + k];
            else
                W_tile[i] = 0.0f;
        }

        __syncthreads();

        /* ---- Compute: 3x3 conv from smem ---- */
        if (oc_local < IG_TILE_OC && (oc_base + oc_local) < C_out &&
            oh_local < IG_TILE_S && (oh_base + oh_local) < H &&
            ow_local < IG_TILE_S && (ow_base + ow_local) < W_) {

            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                int xr = oh_local + kh;  /* X_tile[ti] = X[oh_base + ti - 1], so ti = oh_local + kh */
                #pragma unroll
                for (int kw = 0; kw < 3; kw++) {
                    int xc = ow_local + kw;
                    acc += W_tile[oc_local * 9 + kh * 3 + kw] *
                           X_tile[xr * (IG_TILE_S + 2) + xc];
                }
            }
        }

        __syncthreads();
    }

    /* ---- Write output ---- */
    if (oc_local < IG_TILE_OC && (oc_base + oc_local) < C_out &&
        oh_local < IG_TILE_S && (oh_base + oh_local) < H &&
        ow_local < IG_TILE_S && (ow_base + ow_local) < W_) {

        int oc = oc_base + oc_local;
        int oh = oh_base + oh_local;
        int ow = ow_base + ow_local;
        int out_idx = b * C_out * H * W_ + oc * H * W_ + oh * W_ + ow;

        /* Add bias */
        acc += bias[b * C_out + oc];

        Y[out_idx] = __float2bfloat16(acc);
    }
}

/* ================================================================
 * Host launcher
 * ================================================================ */
void implicit_gemm_fwd_launcher(
    const float* W,
    const void* X_bf16,
    const float* bias,
    void* Y_bf16,
    int B, int C_in, int C_out, int H, int W_
) {
    int tiles_h = (H + IG_TILE_S - 1) / IG_TILE_S;
    int tiles_w = (W_ + IG_TILE_S - 1) / IG_TILE_S;
    int num_spatial_tiles = tiles_h * tiles_w;
    int oc_tiles = (C_out + IG_TILE_OC - 1) / IG_TILE_OC;

    dim3 grid(B, oc_tiles, num_spatial_tiles);
    dim3 block(IG_THREADS);

    int smem = ((IG_TILE_S+2)*(IG_TILE_S+2) + IG_TILE_OC*9) * sizeof(float);

    /* Current-torch-stream launch + error surface (see tiled_conv_fwd). */
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    implicit_gemm_conv3x3_fwd_kernel<<<grid, block, smem, stream>>>(
        W, (const __nv_bfloat16*)X_bf16, bias, (__nv_bfloat16*)Y_bf16, C_in, C_out, H, W_);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "AML kernel launch error (implicit_gemm_fwd): %s\n", cudaGetErrorString(err));
}
