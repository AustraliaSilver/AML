/*
 * batched_hyper_linear.cu
 *
 * Batched matrix multiplication where each batch element has its own
 * weight matrix: C[b] = A[b] @ B[b] + bias[b]
 *
 * Uses tiled shared memory GEMM per sample.
 *
 * Grid:  (B, div_up(N, TILE_N), div_up(M, TILE_M))
 * Block: (TILE_N, TILE_M)
 */

#include "common.h"
#include "cuda_utils.cuh"

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

__global__ void batched_hyper_linear_kernel(
    const float* __restrict__ A,     /* [B, M, K] */
    const float* __restrict__ B,     /* [B, K, N] */
    const float* __restrict__ bias,  /* [B, N] */
    float* __restrict__ C,           /* [B, M, N] */
    const int batch,
    const int M,
    const int K,
    const int N
) {
    const int b = blockIdx.z;
    if (b >= batch) return;

    const int tile_m = blockIdx.y * TILE_M;
    const int tile_n = blockIdx.x * TILE_N;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int a_row = tile_m + ty;
    const int b_col = tile_n + tx;

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    float acc = 0.0f;

    const float* a_base = A + b * M * K;
    const float* b_base = B + b * K * N;

    for (int t = 0; t < K; t += TILE_K) {
        /* Load A tile */
        int a_col = t + tx;
        if (a_row < M && a_col < K)
            As[ty][tx] = a_base[a_row * K + a_col];
        else
            As[ty][tx] = 0.0f;

        /* Load B tile */
        int b_row = t + ty;
        if (b_row < K && b_col < N)
            Bs[ty][tx] = b_base[b_row * N + b_col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            acc += As[ty][kk] * Bs[kk][tx];
        }
        __syncthreads();
    }

    if (a_row < M && b_col < N) {
        float val = acc;
        if (bias != nullptr) {
            val += bias[b * N + b_col];
        }
        C[b * M * N + a_row * N + b_col] = val;
    }
}

/* ================================================================
 * Host launcher
 * ================================================================ */
void batched_hyper_linear_launcher(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    int batch, int M, int K, int N
) {
    dim3 block(TILE_N, TILE_M);
    dim3 grid(div_up(N, TILE_N), div_up(M, TILE_M), batch);

    batched_hyper_linear_kernel<<<grid, block>>>(
        A, B, bias, C, batch, M, K, N
    );
}
