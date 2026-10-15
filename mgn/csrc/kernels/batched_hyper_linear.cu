/*
 * batched_hyper_linear.cu
 *
 * Batched matrix multiplication where each batch element has its own
 * weight matrix: C[b] = A[b] @ B[b] + bias[b]
 *
 * Provides:
 * 1. Standard tiled FP32 GEMM (32x32x32)
 * 2. Hardware Tensor Core WMMA GEMM (FP16 & BF16) on Ampere SM86+ with
 *    shared-memory tile staging for safe arbitrary dimension handling.
 */

#include "common.h"
#include "cuda_utils.cuh"
#include <c10/cuda/CUDAStream.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

using namespace nvcuda;

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

/* ================================================================
 * 1. Standard FP32 Tiled Shared Memory GEMM
 * ================================================================ */
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

    const float* a_base = A + (size_t)b * M * K;
    const float* b_base = B + (size_t)b * K * N;

    for (int t = 0; t < K; t += TILE_K) {
        int a_col = t + tx;
        if (a_row < M && a_col < K)
            As[ty][tx] = a_base[(size_t)a_row * K + a_col];
        else
            As[ty][tx] = 0.0f;

        int b_row = t + ty;
        if (b_row < K && b_col < N)
            Bs[ty][tx] = b_base[(size_t)b_row * N + b_col];
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
        C[(size_t)b * M * N + (size_t)a_row * N + b_col] = val;
    }
}

/* ================================================================
 * 2. Hardware Tensor Core WMMA GEMM (FP16 & BF16)
 * ================================================================ */
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARPS_PER_BLOCK = 4;

__device__ __forceinline__ half make_zero(half) { return __float2half(0.0f); }
__device__ __forceinline__ __nv_bfloat16 make_zero(__nv_bfloat16) { return __float2bfloat16(0.0f); }

template <typename HalfT>
__global__ void batched_hyper_linear_wmma_kernel(
    const HalfT* __restrict__ A,     /* [B, M, K] */
    const HalfT* __restrict__ B,     /* [B, K, N] */
    const float* __restrict__ bias,  /* [B, N] */
    float* __restrict__ C,           /* [B, M, N] */
    const int batch,
    const int M,
    const int K,
    const int N
) {
    const int b = blockIdx.z;
    if (b >= batch) return;

    const int warp_id = threadIdx.y;
    const int lane = threadIdx.x;

    const int warp_m = (blockIdx.y * blockDim.y + warp_id) * WMMA_M;
    const int warp_n = blockIdx.x * WMMA_N;

    if (warp_m >= M || warp_n >= N) return;

    __shared__ HalfT s_a[WARPS_PER_BLOCK][WMMA_M * WMMA_K];
    __shared__ HalfT s_b[WARPS_PER_BLOCK][WMMA_K * WMMA_N];
    __shared__ float s_c[WARPS_PER_BLOCK][WMMA_M * WMMA_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, HalfT, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, HalfT, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const HalfT* a_base = A + (size_t)b * M * K;
    const HalfT* b_base = B + (size_t)b * K * N;

    for (int k = 0; k < K; k += WMMA_K) {
        /* Staged shared-memory load with boundary padding */
        for (int idx = lane; idx < WMMA_M * WMMA_K; idx += 32) {
            int r = idx / WMMA_K;
            int c = idx % WMMA_K;
            int gr = warp_m + r;
            int gc = k + c;
            s_a[warp_id][idx] = (gr < M && gc < K) ? a_base[(size_t)gr * K + gc] : make_zero(HalfT{});
        }

        for (int idx = lane; idx < WMMA_K * WMMA_N; idx += 32) {
            int r = idx / WMMA_N;
            int c = idx % WMMA_N;
            int gr = k + r;
            int gc = warp_n + c;
            s_b[warp_id][idx] = (gr < K && gc < N) ? b_base[(size_t)gr * N + gc] : make_zero(HalfT{});
        }
        __syncwarp();

        wmma::load_matrix_sync(a_frag, s_a[warp_id], WMMA_K);
        wmma::load_matrix_sync(b_frag, s_b[warp_id], WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncwarp();
    }

    wmma::store_matrix_sync(s_c[warp_id], c_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();

    float* c_base = C + (size_t)b * M * N;
    for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 32) {
        int r = idx / WMMA_N;
        int c = idx % WMMA_N;
        int gr = warp_m + r;
        int gc = warp_n + c;
        if (gr < M && gc < N) {
            float val = s_c[warp_id][idx];
            if (bias != nullptr) {
                val += bias[b * N + gc];
            }
            c_base[(size_t)gr * N + gc] = val;
        }
    }
}

/* ================================================================
 * Host Launchers
 * ================================================================ */
void batched_hyper_linear_launcher(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    int batch, int M, int K, int N
) {
    /* gridDim.z is capped at 65535. Every caller passes a small batch, but an
     * oversize one used to launch a kernel that silently covered only the first
     * 65535 batch elements. Fail loudly instead. */
    if (batch > 65535) {
        throw std::runtime_error(
            "batched_hyper_linear: batch exceeds the gridDim.z limit (65535)");
    }
    dim3 block(TILE_N, TILE_M);
    dim3 grid(div_up(N, TILE_N), div_up(M, TILE_M), batch);
    /* Launch on the CURRENT torch stream, not the legacy default stream. With
     * any other current stream the implicit serialisation the default stream
     * provides does not apply, so the kernel can race the surrounding ops (and
     * this is exactly what breaks under CUDA graph capture). */
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    batched_hyper_linear_kernel<<<grid, block, 0, stream>>>(
        A, B, bias, C, batch, M, K, N
    );
    AML_CUDA_CHECK(cudaGetLastError());
}

void batched_hyper_linear_wmma_fp16_launcher(
    const half* A,
    const half* B,
    const float* bias,
    float* C,
    int batch, int M, int K, int N
) {
    if (batch > 65535) {
        throw std::runtime_error(
            "batched_hyper_linear_wmma_fp16: batch exceeds the gridDim.z limit (65535)");
    }
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid(div_up(N, WMMA_N), div_up(div_up(M, WMMA_M), WARPS_PER_BLOCK), batch);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    batched_hyper_linear_wmma_kernel<half><<<grid, block, 0, stream>>>(
        A, B, bias, C, batch, M, K, N
    );
    AML_CUDA_CHECK(cudaGetLastError());
}

void batched_hyper_linear_wmma_bf16_launcher(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    const float* bias,
    float* C,
    int batch, int M, int K, int N
) {
    if (batch > 65535) {
        throw std::runtime_error(
            "batched_hyper_linear_wmma_bf16: batch exceeds the gridDim.z limit (65535)");
    }
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid(div_up(N, WMMA_N), div_up(div_up(M, WMMA_M), WARPS_PER_BLOCK), batch);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    batched_hyper_linear_wmma_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        A, B, bias, C, batch, M, K, N
    );
    AML_CUDA_CHECK(cudaGetLastError());
}
