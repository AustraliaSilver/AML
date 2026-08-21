#pragma once

#include "common.h"

template <int BLOCK_SIZE>
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}
