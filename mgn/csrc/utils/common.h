#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define AML_CUDA_CHECK(call)                                                   \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define AML_MAX_LAYERS 4
#define AML_MAX_DIM 128
#define AML_MAX_K 4
#define AML_KERNEL_SIZE 3

inline int div_up(int a, int b) { return (a + b - 1) / b; }
