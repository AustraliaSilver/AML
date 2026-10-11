#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

void custom_dynamic_conv2d_bf16_launcher(
    const at::BFloat16* X,
    const at::BFloat16* W,
    const at::BFloat16* bias,
    float* Y,
    int B, int C_in, int C_out, int H, int W_spatial
);

void custom_dynamic_conv2d_f32_launcher(
    const float* X,
    const float* W,
    const float* bias,
    float* Y,
    int B, int C_in, int C_out, int H, int W_spatial
);

void custom_group_norm1_launcher(
    const float* X,
    float* Y,
    int B, int C, int H, int W_spatial,
    float eps
);

void custom_instance_norm_launcher(
    const float* X,
    float* Y,
    int B, int C, int H, int W_spatial,
    float eps
);

void custom_relu_dimmask_launcher(
    const float* X,
    const float* dim_mask,
    float* Y,
    int B, int C, int spatial, int D
);

/* Backward launchers */
void custom_dynamic_conv2d_bwd_launcher(
    const at::BFloat16* g_out,
    const at::BFloat16* x_in,
    const at::BFloat16* w_mix,
    float* d_in,
    float* d_w,
    float* d_b,
    int B, int C_in, int C_out, int H, int W_spatial
);

void custom_group_norm1_bwd_launcher(
    const float* g_out,
    const float* raw,
    const float* mean,
    const float* rstd,
    float* d_in,
    int B, int C, int H, int W_spatial
);

void custom_instance_norm_bwd_launcher(
    const float* g_out,
    const float* raw,
    const float* mean,
    const float* rstd,
    float* d_in,
    int B, int C, int H, int W_spatial
);
