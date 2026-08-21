/*
 * fused_backward.cu — CUDA kernels for fused AML backward pipeline.
 *
 * Kernels:
 *   1. fused_post_conv_bwd_kernel  — backward through group_norm(1) + relu + dim_mask + resnet + gate
 *   2. maxpool2x2_bwd_kernel       — backward through 2x2 maxpool
 */
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define FPB_THREADS 256

/* ================================================================
 * fused_post_conv_bwd_kernel
 *
 * Backward through the fused post-conv chain:
 *   normed = GN(h_conv, 1)
 *   hp = relu(normed) * dm
 *   he = (layer0) ? hp : h_prev * dm
 *   hr = relu(he + normed) * dm
 *   ha = g_p * hp + g_r * hr
 *   out = (layer0) ? ha : l_g * ha + (1-l_g) * he
 *
 * Also computes gradients w.r.t. dim_mask, block_mode, layer_gate.
 * ================================================================ */
__global__ void fused_post_conv_bwd_kernel(
    const float* dh_out,
    const float* h_conv,
    const float* h_prev,
    const float* dim_mask,
    const float* block_mode,
    const float* layer_gate,
    float* dh_conv_out,
    float* dh_prev_out,
    float* grad_bm_out,
    float* grad_lg_out,
    float* grad_dm_out,
    int B, int C, int spatial, int D, int is_layer0
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int base = b * C * spatial;
    int total = C * spatial;
    float count = (float)total;

    float g_p = block_mode[b * 2];
    float g_r = block_mode[b * 2 + 1];
    float l_g = layer_gate[b];

    __shared__ float s_partial[FPB_THREADS];
    __shared__ float smean, sinv_std, smean_dn, smean_dn_n;
    __shared__ float s_dgp, s_dgr, s_dlg;

    /* ---- Pass 1 & 2: mean and var (combined) ---- */
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < total; i += FPB_THREADS) {
        float val = h_conv[base + i];
        local_sum += val;
        local_sum_sq += val * val; // use float to avoid FP64 slowdown
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }
    
    // Block-level reduction
    if (threadIdx.x % 32 == 0) {
        s_partial[threadIdx.x / 32] = local_sum;
        s_partial[threadIdx.x / 32 + 32] = local_sum_sq;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        local_sum = (threadIdx.x < FPB_THREADS / 32) ? s_partial[threadIdx.x] : 0.0f;
        local_sum_sq = (threadIdx.x < FPB_THREADS / 32) ? s_partial[threadIdx.x + 32] : 0.0f;
        
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
            local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
        }
        
        if (threadIdx.x == 0) {
            smean = local_sum / count;
            float var = (local_sum_sq / count) - (smean * smean);
            sinv_std = rsqrtf(fmaxf(var, 0.0f) + 1e-5f);
        }
    }
    __syncthreads();
    
    float mean = smean;
    float inv_std = sinv_std;

    /* ---- Pass 3: dnormed + GN stats + scalar param grads ---- */
    float local_dn = 0.0f, local_dn_n = 0.0f;
    float local_dgp = 0.0f, local_dgr = 0.0f, local_dlg = 0.0f;

    for (int i = threadIdx.x; i < total; i += FPB_THREADS) {
        int idx = base + i;
        int c = i / spatial;
        float dm = (c < D) ? dim_mask[b * D + c] : 0.0f;
        float hc = h_conv[idx];
        float normed = (hc - mean) * inv_std;
        float dout = dh_out[idx];

        float hp = fmaxf(normed, 0.0f) * dm;
        float he = is_layer0 ? hp : h_prev[idx] * dm;
        float hr = fmaxf(he + normed, 0.0f) * dm;
        float ha = g_p * hp + g_r * hr;

        float dha = is_layer0 ? dout : l_g * dout;
        float dhe = is_layer0 ? 0.0f : (1.0f - l_g) * dout;
        float dhp = g_p * dha;
        float dhr = g_r * dha;

        float mask_hp = (normed > 0.0f) ? 1.0f : 0.0f;
        float mask_hr = (he + normed > 0.0f) ? 1.0f : 0.0f;

        float dnormed_val = mask_hp * dm * dhp;
        dnormed_val += mask_hr * dm * dhr;
        dhe += mask_hr * dm * dhr;

        if (is_layer0) {
            dhp += dhe;
            dnormed_val += mask_hp * dm * dhe;
        }

        local_dn += dnormed_val;
        local_dn_n += dnormed_val * normed;

        float coeff = is_layer0 ? 1.0f : l_g;
        local_dgp += dout * hp * coeff;
        local_dgr += dout * hr * coeff;
        if (!is_layer0) local_dlg += dout * (ha - he);
    }

    /* Reduce Pass 3 stats using warp shuffles */
    // Reduce local_dn and local_dn_n
    for (int offset = 16; offset > 0; offset /= 2) {
        local_dn += __shfl_down_sync(0xffffffff, local_dn, offset);
        local_dn_n += __shfl_down_sync(0xffffffff, local_dn_n, offset);
    }
    if (threadIdx.x % 32 == 0) {
        s_partial[threadIdx.x / 32] = local_dn;
        s_partial[threadIdx.x / 32 + 32] = local_dn_n;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        float s_dn = (threadIdx.x < FPB_THREADS / 32) ? s_partial[threadIdx.x] : 0.0f;
        float s_dn_n = (threadIdx.x < FPB_THREADS / 32) ? s_partial[threadIdx.x + 32] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            s_dn += __shfl_down_sync(0xffffffff, s_dn, offset);
            s_dn_n += __shfl_down_sync(0xffffffff, s_dn_n, offset);
        }
        if (threadIdx.x == 0) {
            smean_dn = s_dn / count;
            smean_dn_n = s_dn_n / count;
        }
    }

    /* Reduce scalar param grads using warp shuffles */
    for (int offset = 16; offset > 0; offset /= 2) {
        local_dgp += __shfl_down_sync(0xffffffff, local_dgp, offset);
        local_dgr += __shfl_down_sync(0xffffffff, local_dgr, offset);
        local_dlg += __shfl_down_sync(0xffffffff, local_dlg, offset);
    }
    if (threadIdx.x % 32 == 0) {
        s_partial[threadIdx.x / 32] = local_dgp;
        s_partial[threadIdx.x / 32 + 32] = local_dgr;
        s_partial[threadIdx.x / 32 + 64] = local_dlg;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        float s_gp = (threadIdx.x < FPB_THREADS / 32) ? s_partial[threadIdx.x] : 0.0f;
        float s_gr = (threadIdx.x < FPB_THREADS / 32) ? s_partial[threadIdx.x + 32] : 0.0f;
        float s_lg = (threadIdx.x < FPB_THREADS / 32) ? s_partial[threadIdx.x + 64] : 0.0f;
        
        for (int offset = 16; offset > 0; offset /= 2) {
            s_gp += __shfl_down_sync(0xffffffff, s_gp, offset);
            s_gr += __shfl_down_sync(0xffffffff, s_gr, offset);
            s_lg += __shfl_down_sync(0xffffffff, s_lg, offset);
        }
        
        if (threadIdx.x == 0) {
            s_dgp = s_gp;
            s_dgr = s_gr;
            s_dlg = s_lg;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        grad_bm_out[b * 2]     = s_dgp;
        grad_bm_out[b * 2 + 1] = s_dgr;
        grad_lg_out[b]         = s_dlg;
    }
    __syncthreads();

    float mean_dn   = smean_dn;
    float mean_dn_n = smean_dn_n;

    /* ---- Pass 4: write dh_conv + dh_prev + grad_dm ---- */
    for (int i = threadIdx.x; i < total; i += FPB_THREADS) {
        int idx = base + i;
        int c = i / spatial;
        float dm = (c < D) ? dim_mask[b * D + c] : 0.0f;
        float hc = h_conv[idx];
        float normed = (hc - mean) * inv_std;
        float dout = dh_out[idx];

        float hp = fmaxf(normed, 0.0f) * dm;
        float he = is_layer0 ? hp : h_prev[idx] * dm;
        float hr = fmaxf(he + normed, 0.0f) * dm;
        float ha = g_p * hp + g_r * hr;

        float dha = is_layer0 ? dout : l_g * dout;
        float dhe = is_layer0 ? 0.0f : (1.0f - l_g) * dout;
        float dhp = g_p * dha;
        float dhr = g_r * dha;

        float mask_hp = (normed > 0.0f) ? 1.0f : 0.0f;
        float mask_hr = (he + normed > 0.0f) ? 1.0f : 0.0f;

        float dnormed_val = mask_hp * dm * dhp;
        dnormed_val += mask_hr * dm * dhr;
        dhe += mask_hr * dm * dhr;

        if (is_layer0) {
            dhp += dhe;
            dnormed_val += mask_hp * dm * dhe;
        }

        /* GN backward */
        float dhc = inv_std * (dnormed_val - mean_dn - normed * mean_dn_n);
        dh_conv_out[idx] = dhc;

        /* dh_prev from expand/skip path */
        if (!is_layer0) {
            dh_prev_out[idx] = dm * dhe;
        } else {
            dh_prev_out[idx] = 0.0f;
        }

        /* grad_dim_mask */
        if (c < D) {
            float d_out_d_dm;
            if (is_layer0) {
                float u = hp + normed;
                float mask_u = (u > 0.0f) ? 1.0f : 0.0f;
                d_out_d_dm = g_p * fmaxf(normed, 0.0f)
                           + g_r * (fmaxf(u, 0.0f) + dm * mask_u * fmaxf(normed, 0.0f));
            } else {
                float hp_val = fmaxf(normed, 0.0f);
                float he_val = h_prev[idx];
                float u = he_val * dm + normed;
                float mask_u = (u > 0.0f) ? 1.0f : 0.0f;
                float d_dm_hp = hp_val;
                float d_dm_hr = fmaxf(u, 0.0f) + dm * mask_u * he_val;
                float d_dm_he = he_val;
                d_out_d_dm = l_g * (g_p * d_dm_hp + g_r * d_dm_hr) + (1.0f - l_g) * d_dm_he;
            }
            atomicAdd(&grad_dm_out[b * D + c], dout * d_out_d_dm);
        }
    }
}

/* ================================================================
 * maxpool2x2_bwd_kernel
 *
 * Backward through 2x2 maxpool with stride 2.
 * Each input position belongs to exactly one pooling window.
 * ================================================================ */
__global__ void maxpool2x2_bwd_kernel(
    const float* dh_out,
    const float* h_in,
    float* dh_in,
    int B, int C, int H, int W_
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H2 = H / 2, W2 = W_ / 2;
    int total = B * C * H2 * W2;
    if (idx >= total) return;

    int w2 = idx % W2;
    int tmp = idx / W2;
    int h2 = tmp % H2;
    int cb = tmp / H2;
    int b = cb / C;
    int c = cb % C;

    int h0 = h2 * 2, w0 = w2 * 2;
    int base = ((b * C + c) * H + h0) * W_;

    float v00 = h_in[base + w0];
    float v01 = h_in[base + w0 + 1];
    float v10 = h_in[base + W_ + w0];
    float v11 = h_in[base + W_ + w0 + 1];

    float mx = fmaxf(fmaxf(v00, v01), fmaxf(v10, v11));
    float dout = dh_out[idx];

    /* Non-overlapping windows → no race conditions */
    dh_in[base + w0]         = (v00 == mx) ? dout : 0.0f;
    dh_in[base + w0 + 1]     = (v01 == mx) ? dout : 0.0f;
    dh_in[base + W_ + w0]    = (v10 == mx) ? dout : 0.0f;
    dh_in[base + W_ + w0 + 1]= (v11 == mx) ? dout : 0.0f;
}

/* ================================================================
 * Launchers
 * ================================================================ */
void fused_post_conv_bwd_launcher(
    const float* dh_out, const float* h_conv, const float* h_prev,
    const float* dim_mask, const float* block_mode, const float* layer_gate,
    float* dh_conv_out, float* dh_prev_out,
    float* grad_bm_out, float* grad_lg_out, float* grad_dm_out,
    int B, int C, int spatial, int D, int is_layer0
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    fused_post_conv_bwd_kernel<<<B, FPB_THREADS, 0, stream>>>(
        dh_out, h_conv, h_prev, dim_mask, block_mode, layer_gate,
        dh_conv_out, dh_prev_out, grad_bm_out, grad_lg_out, grad_dm_out,
        B, C, spatial, D, is_layer0
    );
}

void maxpool2x2_bwd_launcher(
    const float* dh_out, const float* h_in, float* dh_in,
    int B, int C, int H, int W_
) {
    int H2 = H / 2, W2 = W_ / 2;
    int total = B * C * H2 * W2;
    int blocks = (total + 255) / 256;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    maxpool2x2_bwd_kernel<<<blocks, 256, 0, stream>>>(dh_out, h_in, dh_in, B, C, H, W_);
}

/* ================================================================
 * grouped_conv_bwd_dX_kernel
 *
 * Backward dX for grouped conv2d (groups=B).
 * Forward: Y[b, co, oh, ow] = sum_{ci, kh, kw} X_pad[b, ci, oh+kh, ow+kw] * W[b, co, ci, kh, kw]
 *   where X_pad has padding=1 on each side.
 *
 * dX[b, ci, h, w] = sum_{co, kh, kw valid} dY[b, co, h+1-kh, w+1-kw] * W[b, co, ci, kh, kw]
 *
 * Grid: B * C_in blocks, Block: 256 threads.
 * Each block handles one (b, ci) pair; threads tile over spatial positions.
 * ================================================================ */
__global__ void grouped_conv_bwd_dX_kernel(
    const float* __restrict__ dY,   /* [B, C_out, H, W] */
    const float* __restrict__ W,    /* [B, C_out, C_in, 3, 3] */
    float* __restrict__ dX,         /* [B, C_in, H, W] */
    int B, int C_in, int C_out, int H, int W_
) {
    int bc = blockIdx.x;
    int b = bc / C_in;
    int ci = bc % C_in;
    if (b >= B) return;

    int spatial = H * W_;

    for (int hw = threadIdx.x; hw < spatial; hw += blockDim.x) {
        int h = hw / W_;
        int w = hw % W_;
        float sum = 0.0f;

        for (int co = 0; co < C_out; co++) {
            int w_base = ((b * C_out + co) * C_in + ci) * 9;
            int dy_base = (b * C_out + co) * spatial;

            for (int kh = 0; kh < 3; kh++) {
                int oh = h + 1 - kh;
                if (oh < 0 || oh >= H) continue;
                for (int kw = 0; kw < 3; kw++) {
                    int ow = w + 1 - kw;
                    if (ow < 0 || ow >= W_) continue;
                    sum += dY[dy_base + oh * W_ + ow] * W[w_base + kh * 3 + kw];
                }
            }
        }
        dX[(b * C_in + ci) * spatial + hw] = sum;
    }
}

/* ================================================================
 * grouped_conv_bwd_dW_kernel
 *
 * dW[b, co, ci, kh, kw] = sum_{oh, ow} dY[b, co, oh, ow] * X[b, ci, oh+kh-1, ow+kw-1]
 *
 * Grid: B * C_out blocks, Block: 256 threads.
 * Each block handles one (b, co); threads tile over (ci, kh, kw) with spatial reduction.
 * ================================================================ */
__global__ void grouped_conv_bwd_dW_kernel(
    const float* __restrict__ dY,   /* [B, C_out, H, W] */
    const float* __restrict__ X,    /* [B, C_in, H, W] */
    float* __restrict__ dW,         /* [B, C_out, C_in, 9] */
    int B, int C_in, int C_out, int H, int W_
) {
    int bc = blockIdx.x;
    int b = bc / C_out;
    int co = bc % C_out;
    if (b >= B) return;

    int spatial = H * W_;
    int C9 = C_in * 9;

    for (int ckw = threadIdx.x; ckw < C9; ckw += blockDim.x) {
        int ci = ckw / 9;
        int kk = ckw % 9;
        int kh = kk / 3;
        int kw = kk % 3;

        float sum = 0.0f;
        int dy_base = (b * C_out + co) * spatial;
        int x_base = (b * C_in + ci) * spatial;

        for (int oh = 0; oh < H; oh++) {
            int ih = oh + kh - 1;
            if (ih < 0 || ih >= H) continue;
            for (int ow = 0; ow < W_; ow++) {
                int iw = ow + kw - 1;
                if (iw < 0 || iw >= W_) continue;
                sum += dY[dy_base + oh * W_ + ow] * X[x_base + ih * W_ + iw];
            }
        }
        dW[((b * C_out + co) * C_in + ci) * 9 + kk] = sum;
    }
}

/* ================================================================
 * grouped_conv_bwd_db_kernel
 *
 * db[b, co] = sum_{oh, ow} dY[b, co, oh, ow]
 *
 * Grid: B * C_out blocks, Block: 256 threads with reduction.
 * ================================================================ */
__global__ void grouped_conv_bwd_db_kernel(
    const float* __restrict__ dY,   /* [B, C_out, H, W] */
    float* __restrict__ db,         /* [B, C_out] */
    int B, int C_out, int H, int W_
) {
    int bc = blockIdx.x;
    int b = bc / C_out;
    int co = bc % C_out;
    if (b >= B) return;

    int spatial = H * W_;
    int base = (b * C_out + co) * spatial;

    __shared__ float s_part[256];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < spatial; i += blockDim.x)
        local_sum += dY[base + i];
    s_part[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_part[threadIdx.x] += s_part[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) db[b * C_out + co] = s_part[0];
}

/* ================================================================
 * grouped_conv_bwd_launcher
 *
 * Calls dX, dW, db kernels for grouped conv backward.
 * Input/Output layout: [B, C, H, W] for dY, X, dX; [B, C_out, C_in, 3, 3] for W, dW.
 * ================================================================ */
void grouped_conv_bwd_launcher(
    const float* dY,    /* [B, C_out, H, W] */
    const float* X,     /* [B, C_in, H, W] */
    const float* Wt,    /* [B, C_out, C_in, 3, 3] */
    float* dX,          /* [B, C_in, H, W] */
    float* dW,          /* [B, C_out, C_in, 9] */
    float* db,          /* [B, C_out] */
    int B, int C_in, int C_out, int H, int W_
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    /* dX: B * C_in blocks, 256 threads */
    grouped_conv_bwd_dX_kernel<<<B * C_in, 256, 0, stream>>>(
        dY, Wt, dX, B, C_in, C_out, H, W_
    );

    /* dW: B * C_out blocks, 256 threads */
    grouped_conv_bwd_dW_kernel<<<B * C_out, 256, 0, stream>>>(
        dY, X, dW, B, C_in, C_out, H, W_
    );

    /* db: B * C_out blocks, 256 threads */
    grouped_conv_bwd_db_kernel<<<B * C_out, 256, 0, stream>>>(
        dY, db, B, C_out, H, W_
    );
}
