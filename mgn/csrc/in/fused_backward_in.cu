/*
 * fused_backward_in.cu — InstanceNorm variant of the fused AML backward.
 *
 * Mirrors fused_backward.cu with per-(sample, channel) normalization stats:
 *   normed = IN_c(h_conv)            (stats over spatial, per channel)
 *   hp = relu(normed) * dm
 *   he = (layer0) ? hp : h_prev * dm
 *   hr = relu(he + normed) * dm
 *   ha = g_p * hp + g_r * hr
 *   out = (layer0) ? ha : l_g * ha + (1-l_g) * he
 *
 * Grid is (B*C): one block owns one (sample, channel). Per-sample scalar
 * grads (block_mode / layer_gate) are partial sums here and are combined
 * across channels with atomicAdd into PRE-ZEROED buffers. dim_mask grads
 * use atomicAdd as in the original kernel. dh_conv uses the standard
 * instance-norm backward:
 *   dx = rstd * (dy - mean(dy) - xhat * mean(dy * xhat))
 */
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define INB_THREADS 256

__global__ void in_fused_post_conv_bwd_kernel(
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
    int bc = blockIdx.x;
    int b = bc / C;
    int c = bc % C;
    if (b >= B) return;

    int base = (b * C + c) * spatial;
    float count = (float)spatial;

    float g_p = block_mode[b * 2];
    float g_r = block_mode[b * 2 + 1];
    float l_g = layer_gate[b];
    float dm  = (c < D) ? dim_mask[b * D + c] : 0.0f;

    __shared__ float s_red[24];
    __shared__ float s_mean, s_inv, s_mdn, s_mdnn, s_dgp, s_dgr, s_dlg;

    /* ---- Pass 1: per-channel mean/var (two-pass, numerically stable) ---- */
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < spatial; i += INB_THREADS)
        local_sum += h_conv[base + i];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
    if (threadIdx.x % 32 == 0)
        s_red[threadIdx.x / 32] = local_sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float a = (threadIdx.x < INB_THREADS / 32) ? s_red[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            a += __shfl_down_sync(0xffffffff, a, off);
        if (threadIdx.x == 0)
            s_mean = a / count;
    }
    __syncthreads();
    float mean = s_mean;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < spatial; i += INB_THREADS) {
        float d = h_conv[base + i] - mean;
        local_var += d * d;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_var += __shfl_down_sync(0xffffffff, local_var, off);
    if (threadIdx.x % 32 == 0)
        s_red[threadIdx.x / 32] = local_var;
    __syncthreads();
    if (threadIdx.x < 32) {
        float a = (threadIdx.x < INB_THREADS / 32) ? s_red[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            a += __shfl_down_sync(0xffffffff, a, off);
        if (threadIdx.x == 0)
            s_inv = rsqrtf(a / count + 1e-5f);
    }
    __syncthreads();
    float inv_std = s_inv;

    /* ---- Pass 2: dnormed stats + scalar-grad partials ---- */
    float ldn = 0.0f, ldnn = 0.0f;
    float l_dgp = 0.0f, l_dgr = 0.0f, l_dlg = 0.0f;

    for (int i = threadIdx.x; i < spatial; i += INB_THREADS) {
        int idx = base + i;
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

        float mhp = (normed > 0.0f) ? 1.0f : 0.0f;
        float mhr = (he + normed > 0.0f) ? 1.0f : 0.0f;

        float dn = mhp * dm * dhp + mhr * dm * dhr;
        dhe += mhr * dm * dhr;

        if (is_layer0) {
            dhp += dhe;
            dn += mhp * dm * dhe;
        }

        ldn  += dn;
        ldnn += dn * normed;

        float coeff = is_layer0 ? 1.0f : l_g;
        l_dgp += dout * hp * coeff;
        l_dgr += dout * hr * coeff;
        if (!is_layer0) l_dlg += dout * (ha - he);

        /* stash dhe for pass 3 via recompute (no extra smem needed) */
    }

    /* reduce (ldn, ldnn) */
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ldn  += __shfl_down_sync(0xffffffff, ldn,  off);
        ldnn += __shfl_down_sync(0xffffffff, ldnn, off);
    }
    if (threadIdx.x % 32 == 0) {
        s_red[threadIdx.x / 32]       = ldn;
        s_red[threadIdx.x / 32 + 8]   = ldnn;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float a  = (threadIdx.x < INB_THREADS / 32) ? s_red[threadIdx.x]     : 0.0f;
        float b2 = (threadIdx.x < INB_THREADS / 32) ? s_red[threadIdx.x + 8] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a  += __shfl_down_sync(0xffffffff, a,  off);
            b2 += __shfl_down_sync(0xffffffff, b2, off);
        }
        if (threadIdx.x == 0) {
            s_mdn  = a / count;
            s_mdnn = b2 / count;
        }
    }

    /* reduce scalars (dgp, dgr, dlg) */
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        l_dgp += __shfl_down_sync(0xffffffff, l_dgp, off);
        l_dgr += __shfl_down_sync(0xffffffff, l_dgr, off);
        l_dlg += __shfl_down_sync(0xffffffff, l_dlg, off);
    }
    if (threadIdx.x % 32 == 0) {
        s_red[threadIdx.x / 32]       = l_dgp;
        s_red[threadIdx.x / 32 + 8]   = l_dgr;
        s_red[threadIdx.x / 32 + 16]  = l_dlg;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float a  = (threadIdx.x < INB_THREADS / 32) ? s_red[threadIdx.x]      : 0.0f;
        float b2 = (threadIdx.x < INB_THREADS / 32) ? s_red[threadIdx.x + 8]  : 0.0f;
        float c3 = (threadIdx.x < INB_THREADS / 32) ? s_red[threadIdx.x + 16] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a  += __shfl_down_sync(0xffffffff, a,  off);
            b2 += __shfl_down_sync(0xffffffff, b2, off);
            c3 += __shfl_down_sync(0xffffffff, c3, off);
        }
        if (threadIdx.x == 0) {
            s_dgp = a; s_dgr = b2; s_dlg = c3;
            /* combine this channel's partial into the per-sample totals */
            atomicAdd(&grad_bm_out[b * 2],     s_dgp);
            atomicAdd(&grad_bm_out[b * 2 + 1], s_dgr);
            atomicAdd(&grad_lg_out[b],         s_dlg);
        }
    }
    __syncthreads();

    float mdn  = s_mdn;
    float mdnn = s_mdnn;

    /* ---- Pass 3: dh_conv / dh_prev / grad_dm ---- */
    for (int i = threadIdx.x; i < spatial; i += INB_THREADS) {
        int idx = base + i;
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

        float mhp = (normed > 0.0f) ? 1.0f : 0.0f;
        float mhr = (he + normed > 0.0f) ? 1.0f : 0.0f;

        float dn = mhp * dm * dhp + mhr * dm * dhr;
        dhe += mhr * dm * dhr;

        if (is_layer0) {
            dhp += dhe;
            dn += mhp * dm * dhe;
        }

        /* InstanceNorm backward (per-channel stats) */
        float dhc = inv_std * (dn - mdn - normed * mdnn);
        dh_conv_out[idx] = dhc;

        dh_prev_out[idx] = is_layer0 ? 0.0f : dm * dhe;

        /* grad_dim_mask for this (b, c) — same closed form as original */
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

/* maxpool backward — identical to the GN1 version (norm-independent) */
__global__ void in_maxpool2x2_bwd_kernel(
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

    dh_in[base + w0]           = (v00 == mx) ? dout : 0.0f;
    dh_in[base + w0 + 1]       = (v01 == mx) ? dout : 0.0f;
    dh_in[base + W_ + w0]      = (v10 == mx) ? dout : 0.0f;
    dh_in[base + W_ + w0 + 1]  = (v11 == mx) ? dout : 0.0f;
}

void in_fused_post_conv_bwd_launcher(
    const float* dh_out, const float* h_conv, const float* h_prev,
    const float* dim_mask, const float* block_mode, const float* layer_gate,
    float* dh_conv_out, float* dh_prev_out,
    float* grad_bm_out, float* grad_lg_out, float* grad_dm_out,
    int B, int C, int spatial, int D, int is_layer0
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    in_fused_post_conv_bwd_kernel<<<B * C, INB_THREADS, 0, stream>>>(
        dh_out, h_conv, h_prev, dim_mask, block_mode, layer_gate,
        dh_conv_out, dh_prev_out, grad_bm_out, grad_lg_out, grad_dm_out,
        B, C, spatial, D, is_layer0
    );
}

void in_maxpool2x2_bwd_launcher(
    const float* dh_out, const float* h_in, float* dh_in,
    int B, int C, int H, int W_
) {
    int H2 = H / 2, W2 = W_ / 2;
    int total = B * C * H2 * W2;
    int blocks = (total + 255) / 256;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    in_maxpool2x2_bwd_kernel<<<blocks, 256, 0, stream>>>(dh_out, h_in, dh_in, B, C, H, W_);
}
