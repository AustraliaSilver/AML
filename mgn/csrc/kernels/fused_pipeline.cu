/*
 * fused_pipeline.cu - CUDA kernels for the fused AML forward pipeline.
 *
 * v2 (two-convolution ResNet block): the normalization and both convolutions
 * are executed by ATen inside the C++ driver (bindings.cpp); this file owns
 * only the final per-layer MIX step (fused_mix_conv_kernel), plus maxpool.
 * The v1 single-conv post kernels below remain compiled/exported for the
 * legacy bindings but are no longer used by the training pipeline.
 */
#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdio>

#define FPC_THREADS 256

/* v2: block-mode mix of an already-normed plain path and second conv.
 *   hp  = ReLU(GN(hc1)) * dm           (pre-normed in driver; plain output
 *                                       AND input of the residual conv2)
 *   n2  = GN(hc2)                      (normed residual-branch conv output)
 *   he  = layer0 ? hp : h_prev * dm
 *   hr  = ReLU(he + n2) * dm
 *   ha  = g_p * hp + g_r * hr
 *   out = layer0 ? ha : lg * ha + (1 - lg) * he
 * Purely elementwise: no reductions, no shared memory. */
__global__ void fused_mix_conv_kernel(
    const float* __restrict__ hp,
    const float* __restrict__ n2,
    const float* __restrict__ h_prev,
    const float* __restrict__ dim_mask,
    const float* __restrict__ block_mode,
    const float* __restrict__ layer_gate,
    float* __restrict__ out,
    int B, int C, long spatial, int D, int is_layer0
) {
    long idx = (long)blockIdx.x * FPC_THREADS + threadIdx.x;
    long total = (long)B * C * spatial;
    if (idx >= total) return;

    int b = (int)(idx / ((long)C * spatial));
    int c = (int)((idx / spatial) % C);

    float dm  = (c < D) ? dim_mask[b * D + c] : 0.0f;
    float g_p = block_mode[b * 2];
    float g_r = block_mode[b * 2 + 1];
    float l_g = layer_gate[b];

    float hpv = hp[idx];
    float he  = is_layer0 ? hpv : h_prev[idx] * dm;
    float hr  = fmaxf(he + n2[idx], 0.0f) * dm;
    float ha  = g_p * hpv + g_r * hr;
    out[idx]  = is_layer0 ? ha : l_g * ha + (1.0f - l_g) * he;
}

void fused_mix_conv_launcher(
    const float* hp, const float* n2, const float* h_prev,
    const float* dim_mask, const float* block_mode, const float* layer_gate,
    float* out, int B, int C, long spatial, int D, int is_layer0
) {
    long total = (long)B * C * spatial;
    int blocks = (int)((total + FPC_THREADS - 1) / FPC_THREADS);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    fused_mix_conv_kernel<<<blocks, FPC_THREADS, 0, stream>>>(
        hp, n2, h_prev, dim_mask, block_mode, layer_gate,
        out, B, C, spatial, D, is_layer0
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "AML kernel launch error (fused_mix_conv): %s\n", cudaGetErrorString(err));
}

__global__ void maxpool2x2_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
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
    Y[idx] = fmaxf(fmaxf(X[base + w0], X[base + w0 + 1]),
                    fmaxf(X[base + W_ + w0], X[base + W_ + w0 + 1]));
}

void maxpool2x2_launcher(float* X, float* Y, int B, int C, int H, int W_) {
    int H2 = H / 2, W2 = W_ / 2;
    int total = B * C * H2 * W2;
    int blocks = (total + 255) / 256;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    maxpool2x2_kernel<<<blocks, 256, 0, stream>>>(X, Y, B, C, H, W_);
}

/* ================================================================
 * v2 fused BACKWARD of the mix algebra, two phases:
 *
 *   Phase A consumes d(out) and the recomputed normalized branches and
 *   emits everything that does NOT depend on the conv2 weight gradient:
 *       dn2, d_hp_pre, d_he, gdm_a, gbm0, gbm1, glg.
 *   The caller runs GN2-backward + cuDNN conv2-backward, folds dhp_conv2
 *   into d_hp_pre, then Phase B emits dn1, dh_skip, gdm_b.
 *
 *   Inverse of the forward (see fused_mix_conv_kernel):
 *     r1=max(xhat1,0); hp=r1*dm; he=is0?hp:hpv*dm
 *     u2=he+xhat2;     r2=max(u2,0); hr=r2*dm; ha=gp*hp+gr*hr
 *     g=dout; dha=is0?g:lg*g; d_he0=is0?0:(1-lg)*g
 *     dhr=dha*gr; t2=(u2>0)?dhr*dm:0 (=dn2); d_he=d_he0+t2
 *     d_hp_pre=dha*gp+(is0?d_he:0)
 *     gbm0=SUM dha*hp  gbm1=SUM dha*hr  glg=SUM g*(ha-he)  gdm_a=SUM dhr*r2
 *   phase B (dh=d_hp_pre+dhp_conv2):
 *     dn1=dh*dm*(xhat1>0); dh_skip=is0?0:d_he*dm
 *     gdm_b=SUM dh*r1 + (is0?0:SUM d_he*hpv)
 *
 *   Layout: one block per sample (blockIdx.x = b) so the per-sample [B]
 *   scalars and per-[B,C] gdm reduce within a single block. The inner
 *   loop walks the sample's C*spatial elements with a FLAT grid-stride so
 *   consecutive threads touch consecutive addresses (fully coalesced --
 *   the previous channel-major mapping was non-coalesced and made the
 *   backward slower than the ATen path it replaced). gdm uses a shared
 *   per-channel accumulator flushed with one global store per channel;
 *   float shared atomicAdd order is nondeterministic at ~1e-7 relative,
 *   ~5 orders below the 1e-2 parity gates and consistent with the cuDNN
 *   autotune / bf16 noise the pipeline already carries.
 * ================================================================ */
#define MB_THREADS 256
#define MB_MAX_C   256

__device__ __forceinline__ float mb_tree_sum(float v, float* scratch) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    if (lane == 0) scratch[wid] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < MB_THREADS / 32; i++) s += scratch[i];
        scratch[0] = s;
    }
    __syncthreads();
    return scratch[0];
}

__global__ void fused_mix_bwd_v2_a_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ xhat1,
    const float* __restrict__ xhat2,
    const float* __restrict__ hpv,
    const float* __restrict__ dim_mask,   /* [B, D] */
    const float* __restrict__ block_mode, /* [B, 2] */
    const float* __restrict__ layer_gate, /* [B] */
    float* __restrict__ dn2,
    float* __restrict__ d_hp_pre,
    float* __restrict__ d_he,
    float* __restrict__ gdm_a,            /* [B, C] (zero-init) */
    float* __restrict__ gbm0,             /* [B] (zero-init) */
    float* __restrict__ gbm1,
    float* __restrict__ glg,
    int C, int D, int spatial, int is_layer0
) {
    const int b = blockIdx.x;
    const float* dm_row = dim_mask + (long)b * D;
    const float gp = block_mode[b * 2];
    const float gr = block_mode[b * 2 + 1];
    const float lgv = is_layer0 ? 1.0f : layer_gate[b];

    __shared__ float s_gdm[MB_MAX_C];
    __shared__ float s_part[MB_THREADS / 32];
    for (int t = threadIdx.x; t < C; t += MB_THREADS) s_gdm[t] = 0.0f;
    __syncthreads();

    const long base = (long)b * C * spatial;
    const int CS = C * spatial;
    float ab0 = 0.0f, ab1 = 0.0f, alg = 0.0f;

    for (int j = threadIdx.x; j < CS; j += MB_THREADS) {
        const int i = (int)(base + j);
        const int c = j / spatial;
        const float dm = (c < D) ? dm_row[c] : 0.0f;
        const float g = dout[i];
        const float x1 = xhat1[i];
        const float x2 = xhat2[i];
        const float r1 = fmaxf(x1, 0.0f);
        const float hp = r1 * dm;
        const float he = is_layer0 ? hp : hpv[i] * dm;
        const float u2 = he + x2;
        const float r2 = fmaxf(u2, 0.0f);
        const float hr = r2 * dm;
        const float ha = gp * hp + gr * hr;

        const float dha = is_layer0 ? g : lgv * g;
        const float d_he0 = is_layer0 ? 0.0f : (1.0f - lgv) * g;
        const float dhr = dha * gr;
        const float t2 = (u2 > 0.0f) ? dhr * dm : 0.0f;
        const float dhe = d_he0 + t2;

        dn2[i] = t2;
        d_he[i] = dhe;
        d_hp_pre[i] = dha * gp + (is_layer0 ? dhe : 0.0f);

        ab0 += dha * hp;
        ab1 += dha * hr;
        if (!is_layer0) alg += g * (ha - he);
        atomicAdd(&s_gdm[c], dhr * r2);
    }
    __syncthreads();
    for (int t = threadIdx.x; t < C; t += MB_THREADS)
        gdm_a[(long)b * C + t] = s_gdm[t];

    float f0 = mb_tree_sum(ab0, s_part);
    __syncthreads();
    float f1 = mb_tree_sum(ab1, s_part);
    __syncthreads();
    float f2 = mb_tree_sum(alg, s_part);
    if (threadIdx.x == 0) {
        gbm0[b] = f0;
        gbm1[b] = f1;
        glg[b] = f2;
    }
}

__global__ void fused_mix_bwd_v2_b_kernel(
    const float* __restrict__ d_hp,     /* d_hp_pre + dhp_conv2 */
    const float* __restrict__ d_he,
    const float* __restrict__ xhat1,
    const float* __restrict__ hpv,
    const float* __restrict__ dim_mask, /* [B, D] */
    float* __restrict__ dn1,
    float* __restrict__ dh_skip,
    float* __restrict__ gdm_b,          /* [B, C] (zero-init) */
    int C, int D, int spatial, int is_layer0
) {
    const int b = blockIdx.x;
    const float* dm_row = dim_mask + (long)b * D;

    __shared__ float s_gdm[MB_MAX_C];
    for (int t = threadIdx.x; t < C; t += MB_THREADS) s_gdm[t] = 0.0f;
    __syncthreads();

    const long base = (long)b * C * spatial;
    const int CS = C * spatial;
    for (int j = threadIdx.x; j < CS; j += MB_THREADS) {
        const int i = (int)(base + j);
        const int c = j / spatial;
        const float dm = (c < D) ? dm_row[c] : 0.0f;
        const float dh = d_hp[i];
        const float x1 = xhat1[i];
        const float r1 = fmaxf(x1, 0.0f);
        dn1[i] = (x1 > 0.0f) ? dh * dm : 0.0f;
        float term = dh * r1;
        if (!is_layer0) {
            const float dhe = d_he[i];
            dh_skip[i] = dhe * dm;
            term += dhe * hpv[i];
        }
        atomicAdd(&s_gdm[c], term);
    }
    __syncthreads();
    for (int t = threadIdx.x; t < C; t += MB_THREADS)
        gdm_b[(long)b * C + t] = s_gdm[t];
}

void fused_mix_bwd_v2_a_launcher(
    const float* dout, const float* xhat1, const float* xhat2, const float* hpv,
    const float* dim_mask, const float* block_mode, const float* layer_gate,
    float* dn2, float* d_hp_pre, float* d_he, float* gdm_a,
    float* gbm0, float* gbm1, float* glg,
    int B, int C, int D, int spatial, int is_layer0
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    fused_mix_bwd_v2_a_kernel<<<B, MB_THREADS, 0, stream>>>(
        dout, xhat1, xhat2, hpv, dim_mask, block_mode, layer_gate,
        dn2, d_hp_pre, d_he, gdm_a, gbm0, gbm1, glg,
        C, D, spatial, is_layer0);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "AML kernel launch error (fused_mix_bwd_v2_a): %s\n", cudaGetErrorString(err));
}

void fused_mix_bwd_v2_b_launcher(
    const float* d_hp, const float* d_he, const float* xhat1, const float* hpv,
    const float* dim_mask,
    float* dn1, float* dh_skip, float* gdm_b,
    int B, int C, int D, int spatial, int is_layer0
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    fused_mix_bwd_v2_b_kernel<<<B, MB_THREADS, 0, stream>>>(
        d_hp, d_he, xhat1, hpv, dim_mask, dn1, dh_skip, gdm_b,
        C, D, spatial, is_layer0);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "AML kernel launch error (fused_mix_bwd_v2_b): %s\n", cudaGetErrorString(err));
}
