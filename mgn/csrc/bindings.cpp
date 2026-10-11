/*
 * bindings.cpp — pybind11 bindings for all CUDA kernels + fused pipeline.
 */
#include <torch/extension.h>
#include <vector>

/* ---- Tiled conv launchers ---- */
/* ---- Fused pipeline launchers ---- */
void maxpool2x2_launcher(float* X, float* Y, int B, int C, int H, int W_);

/* v2 block-mix launcher (kernels/fused_pipeline.cu) */
void fused_mix_conv_launcher(
    const float* hp, const float* n2, const float* h_prev,
    const float* dim_mask, const float* block_mode, const float* layer_gate,
    float* out, int B, int C, long spatial, int D, int is_layer0);

/* ---- Fused backward launchers ---- */
void maxpool2x2_bwd_launcher(
    const float* dh_out, const float* h_in, float* dh_in,
    int B, int C, int H, int W_);

/* v2 fused mix-backward launchers (kernels/fused_pipeline.cu) */
void fused_mix_bwd_v2_a_launcher(
    const float* dout, const float* xhat1, const float* xhat2, const float* hpv,
    const float* dim_mask, const float* block_mode, const float* layer_gate,
    float* dn2, float* d_hp_pre, float* d_he, float* gdm_a,
    float* gbm0, float* gbm1, float* glg,
    int B, int C, int D, int spatial, int is_layer0);

void fused_mix_bwd_v2_b_launcher(
    const float* d_hp, const float* d_he, const float* xhat1, const float* hpv,
    const float* dim_mask,
    float* dn1, float* dh_skip, float* gdm_b,
    int B, int C, int D, int spatial, int is_layer0);

#include "kernels/custom_dynamic_ops.h"

/* ================================================================
 * v2 block algebra (real 2-convolution ResNet):
 * Convolutions, Normalizations, and Mix operations are executed
 * natively via custom CUDA kernels (custom_dynamic_conv2d,
 * custom_group_norm1, custom_relu_dimmask, fused_mix_conv_launcher).
 * ================================================================ */
static void block_forward_v2(
    const torch::Tensor& pi, const torch::Tensor& prototypes,
    const torch::Tensor& h_in,
    const torch::Tensor& dm_i,   // [B, C_out] contiguous fp32
    const torch::Tensor& bm_i,   // [B, 2] contiguous
    const torch::Tensor& lg_i,   // [B] contiguous
    int i, int B, int C_in, int C_out, int h_dim, int w_dim, int max_dim,
    const std::vector<int>& w_offsets, const std::vector<int>& b_offsets,
    const std::vector<int>& w2_offsets, const std::vector<int>& b2_offsets,
    torch::Tensor& h_out, torch::Tensor& hc1, torch::Tensor& hc2,
    torch::Tensor& W1_mix, torch::Tensor& W2_mix
) {
    auto opts_f32 = h_in.options().dtype(torch::kFloat32);

    W1_mix = torch::mm(
        pi.to(torch::kBFloat16),
        prototypes.slice(1, w_offsets[i], w_offsets[i] + C_out * C_in * 9).to(torch::kBFloat16)
    ).to(torch::kFloat32).view({B, C_out, C_in, 3, 3}).contiguous();
    torch::Tensor b1_mix = torch::mm(
        pi.to(torch::kBFloat16),
        prototypes.slice(1, b_offsets[i], b_offsets[i] + C_out).to(torch::kBFloat16)
    ).to(torch::kFloat32);

    hc1 = torch::empty({B, C_out, h_dim, w_dim}, opts_f32);
    auto h_in_bf16 = h_in.to(torch::kBFloat16).contiguous();
    auto W1_mix_bf16 = W1_mix.to(torch::kBFloat16).contiguous();
    auto b1_mix_bf16 = b1_mix.to(torch::kBFloat16).contiguous();
    custom_dynamic_conv2d_bf16_launcher(
        h_in_bf16.data_ptr<at::BFloat16>(),
        W1_mix_bf16.data_ptr<at::BFloat16>(),
        b1_mix_bf16.data_ptr<at::BFloat16>(),
        hc1.data_ptr<float>(),
        B, C_in, C_out, h_dim, w_dim
    );

    torch::Tensor n1 = torch::empty_like(hc1);
    custom_group_norm1_launcher(
        hc1.data_ptr<float>(), n1.data_ptr<float>(),
        B, C_out, h_dim, w_dim, 1e-5f
    );

    torch::Tensor h_p = torch::empty_like(n1);
    custom_relu_dimmask_launcher(
        n1.data_ptr<float>(), dm_i.data_ptr<float>(),
        h_p.data_ptr<float>(), B, C_out, h_dim * w_dim, C_out
    );

    W2_mix = torch::mm(
        pi.to(torch::kBFloat16),
        prototypes.slice(1, w2_offsets[i], w2_offsets[i] + C_out * max_dim * 9).to(torch::kBFloat16)
    ).to(torch::kFloat32).view({B, C_out, max_dim, 3, 3}).contiguous();
    torch::Tensor b2_mix = torch::mm(
        pi.to(torch::kBFloat16),
        prototypes.slice(1, b2_offsets[i], b2_offsets[i] + C_out).to(torch::kBFloat16)
    ).to(torch::kFloat32);

    hc2 = torch::empty({B, C_out, h_dim, w_dim}, opts_f32);
    auto h_p_bf16 = h_p.to(torch::kBFloat16).contiguous();
    auto W2_mix_bf16 = W2_mix.to(torch::kBFloat16).contiguous();
    auto b2_mix_bf16 = b2_mix.to(torch::kBFloat16).contiguous();
    custom_dynamic_conv2d_bf16_launcher(
        h_p_bf16.data_ptr<at::BFloat16>(),
        W2_mix_bf16.data_ptr<at::BFloat16>(),
        b2_mix_bf16.data_ptr<at::BFloat16>(),
        hc2.data_ptr<float>(),
        B, max_dim, C_out, h_dim, w_dim
    );

    torch::Tensor n2 = torch::empty_like(hc2);
    custom_group_norm1_launcher(
        hc2.data_ptr<float>(), n2.data_ptr<float>(),
        B, C_out, h_dim, w_dim, 1e-5f
    );

    h_out = h_p.clone();
    fused_mix_conv_launcher(
        h_p.data_ptr<float>(), n2.data_ptr<float>(),
        (i == 0) ? nullptr : h_in.data_ptr<float>(),
        dm_i.data_ptr<float>(), bm_i.data_ptr<float>(), lg_i.data_ptr<float>(),
        h_out.data_ptr<float>(), B, C_out, (long)h_dim * w_dim, max_dim,
        (i == 0) ? 1 : 0);
}

/* ================================================================
 * fused_forward: all layers in one C++ call (no saves, inference).
 * ================================================================ */
torch::Tensor fused_forward(
    torch::Tensor X,
    torch::Tensor prototypes,
    torch::Tensor pi,
    torch::Tensor layer_gates,
    torch::Tensor dim_masks,
    torch::Tensor block_mode,
    std::vector<int> w_offsets,
    std::vector<int> b_offsets,
    std::vector<int> w2_offsets,
    std::vector<int> b2_offsets,
    std::vector<int> C_ins,
    std::vector<int> C_outs,
    std::vector<int> Hs,
    std::vector<int> Ws,
    int max_dim
) {
    int B = X.size(0);
    int L = (int)C_ins.size();
    auto opts = X.options();

    torch::Tensor bm = block_mode;
    if (block_mode.dim() == 3 && block_mode.size(2) > 2)
        bm = block_mode.narrow(2, 0, 2).contiguous();

    torch::Tensor h = X;
    for (int i = 0; i < L; i++) {
        int C_in = C_ins[i], C_out = C_outs[i];
        int h_dim = Hs[i], w_dim = Ws[i];

        torch::Tensor h_out, hc1, hc2, W1_mix, W2_mix;
        block_forward_v2(pi, prototypes, h,
            dim_masks.select(1, i).contiguous(),
            bm.select(1, i).contiguous(),
            layer_gates.select(1, i).contiguous(),
            i, B, C_in, C_out, h_dim, w_dim, max_dim,
            w_offsets, b_offsets, w2_offsets, b2_offsets,
            h_out, hc1, hc2, W1_mix, W2_mix);

        if (i == 5 || i == 11) {
            auto pooled = torch::empty({B, C_out, h_dim / 2, w_dim / 2}, opts);
            maxpool2x2_launcher(
                h_out.data_ptr<float>(), pooled.data_ptr<float>(),
                B, C_out, h_dim, w_dim
            );
            h_out = pooled;
        }
        h = h_out;
    }
    return h;
}

/* ================================================================
 * fused_forward_and_save: forward + save intermediates for backward.
 * Returns flat vector:
 *   [output, hc1(0..L-1), hc2(0..L-1), h_prev(0..L-1), W1_mix(0..L-1), W2_mix(0..L-1)]
 * Total: 1 + 5*L tensors (v2 layout).
 * ================================================================ */
std::vector<torch::Tensor> fused_forward_and_save(
    torch::Tensor X,
    torch::Tensor prototypes,
    torch::Tensor pi,
    torch::Tensor layer_gates,
    torch::Tensor dim_masks,
    torch::Tensor block_mode,
    std::vector<int> w_offsets,
    std::vector<int> b_offsets,
    std::vector<int> w2_offsets,
    std::vector<int> b2_offsets,
    std::vector<int> C_ins,
    std::vector<int> C_outs,
    std::vector<int> Hs,
    std::vector<int> Ws,
    int max_dim
) {
    int B = X.size(0);
    int L = (int)C_ins.size();
    auto opts = X.options();

    torch::Tensor bm = block_mode;
    if (block_mode.dim() == 3 && block_mode.size(2) > 2)
        bm = block_mode.narrow(2, 0, 2).contiguous();

    std::vector<torch::Tensor> saved_hc1(L), saved_hc2(L), saved_h_prev(L),
                               saved_w1(L), saved_w2(L);

    torch::Tensor h = X.to(torch::kFloat32).contiguous();

    for (int i = 0; i < L; i++) {
        int C_in = C_ins[i], C_out = C_outs[i];
        int h_dim = Hs[i], w_dim = Ws[i];

        saved_h_prev[i] = h;

        torch::Tensor h_out;
        block_forward_v2(pi, prototypes, h,
            dim_masks.select(1, i).contiguous(),
            bm.select(1, i).contiguous(),
            layer_gates.select(1, i).contiguous(),
            i, B, C_in, C_out, h_dim, w_dim, max_dim,
            w_offsets, b_offsets, w2_offsets, b2_offsets,
            h_out, saved_hc1[i], saved_hc2[i], saved_w1[i], saved_w2[i]);

        if (i == 5 || i == 11) {
            auto pooled = torch::empty({B, C_out, h_dim / 2, w_dim / 2}, opts);
            maxpool2x2_launcher(
                h_out.data_ptr<float>(), pooled.data_ptr<float>(),
                B, C_out, h_dim, w_dim
            );
            h_out = pooled;
        }
        h = h_out;
    }

    std::vector<torch::Tensor> result;
    result.reserve(1 + 5 * L);
    result.push_back(h);
    for (int i = 0; i < L; i++) result.push_back(saved_hc1[i]);
    for (int i = 0; i < L; i++) result.push_back(saved_hc2[i]);
    for (int i = 0; i < L; i++) result.push_back(saved_h_prev[i]);
    for (int i = 0; i < L; i++) result.push_back(saved_w1[i]);
    for (int i = 0; i < L; i++) result.push_back(saved_w2[i]);
    return result;
}

/* ================================================================
 * maxpool2x2 forward
 * ================================================================ */
/* ================================================================
 * maxpool2x2_bwd: backward through 2x2 maxpool.
 * ================================================================ */
torch::Tensor maxpool2x2_bwd(
    torch::Tensor dh_out, torch::Tensor h_in,
    int B, int C, int H, int W_
) {
    /* zeros_like (not empty_like): the kernel only writes positions that fall
     * inside a complete 2x2 window, so with odd H/W the last row/column would
     * otherwise keep uninitialized memory and pollute the upstream gradient. */
    auto dh_in = torch::zeros_like(h_in);
    dh_out = dh_out.contiguous();
    h_in = h_in.contiguous();
    maxpool2x2_bwd_launcher(
        dh_out.data_ptr<float>(), h_in.data_ptr<float>(), dh_in.data_ptr<float>(),
        B, C, H, W_
    );
    return dh_in;
}

/* ================================================================
 * Module exports
 * ================================================================ */
/* ================================================================
 * v2 fused mix-backward wrappers. C++ driver + Python recomputes the two
 * group_norms (xhat1/xhat2) so the norm statistics path cannot drift from
 * ATen; these kernels invert ONLY the mix algebra and fold the reductions.
 * ================================================================ */
std::vector<torch::Tensor> mix_bwd_v2_a(
    torch::Tensor dout, torch::Tensor xhat1, torch::Tensor xhat2, torch::Tensor hpv,
    torch::Tensor dim_mask, torch::Tensor block_mode, torch::Tensor layer_gate,
    int D, int is_layer0
) {
    TORCH_CHECK(dout.is_cuda() && dout.is_contiguous() && xhat1.is_contiguous()
        && xhat2.is_contiguous() && hpv.is_contiguous() && dim_mask.is_contiguous()
        && block_mode.is_contiguous() && layer_gate.is_contiguous(),
        "mix_bwd_v2_a: all inputs must be contiguous CUDA");
    int B = dout.size(0), C = dout.size(1);
    int spatial = (int)(dout.numel() / ((long)B * C));
    TORCH_CHECK(C <= 256 && (256 % C) == 0,
        "mix_bwd_v2_a: C must divide 256 and be <= 256");
    auto opts = dout.options();
    auto dn2 = torch::empty_like(dout);
    auto d_hp_pre = torch::empty_like(dout);
    auto d_he = torch::empty_like(dout);
    auto gdm_a = torch::empty({B, C}, opts);
    auto gbm0 = torch::empty({B}, opts);
    auto gbm1 = torch::empty({B}, opts);
    auto glg = torch::empty({B}, opts);
    fused_mix_bwd_v2_a_launcher(
        dout.data_ptr<float>(), xhat1.data_ptr<float>(), xhat2.data_ptr<float>(),
        hpv.data_ptr<float>(), dim_mask.data_ptr<float>(),
        block_mode.data_ptr<float>(), layer_gate.data_ptr<float>(),
        dn2.data_ptr<float>(), d_hp_pre.data_ptr<float>(), d_he.data_ptr<float>(),
        gdm_a.data_ptr<float>(), gbm0.data_ptr<float>(), gbm1.data_ptr<float>(),
        glg.data_ptr<float>(), B, C, D, spatial, is_layer0);
    return {dn2, d_hp_pre, d_he, gdm_a, gbm0, gbm1, glg};
}

std::vector<torch::Tensor> mix_bwd_v2_b(
    torch::Tensor d_hp, torch::Tensor d_he, torch::Tensor xhat1, torch::Tensor hpv,
    torch::Tensor dim_mask, int D, int is_layer0
) {
    int B = d_hp.size(0), C = d_hp.size(1);
    int spatial = (int)(d_hp.numel() / ((long)B * C));
    auto opts = d_hp.options();
    auto dn1 = torch::empty_like(d_hp);
    auto dh_skip = torch::empty_like(d_hp);
    auto gdm_b = torch::empty({B, C}, opts);
    fused_mix_bwd_v2_b_launcher(
        d_hp.data_ptr<float>(), d_he.data_ptr<float>(), xhat1.data_ptr<float>(),
        hpv.data_ptr<float>(), dim_mask.data_ptr<float>(),
        dn1.data_ptr<float>(), dh_skip.data_ptr<float>(), gdm_b.data_ptr<float>(),
        B, C, D, spatial, is_layer0);
    return {dn1, dh_skip, gdm_b};
}

std::vector<torch::Tensor> dynamic_conv2d_bwd(
    torch::Tensor g_out, torch::Tensor x_in, torch::Tensor w_mix,
    int B, int C_in, int C_out, int H, int W_spatial
) {
    auto opts_f32 = g_out.options().dtype(torch::kFloat32);
    auto d_in = torch::empty({B, C_in, H, W_spatial}, opts_f32);
    auto d_w  = torch::empty({B, C_out * C_in * 9}, opts_f32);
    auto d_b  = torch::empty({B, C_out}, opts_f32);

    auto g_out_bf16 = g_out.to(torch::kBFloat16).contiguous();
    auto x_in_bf16  = x_in.to(torch::kBFloat16).contiguous();
    auto w_mix_bf16 = w_mix.to(torch::kBFloat16).contiguous();

    custom_dynamic_conv2d_bwd_launcher(
        g_out_bf16.data_ptr<at::BFloat16>(),
        x_in_bf16.data_ptr<at::BFloat16>(),
        w_mix_bf16.data_ptr<at::BFloat16>(),
        d_in.data_ptr<float>(),
        d_w.data_ptr<float>(),
        d_b.data_ptr<float>(),
        B, C_in, C_out, H, W_spatial
    );
    return {d_in, d_w, d_b};
}

torch::Tensor group_norm1_bwd(
    torch::Tensor g_out, torch::Tensor raw, torch::Tensor mean, torch::Tensor rstd,
    int B, int C, int H, int W_spatial
) {
    auto d_in = torch::empty_like(raw);
    custom_group_norm1_bwd_launcher(
        g_out.contiguous().data_ptr<float>(),
        raw.contiguous().data_ptr<float>(),
        mean.contiguous().data_ptr<float>(),
        rstd.contiguous().data_ptr<float>(),
        d_in.data_ptr<float>(),
        B, C, H, W_spatial
    );
    return d_in;
}

torch::Tensor instance_norm_bwd(
    torch::Tensor g_out, torch::Tensor raw, torch::Tensor mean, torch::Tensor rstd,
    int B, int C, int H, int W_spatial
) {
    auto d_in = torch::empty_like(raw);
    custom_instance_norm_bwd_launcher(
        g_out.contiguous().data_ptr<float>(),
        raw.contiguous().data_ptr<float>(),
        mean.contiguous().data_ptr<float>(),
        rstd.contiguous().data_ptr<float>(),
        d_in.data_ptr<float>(),
        B, C, H, W_spatial
    );
    return d_in;
}

void batched_hyper_linear_launcher(
    const float* A, const float* B, const float* bias, float* C,
    int batch, int M, int K, int N);

void batched_hyper_linear_wmma_bf16_launcher(
    const __nv_bfloat16* A, const __nv_bfloat16* B, const float* bias, float* C,
    int batch, int M, int K, int N);

torch::Tensor batched_hyper_linear(torch::Tensor A, torch::Tensor B, c10::optional<torch::Tensor> bias_opt) {
    int batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);
    auto opts = A.options().dtype(torch::kFloat32);
    auto C = torch::empty({batch, M, N}, opts);

    auto A_f32 = A.to(torch::kFloat32).contiguous();
    auto B_f32 = B.to(torch::kFloat32).contiguous();
    const float* bias_ptr = nullptr;
    torch::Tensor bias_contig;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias_contig = bias_opt.value().to(torch::kFloat32).contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }

    batched_hyper_linear_launcher(
        A_f32.data_ptr<float>(),
        B_f32.data_ptr<float>(),
        bias_ptr,
        C.data_ptr<float>(),
        batch, M, K, N
    );
    return C;
}

torch::Tensor batched_hyper_linear_tc(torch::Tensor A, torch::Tensor B, c10::optional<torch::Tensor> bias_opt) {
    int batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);
    auto opts = A.options().dtype(torch::kFloat32);
    auto C = torch::empty({batch, M, N}, opts);

    auto A_bf16 = A.to(torch::kBFloat16).contiguous();
    auto B_bf16 = B.to(torch::kBFloat16).contiguous();
    const float* bias_ptr = nullptr;
    torch::Tensor bias_contig;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias_contig = bias_opt.value().to(torch::kFloat32).contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }

    batched_hyper_linear_wmma_bf16_launcher(
        reinterpret_cast<const __nv_bfloat16*>(A_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(B_bf16.data_ptr<at::BFloat16>()),
        bias_ptr,
        C.data_ptr<float>(),
        batch, M, K, N
    );
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mix_bwd_v2_a", &mix_bwd_v2_a, "v2 fused mix backward phase A");
    m.def("mix_bwd_v2_b", &mix_bwd_v2_b, "v2 fused mix backward phase B");
    m.def("fused_forward", &fused_forward, "Fused 6-layer AML forward");
    m.def("maxpool2x2_bwd", &maxpool2x2_bwd, "MaxPool 2x2 backward");
    m.def("fused_forward_and_save", &fused_forward_and_save, "Forward + save intermediates for backward");
    m.def("dynamic_conv2d_bwd", &dynamic_conv2d_bwd, "Custom native dynamic conv2d backward");
    m.def("group_norm1_bwd", &group_norm1_bwd, "Custom native GroupNorm(1) backward");
    m.def("instance_norm_bwd", &instance_norm_bwd, "Custom native InstanceNorm backward");
    m.def("batched_hyper_linear", &batched_hyper_linear, "Tiled shared memory batched GEMM",
          py::arg("A"), py::arg("B"), py::arg("bias") = py::none());
    m.def("batched_hyper_linear_tc", &batched_hyper_linear_tc, "Tensor Core WMMA BF16 batched GEMM",
          py::arg("A"), py::arg("B"), py::arg("bias") = py::none());
}
