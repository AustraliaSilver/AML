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

/* ================================================================
 * v2 block algebra (real 2-convolution ResNet): the convolutions and
 * group-norms run as ATen ops (cuDNN grouped convs, bf16-rounded weight
 * mixing exactly as v1); only the final mode/gate mix is a custom
 * elementwise kernel. The Python autograd wrapper recomputes this same
 * ATen chain for its backward, so kernel and eager semantics cannot drift.
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
    W1_mix = torch::mm(
        pi.to(torch::kBFloat16),
        prototypes.slice(1, w_offsets[i], w_offsets[i] + C_out * C_in * 9).to(torch::kBFloat16)
    ).to(torch::kFloat32).view({B, C_out, C_in, 3, 3}).contiguous();
    torch::Tensor b1_mix = torch::mm(
        pi.to(torch::kBFloat16),
        prototypes.slice(1, b_offsets[i], b_offsets[i] + C_out).to(torch::kBFloat16)
    ).to(torch::kFloat32);

    hc1 = torch::conv2d(
        h_in.reshape({1, B * C_in, h_dim, w_dim}).to(torch::kBFloat16),
        W1_mix.reshape({B * C_out, C_in, 3, 3}).to(torch::kBFloat16),
        b1_mix.reshape({B * C_out}).to(torch::kBFloat16),
        {1, 1}, {1, 1}, {1, 1}, B
    ).to(torch::kFloat32).reshape({B, C_out, h_dim, w_dim});

    torch::Tensor n1 = at::group_norm(hc1, 1, c10::nullopt, c10::nullopt, 1e-5, true);
    torch::Tensor h_p = at::relu(n1) * dm_i.view({B, C_out, 1, 1});

    W2_mix = torch::mm(
        pi.to(torch::kBFloat16),
        prototypes.slice(1, w2_offsets[i], w2_offsets[i] + C_out * max_dim * 9).to(torch::kBFloat16)
    ).to(torch::kFloat32).view({B, C_out, max_dim, 3, 3}).contiguous();
    torch::Tensor b2_mix = torch::mm(
        pi.to(torch::kBFloat16),
        prototypes.slice(1, b2_offsets[i], b2_offsets[i] + C_out).to(torch::kBFloat16)
    ).to(torch::kFloat32);

    hc2 = torch::conv2d(
        h_p.reshape({1, B * C_out, h_dim, w_dim}).to(torch::kBFloat16),
        W2_mix.reshape({B * C_out, max_dim, 3, 3}).to(torch::kBFloat16),
        b2_mix.reshape({B * C_out}).to(torch::kBFloat16),
        {1, 1}, {1, 1}, {1, 1}, B
    ).to(torch::kFloat32).reshape({B, C_out, h_dim, w_dim});
    torch::Tensor n2 = at::group_norm(hc2, 1, c10::nullopt, c10::nullopt, 1e-5, true);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mix_bwd_v2_a", &mix_bwd_v2_a, "v2 fused mix backward phase A");
    m.def("mix_bwd_v2_b", &mix_bwd_v2_b, "v2 fused mix backward phase B");
    m.def("fused_forward", &fused_forward, "Fused 6-layer AML forward");
    m.def("maxpool2x2_bwd", &maxpool2x2_bwd, "MaxPool 2x2 backward");
    m.def("fused_forward_and_save", &fused_forward_and_save, "Forward + save intermediates for backward");
}
