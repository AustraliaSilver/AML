/*
 * bindings_in.cpp — pybind11 bindings for the InstanceNorm fused pipeline.
 * Mirrors the GN1 entry points in mgn/accel/_C but routes post-conv ops
 * through per-channel (InstanceNorm) kernels.
 */
#include <torch/extension.h>
#include <vector>

void in_maxpool2x2_launcher(float* X, float* Y, int B, int C, int H, int W_);

void in_maxpool2x2_bwd_launcher(
    const float* dh_out, const float* h_in, float* dh_in,
    int B, int C, int H, int W_);

/* v2 block-mix launcher (fused_pipeline_in.cu) */
void in_fused_mix_conv_launcher(
    const float* hp, const float* n2, const float* h_prev,
    const float* dim_mask, const float* block_mode, const float* layer_gate,
    float* out, int B, int C, long spatial, int D, int is_layer0);

#include "../kernels/custom_dynamic_ops.h"

/* v2 block: conv1 -> IN -> relu *dm (also conv2 input) -> conv2 -> IN, then
 * the elementwise mode/gate mix kernel. All ops run natively through custom CUDA kernels. */
static void in_block_forward_v2(
    const torch::Tensor& pi, const torch::Tensor& prototypes,
    const torch::Tensor& h_in,
    const torch::Tensor& dm_i, const torch::Tensor& bm_i, const torch::Tensor& lg_i,
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
    custom_instance_norm_launcher(
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
    custom_instance_norm_launcher(
        hc2.data_ptr<float>(), n2.data_ptr<float>(),
        B, C_out, h_dim, w_dim, 1e-5f
    );

    h_out = h_p.clone();
    in_fused_mix_conv_launcher(
        h_p.data_ptr<float>(), n2.data_ptr<float>(),
        (i == 0) ? nullptr : h_in.data_ptr<float>(),
        dm_i.data_ptr<float>(), bm_i.data_ptr<float>(), lg_i.data_ptr<float>(),
        h_out.data_ptr<float>(), B, C_out, (long)h_dim * w_dim, max_dim,
        (i == 0) ? 1 : 0);
}

/* Forward + save intermediates for backward.
 * Returns [output, hc1(0..L-1), hc2(0..L-1), h_prev(0..L-1), W1_mix(0..L-1), W2_mix(0..L-1)]. */
std::vector<torch::Tensor> in_fused_forward_and_save(
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
        in_block_forward_v2(pi, prototypes, h,
            dim_masks.select(1, i).contiguous(),
            bm.select(1, i).contiguous(),
            layer_gates.select(1, i).contiguous(),
            i, B, C_in, C_out, h_dim, w_dim, max_dim,
            w_offsets, b_offsets, w2_offsets, b2_offsets,
            h_out, saved_hc1[i], saved_hc2[i], saved_w1[i], saved_w2[i]);

        if (i == 5 || i == 11) {
            auto pooled = torch::empty({B, C_out, h_dim / 2, w_dim / 2}, opts);
            in_maxpool2x2_launcher(
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

/* Out-of-place forward (clone + kernel) for recompute during backward. */
torch::Tensor in_maxpool2x2_bwd(
    torch::Tensor dh_out, torch::Tensor h_in,
    int B, int C, int H, int W_
) {
    /* zeros_like: with odd H/W the edge row/column is covered by no pooling
     * window; with empty_like it would keep uninitialized memory. */
    auto dh_in = torch::zeros_like(h_in);
    /* The GN variant (bindings.cpp:maxpool2x2_bwd) normalises both inputs;
     * this one read data_ptr() straight off tensors that come from upstream
     * autograd and may be non-contiguous views (e.g. a sliced grad), which
     * would index the wrong memory. Match the GN variant. */
    dh_out = dh_out.contiguous();
    h_in = h_in.contiguous();
    in_maxpool2x2_bwd_launcher(
        dh_out.data_ptr<float>(), h_in.data_ptr<float>(), dh_in.data_ptr<float>(),
        B, C, H, W_
    );
    return dh_in;
}

std::vector<torch::Tensor> in_dynamic_conv2d_bwd(
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

torch::Tensor in_instance_norm_bwd(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("in_fused_forward_and_save", &in_fused_forward_and_save, "IN fused forward + saves");
    m.def("in_maxpool2x2_bwd", &in_maxpool2x2_bwd, "MaxPool 2x2 backward");
    m.def("dynamic_conv2d_bwd", &in_dynamic_conv2d_bwd, "Custom native dynamic conv2d backward");
    m.def("instance_norm_bwd", &in_instance_norm_bwd, "Custom native InstanceNorm backward");
}
