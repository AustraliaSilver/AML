/*
 * bindings_in.cpp — pybind11 bindings for the InstanceNorm fused pipeline.
 * Mirrors the GN1 entry points in mgn/accel/_C but routes post-conv ops
 * through per-channel (InstanceNorm) kernels.
 */
#include <torch/extension.h>
#include <vector>

void in_fused_post_conv_launcher(
    float* h_conv, const float* h_prev,
    const float* dim_mask, const float* block_mode, const float* layer_gate,
    int B, int C, int spatial, int D, int is_layer0);

void in_maxpool2x2_launcher(float* X, float* Y, int B, int C, int H, int W_);

void in_fused_post_conv_bwd_launcher(
    const float* dh_out, const float* h_conv, const float* h_prev,
    const float* dim_mask, const float* block_mode, const float* layer_gate,
    float* dh_conv_out, float* dh_prev_out,
    float* grad_bm_out, float* grad_lg_out, float* grad_dm_out,
    int B, int C, int spatial, int D, int is_layer0);

void in_maxpool2x2_bwd_launcher(
    const float* dh_out, const float* h_in, float* dh_in,
    int B, int C, int H, int W_);

torch::Tensor in_fused_post_conv_oop(
    torch::Tensor h_conv_in, torch::Tensor h_prev,
    torch::Tensor dim_mask, torch::Tensor block_mode, torch::Tensor layer_gate,
    int B, int C, int spatial, int D, int is_layer0);

/* Forward + save intermediates for backward.
 * Returns [output, h_conv(0..L-1), h_prev(0..L-1), W_mix(0..L-1)]. */
std::vector<torch::Tensor> in_fused_forward_and_save(
    torch::Tensor X,
    torch::Tensor prototypes,
    torch::Tensor pi,
    torch::Tensor layer_gates,
    torch::Tensor dim_masks,
    torch::Tensor block_mode,
    std::vector<int> w_offsets,
    std::vector<int> b_offsets,
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

    std::vector<torch::Tensor> saved_h_conv(L), saved_h_prev(L), saved_w_mix(L);

    torch::Tensor h = X.to(torch::kFloat32).contiguous();

    for (int i = 0; i < L; i++) {
        int C_in = C_ins[i], C_out = C_outs[i];
        int h_dim = Hs[i], w_dim = Ws[i];

        saved_h_prev[i] = h;

        /* Weight mixing */
        torch::Tensor W_mix = torch::mm(
            pi.to(torch::kBFloat16), prototypes.slice(1, w_offsets[i], w_offsets[i] + C_out * C_in * 9).to(torch::kBFloat16)
        ).to(torch::kFloat32).view({B, C_out, C_in, 3, 3}).contiguous();
        torch::Tensor b_mix = torch::mm(
            pi.to(torch::kBFloat16), prototypes.slice(1, b_offsets[i], b_offsets[i] + C_out).to(torch::kBFloat16)
        ).to(torch::kFloat32);

        saved_w_mix[i] = W_mix;

        /* Grouped conv (per-sample weights) via cuDNN */
        torch::Tensor h_conv = torch::conv2d(
            h.reshape({1, B * C_in, h_dim, w_dim}).to(torch::kBFloat16),
            W_mix.reshape({B * C_out, C_in, 3, 3}).to(torch::kBFloat16),
            b_mix.reshape({B * C_out}).to(torch::kBFloat16),
            {1,1}, {1,1}, {1,1}, B
        ).to(torch::kFloat32).reshape({B, C_out, h_dim, w_dim});

        saved_h_conv[i] = h_conv;

        /* Post-conv with InstanceNorm stats */
        auto dm_i = dim_masks.select(1, i).contiguous();
        auto bm_i_full = bm.select(1, i).contiguous();
        auto bm_i = (bm_i_full.dim() == 2 && bm_i_full.size(1) > 2)
                     ? bm_i_full.narrow(1, 0, 2).contiguous()
                     : bm_i_full;

        torch::Tensor h_post = in_fused_post_conv_oop(
            h_conv,
            (i == 0) ? h_conv : h,
            dm_i, bm_i,
            layer_gates.select(1, i).contiguous(),
            B, C_out, h_dim * w_dim, max_dim, (i == 0) ? 1 : 0
        );

        if (i == 5 || i == 11) {
            auto pooled = torch::empty({B, C_out, h_dim / 2, w_dim / 2}, opts);
            in_maxpool2x2_launcher(
                h_post.data_ptr<float>(), pooled.data_ptr<float>(),
                B, C_out, h_dim, w_dim
            );
            h_post = pooled;
        }
        h = h_post;
    }

    std::vector<torch::Tensor> result;
    result.reserve(1 + 3 * L);
    result.push_back(h);
    for (int i = 0; i < L; i++) result.push_back(saved_h_conv[i]);
    for (int i = 0; i < L; i++) result.push_back(saved_h_prev[i]);
    for (int i = 0; i < L; i++) result.push_back(saved_w_mix[i]);
    return result;
}

/* Out-of-place forward (clone + kernel) for recompute during backward. */
torch::Tensor in_fused_post_conv_oop(
    torch::Tensor h_conv_in, torch::Tensor h_prev,
    torch::Tensor dim_mask, torch::Tensor block_mode, torch::Tensor layer_gate,
    int B, int C, int spatial, int D, int is_layer0
) {
    auto h_conv_out = h_conv_in.clone();
    in_fused_post_conv_launcher(
        h_conv_out.data_ptr<float>(), h_prev.data_ptr<float>(),
        dim_mask.data_ptr<float>(), block_mode.data_ptr<float>(),
        layer_gate.data_ptr<float>(), B, C, spatial, D, is_layer0
    );
    return h_conv_out;
}

std::vector<torch::Tensor> in_fused_post_conv_bwd(
    torch::Tensor dh_out, torch::Tensor h_conv, torch::Tensor h_prev,
    torch::Tensor dim_mask, torch::Tensor block_mode, torch::Tensor layer_gate,
    torch::Tensor grad_bm, torch::Tensor grad_lg, torch::Tensor grad_dm,
    int B, int C, int spatial, int D, int is_layer0
) {
    auto dh_conv_out = torch::empty_like(h_conv);
    auto dh_prev_out = is_layer0 ? torch::zeros_like(h_conv) : torch::empty_like(h_prev);

    in_fused_post_conv_bwd_launcher(
        dh_out.data_ptr<float>(), h_conv.data_ptr<float>(),
        is_layer0 ? dh_out.data_ptr<float>() : h_prev.data_ptr<float>(),
        dim_mask.data_ptr<float>(), block_mode.data_ptr<float>(),
        layer_gate.data_ptr<float>(),
        dh_conv_out.data_ptr<float>(), dh_prev_out.data_ptr<float>(),
        grad_bm.data_ptr<float>(), grad_lg.data_ptr<float>(), grad_dm.data_ptr<float>(),
        B, C, spatial, D, is_layer0
    );

    return {dh_conv_out, dh_prev_out};
}

torch::Tensor in_maxpool2x2_bwd(
    torch::Tensor dh_out, torch::Tensor h_in,
    int B, int C, int H, int W_
) {
    auto dh_in = torch::empty_like(h_in);
    in_maxpool2x2_bwd_launcher(
        dh_out.data_ptr<float>(), h_in.data_ptr<float>(), dh_in.data_ptr<float>(),
        B, C, H, W_
    );
    return dh_in;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("in_fused_forward_and_save", &in_fused_forward_and_save, "IN fused forward + saves");
    m.def("in_fused_post_conv_oop", &in_fused_post_conv_oop, "IN out-of-place post-conv forward");
    m.def("in_fused_post_conv_bwd", &in_fused_post_conv_bwd, "IN post-conv backward");
    m.def("in_maxpool2x2_bwd", &in_maxpool2x2_bwd, "MaxPool 2x2 backward");
}
