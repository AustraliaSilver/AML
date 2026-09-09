/*
 * bindings.cpp — pybind11 bindings for all CUDA kernels + fused pipeline.
 */
#include <torch/extension.h>
#include <vector>

/* ---- Tiled conv launchers ---- */
void tiled_conv_fwd_launcher(
    const float* W, const float* X, const float* bias, float* Y,
    int B, int C_in, int C_out, int H, int W_);

void tiled_conv_bwd_launcher(
    const float* grad_Y, const float* W, const float* X,
    float* grad_X, float* grad_W, float* grad_bias,
    int B, int C_in, int C_out, int H, int W_);

void implicit_gemm_fwd_launcher(
    const float* W, const void* X_bf16, const float* bias, void* Y_bf16,
    int B, int C_in, int C_out, int H, int W_);

/* ---- Fused pipeline launchers ---- */
void fused_post_conv_launcher(
    float* h_conv, const float* h_prev,
    const float* dim_mask, const float* block_mode, const float* layer_gate,
    int B, int C, int spatial, int D, int is_layer0);

void maxpool2x2_launcher(float* X, float* Y, int B, int C, int H, int W_);

torch::Tensor fused_post_conv_oop(
    torch::Tensor h_conv_in, torch::Tensor h_prev,
    torch::Tensor dim_mask, torch::Tensor block_mode, torch::Tensor layer_gate,
    int B, int C, int spatial, int D, int is_layer0);


/* ---- Fused backward launchers ---- */
void fused_post_conv_bwd_launcher(
    const float* dh_out, const float* h_conv, const float* h_prev,
    const float* dim_mask, const float* block_mode, const float* layer_gate,
    float* dh_conv_out, float* dh_prev_out,
    float* grad_bm_out, float* grad_lg_out, float* grad_dm_out,
    int B, int C, int spatial, int D, int is_layer0);

void maxpool2x2_bwd_launcher(
    const float* dh_out, const float* h_in, float* dh_in,
    int B, int C, int H, int W_);

void grouped_conv_bwd_launcher(
    const float* dY, const float* X, const float* Wt,
    float* dX, float* dW, float* db,
    int B, int C_in, int C_out, int H, int W_);

/* ================================================================
 * conv_fwd / conv_bwd (existing)
 * ================================================================ */
torch::Tensor conv_fwd(
    torch::Tensor W, torch::Tensor X, torch::Tensor bias
) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && bias.is_cuda(), "All inputs must be CUDA");
    /* The kernels type-pun X/Y pointers as bf16 and W/bias as fp32.
     * An fp32 X used to be silently reinterpreted -> type-punned garbage. */
    TORCH_CHECK(W.scalar_type() == torch::kFloat32, "conv_fwd: W must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "conv_fwd: bias must be float32");
    if (X.scalar_type() != torch::kBFloat16) X = X.to(torch::kBFloat16);
    TORCH_CHECK(W.is_contiguous() && X.is_contiguous() && bias.is_contiguous(),
                "All inputs must be contiguous");

    int B = X.size(0), C_in = X.size(1), H = X.size(2), W_ = X.size(3), C_out = W.size(1);
    auto Y = torch::zeros({B, C_out, H, W_}, X.options());

    if (C_in >= 16) {
        implicit_gemm_fwd_launcher(W.data_ptr<float>(), X.data_ptr(),
            bias.data_ptr<float>(), Y.data_ptr(), B, C_in, C_out, H, W_);
    } else {
        tiled_conv_fwd_launcher(W.data_ptr<float>(), (const float*)X.data_ptr(),
            bias.data_ptr<float>(), (float*)Y.data_ptr(), B, C_in, C_out, H, W_);
    }
    return Y;
}

std::vector<torch::Tensor> conv_bwd(
    torch::Tensor grad_Y, torch::Tensor W, torch::Tensor X
) {
    TORCH_CHECK(grad_Y.is_cuda() && W.is_cuda() && X.is_cuda());
    TORCH_CHECK(W.scalar_type() == torch::kFloat32, "conv_bwd: W must be float32");
    if (grad_Y.scalar_type() != torch::kBFloat16) grad_Y = grad_Y.to(torch::kBFloat16);
    if (X.scalar_type() != torch::kBFloat16) X = X.to(torch::kBFloat16);
    grad_Y = grad_Y.contiguous(); X = X.contiguous(); W = W.contiguous();
    int B = grad_Y.size(0), C_out = grad_Y.size(1), H = grad_Y.size(2), W_ = grad_Y.size(3), C_in = X.size(1);
    auto grad_X = torch::zeros({B, C_in, H, W_}, W.options());
    auto grad_W = torch::zeros({B, C_out, C_in * 9}, W.options());
    auto grad_b = torch::zeros({B, C_out}, W.options());
    tiled_conv_bwd_launcher((const float*)grad_Y.data_ptr(), W.data_ptr<float>(),
        (const float*)X.data_ptr(), grad_X.data_ptr<float>(), grad_W.data_ptr<float>(),
        grad_b.data_ptr<float>(), B, C_in, C_out, H, W_);
    return {grad_X, grad_W, grad_b};
}

/* ================================================================
 * fused_forward: all 6 layers in one C++ call (no saves, inference).
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

        torch::Tensor W_mix = torch::mm(
            pi.to(torch::kBFloat16), prototypes.slice(1, w_offsets[i], w_offsets[i] + C_out * C_in * 9).to(torch::kBFloat16)
        ).to(torch::kFloat32).view({B, C_out, C_in, 3, 3}).contiguous();
        torch::Tensor b_mix = torch::mm(
            pi.to(torch::kBFloat16), prototypes.slice(1, b_offsets[i], b_offsets[i] + C_out).to(torch::kBFloat16)
        ).to(torch::kFloat32);

        torch::Tensor h_conv = torch::conv2d(
            h.reshape({1, B * C_in, h_dim, w_dim}).to(torch::kBFloat16),
            W_mix.reshape({B * C_out, C_in, 3, 3}).to(torch::kBFloat16),
            b_mix.reshape({B * C_out}).to(torch::kBFloat16),
            {1,1}, {1,1}, {1,1}, B
        ).to(torch::kFloat32).reshape({B, C_out, h_dim, w_dim});

        fused_post_conv_launcher(
            h_conv.data_ptr<float>(),
            (i == 0) ? nullptr : h.data_ptr<float>(),
            dim_masks.select(1, i).contiguous().data_ptr<float>(),
            bm.select(1, i).contiguous().data_ptr<float>(),
            layer_gates.select(1, i).contiguous().data_ptr<float>(),
            B, C_out, h_dim * w_dim, max_dim, (i == 0) ? 1 : 0
        );

        if (i == 5 || i == 11) {
            auto pooled = torch::empty({B, C_out, h_dim / 2, w_dim / 2}, opts);
            maxpool2x2_launcher(
                h_conv.data_ptr<float>(), pooled.data_ptr<float>(),
                B, C_out, h_dim, w_dim
            );
            h_conv = pooled;
        }
        h = h_conv;
    }
    return h;
}

/* ================================================================
 * fused_forward_and_save: forward + save intermediates for backward.
 * Returns flat vector: [output, h_conv(0..L-1), h_prev(0..L-1), W_mix(0..L-1)]
 * Total: 1 + 3*L tensors.
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

    /* Pre-allocate save containers */
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

        /* Conv */
        torch::Tensor h_conv = torch::conv2d(
            h.reshape({1, B * C_in, h_dim, w_dim}).to(torch::kBFloat16),
            W_mix.reshape({B * C_out, C_in, 3, 3}).to(torch::kBFloat16),
            b_mix.reshape({B * C_out}).to(torch::kBFloat16),
            {1,1}, {1,1}, {1,1}, B
        ).to(torch::kFloat32).reshape({B, C_out, h_dim, w_dim});

        saved_h_conv[i] = h_conv;

        /* Post-conv */
        auto dm_i = dim_masks.select(1, i).contiguous();
        auto bm_i_full = bm.select(1, i).contiguous();
        auto bm_i = (bm_i_full.dim() == 2 && bm_i_full.size(1) > 2)
                     ? bm_i_full.narrow(1, 0, 2).contiguous()
                     : bm_i_full;

        torch::Tensor h_post = fused_post_conv_oop(
            h_conv,
            (i == 0) ? h_conv : h, // h_prev is ignored if i==0, but pass a valid tensor
            dm_i, bm_i,
            layer_gates.select(1, i).contiguous(),
            B, C_out, h_dim * w_dim, max_dim, (i == 0) ? 1 : 0
        );

        /* MaxPool at layer 5 and 11 */
        if (i == 5 || i == 11) {
            auto pooled = torch::empty({B, C_out, h_dim / 2, w_dim / 2}, opts);
            maxpool2x2_launcher(
                h_post.data_ptr<float>(), pooled.data_ptr<float>(),
                B, C_out, h_dim, w_dim
            );
            h_post = pooled;
        }
        h = h_post;
    }

    /* Pack output: [output, h_conv(0..L-1), h_prev(0..L-1), W_mix(0..L-1)] */
    std::vector<torch::Tensor> result;
    result.reserve(1 + 3 * L);
    result.push_back(h);
    for (int i = 0; i < L; i++) result.push_back(saved_h_conv[i]);
    for (int i = 0; i < L; i++) result.push_back(saved_h_prev[i]);
    for (int i = 0; i < L; i++) result.push_back(saved_w_mix[i]);
    return result;
}

/* ================================================================
 * fused_post_conv_oop: out-of-place forward (clone + in-place kernel).
 * Used to recompute post-conv output during backward.
 * ================================================================ */
void fused_post_conv_inplace(
    torch::Tensor h_conv, torch::Tensor h_prev,
    torch::Tensor dim_mask, torch::Tensor block_mode, torch::Tensor layer_gate,
    int B, int C, int spatial, int D, int is_layer0
) {
    fused_post_conv_launcher(
        h_conv.data_ptr<float>(), h_prev.data_ptr<float>(),
        dim_mask.data_ptr<float>(), block_mode.data_ptr<float>(),
        layer_gate.data_ptr<float>(), B, C, spatial, D, is_layer0
    );
}

torch::Tensor fused_post_conv_oop(
    torch::Tensor h_conv_in, torch::Tensor h_prev,
    torch::Tensor dim_mask, torch::Tensor block_mode, torch::Tensor layer_gate,
    int B, int C, int spatial, int D, int is_layer0
) {
    auto h_conv_out = h_conv_in.clone();
    fused_post_conv_launcher(
        h_conv_out.data_ptr<float>(), h_prev.data_ptr<float>(),
        dim_mask.data_ptr<float>(), block_mode.data_ptr<float>(),
        layer_gate.data_ptr<float>(), B, C, spatial, D, is_layer0
    );
    return h_conv_out;
}

/* ================================================================
 * fused_post_conv_bwd: backward through fused post-conv chain.
 * Returns [dh_conv, dh_prev, grad_block_mode, grad_layer_gate, grad_dim_mask].
 * ================================================================ */
std::vector<torch::Tensor> fused_post_conv_bwd(
    torch::Tensor dh_out, torch::Tensor h_conv, torch::Tensor h_prev,
    torch::Tensor dim_mask, torch::Tensor block_mode, torch::Tensor layer_gate,
    torch::Tensor grad_bm, torch::Tensor grad_lg, torch::Tensor grad_dm,
    int B, int C, int spatial, int D, int is_layer0
) {
    auto dh_conv_out = torch::empty_like(h_conv);
    auto dh_prev_out = is_layer0 ? torch::zeros_like(h_conv) : torch::empty_like(h_prev);

    fused_post_conv_bwd_launcher(
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

/* ================================================================
 * maxpool2x2 forward
 * ================================================================ */
torch::Tensor maxpool2x2_fwd(
    torch::Tensor X, int B, int C, int H, int W_
) {
    int H2 = H / 2, W2 = W_ / 2;
    auto Y = torch::empty({B, C, H2, W2}, X.options());
    maxpool2x2_launcher(X.data_ptr<float>(), Y.data_ptr<float>(), B, C, H, W_);
    return Y;
}

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
 * fused_backward_full: entire backward loop in C++, no Python overhead.
 * Returns [grad_X, grad_prototypes, grad_pi, grad_layer_gates, grad_dim_masks, grad_block_mode].
 * ================================================================ */
std::vector<torch::Tensor> fused_backward_full(
    torch::Tensor grad_output,
    torch::Tensor X,
    torch::Tensor prototypes,
    torch::Tensor pi,
    torch::Tensor layer_gates,
    torch::Tensor dim_masks,
    torch::Tensor block_mode,
    std::vector<torch::Tensor> saved_h_conv,
    std::vector<torch::Tensor> saved_h_prev,
    std::vector<torch::Tensor> saved_w_mix,
    std::vector<int64_t> w_offsets,
    std::vector<int64_t> b_offsets,
    std::vector<int64_t> C_ins,
    std::vector<int64_t> C_outs,
    std::vector<int64_t> Hs,
    std::vector<int64_t> Ws,
    std::vector<int64_t> numel_proto_w,
    int64_t max_dim
) {
    int64_t B = X.size(0);
    int64_t L = (int64_t)C_ins.size();

    auto grad_pi = torch::zeros_like(pi);
    auto grad_prototypes = torch::zeros_like(prototypes);
    auto grad_layer_gates = torch::zeros_like(layer_gates);
    auto grad_dim_masks = torch::zeros_like(dim_masks);
    auto grad_block_mode = torch::zeros_like(block_mode);

    torch::Tensor grad = grad_output;

    for (int64_t i = L - 1; i >= 0; i--) {
        int64_t C_in = C_ins[i];
        int64_t C_out = C_outs[i];
        int64_t h_dim = Hs[i];
        int64_t w_dim = Ws[i];
        torch::Tensor hp = saved_h_prev[i];

        auto dm_i = dim_masks.select(1, i).contiguous();
        auto bm_full = block_mode.select(1, i).contiguous();
        auto bm_i = (bm_full.dim() == 2 && bm_full.size(1) > 2)
                     ? bm_full.narrow(1, 0, 2).contiguous()
                     : bm_full;
        auto lg_i = layer_gates.select(1, i).contiguous();

        /* Maxpool backward at layer 5 and 11 */
        if (i == 5 || i == 11) {
            auto h_post_2 = fused_post_conv_oop(
                saved_h_conv[i], hp, dm_i, bm_i, lg_i,
                B, C_out, h_dim * w_dim, max_dim, 0
            );
            grad = maxpool2x2_bwd(grad, h_post_2, B, C_out, h_dim, w_dim);
        }

        /* Accumulate per-layer grads through CONTIGUOUS scratch buffers.
         * The previous version handed select()/narrow() views (non-contiguous
         * row strides) directly to kernels that index linearly for packed
         * [B,2]/[B]/[B,D] layouts, so the writes landed in wrong memory. */
        auto grad_bm_i = torch::zeros({B, 2}, grad_output.options());
        auto grad_lg_i = torch::zeros({B}, grad_output.options());
        auto grad_dm_i = torch::zeros({B, max_dim}, grad_output.options());

        auto bwd = fused_post_conv_bwd(
            grad.contiguous(), saved_h_conv[i], hp,
            dm_i, bm_i, lg_i,
            grad_bm_i, grad_lg_i, grad_dm_i,
            B, C_out, h_dim * w_dim, max_dim, (i == 0) ? 1 : 0
        );
        grad_block_mode.select(1, i).narrow(1, 0, 2).add_(grad_bm_i);
        grad_layer_gates.select(1, i).add_(grad_lg_i);
        grad_dim_masks.select(1, i).add_(grad_dm_i);
        auto dh_conv = bwd[0];
        auto dh_skip = bwd[1];

        /* Conv backward via cuDNN */
        torch::Tensor W_mix = saved_w_mix[i];

        auto input_4d = hp.reshape({1, B * C_in, h_dim, w_dim});
        auto dh_conv_4d = dh_conv.reshape({1, B * C_out, h_dim, w_dim});

        auto conv_bwd = torch::convolution_backward(
            dh_conv_4d.to(torch::kBFloat16), input_4d.to(torch::kBFloat16),
            W_mix.reshape({B * C_out, C_in, 3, 3}).to(torch::kBFloat16),
            /*bias=*/{},
            /*stride=*/{1, 1}, /*padding=*/{1, 1}, /*dilation=*/{1, 1},
            /*transposed=*/false, /*output_padding=*/{0, 0},
            /*groups=*/B,
            /*output_mask=*/{true, true, true}
        );

        auto dh_from_conv = std::get<0>(conv_bwd).to(torch::kFloat32).reshape({B, C_in, h_dim, w_dim});
        auto dW = std::get<1>(conv_bwd).to(torch::kFloat32).reshape({B, C_out, C_in, 3, 3});
        auto db = std::get<2>(conv_bwd).to(torch::kFloat32).reshape({B, C_out});

        /* Accumulate skip gradient */
        if (i > 0) {
            dh_from_conv.add_(dh_skip);
        }

        /* Weight-mix backward (using in-place addmm_) */
        auto dW_flat = dW.reshape({B, -1});
        int64_t pw = numel_proto_w[i];
        auto proto_w = prototypes.slice(1, w_offsets[i], w_offsets[i] + pw).detach();
        grad_pi.addmm_(dW_flat, proto_w.t());
        grad_prototypes.slice(1, w_offsets[i], w_offsets[i] + pw).addmm_(pi.t(), dW_flat);

        /* Bias-mix backward (using in-place addmm_) */
        auto proto_b = prototypes.slice(1, b_offsets[i], b_offsets[i] + C_out).detach();
        grad_pi.addmm_(db, proto_b.t());
        grad_prototypes.slice(1, b_offsets[i], b_offsets[i] + C_out).addmm_(pi.t(), db);

        grad = dh_from_conv;
    }

    return {grad, grad_prototypes, grad_pi, grad_layer_gates, grad_dim_masks, grad_block_mode};
}

/* ================================================================
 * Module exports
 * ================================================================ */
std::vector<torch::Tensor> grouped_conv_bwd(
    torch::Tensor dY, torch::Tensor X, torch::Tensor Wt
) {
    int B = dY.size(0), C_out = dY.size(1), H = dY.size(2), W_ = dY.size(3);
    int C_in = X.size(1);
    auto opts = dY.options();
    auto dX = torch::empty({B, C_in, H, W_}, opts);
    auto dW = torch::empty({B * C_out * C_in * 9}, opts);
    auto db = torch::empty({B * C_out}, opts);
    grouped_conv_bwd_launcher(
        dY.data_ptr<float>(), X.data_ptr<float>(), Wt.data_ptr<float>(),
        dX.data_ptr<float>(), dW.data_ptr<float>(), db.data_ptr<float>(),
        B, C_in, C_out, H, W_
    );
    return {dX, dW.reshape({B, C_out, C_in, 3, 3}), db.reshape({B, C_out})};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_fwd", &conv_fwd, "Conv forward");
    m.def("conv_bwd", &conv_bwd, "Conv backward");
    m.def("fused_forward", &fused_forward, "Fused 6-layer AML forward");
    m.def("fused_post_conv", &fused_post_conv_inplace, "In-place fused post-conv forward");
    m.def("fused_post_conv_oop", &fused_post_conv_oop, "Out-of-place fused post-conv forward");
    m.def("fused_post_conv_bwd", &fused_post_conv_bwd, "Fused post-conv backward");
    m.def("maxpool2x2_bwd", &maxpool2x2_bwd, "MaxPool 2x2 backward");
    m.def("maxpool2x2", &maxpool2x2_fwd, "MaxPool 2x2 forward");
    m.def("grouped_conv_bwd", &grouped_conv_bwd, "Grouped conv backward (custom CUDA)");
    m.def("fused_backward_full", &fused_backward_full, "Full backward in C++ (no Python loop)");
    m.def("fused_forward_and_save", &fused_forward_and_save, "Forward + save intermediates for backward");
}
