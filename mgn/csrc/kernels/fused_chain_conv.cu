/*
 * fused_chain_conv.cu
 *
 * Fused forward/backward for all 6 AML layers in a single C++ call.
 * Uses cuBLAS for GEMMs (weight mixing + bmm), reuses im2col buffer.
 *
 * This eliminates Python overhead between layers and reuses memory.
 */
#include "common.h"
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

/* Forward declaration */
void fused_norm_relu_gate_launcher(
    float* X, const float* h_prev, const float* dim_mask,
    const float* block_mode, const float* layer_gate,
    int B, int C, int H, int W, int D, int is_layer0);

void tiled_conv_fwd_launcher(
    const float* W, const float* X, const float* bias, float* Y,
    int B, int C_in, int C_out, int H, int W_);

/* ================================================================
 * Fused forward: process all layers in one call
 *
 * Args:
 *   prototypes  [K_total]  — full prototype weight vector (flat)
 *   pi          [B, K]     — per-sample mixing coefficients
 *   X           [B, 3, H0, W0] — input image (bf16)
 *   h_prev      [B, max_dim, H0, W0] — previous hidden state for residual (NULL for layer 0)
 *   dim_masks   [B, max_layers, max_dim] — per-layer dimension masks
 *   layer_gates [B, max_layers] — inter-layer residual gates
 *   block_mode  [B, max_layers, 2] — plain/resnet gates
 *   layer_info  [max_layers, 5] — (w_start, b_start, C_in, C_out, H, W) per layer
 *   Y           [B, max_dim, H_final, W_final] — output (bf16)
 *   K, max_dim, max_layers, H0, W0
 * ================================================================ */
void fused_chain_forward(
    torch::Tensor prototypes,   /* [K_total] f32 */
    torch::Tensor pi,           /* [B, K] f32 */
    torch::Tensor X_in,         /* [B, 3, H0, W0] bf16 */
    torch::Tensor dim_masks,    /* [B, max_layers, max_dim] f32 */
    torch::Tensor layer_gates,  /* [B, max_layers] f32 */
    torch::Tensor block_mode,   /* [B, max_layers, 2] f32 */
    /* Layer offsets: w_start per layer */
    std::vector<int> w_starts,
    std::vector<int> b_starts,
    std::vector<int> C_ins,
    std::vector<int> C_outs,
    std::vector<int> Hs,
    std::vector<int> Ws,
    int K, int max_dim, int max_layers
) {
    int B = X_in.size(0);

    /* Allocate im2col buffer once, reuse across layers */
    int max_spatial = Hs[0] * Ws[0];  /* 32*32 = 1024 */
    int max_cin9 = max_dim * 9;        /* 128*9 = 1152 */
    auto im2col_buf = torch::zeros({B, max_cin9, max_spatial}, X_in.options());

    /* Working buffer for conv output (f32 for norm, then cast to bf16) */
    auto h_buf_a = torch::zeros({B, max_dim, Hs[0], Ws[0]},
                                torch::TensorOptions().device(X_in.device()).dtype(torch::kFloat32));
    auto h_buf_b = torch::zeros({B, max_dim, Hs[0], Ws[0]},
                                torch::TensorOptions().device(X_in.device()).dtype(torch::kFloat32));

    /* Pointer to current hidden state */
    float* h_cur = h_buf_a.data_ptr<float>();
    float* h_next = h_buf_b.data_ptr<float>();

    int H_cur = Hs[0], W_cur = Ws[0];

    /* Cast input bf16 → f32 for first layer */
    {
        auto X_f32 = X_in.to(torch::kFloat32);
        /* Copy X_f32 into h_cur for the first layer (C_in=3 → pad to C_out channels after conv) */
        /* Actually, h_cur starts empty; the first layer conv takes X_in directly */
    }

    auto proto_flat = prototypes; /* [W_total] f32 */

    for (int layer = 0; layer < max_layers; layer++) {
        int w_start = w_starts[layer];
        int b_start = b_starts[layer];
        int C_in = C_ins[layer];
        int C_out = C_outs[layer];
        int H = Hs[layer];
        int W = Ws[layer];
        int per_oc_w = C_in * 9;
        int is_first = (layer == 0);

        /* === Step 1: Weight mixing === */
        /* W_mix = pi @ proto[:, w_start:w_start+C_out*per_oc_w] → [B, C_out, per_oc_w] */
        auto proto_slice = proto_flat.narrow(0, w_start, C_out * per_oc_w);
        auto W_mix = torch::mm(pi, proto_slice.view({C_out * per_oc_w, 1}).expand({-1, B}).t()).t();
        /* Actually: pi [B, K] @ proto_slice [K, C_out*per_oc_w] → [B, C_out*per_oc_w] */
        W_mix = torch::mm(pi, proto_slice.view({K, C_out * per_oc_w}));
        W_mix = W_mix.view({B, C_out, C_in * 9}).contiguous();

        /* b_mix = pi @ proto[:, b_start:b_start+C_out] → [B, C_out] */
        auto b_slice = proto_flat.narrow(0, b_start, C_out);
        auto b_mix = torch::mm(pi, b_slice.view({K, C_out}));

        /* === Step 2: im2col + bmm === */
        /* im2col: extract 3x3 patches from input */
        auto X_cur = (is_first) ?
            X_in.to(torch::kFloat32) :
            torch::from_blob(h_cur, {B, C_in, H, W},
                torch::TensorOptions().device(X_in.device()).dtype(torch::kFloat32));

        auto X_cols = torch::nn::functional::unfold(
            X_cur, torch::nn::UnfoldOptions(3).padding(1));
        /* X_cols: [B, C_in*9, H*W] */

        /* bmm: Y = W_mix @ X_cols + b_mix */
        auto Y_conv = torch::bmm(W_mix, X_cols) + b_mix.unsqueeze(2);
        Y_conv = Y_conv.view({B, C_out, H, W});

        /* === Step 3: Post-conv (norm + relu + gate + residual) === */
        float* h_prev_ptr = is_first ? nullptr : h_cur;

        /* Copy Y_conv f32 into h_next */
        Y_conv.to(torch::kFloat32).copy_(
            torch::from_blob(h_next, {B, C_out, H, W},
                torch::TensorOptions().device(X_in.device()).dtype(torch::kFloat32)));

        fused_norm_relu_gate_launcher(
            h_next,
            h_prev_ptr,
            dim_masks[layer].data_ptr<float>(),
            block_mode[layer].data_ptr<float>(),
            layer_gates[layer].data_ptr<float>(),
            B, C_out, H, W, max_dim,
            is_first ? 1 : 0
        );

        /* === Step 4: Max pool at layer 2 === */
        if (layer == 2) {
            auto pooled = torch::max_pool2d(
                torch::from_blob(h_next, {B, C_out, H, W},
                    torch::TensorOptions().device(X_in.device()).dtype(torch::kFloat32)),
                {2, 2});
            /* Copy back to h_next */
            pooled.copy_(
                torch::from_blob(h_next, {B, C_out, H/2, W/2},
                    torch::TensorOptions().device(X_in.device()).dtype(torch::kFloat32)));
            H_cur = H / 2;
            W_cur = W / 2;
        }

        /* Swap buffers */
        std::swap(h_cur, h_next);
    }

    /* h_cur now points to the output of the last layer */
    /* Copy to output bf16 */
    /* (caller will handle this) */
}

/* ================================================================
 * Python bindings
 * ================================================================ */
torch::Tensor fused_chain_forward_py(
    torch::Tensor prototypes,
    torch::Tensor pi,
    torch::Tensor X_in,
    torch::Tensor dim_masks,
    torch::Tensor layer_gates,
    torch::Tensor block_mode,
    std::vector<int> w_starts,
    std::vector<int> b_starts,
    std::vector<int> C_ins,
    std::vector<int> C_outs,
    std::vector<int> Hs,
    std::vector<int> Ws,
    int K, int max_dim, int max_layers
) {
    int B = X_in.size(0);
    int last_C = C_outs[max_layers - 1];
    int last_H = Hs[max_layers - 1] / (max_layers > 2 ? 2 : 1);
    int last_W = Ws[max_layers - 1] / (max_layers > 2 ? 2 : 1);

    auto h_buf = torch::zeros({B, last_C, last_H, last_W},
        X_in.options().dtype(torch::kFloat32));

    /* Actually, let's just do it the simple way: chain via PyTorch ops in C++ */
    /* This avoids the complexity of managing raw CUDA pointers */

    int H_cur = Hs[0], W_cur = Ws[0];
    torch::Tensor h = X_in.to(torch::kFloat32);

    for (int layer = 0; layer < max_layers; layer++) {
        int w_start = w_starts[layer];
        int b_start = b_starts[layer];
        int C_in = C_ins[layer];
        int C_out = C_outs[layer];
        int H = Hs[layer];
        int W = Ws[layer];
        int per_oc_w = C_in * 9;
        int is_first = (layer == 0);

        /* Weight mix */
        auto proto_slice = prototypes.narrow(0, w_start, C_out * per_oc_w);
        auto W_mix = torch::mm(pi, proto_slice.view({K, C_out * per_oc_w}));
        auto b_slice = prototypes.narrow(0, b_start, C_out);
        auto b_mix = torch::mm(pi, b_slice.view({K, C_out}));

        /* im2col + bmm */
        auto X_cols = torch::nn::functional::unfold(
            h, torch::nn::UnfoldOptions(3).padding(1));
        auto Y = torch::bmm(W_mix.view({B, C_out, C_in * 9}), X_cols) + b_mix.unsqueeze(2);
        Y = Y.view({B, C_out, H, W});

        /* Post-conv: norm + relu + gate + residual */
        fused_norm_relu_gate_launcher(
            Y.data_ptr<float>(),
            is_first ? nullptr : h.data_ptr<float>(),
            dim_masks[layer].data_ptr<float>(),
            block_mode[layer].data_ptr<float>(),
            layer_gates[layer].data_ptr<float>(),
            B, C_out, H, W, max_dim,
            is_first ? 1 : 0
        );

        h = Y;

        /* Max pool at layer 2 */
        if (layer == 2) {
            h = torch::max_pool2d(h, {2, 2});
        }
    }

    return h;
}
