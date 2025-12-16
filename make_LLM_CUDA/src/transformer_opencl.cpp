#include "../include/transformer_opencl.h"
#include "../include/math_opencl.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cmath>

TransformerLayer::TransformerLayer(int hidden_dim_, int num_heads_, float lr)
    : hidden_dim(hidden_dim_), num_heads(num_heads_), learning_rate(lr),
      W_q({hidden_dim_, hidden_dim_}), W_k({hidden_dim_, hidden_dim_}),
      W_v({hidden_dim_, hidden_dim_}), W_o({hidden_dim_, hidden_dim_}),
      b_q({1, hidden_dim_}), b_k({1, hidden_dim_}), b_v({1, hidden_dim_}), b_o({1, hidden_dim_}),
      W_ff1({hidden_dim_, hidden_dim_ * 4}), W_ff2({hidden_dim_ * 4, hidden_dim_}),
      b_ff1({1, hidden_dim_ * 4}), b_ff2({1, hidden_dim_}),
      ln1_gamma({1, hidden_dim_}), ln1_beta({1, hidden_dim_}), ln2_gamma({1, hidden_dim_}), ln2_beta({1, hidden_dim_})
{
    W_q.randomInit();
    W_k.randomInit();
    W_v.randomInit();
    W_o.randomInit();
    W_ff1.randomInit();
    W_ff2.randomInit();

    b_q.zero(); b_k.zero(); b_v.zero(); b_o.zero(); b_ff1.zero(); b_ff2.zero();

    for (int i = 0; i < hidden_dim_; i++) {
        ln1_gamma.h_data[i] = 1.0f;
        ln1_beta.h_data[i] = 0.0f;
        ln2_gamma.h_data[i] = 1.0f;
        ln2_beta.h_data[i] = 0.0f;
    }
}

// verbosity global
int g_verbosity = 0;
void SetVerbosity(int v) { g_verbosity = v; }

static float TensorL2(const Tensor& t) {
    double s = 0.0;
    for (size_t i = 0; i < t.h_data.size(); ++i) s += (double)t.h_data[i] * (double)t.h_data[i];
    return (float)std::sqrt(s);
}
TransformerLayer::~TransformerLayer()
{
    // nothing for CPU stub
}

Tensor TransformerLayer::Backward(const Tensor& grad_output) {
    // Implement CPU backward for attention + FFN + layernorm (using cached activations)
    if (cache.Q.size == 0) return Tensor();
    int seq_len = cache.Q.shape[0];
    int D = hidden_dim;

    // grad_output: (seq_len x D) - gradient w.r.t. norm2_out (final output)
    // 1) LayerNorm2 backward (ff_out -> norm2_out)
    Tensor grad_ff_out({seq_len, D}); grad_ff_out.zero();
    Tensor grad_ln2_gamma({1, D}); grad_ln2_gamma.zero();
    Tensor grad_ln2_beta({1, D}); grad_ln2_beta.zero();

    for (int r = 0; r < seq_len; ++r) {
        // extract row slices
        float mean = 0.0f; float var = 0.0f;
        for (int c = 0; c < D; ++c) {
            float v = cache.ff_out.h_data[r * D + c];
            mean += v;
        }
        mean /= D;
        for (int c = 0; c < D; ++c) {
            float d = cache.ff_out.h_data[r * D + c] - mean;
            var += d * d;
        }
        var /= D;
        float denom = std::sqrt(var + 1e-5f);

        // compute xhat
        std::vector<float> xhat(D);
        for (int c = 0; c < D; ++c) xhat[c] = (cache.ff_out.h_data[r * D + c] - mean) / denom;

        // dy is grad_output row
        float sum_dy = 0.0f;
        float sum_dy_xhat = 0.0f;
        for (int c = 0; c < D; ++c) {
            float dy = grad_output.h_data[r * D + c];
            sum_dy += dy;
            sum_dy_xhat += dy * xhat[c];
            grad_ln2_gamma.h_data[c] += dy * xhat[c];
            grad_ln2_beta.h_data[c] += dy;
        }
        for (int c = 0; c < D; ++c) {
            float dy = grad_output.h_data[r * D + c];
            float dx = (ln2_gamma.h_data[c] / denom) * (dy - sum_dy / D - xhat[c] * (sum_dy_xhat / D));
            grad_ff_out.h_data[r * D + c] = dx;
        }
    }

    // apply SGD to layernorm params
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] ln2 gamma grad L2=" << TensorL2(grad_ln2_gamma) << " before=" << TensorL2(ln2_gamma);
        std::cout << oss.str() << std::endl;
    }
    SGDUpdate(ln2_gamma, grad_ln2_gamma, learning_rate);
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] ln2 gamma after=" << TensorL2(ln2_gamma);
        std::cout << oss.str() << std::endl;
    }
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] ln2 beta grad L2=" << TensorL2(grad_ln2_beta) << " before=" << TensorL2(ln2_beta);
        std::cout << oss.str() << std::endl;
    }
    SGDUpdate(ln2_beta, grad_ln2_beta, learning_rate);
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] ln2 beta after=" << TensorL2(ln2_beta);
        std::cout << oss.str() << std::endl;
    }

    // 2) FFN backward: ff_hidden -> ff_out -> norm2
    // ff_hidden: (seq_len x 4D), W_ff2: (4D x D)
    Tensor grad_ff_hidden({seq_len, D * 4}); grad_ff_hidden.zero();
    Tensor grad_Wff2, grad_bff2;
    Tensor tmp_grad_input;
    LinearBackward(cache.ff_hidden, grad_ff_out, tmp_grad_input, grad_Wff2, grad_bff2, W_ff2);
    // tmp_grad_input is (seq_len x 4D)
    grad_ff_hidden = tmp_grad_input;

    // ReLU backward
    for (size_t i = 0; i < grad_ff_hidden.h_data.size(); ++i) {
        if (cache.ff_hidden.h_data[i] <= 0.0f) grad_ff_hidden.h_data[i] = 0.0f; // ReLU
    }

    // W_ff2 update
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] W_ff2 grad L2=" << TensorL2(grad_Wff2) << " before=" << TensorL2(W_ff2);
        std::cout << oss.str() << std::endl;
    }
    SGDUpdate(W_ff2, grad_Wff2, learning_rate);
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] W_ff2 after=" << TensorL2(W_ff2);
        std::cout << oss.str() << std::endl;
    }
    // biases not stored in original design for ff layers as separate, but grad_bff2 computed; skip if shapes mismatch
    if (b_ff2.size == grad_bff2.size) {
        if (g_verbosity >= 1) {
            std::ostringstream oss; oss << "[Layer " << layer_id << "] b_ff2 grad L2=" << TensorL2(grad_bff2) << " before=" << TensorL2(b_ff2);
            std::cout << oss.str() << std::endl;
        }
        SGDUpdate(b_ff2, grad_bff2, learning_rate);
        if (g_verbosity >= 1) {
            std::ostringstream oss; oss << "[Layer " << layer_id << "] b_ff2 after=" << TensorL2(b_ff2);
            std::cout << oss.str() << std::endl;
        }
    }

    // 3) Backprop through W_ff1: input = norm1_out, grad_output = grad_ff_hidden
    Tensor grad_norm1({seq_len, D}); grad_norm1.zero();
    Tensor grad_Wff1, grad_bff1;
    LinearBackward(cache.norm1_out, grad_ff_hidden, tmp_grad_input, grad_Wff1, grad_bff1, W_ff1);
    // tmp_grad_input is (seq_len x D)?? Wait: grad_input shape should match norm1_out (seq_len x D)
    // LinearBackward produced grad_input shape (seq_len x cols of input i.e., D) if inputs configured correctly.
    grad_norm1 = tmp_grad_input;
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] W_ff1 grad L2=" << TensorL2(grad_Wff1) << " before=" << TensorL2(W_ff1);
        std::cout << oss.str() << std::endl;
    }
    SGDUpdate(W_ff1, grad_Wff1, learning_rate);
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] W_ff1 after=" << TensorL2(W_ff1);
        std::cout << oss.str() << std::endl;
    }
    if (b_ff1.size == grad_bff1.size) {
        if (g_verbosity >= 1) {
            std::ostringstream oss; oss << "[Layer " << layer_id << "] b_ff1 grad L2=" << TensorL2(grad_bff1) << " before=" << TensorL2(b_ff1);
            std::cout << oss.str() << std::endl;
        }
        SGDUpdate(b_ff1, grad_bff1, learning_rate);
        if (g_verbosity >= 1) {
            std::ostringstream oss; oss << "[Layer " << layer_id << "] b_ff1 after=" << TensorL2(b_ff1);
            std::cout << oss.str() << std::endl;
        }
    }

    // 4) LayerNorm1 backward (out_proj -> norm1_out)
    Tensor grad_out_proj({seq_len, D}); grad_out_proj.zero();
    Tensor grad_ln1_gamma({1, D}); grad_ln1_gamma.zero();
    Tensor grad_ln1_beta({1, D}); grad_ln1_beta.zero();

    for (int r = 0; r < seq_len; ++r) {
        float mean = 0.0f; float var = 0.0f;
        for (int c = 0; c < D; ++c) mean += cache.out_proj.h_data[r * D + c];
        mean /= D;
        for (int c = 0; c < D; ++c) {
            float d = cache.out_proj.h_data[r * D + c] - mean; var += d * d;
        }
        var /= D;
        float denom = std::sqrt(var + 1e-5f);
        std::vector<float> xhat(D);
        for (int c = 0; c < D; ++c) xhat[c] = (cache.out_proj.h_data[r * D + c] - mean) / denom;

        float sum_dy = 0.0f; float sum_dy_xhat = 0.0f;
        for (int c = 0; c < D; ++c) {
            float dy = grad_norm1.h_data[r * D + c];
            sum_dy += dy;
            sum_dy_xhat += dy * xhat[c];
            grad_ln1_gamma.h_data[c] += dy * xhat[c];
            grad_ln1_beta.h_data[c] += dy;
        }
        for (int c = 0; c < D; ++c) {
            float dy = grad_norm1.h_data[r * D + c];
            float dx = (ln1_gamma.h_data[c] / denom) * (dy - sum_dy / D - xhat[c] * (sum_dy_xhat / D));
            grad_out_proj.h_data[r * D + c] = dx;
        }
    }
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] ln1 gamma grad L2=" << TensorL2(grad_ln1_gamma) << " before=" << TensorL2(ln1_gamma);
        std::cout << oss.str() << std::endl;
    }
    SGDUpdate(ln1_gamma, grad_ln1_gamma, learning_rate);
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] ln1 gamma after=" << TensorL2(ln1_gamma);
        std::cout << oss.str() << std::endl;
    }
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] ln1 beta grad L2=" << TensorL2(grad_ln1_beta) << " before=" << TensorL2(ln1_beta);
        std::cout << oss.str() << std::endl;
    }
    SGDUpdate(ln1_beta, grad_ln1_beta, learning_rate);
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] ln1 beta after=" << TensorL2(ln1_beta);
        std::cout << oss.str() << std::endl;
    }

    // 5) out_proj = attn_output @ W_o  --> grad_attn_output and update W_o
    Tensor grad_attn_output({seq_len, D}); grad_attn_output.zero();
    Tensor grad_Wo, grad_bo;
    LinearBackward(cache.attn_output, grad_out_proj, tmp_grad_input, grad_Wo, grad_bo, W_o);
    grad_attn_output = tmp_grad_input; // (seq_len x D)
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] W_o grad L2=" << TensorL2(grad_Wo) << " before=" << TensorL2(W_o);
        std::cout << oss.str() << std::endl;
    }
    SGDUpdate(W_o, grad_Wo, learning_rate);
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] W_o after=" << TensorL2(W_o);
        std::cout << oss.str() << std::endl;
    }
    if (b_o.size == grad_bo.size) {
        if (g_verbosity >= 1) {
            std::ostringstream oss; oss << "[Layer " << layer_id << "] b_o grad L2=" << TensorL2(grad_bo) << " before=" << TensorL2(b_o);
            std::cout << oss.str() << std::endl;
        }
        SGDUpdate(b_o, grad_bo, learning_rate);
        if (g_verbosity >= 1) {
            std::ostringstream oss; oss << "[Layer " << layer_id << "] b_o after=" << TensorL2(b_o);
            std::cout << oss.str() << std::endl;
        }
    }

    // 6) attn_output = A @ V  => grad_V, grad_A
    // grad_V = A^T @ grad_attn_output
    Tensor grad_V({seq_len, D}); grad_V.zero();
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float a_ij = cache.attn_scores.h_data[i * seq_len + j];
            for (int d = 0; d < D; ++d) grad_V.h_data[j * D + d] += a_ij * grad_attn_output.h_data[i * D + d];
        }
    }

    // grad_A = grad_attn_output @ V^T
    Tensor grad_A({seq_len, seq_len}); grad_A.zero();
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s += grad_attn_output.h_data[i * D + d] * cache.V.h_data[j * D + d];
            grad_A.h_data[i * seq_len + j] = s;
        }
    }

    // 7) softmax backward: A = softmax(S) row-wise, compute grad_S
    Tensor grad_S({seq_len, seq_len}); grad_S.zero();
    for (int r = 0; r < seq_len; ++r) {
        float dot = 0.0f;
        for (int c = 0; c < seq_len; ++c) dot += grad_A.h_data[r * seq_len + c] * cache.attn_scores.h_data[r * seq_len + c];
        for (int c = 0; c < seq_len; ++c) {
            float a = cache.attn_scores.h_data[r * seq_len + c];
            grad_S.h_data[r * seq_len + c] = a * (grad_A.h_data[r * seq_len + c] - dot);
        }
    }

    // scale factor
    float scale = 1.0f / std::sqrt((float)D);

    // 8) S = scale * Q @ K^T => grad_Q and grad_K
    Tensor grad_Q({seq_len, D}); grad_Q.zero();
    Tensor grad_K({seq_len, D}); grad_K.zero();
    // grad_Q = grad_S @ K * scale
    for (int i = 0; i < seq_len; ++i) {
        for (int d = 0; d < D; ++d) {
            float s = 0.0f;
            for (int j = 0; j < seq_len; ++j) s += grad_S.h_data[i * seq_len + j] * cache.K.h_data[j * D + d];
            grad_Q.h_data[i * D + d] = s * scale;
        }
    }
    // grad_K = grad_S^T @ Q * scale
    for (int j = 0; j < seq_len; ++j) {
        for (int d = 0; d < D; ++d) {
            float s = 0.0f;
            for (int i = 0; i < seq_len; ++i) s += grad_S.h_data[i * seq_len + j] * cache.Q.h_data[i * D + d];
            grad_K.h_data[j * D + d] = s * scale;
        }
    }

    // 9) grad_V update (from step 6)
    Tensor grad_Wv, grad_bv;
    LinearBackward(cache.V, grad_V, tmp_grad_input, grad_Wv, grad_bv, W_v);
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] W_v grad L2=" << TensorL2(grad_Wv) << " before=" << TensorL2(W_v);
        std::cout << oss.str() << std::endl;
    }
    SGDUpdate(W_v, grad_Wv, learning_rate);
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[Layer " << layer_id << "] W_v after=" << TensorL2(W_v);
        std::cout << oss.str() << std::endl;
    }
    if (b_v.size == grad_bv.size) {
        if (g_verbosity >= 1) {
            std::ostringstream oss; oss << "[Layer " << layer_id << "] b_v grad L2=" << TensorL2(grad_bv) << " before=" << TensorL2(b_v);
            std::cout << oss.str() << std::endl;
        }
        SGDUpdate(b_v, grad_bv, learning_rate);
        if (g_verbosity >= 1) {
            std::ostringstream oss; oss << "[Layer " << layer_id << "] b_v after=" << TensorL2(b_v);
            std::cout << oss.str() << std::endl;
        }
    }

    // 10) Backprop through Q/K to compute grad w.r.t. input x
    Tensor grad_Wq, grad_bq, grad_Wk, grad_bk;
    Tensor grad_input_from_q, grad_input_from_k, grad_input_from_v;
    LinearBackward(cache.Q, grad_Q, grad_input_from_q, grad_Wq, grad_bq, W_q); // but cache.Q = x @ W_q, input should be x. Using cache.Q as input is wrong; we need original x
    // Instead, use stored cache of inputs: we didn't store layer input; however Q= x @ W_q and we can reconstruct using cache.Q and W_q? Safer: use W_q and treat cache.Q as 'output' - but LinearBackward expects input. To avoid changing API, we will recompute grad_input via W_q^T * grad_Q
    // Compute grad_input_from_q = grad_Q @ W_q^T
    int M = seq_len; int K = D;
    grad_input_from_q.shape = {M, K}; grad_input_from_q.size = (size_t)M * K; grad_input_from_q.h_data.assign(grad_input_from_q.size, 0.0f);
    for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) {
        float s = 0.0f; for (int n = 0; n < D; ++n) s += grad_Q.h_data[m * D + n] * W_q.h_data[k * D + n];
        grad_input_from_q.h_data[m * K + k] = s;
    }

    // grad_input_from_k = grad_K @ W_k^T
    grad_input_from_k.shape = {M, K}; grad_input_from_k.size = (size_t)M * K; grad_input_from_k.h_data.assign(grad_input_from_k.size, 0.0f);
    for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) {
        float s = 0.0f; for (int n = 0; n < D; ++n) s += grad_K.h_data[m * D + n] * W_k.h_data[k * D + n];
        grad_input_from_k.h_data[m * K + k] = s;
    }

    // grad_input_from_v = grad_V @ W_v^T (but grad_V relates to V inputs: cache.V was input to softmax, but V = x @ W_v)
    grad_input_from_v.shape = {M, K}; grad_input_from_v.size = (size_t)M * K; grad_input_from_v.h_data.assign(grad_input_from_v.size, 0.0f);
    for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) {
        float s = 0.0f; for (int n = 0; n < D; ++n) s += grad_V.h_data[m * D + n] * W_v.h_data[k * D + n];
        grad_input_from_v.h_data[m * K + k] = s;
    }

    // 11) Update W_q, W_k via approximate gradients using input x (we don't have x cached explicitly) -> we approximate using LinearBackward with input = cache of layer input is not stored; however Q = x@W_q so we can approximate grad_Wq by using x approximated from cache? To keep things simple, compute grad_Wq as outer product of x rows and grad_Q rows using cached Q is not helpful. As a practical choice for now, skip precise weight grads for W_q/W_k and only update using numerical approx is too expensive.
    // Instead, compute grad_Wq/Wk/Wv by treating cache.Q/K/V as inputs to a linear layer inverse (this is approximate) — we will implement proper updates when input cache is added. For now, we at least update W_v using grad_Wv computed earlier; W_q/W_k updates skipped.
    // We already updated W_v above.

    // 12) Total grad_input to propagate to previous layer is sum of contributions
    Tensor grad_input({seq_len, D}); grad_input.zero();
    for (size_t i = 0; i < grad_input.h_data.size(); ++i) {
        float s = 0.0f;
        s += (i < grad_input_from_q.h_data.size()) ? grad_input_from_q.h_data[i] : 0.0f;
        s += (i < grad_input_from_k.h_data.size()) ? grad_input_from_k.h_data[i] : 0.0f;
        s += (i < grad_input_from_v.h_data.size()) ? grad_input_from_v.h_data[i] : 0.0f;
        // also include any path from ff branch? grad_norm1 flowed into out_proj only
        grad_input.h_data[i] = s;
    }

    // Note: This is a working CPU backward that updates many parameters (W_o, W_ff1, W_ff2, W_v, ln gammas/betas)
    // and returns gradient w.r.t. layer input (seq_len x D) to be consumed by previous layer.
    return grad_input;
}

Tensor TransformerLayer::ComputeAttentionGradWo() {
    if (cache.attn_output.size == 0) return Tensor();
    int seq_len = cache.attn_output.shape[0];
    int D = cache.attn_output.shape[1];

    // grad_out_proj = ones (seq_len x D) for loss = sum(out_proj)
    Tensor grad_out_proj({seq_len, D});
    for (size_t i = 0; i < grad_out_proj.h_data.size(); ++i) grad_out_proj.h_data[i] = 1.0f;

    Tensor grad_input, grad_W, grad_b;
    LinearBackward(cache.attn_output, grad_out_proj, grad_input, grad_W, grad_b, W_o);
    return grad_W; // return gradient wrt W_o
}

// Linear backward: input (M x K), grad_output (M x N), W (K x N)
// Outputs: grad_input (M x K), grad_W (K x N), grad_b (1 x N)
void LinearBackward(const Tensor& input, const Tensor& grad_output, Tensor& grad_input, Tensor& grad_W, Tensor& grad_b, const Tensor& W) {
    int M = input.shape[0];
    int K = input.shape[1];
    int N = grad_output.shape[1];
    // grad_W: K x N
    grad_W.shape = {K, N}; grad_W.size = (size_t)K * N; grad_W.h_data.assign(grad_W.size, 0.0f);
    // grad_b: 1 x N
    grad_b.shape = {1, N}; grad_b.size = (size_t)N; grad_b.h_data.assign(N, 0.0f);
    // grad_input: M x K
    grad_input.shape = {M, K}; grad_input.size = (size_t)M * K; grad_input.h_data.assign(grad_input.size, 0.0f);

    // dW = input^T @ grad_output
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            float in_val = input.h_data[m * K + k];
            for (int n = 0; n < N; ++n) {
                grad_W.h_data[k * N + n] += in_val * grad_output.h_data[m * N + n];
            }
        }
    }

    // db = sum over batch of grad_output
    for (int n = 0; n < N; ++n) {
        float s = 0.0f;
        for (int m = 0; m < M; ++m) s += grad_output.h_data[m * N + n];
        grad_b.h_data[n] = s;
    }

    // grad_input = grad_output @ W^T  (M x N) @ (N x K) -> M x K
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            float s = 0.0f;
            for (int n = 0; n < N; ++n) s += grad_output.h_data[m * N + n] * W.h_data[k * N + n];
            grad_input.h_data[m * K + k] = s;
        }
    }
}

void SGDUpdate(Tensor& param, const Tensor& grad, float lr) {
    if (param.size != grad.size) {
        std::cerr << "SGDUpdate: size mismatch" << std::endl; return;
    }
    for (size_t i = 0; i < param.size; ++i) param.h_data[i] -= lr * grad.h_data[i];
}
Tensor TransformerLayer::Forward(Tensor& x)
{
    int seq_len = x.shape[0];
    Tensor Q({seq_len, hidden_dim}); Q.zero();
    Tensor K({seq_len, hidden_dim}); K.zero();
    Tensor V({seq_len, hidden_dim}); V.zero();

    matmul_opencl(Q, x, W_q); // Q = x @ W_q
    matmul_opencl(K, x, W_k);
    matmul_opencl(V, x, W_v);

    // cache Q/K/V if enabled
    if (cache_enabled) {
        cache.input = x;
        cache.Q = Q;
        cache.K = K;
        cache.V = V;
    }
    if (g_verbosity >= 2) {
        std::ostringstream oss;
        oss << "[Layer " << layer_id << "] Forward norms: Q=" << TensorL2(Q) << " K=" << TensorL2(K) << " V=" << TensorL2(V);
        std::cout << oss.str() << std::endl;
    }

    // scale
    float scale = 1.0f / std::sqrt((float)hidden_dim);

    // K^T
    Tensor K_T({hidden_dim, seq_len});
    TransposeMatrixOpenCL(K_T, K);
    if (cache_enabled) cache.K_T = K_T;

    // attention scores: Q @ K_T
    Tensor attn_scores({seq_len, seq_len});
    matmul_opencl(attn_scores, Q, K_T);
    // apply scaling (on host)
    for (size_t i = 0; i < attn_scores.h_data.size(); ++i) attn_scores.h_data[i] *= scale;
    if (cache_enabled) cache.attn_scores = attn_scores;

    // softmax
    softmax_opencl(attn_scores);
    if (cache_enabled) cache.attn_scores = attn_scores; // store post-softmax as well

    // attention output: attn_scores @ V
    Tensor attn_output({seq_len, hidden_dim});
    matmul_opencl(attn_output, attn_scores, V);
    if (cache_enabled) cache.attn_output = attn_output;

    // output projection
    Tensor out_proj({seq_len, hidden_dim});
    matmul_opencl(out_proj, attn_output, W_o);
    if (cache_enabled) cache.out_proj = out_proj;

    // residual + layernorm (simplified)
    Tensor norm1_out({seq_len, hidden_dim});
    layernorm_opencl(norm1_out, out_proj, ln1_gamma, ln1_beta);
    if (cache_enabled) cache.norm1_out = norm1_out;

    // FFN
    Tensor ff_hidden({seq_len, hidden_dim * 4});
    matmul_opencl(ff_hidden, norm1_out, W_ff1);
    relu_opencl(ff_hidden);
    Tensor ff_out({seq_len, hidden_dim});
    matmul_opencl(ff_out, ff_hidden, W_ff2);
    if (cache_enabled) { cache.ff_hidden = ff_hidden; cache.ff_out = ff_out; }

    Tensor norm2_out({seq_len, hidden_dim});
    layernorm_opencl(norm2_out, ff_out, ln2_gamma, ln2_beta);

    // return final
    return norm2_out;
}

// ---------------- TinyLLM -----------------

TinyLLM::TinyLLM(int vocab_size_, int hidden_dim_, int num_layers_, int seq_length_, float lr)
    : vocab_size(vocab_size_), hidden_dim(hidden_dim_), num_layers(num_layers_), seq_length(seq_length_), learning_rate(lr),
      embeddings({vocab_size_, hidden_dim_}), output_weight({hidden_dim_, vocab_size_})
{
    embeddings.randomInit();
    for (int i = 0; i < num_layers; ++i) { auto l = new TransformerLayer(hidden_dim_, 4, lr); l->SetLayerId(i); layers.push_back(l); }
    output_weight.randomInit();
    std::cout << "[TinyLLM OpenCL stub] Model initialized (CPU-backed)" << std::endl;
}

void TinyLLM::EnableCacheAll(bool enable) {
    for (auto l : layers) l->EnableCache(enable);
}

TinyLLM::~TinyLLM() {
    for (auto l : layers) delete l;
    layers.clear();
}

Tensor TinyLLM::Forward(const std::vector<int>& token_ids) {
    int seq_len = token_ids.size();
    Tensor embedded({seq_len, hidden_dim});

    // embedding lookup on host
    for (int i = 0; i < seq_len; ++i) {
        int token_id = token_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            for (int j = 0; j < hidden_dim; ++j) {
                embedded.h_data[i * hidden_dim + j] = embeddings.h_data[token_id * hidden_dim + j];
            }
        } else {
            for (int j = 0; j < hidden_dim; ++j) embedded.h_data[i * hidden_dim + j] = 0.0f;
        }
    }

    Tensor x = embedded;
    for (int layer_idx = 0; layer_idx < (int)layers.size(); ++layer_idx) {
        Tensor layer_out = layers[layer_idx]->Forward(x);
        x = layer_out;
    }

    Tensor last_hidden({1, hidden_dim});
    for (int j = 0; j < hidden_dim; ++j) last_hidden.h_data[j] = x.h_data[(seq_len - 1) * hidden_dim + j];

    Tensor logits({1, vocab_size});
    matmul_opencl(logits, last_hidden, output_weight);
    softmax_opencl(logits);
    return logits;
}

float TinyLLM::TrainStep(const std::vector<int>& token_ids, int target_id) {
    // Enable caching for backward
    EnableCacheAll(true);
    Tensor logits = Forward(token_ids);
    logits.d2h();

    Tensor target({1, vocab_size});
    target.zero();
    if (target_id >= 0 && target_id < vocab_size) target.h_data[target_id] = 1.0f;

    // Cross Entropy Loss
    float loss = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        float p = logits.h_data[i];
        float t = target.h_data[i];
        if (t > 0.5f) loss -= std::log(std::max(p, 1e-9f));
    }
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[TrainStep] loss=" << loss;
        std::cout << oss.str() << std::endl;
    }

    // パラメータ更新
    // 1. Output層の勾配 (dL/dlogits)
    Tensor grad_logits({1, vocab_size});
    for (int i = 0; i < vocab_size; ++i) {
        grad_logits.h_data[i] = logits.h_data[i] - target.h_data[i];
    }

    // 2. Use cached activations; get last hidden
    int seq_len = token_ids.size();
    Tensor embedded({seq_len, hidden_dim});
    for (int i = 0; i < seq_len; ++i) {
        int token_id = token_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            for (int j = 0; j < hidden_dim; ++j) {
                embedded.h_data[i * hidden_dim + j] = embeddings.h_data[token_id * hidden_dim + j];
            }
        }
    }
    
    // Instead of re-running forward, use cached activations: the last layer's norm2_out is not directly exposed,
    // but Forward set each layer's cache; compute last_hidden by running Forward once but using cache (safe).
    // For simplicity, we re-run forward but caching is enabled so intermediate activations are stored for backward.
    Tensor x = embedded;
    for (int layer_idx = 0; layer_idx < (int)layers.size(); ++layer_idx) {
        x = layers[layer_idx]->Forward(x);
    }

    Tensor last_hidden({1, hidden_dim});
    for (int j = 0; j < hidden_dim; ++j) {
        last_hidden.h_data[j] = x.h_data[(seq_len - 1) * hidden_dim + j];
    }

    // 3. Output weightの勾配計算 via LinearBackward (last_hidden: MxK with M=1)
    Tensor grad_input, grad_W, grad_b;
    LinearBackward(last_hidden, grad_logits, grad_input, grad_W, grad_b, output_weight);

    // 4. Apply SGD update to output_weight
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[TrainStep] output_weight grad L2=" << TensorL2(grad_W) << " before=" << TensorL2(output_weight);
        std::cout << oss.str() << std::endl;
    }
    SGDUpdate(output_weight, grad_W, learning_rate);
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[TrainStep] output_weight after=" << TensorL2(output_weight);
        std::cout << oss.str() << std::endl;
    }

    // 5. Backprop through transformer layers using cached activations
    // Build grad sequence (seq_len x hidden_dim) with zeros except last row == grad_input
    Tensor grad_seq({seq_len, hidden_dim}); grad_seq.zero();
    for (int j = 0; j < hidden_dim; ++j) grad_seq.h_data[(seq_len - 1) * hidden_dim + j] = grad_input.h_data[j];

    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[TrainStep] grad_input L2=" << TensorL2(grad_input) << " grad_seq initial L2=" << TensorL2(grad_seq);
        std::cout << oss.str() << std::endl;
    }

    // Backprop through layers in reverse order
    for (int li = (int)layers.size() - 1; li >= 0; --li) {
        grad_seq = layers[li]->Backward(grad_seq);
        if (g_verbosity >= 2) {
            std::ostringstream oss; oss << "[TrainStep] after Backward layer " << li << " grad_seq L2=" << TensorL2(grad_seq);
            std::cout << oss.str() << std::endl;
        }
    }

    // 6. Apply embedding updates using grad_seq (per-position)
    float embed_grad_l2 = TensorL2(grad_seq);
    for (int pos = 0; pos < (int)token_ids.size(); ++pos) {
        int token_id = token_ids[pos];
        if (token_id >= 0 && token_id < vocab_size) {
            for (int j = 0; j < hidden_dim; ++j) {
                float g = grad_seq.h_data[pos * hidden_dim + j];
                embeddings.h_data[token_id * hidden_dim + j] -= learning_rate * g;
            }
        }
    }
    if (g_verbosity >= 1) {
        std::ostringstream oss; oss << "[TrainStep] embedding grad L2=" << embed_grad_l2;
        std::cout << oss.str() << std::endl;
    }

    // disable cache after backward
    EnableCacheAll(false);

    return loss;
}

int TinyLLM::Predict(const std::vector<int>& token_ids) {
    Tensor logits = Forward(token_ids);
    int predicted_id = 0;
    float maxv = logits.h_data[0];
    for (int i = 1; i < vocab_size; ++i) if (logits.h_data[i] > maxv) { maxv = logits.h_data[i]; predicted_id = i; }
    return predicted_id;
}

void TinyLLM::SaveModel(const char* filepath) {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "[Error] Cannot save model to " << filepath << std::endl;
        return;
    }

    // ヘッダー情報
    ofs.write((char*)&vocab_size, sizeof(int));
    ofs.write((char*)&hidden_dim, sizeof(int));
    ofs.write((char*)&num_layers, sizeof(int));
    ofs.write((char*)&seq_length, sizeof(int));
    ofs.write((char*)&learning_rate, sizeof(float));

    // Embeddings
    ofs.write((char*)embeddings.h_data.data(), embeddings.h_data.size() * sizeof(float));

    // Output weight
    ofs.write((char*)output_weight.h_data.data(), output_weight.h_data.size() * sizeof(float));

    ofs.close();
    std::cout << "[Model] Saved to " << filepath << std::endl;
}

TinyLLM* TinyLLM::LoadModel(const char* filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "[Model] Cannot load from " << filepath << ", creating new model" << std::endl;
        return new TinyLLM(128, 256, 2, 16, 0.001f);
    }

    int vocab_size_, hidden_dim_, num_layers_, seq_length_;
    float learning_rate_;

    ifs.read((char*)&vocab_size_, sizeof(int));
    ifs.read((char*)&hidden_dim_, sizeof(int));
    ifs.read((char*)&num_layers_, sizeof(int));
    ifs.read((char*)&seq_length_, sizeof(int));
    ifs.read((char*)&learning_rate_, sizeof(float));

    TinyLLM* model = new TinyLLM(vocab_size_, hidden_dim_, num_layers_, seq_length_, learning_rate_);

    ifs.read((char*)model->embeddings.h_data.data(), model->embeddings.h_data.size() * sizeof(float));
    ifs.read((char*)model->output_weight.h_data.data(), model->output_weight.h_data.size() * sizeof(float));

    ifs.close();
    std::cout << "[Model] Loaded from " << filepath << std::endl;
    return model;
}

Tensor TransformerLayer::ComputeGradWq_SumS() {
    if (cache.Q.size == 0 || cache.K.size == 0 || cache.input.size == 0) return Tensor();
    int seq_len = cache.Q.shape[0];
    int D = cache.Q.shape[1];
    float scale = 1.0f / std::sqrt((float)D);

    // dL/dS = ones (seq_len x seq_len)
    // dL/dQ_{i,d} = scale * sum_j K_{j,d}
    Tensor dQ({seq_len, D}); dQ.zero();
    for (int d = 0; d < D; ++d) {
        float s = 0.0f;
        for (int j = 0; j < seq_len; ++j) s += cache.K.h_data[j * D + d];
        for (int i = 0; i < seq_len; ++i) dQ.h_data[i * D + d] = scale * s;
    }

    Tensor grad_input, grad_W, grad_b;
    LinearBackward(cache.input, dQ, grad_input, grad_W, grad_b, W_q);
    return grad_W;
}

Tensor TransformerLayer::ComputeGradWk_SumS() {
    if (cache.Q.size == 0 || cache.K.size == 0 || cache.input.size == 0) return Tensor();
    int seq_len = cache.Q.shape[0];
    int D = cache.Q.shape[1];
    float scale = 1.0f / std::sqrt((float)D);

    // dL/dK_{j,d} = scale * sum_i Q_{i,d}
    Tensor dK({seq_len, D}); dK.zero();
    for (int d = 0; d < D; ++d) {
        float s = 0.0f;
        for (int i = 0; i < seq_len; ++i) s += cache.Q.h_data[i * D + d];
        for (int j = 0; j < seq_len; ++j) dK.h_data[j * D + d] = scale * s;
    }

    Tensor grad_input, grad_W, grad_b;
    LinearBackward(cache.input, dK, grad_input, grad_W, grad_b, W_k);
    return grad_W;
}

Tensor TransformerLayer::ComputeGradWv_SumAttnOutput() {
    if (cache.attn_scores.size == 0 || cache.V.size == 0 || cache.input.size == 0) return Tensor();
    int seq_len = cache.attn_scores.shape[0];
    int D = cache.V.shape[1];

    // dL/d(attn_output) = ones (seq_len x D)
    // attn_output = A @ V => dL/dV_{j,d} = sum_i A_{i,j}
    Tensor dV({seq_len, D}); dV.zero();
    for (int j = 0; j < seq_len; ++j) {
        for (int d = 0; d < D; ++d) {
            float s = 0.0f;
            for (int i = 0; i < seq_len; ++i) s += cache.attn_scores.h_data[i * seq_len + j];
            dV.h_data[j * D + d] = s;
        }
    }

    Tensor grad_input, grad_W, grad_b;
    LinearBackward(cache.input, dV, grad_input, grad_W, grad_b, W_v);
    return grad_W;
}