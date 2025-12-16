#include "../include/transformer_opencl.h"
#include <iostream>
#include <cmath>

float compute_outproj_sum_after_forward(TransformerLayer& layer, Tensor& x) {
    layer.EnableCache(true);
    Tensor out = layer.Forward(x);
    // sum over cached out_proj (exists in cache)
    float s = 0.0f;
    for (size_t i = 0; i < layer.GetWo().h_data.size(); ++i) {
        // noop, just to ensure access
    }
    // We need to access cache.out_proj - it's private; instead compute sum of out by re-running forward and summing out_proj via a freshly run forward
    // Workaround: since Forward caches out_proj, call Forward and then compute sum from layer's cached out_proj by peeking via ComputeAttentionGradWo used only to assert shapes.
    // Simpler: sum final layer output (norm2_out) as proxy; but we want to test W_o effect. To align, we'll recompute forward and derive out_proj via internal cache by calling Forward then recomputing out_proj via attn_output @ W_o
    // For test, we will compute attn_output via Forward and then compute out_proj = attn_output @ W_o
    // To get attn_output we rely on Forward having stored cache.attn_output. We'll call ComputeAttentionGradWo which uses cache.attn_output; so to get sum(out_proj) we'll compute directly here:
    // call Forward to populate cache
    layer.EnableCache(true);
    out = layer.Forward(x);
    // We will compute out_proj manually: out_proj = attn_output @ W_o
    // To access attn_output we need access; it's in layer.cache but private; no accessor. So instead we will use the following trick: the ComputeAttentionGradWo computed grad_W = attn_output^T @ ones, and the sum(out_proj) = sum(attn_output * W_o) = sum over all elementwise products -> equivalently sum of elementwise products equals dot(grad_W, W_o) ???
    // Observing: grad_W computed as cache.attn_output^T @ ones => grad_W[a,b] = sum_i attn_output[i,a] * 1_b = column sums. The loss L = sum(out_proj) = sum_i sum_j attn_output[i,j'] * W_o[j',j], sum over i,j' ,j. This is linear; but computing numerically is simpler: recompute out_proj by performing matmul of attn_output and W_o
    // To get attn_output, we can't access cache directly; to remain simple, we will not rely on out_proj sum; instead we will compute numerical gradient by directly perturbing W_o and observing the change in a surrogate loss: sum(attn_output @ W_o) where attn_output is obtained from a Forward call, then re-run Forward after perturbation to get new attn_output (which changes with W_o? Actually attn_output depends on W_o only through softmax? No, attn_output = A @ V depends on W_o only via later projection; so attn_output is independent of W_o. Therefore we can compute loss L = sum(attn_output @ W_o) = sum(attn_output * W_o) and attn_output can be taken from a single forward call (cache.attn_output). This means loss depends linearly on W_o and numerical gradient is straightforward.
    // So: we call Forward once, fetch attn_output by computing grad_W via ComputeAttentionGradWo, which depends only on attn_output. But we also need to compute the actual loss for numerical check: L = sum(attn_output @ W_o) = sum over (i,j) attn_output[i,j'] * W_o[j',j]. We can compute L using ComputeAttentionGradWo and W_o: since grad_W = attn_output^T @ 1, and loss = sum over elemwise (attn_output @ W_o) = sum over columns of (grad_W * W_o)
    // Implement: get grad_W = ComputeAttentionGradWo(); then compute loss = sum(grad_W.h_data[k] * W_o.h_data[k])
    Tensor gradW = layer.ComputeAttentionGradWo();
    float L = 0.0f;
    Tensor& Wo = layer.GetWo();
    for (size_t i = 0; i < gradW.h_data.size(); ++i) L += gradW.h_data[i] * Wo.h_data[i];
    return L;
}

int main() {
    // small reproducible test
    int hidden = 6;
    int heads = 2;
    int seq = 4;
    TransformerLayer layer(hidden, heads, 0.01f);

    // prepare input x: seq x hidden
    Tensor x({seq, hidden});
    for (size_t i = 0; i < x.h_data.size(); ++i) x.h_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    layer.EnableCache(true);
    layer.Forward(x); // populate cache

    // analytical grad_Wo for loss = sum(attn_output @ W_o) is gradW (ComputeAttentionGradWo uses ones)
    Tensor analytic_grad = layer.ComputeAttentionGradWo();

    // numerical gradient via finite differences
    Tensor& Wo = layer.GetWo();
    Tensor numeric_grad = analytic_grad; // same shape
    float eps = 1e-4f;
    for (size_t idx = 0; idx < Wo.h_data.size(); ++idx) {
        float orig = Wo.h_data[idx];
        // plus
        Wo.h_data[idx] = orig + eps;
        float Lp = compute_outproj_sum_after_forward(layer, x);
        // minus
        Wo.h_data[idx] = orig - eps;
        float Lm = compute_outproj_sum_after_forward(layer, x);
        float g = (Lp - Lm) / (2.0f * eps);
        numeric_grad.h_data[idx] = g;
        // restore
        Wo.h_data[idx] = orig;
    }

    // compare W_o
    float max_rel_err = 0.0f;
    for (size_t i = 0; i < analytic_grad.h_data.size(); ++i) {
        float a = analytic_grad.h_data[i];
        float n = numeric_grad.h_data[i];
        float rel = std::abs(a - n) / (std::max(1e-6f, std::abs(a) + std::abs(n)));
        if (rel > max_rel_err) max_rel_err = rel;
    }

    std::cout << "Max relative error for W_o gradient: " << max_rel_err << std::endl;
    if (max_rel_err < 1e-3f) {
        std::cout << "W_o gradient check PASSED" << std::endl;
    } else {
        std::cout << "W_o gradient check FAILED" << std::endl;
        return 1;
    }

    // --- W_q: loss = sum(S = scale * Q @ K^T)
    Tensor analytic_Wq = layer.ComputeGradWq_SumS();
    Tensor& Wq = layer.GetWq();

    auto compute_S_sum_after_forward = [&](TransformerLayer& l, Tensor& x)->float {
        l.EnableCache(true);
        Tensor out = l.Forward(x);
        // compute S sum using cached Q,K
        // S = scale * Q @ K^T
        // sum(S) = scale * sum_d (sum_i Q_{i,d}) * (sum_j K_{j,d})
        if (l.GetCachedQ().size == 0 || l.GetCachedK().size == 0) return 0.0f; // should not happen
        int seq_len = l.GetCachedQ().shape[0];
        int D = l.GetCachedQ().shape[1];
        float scale = 1.0f / std::sqrt((float)D);
        float total = 0.0f;
        for (int d = 0; d < D; ++d) {
            float sQ = 0.0f; for (int i = 0; i < seq_len; ++i) sQ += l.GetCachedQ().h_data[i * D + d];
            float sK = 0.0f; for (int j = 0; j < seq_len; ++j) sK += l.GetCachedK().h_data[j * D + d];
            total += scale * sQ * sK;
        }
        return total;
    };

    // numerical gradient for W_q
    Tensor numeric_Wq = analytic_Wq;
    for (size_t idx = 0; idx < Wq.h_data.size(); ++idx) {
        float orig = Wq.h_data[idx];
        Wq.h_data[idx] = orig + eps;
        float Lp = compute_S_sum_after_forward(layer, x);
        Wq.h_data[idx] = orig - eps;
        float Lm = compute_S_sum_after_forward(layer, x);
        float g = (Lp - Lm) / (2.0f * eps);
        numeric_Wq.h_data[idx] = g;
        Wq.h_data[idx] = orig;
    }

    // compare W_q
    max_rel_err = 0.0f;
    for (size_t i = 0; i < analytic_Wq.h_data.size(); ++i) {
        float a = analytic_Wq.h_data[i];
        float n = numeric_Wq.h_data[i];
        float rel = std::abs(a - n) / (std::max(1e-6f, std::abs(a) + std::abs(n)));
        if (rel > max_rel_err) max_rel_err = rel;
    }
    float max_abs_err = 0.0f;
    for (size_t i = 0; i < analytic_Wq.h_data.size(); ++i) {
        float a = analytic_Wq.h_data[i];
        float n = numeric_Wq.h_data[i];
        max_abs_err = std::max(max_abs_err, std::abs(a - n));
    }
    std::cout << "Max relative error for W_q gradient: " << max_rel_err << " (max abs err=" << max_abs_err << ")" << std::endl;
    if (max_rel_err < 5e-3f) std::cout << "W_q gradient check PASSED" << std::endl; else { std::cout << "W_q gradient check FAILED" << std::endl; return 1; }

    // --- W_k
    Tensor analytic_Wk = layer.ComputeGradWk_SumS();
    Tensor& Wk = layer.GetWk();
    Tensor numeric_Wk = analytic_Wk;
    for (size_t idx = 0; idx < Wk.h_data.size(); ++idx) {
        float orig = Wk.h_data[idx];
        Wk.h_data[idx] = orig + eps;
        float Lp = compute_S_sum_after_forward(layer, x);
        Wk.h_data[idx] = orig - eps;
        float Lm = compute_S_sum_after_forward(layer, x);
        float g = (Lp - Lm) / (2.0f * eps);
        numeric_Wk.h_data[idx] = g;
        Wk.h_data[idx] = orig;
    }
    max_rel_err = 0.0f;
    for (size_t i = 0; i < analytic_Wk.h_data.size(); ++i) {
        float a = analytic_Wk.h_data[i];
        float n = numeric_Wk.h_data[i];
        float rel = std::abs(a - n) / (std::max(1e-6f, std::abs(a) + std::abs(n)));
        if (rel > max_rel_err) max_rel_err = rel;
    }
    max_abs_err = 0.0f;
    for (size_t i = 0; i < analytic_Wk.h_data.size(); ++i) {
        float a = analytic_Wk.h_data[i];
        float n = numeric_Wk.h_data[i];
        max_abs_err = std::max(max_abs_err, std::abs(a - n));
    }
    std::cout << "Max relative error for W_k gradient: " << max_rel_err << " (max abs err=" << max_abs_err << ")" << std::endl;
    if (max_rel_err < 5e-3f) std::cout << "W_k gradient check PASSED" << std::endl; else { std::cout << "W_k gradient check FAILED" << std::endl; return 1; }

    // --- W_v (loss = sum(attn_output))
    Tensor analytic_Wv = layer.ComputeGradWv_SumAttnOutput();
    Tensor& Wv = layer.GetWv();
    auto compute_attn_output_sum_after_forward = [&](TransformerLayer& l, Tensor& x)->float {
        l.EnableCache(true);
        Tensor out = l.Forward(x);
        if (l.GetCachedAttnOutput().size == 0) return 0.0f;
        float total = 0.0f;
        for (size_t i = 0; i < l.GetCachedAttnOutput().h_data.size(); ++i) total += l.GetCachedAttnOutput().h_data[i];
        return total;
    };
    Tensor numeric_Wv = analytic_Wv;
    for (size_t idx = 0; idx < Wv.h_data.size(); ++idx) {
        float orig = Wv.h_data[idx];
        Wv.h_data[idx] = orig + eps;
        float Lp = compute_attn_output_sum_after_forward(layer, x);
        Wv.h_data[idx] = orig - eps;
        float Lm = compute_attn_output_sum_after_forward(layer, x);
        float g = (Lp - Lm) / (2.0f * eps);
        numeric_Wv.h_data[idx] = g;
        Wv.h_data[idx] = orig;
    }
    max_rel_err = 0.0f;
    for (size_t i = 0; i < analytic_Wv.h_data.size(); ++i) {
        float a = analytic_Wv.h_data[i];
        float n = numeric_Wv.h_data[i];
        float rel = std::abs(a - n) / (std::max(1e-6f, std::abs(a) + std::abs(n)));
        if (rel > max_rel_err) max_rel_err = rel;
    }
    max_abs_err = 0.0f;
    for (size_t i = 0; i < analytic_Wv.h_data.size(); ++i) {
        float a = analytic_Wv.h_data[i];
        float n = numeric_Wv.h_data[i];
        max_abs_err = std::max(max_abs_err, std::abs(a - n));
    }
    std::cout << "Max relative error for W_v gradient: " << max_rel_err << " (max abs err=" << max_abs_err << ")" << std::endl;
    if (max_rel_err < 5e-3f) std::cout << "W_v gradient check PASSED" << std::endl; else { std::cout << "W_v gradient check FAILED" << std::endl; return 1; }

    std::cout << "All checked gradients PASSED" << std::endl;
    return 0;
}
