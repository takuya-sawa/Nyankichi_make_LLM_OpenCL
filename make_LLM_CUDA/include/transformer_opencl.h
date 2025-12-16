#ifndef TRANSFORMER_OPENCL_H
#define TRANSFORMER_OPENCL_H

#include "tensor_opencl.h"
#include <vector>

class TransformerLayer
{
private:
    int hidden_dim;
    int num_heads;
    float learning_rate;
    int layer_id = -1; // optional id used for logging

    Tensor W_q, W_k, W_v, W_o;
    Tensor b_q, b_k, b_v, b_o;
    Tensor W_ff1, W_ff2;
    Tensor b_ff1, b_ff2;
    Tensor ln1_gamma, ln1_beta;
    Tensor ln2_gamma, ln2_beta;

    // Activation cache for backward
    struct Cache {
        Tensor input; // cached input x (seq_len x hidden_dim)
        Tensor Q, K, V, K_T;
        Tensor attn_scores, attn_output, out_proj;
        Tensor ff_hidden, ff_out;
        Tensor norm1_out, norm2_out;
    } cache;
    bool cache_enabled = false;

public:
    TransformerLayer(int hidden_dim_, int num_heads_, float lr = 0.001f);
    ~TransformerLayer();
    Tensor Forward(Tensor& x);

    // Enable or disable caching of intermediate activations (for backprop)
    void EnableCache(bool enable) { cache_enabled = enable; if (!enable) cache = Cache(); }

    // Backward: take gradient wrt layer output (same shape as output) and return gradient wrt input x
    // This is an interface implementation; full backward math to be implemented later.
    Tensor Backward(const Tensor& grad_output);

    // For testing: compute gradient of W_o for loss = sum(out_proj)
    Tensor ComputeAttentionGradWo();

    // Accessor for W_o (used by tests to perturb weights)
    Tensor& GetWo() { return W_o; }
    Tensor& GetWq() { return W_q; }
    Tensor& GetWk() { return W_k; }
    Tensor& GetWv() { return W_v; }

    // Cached accessors for tests
    const Tensor& GetCachedQ() const { return cache.Q; }
    const Tensor& GetCachedK() const { return cache.K; }
    const Tensor& GetCachedAttnOutput() const { return cache.attn_output; }

    // Layer id accessors for logging
    void SetLayerId(int id) { layer_id = id; }
    int GetLayerId() const { return layer_id; }
    
    // Compute analytical gradients for W_q, W_k, W_v for simple losses:
    // - For W_q/W_k: loss = sum(scale * Q @ K^T) (i.e., sum of attention scores S before softmax)
    // - For W_v: loss = sum(attn_output)
    Tensor ComputeGradWq_SumS();
    Tensor ComputeGradWk_SumS();
    Tensor ComputeGradWv_SumAttnOutput();
};

// Helper linear backward and optimizer (simple SGD)
void LinearBackward(const Tensor& input, const Tensor& grad_output, Tensor& grad_input, Tensor& grad_W, Tensor& grad_b, const Tensor& W);
void SGDUpdate(Tensor& param, const Tensor& grad, float lr);

// NOTE: EnableCacheAll is declared on the full TinyLLM definition below.

class TinyLLM
{
private:
    int vocab_size;
    int hidden_dim;
    int num_layers;
    int seq_length;
    float learning_rate;

    Tensor embeddings;
    std::vector<TransformerLayer*> layers;
    Tensor output_weight;

public:
    TinyLLM(int vocab_size_, int hidden_dim_, int num_layers_ = 2, int seq_length_ = 16, float lr = 0.001f);
    ~TinyLLM();

    Tensor Forward(const std::vector<int>& token_ids);
    float TrainStep(const std::vector<int>& token_ids, int target_id);
    int Predict(const std::vector<int>& token_ids);

    void SaveModel(const char* filepath);
    static TinyLLM* LoadModel(const char* filepath);

    // Enable/disable caching in all layers (useful for training/backprop)
    void EnableCacheAll(bool enable);

    int GetVocabSize() const { return vocab_size; }
};

// Global verbosity control for training logs (0=off, higher -> more verbose)
extern int g_verbosity;
void SetVerbosity(int v);

#endif // TRANSFORMER_OPENCL_H