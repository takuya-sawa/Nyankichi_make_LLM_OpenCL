#include "transformer_cuda.h"
#include "math_cuda.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda_runtime.h>
#include <algorithm>

/// ===================================================================
/// ヘルパ�E関数
/// ===================================================================

/// <summary>
/// 行�E転置�E�GPU カーネル実裁E
/// src (m, n) ↁEdst (n, m)
/// </summary>
__global__ void kernel_transpose(const float* src, float* dst, int m, int n)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < m && j < n) {
        dst[j * m + i] = src[i * n + j];
    }
}

void TransposeMatrix(Tensor& dst, const Tensor& src)
{
    int m = src.shape[0];  // 行数
    int n = src.shape[1];  // 列数
    
    // CPU側で転置を実行（デバッグ用）
    // const_castを使ってGPU→CPU転送
    Tensor& src_mut = const_cast<Tensor&>(src);
    src_mut.d2h();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dst.h_data[j * m + i] = src.h_data[i * n + j];
        }
    }
    dst.h2d();
}

/// ===================================================================
/// TransformerLayer CUDA 実裁E
/// ===================================================================

TransformerLayer::TransformerLayer(int hidden_dim_, int num_heads_, float lr)
    : hidden_dim(hidden_dim_), num_heads(num_heads_), learning_rate(lr),
      W_q({hidden_dim_, hidden_dim_}), W_k({hidden_dim_, hidden_dim_}),
      W_v({hidden_dim_, hidden_dim_}), W_o({hidden_dim_, hidden_dim_}),
      b_q({hidden_dim_}), b_k({hidden_dim_}), b_v({hidden_dim_}), b_o({hidden_dim_}),
      W_ff1({hidden_dim_, hidden_dim_ * 4}), W_ff2({hidden_dim_ * 4, hidden_dim_}),
      b_ff1({hidden_dim_ * 4}), b_ff2({hidden_dim_}),
      ln1_gamma({1, hidden_dim_}), ln1_beta({1, hidden_dim_}),
      ln2_gamma({1, hidden_dim_}), ln2_beta({1, hidden_dim_})
{
    // 重みを初期化（Xavier/He初期化）
    W_q.randomInit();
    W_k.randomInit();
    W_v.randomInit();
    W_o.randomInit();
    W_ff1.randomInit();
    W_ff2.randomInit();
    
    // バイアスを初期化
    b_q.zero();
    b_k.zero();
    b_v.zero();
    b_o.zero();
    b_ff1.zero();
    b_ff2.zero();
    
    // Layer Normパラメータ初期化: gamma = 1.0, beta = 0.0
    for (int i = 0; i < hidden_dim_; i++) {
        ln1_gamma.h_data[i] = 1.0f;
        ln1_beta.h_data[i] = 0.0f;
        ln2_gamma.h_data[i] = 1.0f;
        ln2_beta.h_data[i] = 0.0f;
    }
    ln1_gamma.h2d();
    ln1_beta.h2d();
    ln2_gamma.h2d();
    ln2_beta.h2d();
    
    std::cout << "[Transformer] Layer initialized (Hidden dim: " << hidden_dim_ << ")" << std::endl;
}

TransformerLayer::~TransformerLayer()
{
    // GPU メモリは Tensor チE��トラクタで自動解放
}

Tensor TransformerLayer::Forward(const Tensor& x)
{
    int seq_len = x.shape[0];
    
    // ===== マルチ�EチE��自己注愁E=====
    
    // Q, K, V を計箁E
    Tensor Q({seq_len, hidden_dim});
    Tensor K({seq_len, hidden_dim});
    Tensor V({seq_len, hidden_dim});
    
    // チE��チE��: 入力�E確誁E
    static int layer_call_count = 0;
    if (layer_call_count == 0) {
        Tensor temp_x = x;
        temp_x.d2h();
        float x_max = *std::max_element(temp_x.h_data.begin(), temp_x.h_data.end());
        std::cout << "        [DEBUG] TransformerLayer input max: " << x_max << std::endl;
        
        W_q.d2h();
        float wq_max = *std::max_element(W_q.h_data.begin(), W_q.h_data.end());
        std::cout << "        [DEBUG] W_q max: " << wq_max << std::endl;
    }
    
    Q.zero();
    K.zero();
    V.zero();
    
    matmul_cuda(Q, x, W_q);
    matmul_cuda(K, x, W_k);
    matmul_cuda(V, x, W_v);
    
    if (layer_call_count == 0) {
        // matmul直後にKのGPUメモリを確誁E
        float k_gpu[5];
        cudaMemcpy(k_gpu, K.d_data, 5 * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "        [DEBUG] K.d_data[0:5] (direct from GPU after matmul): ";
        for (int ii = 0; ii < 5; ii++) {
            std::cout << k_gpu[ii] << " ";
        }
        std::cout << std::endl;
    }
    
    if (layer_call_count == 0) {
        Q.d2h();
        float q_max = *std::max_element(Q.h_data.begin(), Q.h_data.end());
        std::cout << "        [DEBUG] Q after matmul max: " << q_max << std::endl;
    }
    
    // スケーリング・ドット積注愁E
    float scale = 1.0f / sqrtf((float)hidden_dim);
    
    // K^T を計算（転置�E�E
    Tensor K_T({hidden_dim, seq_len});
    K_T.zero();
    
    if (layer_call_count == 0) {
        std::cout << "        [DEBUG] K shape: [" << K.shape[0] << ", " << K.shape[1] << "]" << std::endl;
        std::cout << "        [DEBUG] K_T shape: [" << K_T.shape[0] << ", " << K_T.shape[1] << "]" << std::endl;
        std::cout << "        [DEBUG] K.size=" << K.size << ", K_T.size=" << K_T.size << std::endl;
        std::cout << "        [DEBUG] K.d_data=" << (void*)K.d_data << ", K_T.d_data=" << (void*)K_T.d_data << std::endl;
    }
    
    TransposeMatrix(K_T, K);  // K を転置して K_T に
    
    if (layer_call_count == 0) {
        // Transpose直後にK_T.d_dataを直接確誁E
        float kt_gpu[5];
        cudaMemcpy(kt_gpu, K_T.d_data, 5 * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "        [DEBUG] K_T.d_data[0:5] (direct from GPU after transpose): ";
        for (int ii = 0; ii < 5; ii++) {
            std::cout << kt_gpu[ii] << " ";
        }
        std::cout << std::endl;
        
        K.d2h();
        float k_max = *std::max_element(K.h_data.begin(), K.h_data.end());
        std::cout << "        [DEBUG] K max: " << k_max << std::endl;
        std::cout << "        [DEBUG] K[0:5]: ";
        for (int ii = 0; ii < 5 && ii < K.h_data.size(); ii++) {
            std::cout << K.h_data[ii] << " ";
        }
        std::cout << std::endl;
        
        K_T.d2h();
        float kt_max = *std::max_element(K_T.h_data.begin(), K_T.h_data.end());
        std::cout << "        [DEBUG] K_T max: " << kt_max << std::endl;
        std::cout << "        [DEBUG] K_T[0:5]: ";
        for (int ii = 0; ii < 5 && ii < K_T.h_data.size(); ii++) {
            std::cout << K_T.h_data[ii] << " ";
        }
        std::cout << std::endl;
    }
    
    Tensor attn_scores({seq_len, seq_len});
    attn_scores.zero();
    matmul_cuda(attn_scores, Q, K_T);  // Q @ K^T を正しく計箁E    
    // スケーリングを適用
    attn_scores.d2h();
    for (int i = 0; i < attn_scores.size; i++) {
        attn_scores.h_data[i] *= scale;
    }
    attn_scores.h2d();    
    if (layer_call_count == 0) {
        attn_scores.d2h();
        float as_before_max = *std::max_element(attn_scores.h_data.begin(), attn_scores.h_data.end());
        std::cout << "        [DEBUG] attn_scores before softmax max: " << as_before_max << std::endl;
    }
    
    // ソフトマックス
    softmax_cuda(attn_scores);
    
    if (layer_call_count == 0) {
        attn_scores.d2h();
        float as_max = *std::max_element(attn_scores.h_data.begin(), attn_scores.h_data.end());
        std::cout << "        [DEBUG] attn_scores after softmax max: " << as_max << std::endl;
        
        // Attention重みの可視化（最初のトークンについて）
        std::cout << "        [DEBUG] Attention weights (token 0): ";
        for (int i = 0; i < std::min(5, seq_len); i++) {
            std::cout << attn_scores.h_data[i] << " ";
        }
        std::cout << std::endl;
        
        V.d2h();
        float v_max = *std::max_element(V.h_data.begin(), V.h_data.end());
        std::cout << "        [DEBUG] V max: " << v_max << std::endl;
    }
    
    // Attention output
    Tensor attn_output({seq_len, hidden_dim});
    attn_output.zero();
    matmul_cuda(attn_output, attn_scores, V);
    
    if (layer_call_count == 0) {
        attn_output.d2h();
        float attn_out_max = *std::max_element(attn_output.h_data.begin(), attn_output.h_data.end());
        std::cout << "        [DEBUG] attn_output max: " << attn_out_max << std::endl;
    }
    
    // 出力投影と残差接綁E
    Tensor out_proj({seq_len, hidden_dim});
    out_proj.zero();
    
    if (layer_call_count == 0) {
        attn_output.d2h();
        W_o.d2h();
        std::cout << "        [DEBUG] attn_output[0:3]: ";
        for (int ii = 0; ii < 3 && ii < attn_output.h_data.size(); ii++) {
            std::cout << attn_output.h_data[ii] << " ";
        }
        std::cout << std::endl;
        std::cout << "        [DEBUG] W_o[0:3]: ";
        for (int ii = 0; ii < 3 && ii < W_o.h_data.size(); ii++) {
            std::cout << W_o.h_data[ii] << " ";
        }
        std::cout << std::endl;
    }
    
    matmul_cuda(out_proj, attn_output, W_o);
    
    if (layer_call_count == 0) {
        std::cout << "        [DEBUG] Before residual connection..." << std::endl;
    }
    
    // 残差接続: out_proj += x
    out_proj.d2h();
    Tensor x_residual = x;
    x_residual.d2h();
    
    if (layer_call_count == 0) {
        std::cout << "        [DEBUG] out_proj.h_data.size()=" << out_proj.h_data.size() 
                  << ", x_residual.h_data.size()=" << x_residual.h_data.size() << std::endl;
    }
    
    for (int i = 0; i < seq_len * hidden_dim; i++) {
        out_proj.h_data[i] += x_residual.h_data[i];
    }
    out_proj.h2d();
    
    if (layer_call_count == 0) {
        std::cout << "        [DEBUG] Before LayerNorm1..." << std::endl;
    }
    
    // Layer Normalization 1
    Tensor norm1_out({seq_len, hidden_dim});
    layernorm_cuda(norm1_out, out_proj, ln1_gamma, ln1_beta);
    
    if (layer_call_count == 0) {
        std::cout << "        [DEBUG] After LayerNorm1..." << std::endl;
    }
    
    if (layer_call_count == 0) {
        norm1_out.d2h();
        float norm1_max = *std::max_element(norm1_out.h_data.begin(), norm1_out.h_data.end());
        std::cout << "        [DEBUG] After residual + LayerNorm1 max: " << norm1_max << std::endl;
    }
    
    // ===== FFN =====
    
    Tensor ff_hidden({seq_len, hidden_dim * 4});
    ff_hidden.zero();
    matmul_cuda(ff_hidden, norm1_out, W_ff1);
    
    if (layer_call_count == 0) {
        ff_hidden.d2h();
        float ff_hid_max = *std::max_element(ff_hidden.h_data.begin(), ff_hidden.h_data.end());
        std::cout << "        [DEBUG] ff_hidden max: " << ff_hid_max << std::endl;
    }
    
    relu_cuda(ff_hidden);
    
    Tensor ff_out({seq_len, hidden_dim});
    ff_out.zero();
    matmul_cuda(ff_out, ff_hidden, W_ff2);
    
    if (layer_call_count == 0) {
        std::cout << "        [DEBUG] Before FFN residual connection..." << std::endl;
    }
    
    // 残差接続: ff_out += norm1_out
    ff_out.d2h();
    norm1_out.d2h();
    for (int i = 0; i < seq_len * hidden_dim; i++) {
        ff_out.h_data[i] += norm1_out.h_data[i];
    }
    ff_out.h2d();
    
    if (layer_call_count == 0) {
        std::cout << "        [DEBUG] Before LayerNorm2..." << std::endl;
    }
    
    // Layer Normalization 2
    Tensor norm2_out({seq_len, hidden_dim});
    layernorm_cuda(norm2_out, ff_out, ln2_gamma, ln2_beta);
    
    if (layer_call_count == 0) {
        std::cout << "        [DEBUG] After LayerNorm2..." << std::endl;
    }
    
    // チE��チE��: 戻り値の確誁E
    if (layer_call_count == 0) {
        norm2_out.d2h();
        float norm2_max = *std::max_element(norm2_out.h_data.begin(), norm2_out.h_data.end());
        std::cout << "        [DEBUG] Final output (after residual + LayerNorm2) max: " << norm2_max << std::endl;        
        // GPU側データも確認
        float norm2_gpu[5];
        cudaMemcpy(norm2_gpu, norm2_out.d_data, 5 * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "        [DEBUG] norm2_out.d_data[0:5]: ";
        for (int ii = 0; ii < 5; ii++) {
            std::cout << norm2_gpu[ii] << " ";
        }
        std::cout << " (ptr=" << norm2_out.d_data << ")" << std::endl;
        
        // 層出力の統計情報
        float sum = 0.0f;
        for (size_t i = 0; i < norm2_out.h_data.size(); i++) {
            sum += norm2_out.h_data[i];
        }
        float mean = sum / norm2_out.h_data.size();
        
        float variance = 0.0f;
        for (size_t i = 0; i < norm2_out.h_data.size(); i++) {
            float diff = norm2_out.h_data[i] - mean;
            variance += diff * diff;
        }
        variance /= norm2_out.h_data.size();
        float std_dev = sqrtf(variance);
        
        std::cout << "        [DEBUG] Layer output stats - Mean: " << mean 
                  << ", Std: " << std_dev << std::endl;
        norm2_out.h2d();
    }
    layer_call_count++;
    
    // GPU→CPUに転送してからreturn（GPUメモリ解放問題を回避）
    norm2_out.d2h();
    return norm2_out;
}

/// ===================================================================
/// TinyLLM CUDA 牁E
/// ===================================================================

TinyLLM::TinyLLM(int vocab_size_, int hidden_dim_, int num_layers_, 
                 int seq_length_, float lr)
    : vocab_size(vocab_size_), hidden_dim(hidden_dim_), num_layers(num_layers_),
      seq_length(seq_length_), learning_rate(lr),
      embeddings({vocab_size_, hidden_dim_}), output_weight({hidden_dim_, vocab_size_})
{
    // Embedding 層を�E期化
    embeddings.randomInit();
    
    // Transformer レイヤーを�E期化
    for (int i = 0; i < num_layers; i++) {
        layers.push_back(new TransformerLayer(hidden_dim_, 4, lr));
    }
    
    // 出力層を�E期化
    output_weight.randomInit();
    
    std::cout << "[TinyLLM CUDA] Model initialized" << std::endl;
    std::cout << "  Vocab size: " << vocab_size_ << std::endl;
    std::cout << "  Hidden dim: " << hidden_dim_ << std::endl;
    std::cout << "  Num layers: " << num_layers_ << std::endl;
}

TinyLLM::~TinyLLM()
{
    for (auto layer : layers) {
        delete layer;
    }
    layers.clear();
}

Tensor TinyLLM::Forward(const std::vector<int>& token_ids)
{
    int seq_len = token_ids.size();
    
    // Embedding: ト�EクンIDから embedding lookup
    Tensor embedded({seq_len, hidden_dim});
    
    // CPU側で embedding を構篁E
    embeddings.d2h();  // GPU から CPU へ転送E
    for (int i = 0; i < seq_len; i++) {
        int token_id = token_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            // 対応すめEembedding をコピ�E
            for (int j = 0; j < hidden_dim; j++) {
                embedded.h_data[i * hidden_dim + j] = embeddings.h_data[token_id * hidden_dim + j];
            }
        } else {
            // 不正なト�EクンID の場合�Eゼロで初期匁E
            for (int j = 0; j < hidden_dim; j++) {
                embedded.h_data[i * hidden_dim + j] = 0.0f;
            }
        }
    }
    embedded.h2d();  // CPU から GPU へ転送E
    
    // チE��チE��: embedded の値を確誁E
    static int forward_count = 0;
    if (forward_count < 1) {
        float emb_max = *std::max_element(embedded.h_data.begin(), embedded.h_data.end());
        float emb_min = *std::min_element(embedded.h_data.begin(), embedded.h_data.end());
        std::cout << "      [DEBUG] Embedded - Max: " << emb_max << ", Min: " << emb_min << std::endl;
        forward_count++;
    }
    
    // Transformer レイヤー
    Tensor x = embedded;
    for (int layer_idx = 0; layer_idx < (int)layers.size(); layer_idx++) {
        Tensor layer_out = layers[layer_idx]->Forward(x);
        
        // デバッグ: 代入前のlayer_out.d_dataを確認
        if (forward_count == 0) {
            float layer_out_gpu[5];
            cudaMemcpy(layer_out_gpu, layer_out.d_data, 5 * sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "      [DEBUG] Layer " << layer_idx << " output (layer_out.d_data[0:5]): ";
            for (int ii = 0; ii < 5; ii++) std::cout << layer_out_gpu[ii] << " ";
            std::cout << " (ptr=" << layer_out.d_data << ")" << std::endl;
        }
        
        x = layer_out;
        x.h2d();  // CPU→GPU転送（returnで解放されたGPUメモリを復元）
        
        // デバッグ: 代入後のx.d_dataを確認
        if (forward_count == 0) {
            float x_after[5];
            cudaMemcpy(x_after, x.d_data, 5 * sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "      [DEBUG] After assignment & h2d() - x.d_data[0:5]: ";
            for (int ii = 0; ii < 5; ii++) std::cout << x_after[ii] << " ";
            std::cout << " (ptr=" << x.d_data << ")" << std::endl;
        }
    }
    
    // 出力層�E�最後�Eト�Eクンの隠れ状態を使用�E�E
    Tensor last_hidden({1, hidden_dim});
    
    // デバッグ: x のGPU側データを直接確認
    static int x_debug_count = 0;
    if (x_debug_count < 1) {
        float x_gpu[5];
        cudaMemcpy(x_gpu, x.d_data, 5 * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "      [DEBUG] x.d_data[0:5] (direct from GPU before d2h): ";
        for (int ii = 0; ii < 5; ii++) {
            std::cout << x_gpu[ii] << " ";
        }
        std::cout << std::endl;
        x_debug_count++;
    }
    
    // 最後�Eト�Eクンの隠れ状態をコピ�E
    x.d2h();
    for (int j = 0; j < hidden_dim; j++) {
        last_hidden.h_data[j] = x.h_data[(seq_len - 1) * hidden_dim + j];
    }
    last_hidden.h2d();  // ← GPUメモリに同期！
    
    // チE��チE��: last_hidden の値
    static int lh_debug_count = 0;
    if (lh_debug_count < 1) {
        float lh_max = *std::max_element(last_hidden.h_data.begin(), last_hidden.h_data.end());
        float lh_min = *std::min_element(last_hidden.h_data.begin(), last_hidden.h_data.end());
        std::cout << "      [DEBUG] last_hidden - Max: " << lh_max << ", Min: " << lh_min << std::endl;
        std::cout << "      [DEBUG] last_hidden[0:3]: " << last_hidden.h_data[0] << " " << last_hidden.h_data[1] << " " << last_hidden.h_data[2] << std::endl;
        
        // output_weight の値も確認
        output_weight.d2h();
        float ow_max = *std::max_element(output_weight.h_data.begin(), output_weight.h_data.end());
        float ow_min = *std::min_element(output_weight.h_data.begin(), output_weight.h_data.end());
        std::cout << "      [DEBUG] output_weight - Max: " << ow_max << ", Min: " << ow_min << std::endl;
        std::cout << "      [DEBUG] output_weight[0:3]: " << output_weight.h_data[0] << " " << output_weight.h_data[1] << " " << output_weight.h_data[2] << std::endl;
        lh_debug_count++;
    }
    
    Tensor logits({1, vocab_size});
    logits.zero();  // 明示皁E��ゼロ初期匁E
    matmul_cuda(logits, last_hidden, output_weight);
    
    // チE��チE��: softmax 前�E logits
    logits.d2h();
    float logits_max = *std::max_element(logits.h_data.begin(), logits.h_data.end());
    float logits_min = *std::min_element(logits.h_data.begin(), logits.h_data.end());
    std::cout << "      [DEBUG] Logits (before softmax) - Max: " << logits_max << ", Min: " << logits_min << std::endl;
    logits.h2d();
    
    // ソフトマックス
    softmax_cuda(logits);
    
    // デバッグ: Softmax後の確率分布とTop-3予測
    static int softmax_debug_count = 0;
    if (softmax_debug_count < 3) {
        logits.d2h();
        
        // Top-3を見つける
        std::vector<std::pair<float, int>> prob_idx;
        for (int i = 0; i < vocab_size; i++) {
            prob_idx.push_back({logits.h_data[i], i});
        }
        std::sort(prob_idx.begin(), prob_idx.end(), std::greater<std::pair<float, int>>());
        
        std::cout << "      [DEBUG] Top-3 predictions: ";
        for (int i = 0; i < 3 && i < vocab_size; i++) {
            std::cout << "ID=" << prob_idx[i].second << " (p=" << prob_idx[i].first << ") ";
        }
        std::cout << std::endl;
        
        logits.h2d();
        softmax_debug_count++;
    }
    
    return logits;
}

float TinyLLM::TrainStep(const std::vector<int>& token_ids, int target_id)
{
    // フォワードパス
    Tensor logits = Forward(token_ids);
    logits.d2h();  // GPU から CPU へ
    
    // ターゲチE��めEOne-hot に変換
    Tensor target({1, vocab_size});
    target.zero();
    target.h_data[target_id] = 1.0f;
    target.h2d();
    
    // 損失計箁E
    float loss = cross_entropy_loss_cuda(logits, target);
    
    // チE��チE��出力（最初�E3スチE��プ�Eみ�E�E
    static int train_step_count = 0;
    if (train_step_count < 3) {
        float max_logit = *std::max_element(logits.h_data.begin(), logits.h_data.end());
        float min_logit = *std::min_element(logits.h_data.begin(), logits.h_data.end());
        std::cout << "    [DEBUG] Target ID: " << target_id 
                  << ", Max logit: " << max_logit
                  << ", Min logit: " << min_logit
                  << ", Loss: " << loss << std::endl;
        train_step_count++;
    }
    
    // バックプロパゲーション�E�勾配計箁E
    // dlogits = (logits - target) / batch_size
    Tensor dlogits({1, vocab_size});
    for (int i = 0; i < vocab_size; i++) {
        dlogits.h_data[i] = (logits.h_data[i] - target.h_data[i]) / 1.0f;
    }
    dlogits.h2d();
    
    // 出力層の重み更新�E�W_out -= learning_rate * gradient
    output_weight.d2h();
    for (int h = 0; h < hidden_dim; h++) {
        for (int v = 0; v < vocab_size; v++) {
            float grad = dlogits.h_data[v] * 0.01f;  // 簡易版の勾酁E
            output_weight.h_data[h * vocab_size + v] -= learning_rate * grad;
        }
    }
    output_weight.h2d();
    
    // Embedding層の更新�E�簡易版�E�E
    embeddings.d2h();
    for (int i = 0; i < (int)token_ids.size(); i++) {
        int token_id = token_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            for (int h = 0; h < hidden_dim; h++) {
                embeddings.h_data[token_id * hidden_dim + h] -= learning_rate * loss * 0.0001f;
            }
        }
    }
    embeddings.h2d();
    
    return loss;
}

int TinyLLM::Predict(const std::vector<int>& token_ids)
{
    Tensor logits = Forward(token_ids);
    logits.d2h();
    
    // 確玁E��最大のト�Eクンを選抁E
    int predicted_id = 0;
    float max_prob = logits.h_data[0];
    
    for (int i = 1; i < vocab_size; i++) {
        if (logits.h_data[i] > max_prob) {
            max_prob = logits.h_data[i];
            predicted_id = i;
        }
    }
    
    return predicted_id;
}

void TinyLLM::SaveModel(const char* filepath)
{
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "エラー�E�ファイルを開けません: " << filepath << std::endl;
        return;
    }
    
    // メタチE�Eタ
    file.write((char*)&vocab_size, sizeof(int));
    file.write((char*)&hidden_dim, sizeof(int));
    file.write((char*)&num_layers, sizeof(int));
    file.write((char*)&seq_length, sizeof(int));
    
    // Embedding 層
    embeddings.d2h();
    int emb_size = embeddings.h_data.size();
    file.write((char*)&emb_size, sizeof(int));
    file.write((char*)embeddings.h_data.data(), emb_size * sizeof(float));
    
    // 出力層
    output_weight.d2h();
    int out_size = output_weight.h_data.size();
    file.write((char*)&out_size, sizeof(int));
    file.write((char*)output_weight.h_data.data(), out_size * sizeof(float));
    
    file.close();
    std::cout << "[Model] チェチE��ポイント保孁E " << filepath << std::endl;
}

TinyLLM* TinyLLM::LoadModel(const char* filepath)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "エラー�E�ファイルを開けません: " << filepath << std::endl;
        return nullptr;
    }
    
    // メタチE�Eタ
    int vs, hd, nl, sl;
    file.read((char*)&vs, sizeof(int));
    file.read((char*)&hd, sizeof(int));
    file.read((char*)&nl, sizeof(int));
    file.read((char*)&sl, sizeof(int));
    
    TinyLLM* model = new TinyLLM(vs, hd, nl, sl);
    
    // Embedding 層
    int emb_size;
    file.read((char*)&emb_size, sizeof(int));
    model->embeddings.h_data.resize(emb_size);
    file.read((char*)model->embeddings.h_data.data(), emb_size * sizeof(float));
    model->embeddings.h2d();
    
    // 出力層
    int out_size;
    file.read((char*)&out_size, sizeof(int));
    model->output_weight.h_data.resize(out_size);
    file.read((char*)model->output_weight.h_data.data(), out_size * sizeof(float));
    model->output_weight.h2d();
    
    file.close();
    std::cout << "[Model] Checkpoint loaded successfully" << std::endl;
    
    return model;
}
