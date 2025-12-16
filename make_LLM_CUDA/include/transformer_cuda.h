#ifndef TRANSFORMER_CUDA_H
#define TRANSFORMER_CUDA_H

#include "tensor_cuda.h"
#include <vector>

/// ===================================================================
/// TransformerLayer CUDA 実装：マルチヘッド自己注意 + FFN
/// 
/// 構成：
/// 1. マルチヘッド自己注意（Multi-Head Self-Attention）
///    - cuBLAS による高速 Q, K, V 計算
///    - 注意重み計算（GPU カーネル）
///    - スケーリング・ソフトマックス（GPU 最適化）
/// 
/// 2. フィードフォワード（FFN）
///    - cuBLAS 行列乗算
///    - ReLU 活性化
///    - 出力投影
/// 
/// 3. 正規化・残差接続
///    - 層正規化（LayerNorm）
///    - 残差接続
/// ===================================================================
class TransformerLayer
{
private:
    int hidden_dim;
    int num_heads;
    float learning_rate;
    
    // マルチヘッド注意用の重み
    Tensor W_q, W_k, W_v, W_o;
    Tensor b_q, b_k, b_v, b_o;
    
    // フィードフォワード層の重み
    Tensor W_ff1, W_ff2;
    Tensor b_ff1, b_ff2;
    
    // Layer Normalization パラメータ
    Tensor ln1_gamma, ln1_beta;  // Attention後の正規化
    Tensor ln2_gamma, ln2_beta;  // FFN後の正規化
    
public:
    /// <summary>
    /// コンストラクタ：重み初期化
    /// </summary>
    TransformerLayer(int hidden_dim_, int num_heads_, float lr = 0.001f);
    
    /// <summary>
    /// デストラクタ：GPU メモリ解放
    /// </summary>
    ~TransformerLayer();
    
    /// <summary>
    /// フォワードパス：GPU 実装
    /// 
    /// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
    /// </summary>
    Tensor Forward(const Tensor& x);
};

/// ===================================================================
/// TinyLLM CUDA 版：GPU 最適化言語モデル
/// 
/// 構成：
/// - Embedding層：トークン → ベクトル（GPU）
/// - Transformerレイヤー × 複数（GPU）
/// - 出力層：隠れ状態 → 語彙確率分布（GPU）
/// ===================================================================
class TinyLLM
{
private:
    int vocab_size;
    int hidden_dim;
    int num_layers;
    int seq_length;
    float learning_rate;
    
    Tensor embeddings;                           // 語彙 × 隠れ次元
    std::vector<TransformerLayer*> layers;       // Transformerレイヤー配列
    Tensor output_weight;                        // 隠れ × 語彙
    
public:
    /// <summary>
    /// コンストラクタ：モデルパラメータを指定
    /// </summary>
    TinyLLM(int vocab_size_, int hidden_dim_, int num_layers_ = 2, 
            int seq_length_ = 16, float lr = 0.001f);
    
    /// <summary>
    /// デストラクタ：GPU メモリ解放
    /// </summary>
    ~TinyLLM();
    
    /// <summary>
    /// フォワードパス：トークンID → 予測確率分布
    /// </summary>
    Tensor Forward(const std::vector<int>& token_ids);
    
    /// <summary>
    /// 訓練ステップ：フォワード・バックワード・更新
    /// </summary>
    float TrainStep(const std::vector<int>& token_ids, int target_id);
    
    /// <summary>
    /// 推論：次のトークンを予測
    /// </summary>
    int Predict(const std::vector<int>& token_ids);
    
    /// <summary>
    /// モデルをファイルに保存
    /// </summary>
    void SaveModel(const char* filepath);
    
    /// <summary>
    /// モデルをファイルから読み込み
    /// </summary>
    static TinyLLM* LoadModel(const char* filepath);
    
    /// <summary>
    /// ゲッター：語彙サイズ
    /// </summary>
    int GetVocabSize() const { return vocab_size; }
};

#endif // TRANSFORMER_CUDA_H
