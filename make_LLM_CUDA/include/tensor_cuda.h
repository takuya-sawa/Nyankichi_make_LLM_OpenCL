#ifndef TENSOR_CUDA_H
#define TENSOR_CUDA_H

#include <vector>
#include <cuda_runtime.h>

/// ===================================================================
/// CUDA テンソル クラス：GPU メモリ管理
/// 
/// GPU（NVIDIA）上での高速テンソル操作を実現。
/// host_data（CPU）と device_data（GPU）の両方を管理。
/// 
/// 主な機能：
/// - Shape管理（多次元対応）
/// - GPU ↔ CPU メモリ転送
/// - GPU 上でのランダム初期化・ゼロ初期化
/// ===================================================================
class Tensor
{
public:
    std::vector<int> shape;      // テンソルの形状
    size_t size;                  // 総要素数
    
    // CPU メモリ
    std::vector<float> h_data;    // ホスト（CPU）側データ
    
    // GPU メモリ
    float* d_data;                // デバイス（GPU）側データ
    
    // デフォルトコンストラクタ
    Tensor();
    
    // 形状を指定してコンストラクタ
    Tensor(const std::vector<int>& shape_);
    
    // コピーコンストラクタ
    Tensor(const Tensor& other);
    
    // 代入演算子
    Tensor& operator=(const Tensor& other);
    
    // デストラクタ
    ~Tensor();
    
    // GPU メモリを確保
    void allocate();
    
    // GPU メモリを解放
    void deallocate();
    
    // CPU から GPU へコピー
    void h2d();
    
    // GPU から CPU へコピー
    void d2h();
    
    // ランダム初期化（GPU）
    void randomInit();
    
    // ゼロ初期化（GPU）
    void zero();
    
    // GPU メモリ上での値を設定（デバッグ用）
    void setValue(int idx, float value);
};

#endif // TENSOR_CUDA_H
