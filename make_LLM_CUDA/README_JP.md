# TinyLLM - CUDA 版（GPU 最適化）

学習用自作LLM - NVIDIA GPU（CUDA）を活用した高速言語モデル

## 特徴

- 🚀 **GPU 加速**: cuBLAS によるハイパフォーマンス行列演算
- ⚡ **CUDA カーネル**: カスタム最適化された活性化関数・ソフトマックス
- 🎓 **教育向け**: LLM と CUDA プログラミングの学習に最適
- 📊 **スケーラブル**: 大規模モデル対応可能
- 💾 **永続性**: チェックポイント保存・読み込み対応

## プロジェクト構成

```
LLMCUDA/
├── src/
│   ├── main.cu              # メインプログラム
│   ├── tensor_cuda.cu       # GPU テンソル実装
│   ├── math_cuda.cu         # CUDA 数学演算（cuBLAS）
│   └── transformer_cuda.cu  # Transformer レイヤー
├── include/
│   ├── tensor_cuda.h
│   ├── math_cuda.h
│   └── transformer_cuda.h
├── data/
│   └── training_data.txt    # 訓練用テキストデータ
├── CMakeLists.txt           # CUDA ビルド設定
└── README_JP.md             # このファイル
```

## システム要件

- **GPU**: NVIDIA GeForce RTX シリーズ以上（推奨: RTX 2060 以上）
- **CUDA Toolkit**: 11.0 以上
- **CMake**: 3.18 以上
- **Visual Studio**: 2019 以上（CUDA コンパイラ対応）

## ビルド方法

### Windows (Visual Studio)

```bash
# ビルドディレクトリを作成
mkdir build
cd build

# CMake でプロジェクトを生成
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=75

# ビルド
cmake --build . --config Release
```

### Linux/macOS

```bash
mkdir build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build . --config Release
```

## 実行方法

### 訓練 + 推論（デフォルト）
```bash
./bin/TinyLLM_CUDA
```

### 訓練のみ
```bash
./bin/TinyLLM_CUDA train
```

### 推論のみ
```bash
./bin/TinyLLM_CUDA infer
```

## モデル構成

| パラメータ | 値 |
|-----------|-----|
| 語彙サイズ | 128+ |
| 隠れ次元 | 256 |
| Transformer レイヤー | 3 |
| マルチヘッド数 | 4 |
| FFN 拡張係数 | 4 |

## ファイル説明

### main.cu
メインプログラム。訓練・推論ループとトークナイザーを実装。

**主要機能:**
- トレーニングデータ読み込み
- トークナイザー（動的語彙構築）
- 訓練ステップ（フォワード・バックワード）
- 推論（次のトークン予測）

### tensor_cuda.cu/h
GPU テンソル クラス実装。

**主要メソッド:**
- `allocate()`: GPU メモリ確保
- `h2d()`: CPU → GPU メモリ転送
- `d2h()`: GPU → CPU メモリ転送
- `randomInit()`: GPU 上でランダム初期化
- `zero()`: GPU 上でゼロ初期化

### math_cuda.cu/h
CUDA 数学演算 - cuBLAS と カスタムカーネル

**実装関数:**
- `matmul_cuda()`: 行列乗算（cuBLAS）
- `relu_cuda()`: ReLU 活性化（カスタムカーネル）
- `softmax_cuda()`: ソフトマックス（カスタムカーネル）
- `cross_entropy_loss_cuda()`: 損失計算（GPU）

**CUDA カーネル:**
```cuda
__global__ void kernel_relu(float* data, int size)
__global__ void kernel_softmax(float* data, int batch_size, int vocab_size)
__global__ void kernel_cross_entropy_loss(...)
```

### transformer_cuda.cu/h
Transformer レイヤーとモデル実装。

**TransformerLayer:**
- Q, K, V 計算（cuBLAS）
- Self-Attention スコア計算
- ソフトマックス正規化
- FFN（Feed-Forward Network）
- 残差接続

**TinyLLM:**
- Embedding 層
- Transformer × 複数レイヤー
- 出力層（語彙確率分布）
- `SaveModel()`: バイナリ形式保存
- `LoadModel()`: チェックポイント復元

## パフォーマンス比較

| 実装 | 処理速度 | メモリ効率 | 可読性 |
|------|---------|-----------|--------|
| **C++（CPU）** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **C#（CPU）** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **CUDA（GPU）** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

CUDA 版は **CPU 版の 10-100 倍** の速度向上を期待できます。

## CUDA プログラミングの学習ポイント

このプロジェクトで学べること：

1. **GPU メモリ管理**
   - `cudaMalloc()` / `cudaFree()`
   - `cudaMemcpy()` による CPU ↔ GPU 転送

2. **CUDA カーネル開発**
   - `__global__` カーネル関数
   - ブロック・スレッド構造（Grid, Block）
   - 同期化（`__syncthreads()`）

3. **cuBLAS ライブラリ**
   - `cublasCreate()` / `cublasDestroy()`
   - `cublasSgemm()` による行列乗算

4. **最適化テクニック**
   - メモリ転送の最小化
   - スレッド並列化
   - ワープの効率的利用

## トラブルシューティング

### CUDA がコンパイルエラー
```
error: CUDA is not properly installed
```
→ CUDA Toolkit のインストール確認: `nvcc --version`

### cuBLAS ラインカーエラー
```
error: undefined reference to `cublasSgemm'
```
→ CMakeLists.txt で `target_link_libraries` に cuBLAS が指定されているか確認

### Out of Memory（メモリ不足）
→ `hidden_dim` を削減するか、バッチサイズを縮小

## 今後の拡張

1. **マルチ GPU 対応**: NCCL による分散学習
2. **混合精度演算**: FP16 計算による高速化
3. **テンソルコア利用**: より高速な行列演算
4. **ディープラーニングフレームワーク**: cuDNN/TensorRT 統合
5. **より複雑な Attention**: Flash-Attention 実装

## ライセンス

教育目的での使用を想定しています。

## 関連リンク

- [C++ 版 TinyLLM](https://github.com/takuya-sawa/Nyankichi_make_LLM)
- [C# 版 TinyLLM](https://github.com/takuya-sawa/Nyankichi_make_LLM_CSHARP)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)

---

作成日: 2025年12月14日
学習用自作LLM - CUDA 版
