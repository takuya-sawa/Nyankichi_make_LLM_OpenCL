# TinyLLM CUDA 版 - ビルド手順書

## クイックスタート

### 1. ビルド（初回のみ）

```bash
build_cuda.bat
```

このスクリプトが行うこと：
- CUDA Toolkit のインストール確認
- cuBLAS ライブラリ確認
- CMake でプロジェクトを生成
- Visual Studio でコンパイル（Release 版）

**初回ビルド時間**: 3-5 分（GPU 依存）

### 2. 実行

#### 訓練 + 推論（デフォルト）
```bash
both.bat
```
またはコマンドラインで：
```bash
.\build\bin\Release\TinyLLM_CUDA.exe
```

#### 訓練のみ
```bash
train.bat
```
またはコマンドラインで：
```bash
.\build\bin\Release\TinyLLM_CUDA.exe train
```

#### 推論のみ
```bash
infer.bat
```
またはコマンドラインで：
```bash
.\build\bin\Release\TinyLLM_CUDA.exe infer
```

## システム要件

### ハードウェア
- **GPU**: NVIDIA GeForce RTX 20 シリーズ以上推奨
  - RTX 2060 (Turing, Compute Capability 7.5)
  - RTX 30 シリーズ (Ampere, CC 8.6)
  - RTX 40 シリーズ (Ada, CC 8.9)

### ソフトウェア
- **CUDA Toolkit**: 11.0 以上（推奨: 12.0+）
  - インストール: [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
  - 確認: `nvcc --version`
  
- **Visual Studio**: 2019 以上
  - C++ デスクトップ開発ツール
  - CUDA コンパイラ対応
  
- **CMake**: 3.18 以上
  - インストール: [cmake.org](https://cmake.org/download/)
  - 確認: `cmake --version`

## トラブルシューティング

### エラー: `CUDA_PATH` が設定されていない
```
エラー: CUDA_PATH 環境変数が設定されていません
```

**解決方法:**
1. CUDA Toolkit をインストール（再度実行）
2. または手動で `CUDA_PATH` を設定：
   ```
   set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0
   ```

### エラー: `cuBLAS ライブラリが見つかりません`
```
エラー: cuBLAS ライブラリが見つかりません
場所: C:\...\lib\x64\cublas.lib
```

**解決方法:**
- CUDA Toolkit の完全インストール確認
- インストール時に「CUDA」チェックボックスを確認

### エラー: `CMake 生成に失敗しました`
```
error: generator platform not found
```

**解決方法:**
- Visual Studio 2022 がインストールされているか確認
- CMake をアップデート: `cmake --version`

### エラー: `ビルドに失敗しました`
```
error: CUDA compilation failure
```

**解決方法:**
1. `build` フォルダを削除
2. 再度 `build_cuda.bat` を実行
3. Visual Studio で `build/TinyLLM_CUDA.sln` を開いてビルド

## パフォーマンス最適化

### CUDA Compute Capability の確認

`build_cuda.bat` で自動設定される Compute Capability：
```
CMAKE_CUDA_ARCHITECTURES=75;80;86;89
```

自分の GPU に合わせて最適化：
```
// RTX 2060 のみ
CMAKE_CUDA_ARCHITECTURES=75

// RTX 3090 のみ
CMAKE_CUDA_ARCHITECTURES=86

// 複数 GPU 対応
CMAKE_CUDA_ARCHITECTURES=75;80;86;89
```

### メモリ使用量の削減

`main.cu` の設定を変更：
```cpp
int hidden_dim = 256;   // 128 に削減すると高速化
int num_layers = 3;     // 2 に削減
```

## ビルド結果

ビルド成功後：
```
.\build\bin\Release\TinyLLM_CUDA.exe        実行ファイル
.\build\bin\Release\TinyLLM_CUDA.lib        スタティックライブラリ
.\build\CMakeFiles\                         ビルド情報
```

## クリーンビルド

```bash
REM build フォルダを削除
rmdir /s /q build

REM 再度ビルド
build_cuda.bat
```

## Visual Studio での開発

### デバッグ方法

1. CMake が生成した Visual Studio プロジェクトを開く：
   ```
   .\build\TinyLLM_CUDA.sln
   ```

2. Visual Studio でブレークポイント設定

3. F5 キーでデバッグ実行

### カスタムカーネルのプロファイル

NVIDIA Nsight Compute でプロファイル：
```bash
"C:\Program Files\NVIDIA Corporation\NVIDIA Nsight Compute\ncu.exe" .\build\bin\Release\TinyLLM_CUDA.exe
```

## 関連ドキュメント

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [CMake CUDA Documentation](https://cmake.org/cmake/help/latest/language/CUDA/)

---

最終更新: 2025年12月14日
