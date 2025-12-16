# TinyLLM — トレーニング & ロギング ドキュメント ✅

## 概要 ✨
本ドキュメントは本リポジトリの学習・推論ワークフローと、詳細ログ（各層・パラメータ勾配）を確認する方法をまとめたものです。主に学習用途での**可観測性（verbosity）**に重点を置いています。

---

## 前提条件 🔧
- Windows 環境（本ドキュメントは Windows 向けコマンドを示します）
- Visual Studio / MSVC がインストールされていること
- `vcpkg` を用いて OpenCL が導入される場合があります（`build_opencl.bat` が vcpkg を検出します）

---

## ビルド方法（OpenCL バックエンド） 🏗️
1. リポジトリルートへ移動:

```powershell
cd C:\div
```

2. OpenCL のビルドを行う（vcpkg を検出して使います）:

```powershell
make_LLM_CUDA\build_opencl.bat
```

- 成功するとバイナリは `build\bin\Release\TinyLLM_OPENCL.exe` に生成されます。

---

## CLI オプション（重要） 🧭
- `--list-devices` : 利用可能な OpenCL デバイスを一覧表示
- `--device <N>` : デフォルトで使うデバイスを選択（インデックス N）
- `--opencl` / `--gpu` : 明示的に OpenCL (GPU) を有効化して実行
- `--cpu` : 明示的に CPU 実行（OpenCL を使わない）
- **注意**: デフォルトは **CPU 実行** です。OpenCL を使う場合は明示的に `--opencl` を渡してください。
- `--verbosity <N>` : ログ出力の詳細度（数値で制御）
  - **0**: ログ出力オフ（デフォルト）
  - **1**: 学習サマリ（Loss、パラメータ勾配 L2、更新の前後 L2 など）✅
  - **2**: 層ごとの詳細（Forward の Q/K/V L2、Backward の grad L2 等）✅
  - **3+**: （将来拡張予定 — 層内のスライス等のさらに細かい情報）

---

## 実行例（推論 / 訓練） 🚀
- 推論（詳細フォワードノルムを表示）:

```powershell
build\bin\Release\TinyLLM_OPENCL.exe --verbosity 2 infer
```

- 学習（要約ログを表示）:

```powershell
build\bin\Release\TinyLLM_OPENCL.exe --verbosity 1 train
```

- デバイスの一覧表示:

```powershell
build\bin\Release\TinyLLM_OPENCL.exe --list-devices
```

---

## 勾配テスト（ユニット）✅
解析勾配と数値微分を比較するテストバイナリがあります。

- ビルド済み: `build\Release\TinyLLM_GRADTEST.exe`
- 実行例:

```powershell
build\Release\TinyLLM_GRADTEST.exe
```

- 現在確認済みのテスト: `W_o`, `W_q`, `W_k`, `W_v` の数値勾配チェック（許容誤差内で合格）

---

## ログ出力の内容（verbosity 別） 📝
- **verbosity = 1**（学習向け）
  - 各 TrainStep の Loss
  - 出力層・各種パラメータの **勾配 L2** 値と **更新前後のパラメータ L2**
  - 埋め込みへの勾配 L2
- **verbosity = 2**（デバッグ向け）
  - 各レイヤーの Forward での **Q/K/V の L2 ノルム**
  - Backward 実行後の層ごとの grad_seq の L2
  - （verbosity=1 の情報含む）

> これらは標準出力（stdout）へ出力されます。必要であればファイル保存（`logs/train.log` 等）・JSON 形式の出力を追加可能です。

---

## 設定・保存関連 🗂️
- チェックポイント保存先: `model_checkpoint.bin`（Train の実行中に定期保存）
- 設定可能項目: `learning_rate`, `num_layers`, `hidden_dim` などは `main_opencl.cpp` のモデル初期化部で変更可

---

## トラブルシューティング ⚠️
- CMake が OpenCL を見つけない場合:
  - `vcpkg` が適切に設定されているか確認し、`build_opencl.bat` の出力を参照してください
  - 必要なら OpenCL ランタイム/SDK をインストールしてください（Intel/AMD/ROCm のドライバ）
- 文字化け（日本語）:
  - Windows の場合、UTF-8 を使用してビルドしています。端末の設定（PowerShell の場合 `chcp 65001`）やフォントを確認してください

---

## 今後の拡張案 💡
- ログを **JSON** 出力にして可視化ツールで解析（TensorBoard 風）
- パラメータ更新を GPU 上で行いパフォーマンス向上
- さらなる verbosity レベル（レイヤ内部統計、勾配分布のヒストグラム等）

---

## 最後に ⭐
詳細ログは学習の健全性確認やデバッグに非常に役立ちます。必要ならログをファイルに保存する機能や、より細かいメトリクス（勾配分布、勾配ノルム推移など）の追加を行います。

もしよければ次は **ログのファイル出力（JSON/CSV）** を追加しましょうか？
