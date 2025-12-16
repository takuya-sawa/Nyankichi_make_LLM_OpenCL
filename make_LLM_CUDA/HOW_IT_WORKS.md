# TinyLLM (OpenCL) — 学習と推論の仕組み 🧠

このドキュメントは、本プロジェクト（TinyLLM OpenCL）の**学習（training）**と**推論（inference）**の内部処理について、実装上の観点からわかりやすく整理したものです。

---

## 目次
- 概要
- モデル構成（アーキテクチャ）
- フォワード（推論）フロー
- バックワード（逆伝播）フロー
- パラメータ更新（SGD）
- キャッシュ（activation caching）
- OpenCL (GPU) の関与
- ログと可観測性（verbosity）
- テスト・検証（勾配チェック）
- 実行方法（コマンド参照）
- 注意点と今後の拡張

---

## 概要
- TinyLLM は小さめの Transformer ベースの言語モデルを模した実験実装で、**自己注意（Self-Attention）** と **FFN（2 層 FFN）**、**層正規化（LayerNorm）** を含みます。
- 実装は CPU と OpenCL の両方をサポートし、OpenCL が利用可能ならカーネルを使って一部の行列演算を高速化します。

---

## モデル構成（アーキテクチャ）
- Embedding 層: 語彙サイズ × 隠れ次元
- L 層の TransformerLayer。各 Layer:
  - **Q/K/V 投影 (W_q, W_k, W_v)**：各入力ベクトル x（形状: M×D, M はシーケンス長）に対して線形変換を行い、Query/Key/Value を得ます。数式で表すと `Q = x @ W_q`, `K = x @ W_k`, `V = x @ W_v`（`W_*` の形状は D×D が一般的）。Q/K/V の役割は直感的には **Q=問い（query）**、**K=鍵（key）**、**V=値（value）** で、Q と K の内積（スケーリング付き）で位置間の類似度 `S` を計算し、softmax により重み `A` を得て `A @ V` により出力を合成します。
  - Attention スコア `S = (Q @ K^T) / sqrt(d)`
  - softmax による重み `A = softmax(S)`（行ごと）
  - Attention 出力 `= A @ V`

  ### マルチヘッド注意 (Multi-Head Attention)
  - マルチヘッドでは隠れ次元 `D` を `h` 個のヘッドに分割し、各ヘッドが次元 `d_k = D / h` を持ちます。
  - 各ヘッドで独立に `Q_i, K_i, V_i` を計算し、`A_i = softmax( (Q_i @ K_i^T) / sqrt(d_k) )`、`head_i = A_i @ V_i` を得ます。
  - すべての `head_i` を結合して `Concat(head_1, ..., head_h) @ W_o` を適用するとマルチヘッド注意の出力が得られます。

  ### 小さな数値例（理解のための手計算）
  - **設定**: シーケンス長 `M=2`, 隠れ次元 `D=4`, ヘッド数 `h=2` → `d_k = 2`.
  - **入力**（埋め込み）:

    ```text
    x0 = [1, 0, 0, 0]
    x1 = [0, 1, 0, 0]
    X = [[1,0,0,0], [0,1,0,0]]  (2×4)
    ```

  - 単純化して `W_q = W_k = W_v = I`（恒等行列）とする。
  - ヘッド0（最初の2次元）について、`Q0 = [[1,0],[0,1]]`, `K0 = [[1,0],[0,1]]`, `V0 = [[1,0],[0,1]]`。
  - スケーリング `scale = 1/sqrt(d_k) ≈ 0.7071`。
  - スコア `S0 = scale * Q0 @ K0^T = 0.7071 * [[1,0],[0,1]]`。
  - 行ごと softmax の結果（近似）:
    - row0: softmax([0.7071, 0]) ≈ [0.67, 0.33]
    - row1: softmax([0, 0.7071]) ≈ [0.33, 0.67]
  - Attention 出力（ヘッド0）:
    - row0: 0.67*[1,0] + 0.33*[0,1] = [0.67, 0.33]
    - row1: [0.33, 0.67]
  - ヘッド1 はゼロなので出力はゼロ。ヘッドを結合すると行0: `[0.67,0.33,0,0]`、行1: `[0.33,0.67,0,0]`。
  - `W_o` を恒等とすれば `out_proj` は上と同じ値になります。

  - この例で、Q/K の内積 → softmax → A @ V により入力の特定部分（ここでは初期次元）に重みが付き、出力に反映される仕組みが分かります。
  - 出力投影 W_o
  - 残差 + 層正規化 (ln1)
  - FFN: W_ff1 (d -> 4d), ReLU, W_ff2 (4d -> d)
  - 残差 + 層正規化 (ln2)

- 出力層: 最終隠れ状態（最後のトークン）に対して線形変換で語彙ロジットを計算 -> softmax -> 確率。

---

## フォワード（推論）フロー（高水準）
1. トークン列 -> 埋め込み行列（embedding lookup）
2. 各 TransformerLayer に順次入力
   - x -> Q,K,V = x @ W_q/k/v
   - S = (Q @ K^T)/sqrt(d)
   - A = softmax(S)  (行ごと)
   - attn_output = A @ V
   - out_proj = attn_output @ W_o
   - norm1_out = LayerNorm(out_proj)
   - ff_hidden = ReLU(norm1_out @ W_ff1)
   - ff_out = ff_hidden @ W_ff2
   - norm2_out = LayerNorm(ff_out)
3. 最後の層から最後の位置を取り出し、logits = last_hidden @ output_weight
4. softmax(logits) で確率を得る

※ 実装では、行列積等の重い演算を `math_opencl.cpp` の OpenCL カーネルで実行し、存在しない場合は CPU 実装にフォールバックします。

---

## バックワード（逆伝播）フロー（高水準）
1. 損失: ここではクロスエントロピー (one-hot target) を使用
   - L = -log p(target)
2. 出力層の勾配: dL/dlogits = p - t
3. 出力層から最後の隠れ状態へ逆伝播（LinearBackward）
   - dW_output, db_output を計算し、SGDUpdate で更新
4. 最後の隠れ状態の勾配をシーケンス長サイズのテンソルに配置（最後の位置のみ非ゼロ）
5. レイヤを逆順で Backward
   - LayerNorm の逆伝播（gamma/beta の勾配と入力勾配）
   - FFN の逆伝播（W_ff2, W_ff1 の勾配、ReLU の逆伝播）
   - Attention の逆伝播:
     - d(out_proj) -> d(attn_output) via W_o^T
     - d(attn_output) -> dV, dA
     - softmax の逆伝播: dA -> dS
     - dS -> dQ, dK (S = scale * Q @ K^T)
     - dQ/dK は入力 x と W_q/W_k の勾配に変換される
   - 各線形層の勾配は LinearBackward で算出
6. 埋め込みは位置ごとに勾配を集計して更新

注: 現行実装では、Attention の W_q/W_k に関する厳密な勾配の一部やパラメータ更新の簡略化が行われている箇所があります（コード中に該当コメントあり）。

---

## パラメータ更新（SGD）
- 単純 SGD を用いており、各パラメータは次のルールで更新されます:
  - param -= lr * grad
- 実験的に Adam 等の最適化器を追加する余地があります（TODO）。

---

## Activation caching（逆伝播のための中間保存）
- 逆伝播を効率化するために、Forward 時に以下をキャッシュできます（`EnableCache(true)`）:
  - 層の入力 `x`
  - Q, K, V, K^T
  - attn_scores（softmax 前/後）
  - attn_output, out_proj
  - ff_hidden, ff_out
  - norm1_out, norm2_out
- TrainStep 実行中はキャッシュを有効にして、逆伝播後に `EnableCacheAll(false)` でキャッシュをクリアします。

---

## OpenCL (GPU) の関与
- OpenCL の有無は CMake と実行時オプションで切り替えられます。vcpkg を通じて OpenCL を検出します。
- 行列積、softmax、layernorm、relu 等の重い処理には OpenCL カーネルを用意し、デバイスバッファ (cl_mem) を介して処理します。
- GPU パスが未使用の場合は CPU 実装で安全にフォールバックします。

---

## ログと可観測性（verbosity）
- CLI: `--verbosity <N>` で出力レベルを制御
  - `N >= 1`: TrainStep の損失、各パラメータの勾配 L2、更新前後の L2 を出力
  - `N >= 2`: 層ごとの Forward（Q/K/V の L2）、Backward（grad_seq の L2）などより詳細を出力
- ログは stdout に出力されます。将来的にはファイル/JSON ロギングのオプション追加を予定しています。

---

## テスト・検証
- 数値微分（中心差分）による勾配チェックを `src/test_gradients.cpp` に実装しています。
  - 現在チェック対象: W_o, W_q, W_k, W_v
  - `TinyLLM_GRADTEST` 実行で解析勾配 vs 数値勾配を比較し、誤差がしきい値内なら合格します。

---

## 実行方法（要約）
- ビルド（OpenCL）: `make_LLM_CUDA\build_opencl.bat`
- 訓練実行: `build\bin\Release\TinyLLM_OPENCL.exe --verbosity 1 train`
- 推論実行: `build\bin\Release\TinyLLM_OPENCL.exe --verbosity 2 infer`
- 勾配テスト: `build\Release\TinyLLM_GRADTEST.exe`

---

## 注意点・制限
- 小規模実験実装のため、大規模な日本語コーパスでの高品質な生成にはモデル容量とデータが不足します。
- Attention の一部勾配処理や W_q/W_k の更新に簡略化が残っている箇所があります。厳密性が必要な実験では該当部分の見直しを推奨します。

---

## 今後の拡張案
- Adam 等の最適化器の追加
- Attention の完全なパラメータ勾配（W_q, W_k の厳密更新）と GPU 上での更新
- トークナイザーを日本語対応（文字トークナイザー / SentencePiece）に変更して多言語学習に対応
- ログの JSON 出力・可視化（TensorBoard 風）

---

## 参考ソースコード（主な関連ファイル）
- `src/main_opencl.cpp` — CLI / トレーニングループ / 推論フロー
- `src/transformer_opencl.cpp` — Transformer 層の forward / backward / TrainStep 実装
- `src/math_opencl.cpp` — OpenCL カーネルのラッパ（matmul, softmax, layernorm, relu 等）
- `src/test_gradients.cpp` — 勾配チェックのユニットテスト
- `TRAINING_AND_LOGGING.md` — 訓練・ログ出力の使い方（別ドキュメント）

---

疑問点や、より詳しく書いてほしい箇所（例: 数式の厳密導出、OpenCL カーネルの実装例、JSON ログ出力例）があれば教えてください。対応して追加ドキュメントを作成します。