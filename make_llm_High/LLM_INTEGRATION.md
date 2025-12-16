# LLM Integration Design (AVX2 CPU backend)

目的
- AVX2 8x8 micro-kernel を利用して、Transformer の線形変換（Q/K/V、出力 W_o、FFN の線形層）を高速化する。
- フォールバックとテストを用意し、安全に置換を行う。

設計要点

1) API（batched / strided GEMM）

関数:

```
// Batched strided GEMM
// Performs for b in [0, batch): C_b = alpha * A_b * B_b + beta * C_b
void batched_gemm_strided(const float* A, const float* B, float* C,
                          int batch, int M, int N, int K,
                          int lda, int ldb, int ldc,
                          ptrdiff_t strideA, ptrdiff_t strideB, ptrdiff_t strideC,
                          bool transposeB = false,
                          int flags = 0);
```

- strideX はバッチ間のバイトオフセット（要は行列ごとのポインタステップ）
- transposeB を使えば B を列優先扱い（または transposed）に合わせられる
- flags に "pack B" や "high-accuracy" などのフラグを追加可能

2) ランタイムディスパッチ
- CPU が AVX2 をサポートすれば AVX2/rec_gemm ベースの経路を選択
- それ以外は汎用 simd/omp パスへフォールバック
- env var またはビルド定義で強制的にフォールバック/高精度モードを指定可能

3) B のパッキング（将来）
- 同一 B を複数の A と掛けるケース（照会バッチ化）向けに B をパックする API を追加予定
- パックはキャッシュフレンドリなレイアウト（小ブロック x K）

4) 数値／テスト基準
- 単体テスト: バッチごとの参照実装（double の naive_gemm）と比較して Frobenius 差/要素最大差を検証
- E2E: Transformer 層単位で forward の出力差と、学習での勾配差（数値チェック）を確認
- 受け入れ基準（初期）: Frobenius < 1e-3, 要素の大多数が rel < 1e-4、重大な要素(rel >= 1e-2) は非常に稀

5) パフォーマンス検証
- 既存の `make_llm_high_bench` を拡張して Transformer の 1 層（複数シーケンス長/バッチ）を測定
- CI にベンチを追加し、アーティファクトを保存（既に実装済）

6) 導入手順 (短期)
- PoC: Transformer の 1 レイヤのみ新 API に切替（比較、精度・速度を評価）
- フル導入: 問題なければ全線形層を置換、E2E テスト追加

実装ノート（PoC 用）
- まずは `batched_gemm_strided` を実装し、既存の `rec_gemm` を内部でバッチ分ループして呼ぶ（正しさ優先）
- その後、B パッキングと AVX2 パスを統合し、性能改善を図る

---

次のステップ
1. PoC のヘッダーとパススルー実装を追加して単体テストを追加
2. Transformer の 1 層を PoC API で置換してベンチと数値検証
3. 問題なければ全層に適用し CI に E2E とベンチ検証を追加
