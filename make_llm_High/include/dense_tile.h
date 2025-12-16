#pragma once
#include <cstddef>

namespace make_llm_high {

// dense_tile: 再帰ブロック GEMM のインターフェース
// - 行列は row-major を想定
// - small_gemm_base がベースケースで、ここを SIMD マイクロカーネルに差し替え可能

// naive_gemm: ベースライン（正しさ確認用）
void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K,
                int lda, int ldb, int ldc);

// rec_gemm: cache-oblivious な再帰 GEMM
void rec_gemm(const float* A, const float* B, float* C,
              int M, int N, int K,
              int lda, int ldb, int ldc,
              int threshold = 64*64*64);

// small_gemm_base: base-case（小ブロック）
void small_gemm_base(const float* A, const float* B, float* C,
                     int M, int N, int K,
                     int lda, int ldb, int ldc);

} // namespace make_llm_high