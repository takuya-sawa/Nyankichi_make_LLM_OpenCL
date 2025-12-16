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

// Batched strided GEMM (PoC - simple stride units: elements)
// Performs for b in [0, batch): C_b = A_b * B_b + C_b
// strideA/strideB/strideC are in number of elements (not bytes)
void batched_gemm_strided(const float* A, const float* B, float* C,
                          int batch, int M, int N, int K,
                          int lda, int ldb, int ldc,
                          ptrdiff_t strideA, ptrdiff_t strideB, ptrdiff_t strideC,
                          bool transposeB = false);

// Runtime CPU feature query
bool cpu_has_avx2_available();
} // namespace make_llm_high