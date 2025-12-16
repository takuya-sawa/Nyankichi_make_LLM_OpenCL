#include "../include/dense_tile.h"
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <iostream>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <intrin.h>

namespace make_llm_high {

static bool cpu_has_avx2() {
    int regs[4];
    __cpuidex(regs, 1, 0);
    bool osxsave = (regs[2] & (1 << 27)) != 0;
    bool avx = (regs[2] & (1 << 28)) != 0;
    unsigned long long xcr0 = 0;
    if (osxsave) xcr0 = _xgetbv(0);
    bool avx_os = osxsave && ((xcr0 & 0x6ULL) == 0x6ULL);
    __cpuidex(regs, 7, 0);
    bool avx2 = (regs[1] & (1 << 5)) != 0;
    return avx && avx_os && avx2;
}

static bool cpu_has_fma() {
    int regs[4];
    __cpuidex(regs, 1, 0);
    bool osxsave = (regs[2] & (1 << 27)) != 0;
    unsigned long long xcr0 = 0;
    if (osxsave) xcr0 = _xgetbv(0);
    bool avx_os = osxsave && ((xcr0 & 0x6ULL) == 0x6ULL);
    bool fma = (regs[2] & (1 << 12)) != 0;
    return fma && avx_os;
}

// AVX2 8x8 micro-kernel
static void small_gemm_avx2_8x8(const float* A, const float* B, float* C,
                                int M, int N, int K,
                                int lda, int ldb, int ldc) {
#ifdef __AVX2__
    for (int i0 = 0; i0 < M; i0 += 8) {
        int mb = std::min(8, M - i0);
        for (int j0 = 0; j0 < N; j0 += 8) {
            int nb = std::min(8, N - j0);
            if (mb == 8 && nb == 8) {
                // Chunked K accumulation: for each K chunk, compute a float 8x8 partial
                // result using AVX float operations, then promote to double and add
                // to double accumulators. This balances speed and numerical stability.
                const int CHUNK = 64;
                double acc[8][8];
                for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) acc[i][j] = double(C[(i0 + i) * ldc + (j0 + j)]);

                static bool fma = cpu_has_fma();

                for (int kc = 0; kc < K; kc += CHUNK) {
                    int kend = std::min(K, kc + CHUNK);
                    // tmp accumulators in float (8 rows x 8 cols)
                    __m256 tmpv[8]; // each row accumulates 8 floats in two __m256 halves; we'll handle as two halves
                    // We will keep two halves per row: cols 0..3 and 4..7
                    __m256 tmp_lo[8];
                    __m256 tmp_hi[8];
                    for (int i = 0; i < 8; ++i) { tmp_lo[i] = _mm256_setzero_ps(); tmp_hi[i] = _mm256_setzero_ps(); }

                    // Unrolled K loop with prefetching (4-way unroll)
                    int k = kc;
                    for (; k + 3 < kend; k += 4) {
                        // prefetch ahead
                        _mm_prefetch((const char*)&B[(k + 16) * ldb + j0], _MM_HINT_T0);
                        for (int ii = 0; ii < 8; ++ii)
                            _mm_prefetch((const char*)&A[(i0 + ii) * lda + k + 16], _MM_HINT_T0);

                        __m256 b0 = _mm256_loadu_ps(&B[k * ldb + j0]);
                        __m256 b1 = _mm256_loadu_ps(&B[(k + 1) * ldb + j0]);
                        __m256 b2 = _mm256_loadu_ps(&B[(k + 2) * ldb + j0]);
                        __m256 b3 = _mm256_loadu_ps(&B[(k + 3) * ldb + j0]);

                        for (int i = 0; i < 8; ++i) {
                            __m256 a0 = _mm256_set1_ps(A[(i0 + i) * lda + k + 0]);
                            __m256 a1 = _mm256_set1_ps(A[(i0 + i) * lda + k + 1]);
                            __m256 a2 = _mm256_set1_ps(A[(i0 + i) * lda + k + 2]);
                            __m256 a3 = _mm256_set1_ps(A[(i0 + i) * lda + k + 3]);
#ifdef __FMA__
                            tmp_lo[i] = _mm256_fmadd_ps(a0, b0, tmp_lo[i]);
                            tmp_lo[i] = _mm256_fmadd_ps(a1, b1, tmp_lo[i]);
                            tmp_lo[i] = _mm256_fmadd_ps(a2, b2, tmp_lo[i]);
                            tmp_lo[i] = _mm256_fmadd_ps(a3, b3, tmp_lo[i]);
#else
                            tmp_lo[i] = _mm256_add_ps(tmp_lo[i], _mm256_mul_ps(a0, b0));
                            tmp_lo[i] = _mm256_add_ps(tmp_lo[i], _mm256_mul_ps(a1, b1));
                            tmp_lo[i] = _mm256_add_ps(tmp_lo[i], _mm256_mul_ps(a2, b2));
                            tmp_lo[i] = _mm256_add_ps(tmp_lo[i], _mm256_mul_ps(a3, b3));
#endif
                        }
                    }
                    for (; k < kend; ++k) {
                        __m256 b_lo = _mm256_loadu_ps(&B[k * ldb + j0 + 0]);
                        for (int i = 0; i < 8; ++i) {
                            __m256 a = _mm256_set1_ps(A[(i0 + i) * lda + k]);
#ifdef __FMA__
                            tmp_lo[i] = _mm256_fmadd_ps(a, b_lo, tmp_lo[i]);
#else
                            tmp_lo[i] = _mm256_add_ps(tmp_lo[i], _mm256_mul_ps(a, b_lo));
#endif
                        }
                    }

                    // Now convert each row's tmp_lo (8 floats) into two __m256d halves and add to acc
                    for (int i = 0; i < 8; ++i) {
                        // extract low 4 and high 4 floats
                        __m128 lo = _mm256_castps256_ps128(tmp_lo[i]);
                        __m128 hi = _mm256_extractf128_ps(tmp_lo[i], 1);
                        __m256d bd0 = _mm256_cvtps_pd(lo);
                        __m256d bd1 = _mm256_cvtps_pd(hi);
                        double tmpd0[4], tmpd1[4];
                        _mm256_storeu_pd(tmpd0, bd0);
                        _mm256_storeu_pd(tmpd1, bd1);
                        for (int j = 0; j < 4; ++j) acc[i][j] += tmpd0[j];
                        for (int j = 0; j < 4; ++j) acc[i][4 + j] += tmpd1[j];
                    }
                }

                // store back to C (convert to float)
                for (int i = 0; i < 8; ++i) {
                    for (int j = 0; j < 8; ++j) {
                        C[(i0 + i) * ldc + (j0 + j)] = float(acc[i][j]);
                    }
                }
            } else {
                // handle tails with scalar loops
                for (int ii = i0; ii < i0 + mb; ++ii) {
                    for (int k = 0; k < K; ++k) {
                        float aik = A[ii * lda + k];
                        for (int j = j0; j < j0 + nb; ++j) {
                            C[ii * ldc + j] += aik * B[k * ldb + j];
                        }
                    }
                }
            }
        }
    }
#else
    // Should not be called when no AVX2 support; keep fallback
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float aik = A[i * lda + k];
            #pragma omp simd
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] += aik * B[k * ldb + j];
            }
        }
    }
#endif
}

void small_gemm_base(const float* A, const float* B, float* C,
                     int M, int N, int K,
                     int lda, int ldb, int ldc) {
    static bool avx2 = cpu_has_avx2();
    if (avx2) {
        small_gemm_avx2_8x8(A, B, C, M, N, K, lda, ldb, ldc);
        return;
    }

    // ベースケース: ループ順は i-k-j。
    // 内側の j を simd を使ってベクトル化しやすくする。
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float aik = A[i * lda + k];
            #pragma omp simd
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] += aik * B[k * ldb + j];
            }
        }
    }
}

void rec_gemm(const float* A, const float* B, float* C,
              int M, int N, int K,
              int lda, int ldb, int ldc,
              int threshold) {
    long long work = 1LL * M * N * K;
    if (work <= threshold) {
        small_gemm_base(A, B, C, M, N, K, lda, ldb, ldc);
        return;
    }

    if (M >= std::max(N, K)) {
        int m2 = M / 2;
        rec_gemm(A, B, C, m2, N, K, lda, ldb, ldc, threshold);
        rec_gemm(A + m2 * lda, B, C + m2 * ldc, M - m2, N, K, lda, ldb, ldc, threshold);
    } else if (N >= K) {
        int n2 = N / 2;
        rec_gemm(A, B, C, M, n2, K, lda, ldb, ldc, threshold);
        rec_gemm(A, B + n2, C + n2, M, N - n2, K, lda, ldb, ldc, threshold);
    } else {
        int k2 = K / 2;
        rec_gemm(A, B, C, M, N, k2, lda, ldb, ldc, threshold);
        rec_gemm(A + k2, B + k2 * ldb, C, M, N, K - k2, lda, ldb, ldc, threshold);
    }
}

void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K,
                int lda, int ldb, int ldc) {
    std::memset(C, 0, sizeof(float) * (size_t)M * (size_t)ldc);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = acc;
        }
    }
}

} // namespace make_llm_high