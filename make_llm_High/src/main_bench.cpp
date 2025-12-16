#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#ifdef _WIN32
#include <windows.h>
#endif
#include "../include/dense_tile.h"

using namespace make_llm_high;
using namespace std::chrono;

#include <iomanip>

// Frobenius ノルム差
static double frob_norm_diff(const std::vector<float>& A, const std::vector<float>& B) {
    double s = 0.0;
    size_t n = A.size();
    for (size_t i = 0; i < n; ++i) {
        double d = double(A[i]) - double(B[i]);
        s += d * d;
    }
    return std::sqrt(s);
}

static void naive_gemm_double(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                              int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += double(A[i * lda + k]) * double(B[k * ldb + j]);
            }
            C[i * ldc + j] = float(acc);
        }
    }
}

int main() {
#ifdef _WIN32
    // Ensure console outputs use UTF-8 so Japanese messages are not garbled
    SetConsoleOutputCP(CP_UTF8);
#endif
    const int M = 512;
    const int N = 512;
    const int K = 512;
    const int threshold = 64*64*64;

    std::cout << "Benchmark: rec_gemm vs naive_gemm\n";
    std::cout << "M=" << M << " N=" << N << " K=" << K << "\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A((size_t)M*K), B((size_t)K*N);
    std::vector<float> C_ref((size_t)M*N), C_rec((size_t)M*N);

    for (auto &v : A) v = dist(rng);
    for (auto &v : B) v = dist(rng);
    std::fill(C_ref.begin(), C_ref.end(), 0.0f);
    std::fill(C_rec.begin(), C_rec.end(), 0.0f);

    auto t0 = high_resolution_clock::now();
    naive_gemm_double(A, B, C_ref, M, N, K, K, N, N);
    auto t1 = high_resolution_clock::now();
    double naive_ms = duration<double, std::milli>(t1 - t0).count();
    std::cout << "naive_gemm (double acc) time: " << naive_ms << " ms\n";

    t0 = high_resolution_clock::now();
    rec_gemm(A.data(), B.data(), C_rec.data(), M, N, K, K, N, N, threshold);
    t1 = high_resolution_clock::now();
    double rec_ms = duration<double, std::milli>(t1 - t0).count();
    std::cout << "rec_gemm time:                " << rec_ms << " ms\n";

    double frob = frob_norm_diff(C_ref, C_rec);
    std::cout << "Frobenius norm diff: " << std::setprecision(12) << frob << "\n";

    // detailed error statistics
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t max_idx = 0;
    std::vector<std::pair<double,size_t>> diffs; diffs.reserve(C_ref.size());

    for (size_t i = 0; i < C_ref.size(); ++i) {
        double a = double(C_ref[i]);
        double b = double(C_rec[i]);
        double d = std::abs(a - b);
        double rel = d / (std::abs(a) + 1e-12);
        diffs.emplace_back(d, i);
        if (d > max_abs) { max_abs = d; }
        if (rel > max_rel) { max_rel = rel; max_idx = i; }
    }

    std::sort(diffs.begin(), diffs.end(), [](auto &x, auto &y){ return x.first > y.first; });

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "Max abs diff: " << max_abs << " at index (top element): ";
    if (!diffs.empty()) std::cout << diffs[0].second;
    std::cout << "\n";
    std::cout << "Max rel diff: " << max_rel << " at idx " << max_idx << "\n";

    // histogram of relative errors
    int buckets[6] = {0};
    for (size_t i = 0; i < C_ref.size(); ++i) {
        double a = double(C_ref[i]);
        double b = double(C_rec[i]);
        double d = std::abs(a - b);
        double rel = d / (std::abs(a) + 1e-12);
        if (rel < 1e-6) buckets[0]++;
        else if (rel < 1e-5) buckets[1]++;
        else if (rel < 1e-4) buckets[2]++;
        else if (rel < 1e-3) buckets[3]++;
        else if (rel < 1e-2) buckets[4]++;
        else buckets[5]++;
    }
    std::cout << "Relative error buckets:\n";
    std::cout << "  <1e-6: " << buckets[0] << "\n";
    std::cout << "  <1e-5: " << buckets[1] << "\n";
    std::cout << "  <1e-4: " << buckets[2] << "\n";
    std::cout << "  <1e-3: " << buckets[3] << "\n";
    std::cout << "  <1e-2: " << buckets[4] << "\n";
    std::cout << "  >=1e-2: " << buckets[5] << "\n";

    std::cout << "Top-5 absolute diffs (value,index,ref,rec):\n";
    for (int t = 0; t < 5 && t < (int)diffs.size(); ++t) {
        size_t idx = diffs[t].second;
        int i = idx / N;
        int j = idx % N;
        std::cout << "  " << t << ": " << diffs[t].first << " at (" << i << "," << j << ") ref=" << C_ref[idx] << " rec=" << C_rec[idx] << "\n";
    }

    // Diagnose the element with maximum relative error
    {
        size_t idx = max_idx;
        int ii = idx / N;
        int jj = idx % N;
        std::cout << "\nDiagnosing max-relative-error element at (" << ii << "," << jj << ") idx=" << idx << "\n";
        std::cout << "  ref=" << C_ref[idx] << " rec=" << C_rec[idx] << " abs=" << std::abs(double(C_ref[idx]) - double(C_rec[idx])) << " rel=" << max_rel << "\n";

        // compute per-k terms and stats
        std::vector<std::pair<double,int>> term_abs_idx; term_abs_idx.reserve(K);
        std::vector<double> terms(K);
        for (int k = 0; k < K; ++k) {
            double val = double(A[ii * K + k]) * double(B[k * N + jj]);
            terms[k] = val;
            term_abs_idx.emplace_back(std::abs(val), k);
        }
        std::sort(term_abs_idx.begin(), term_abs_idx.end(), [](auto &a, auto &b){ return a.first > b.first; });

        // compute cumulative sums with different accumulation strategies
        double cum_double = 0.0;
        float cum_float_add = 0.0f;
        float cum_float_fma = 0.0f;
        for (int k = 0; k < K; ++k) {
            double dv = terms[k];
            cum_double += dv;
            float af = float(A[ii * K + k]);
            float bf = float(B[k * N + jj]);
            cum_float_add += af * bf;
            cum_float_fma = std::fmaf(af, bf, cum_float_fma);
        }

        std::cout << std::setprecision(12);
        std::cout << "  double-sum final = " << cum_double << "\n";
        std::cout << "  float-add final  = " << cum_float_add << "\n";
        std::cout << "  float-fma final  = " << cum_float_fma << "\n";

        std::cout << "  diff(rec - double) = " << double(C_rec[idx]) - cum_double << "\n";
        std::cout << "  diff(rec - float-add) = " << double(C_rec[idx]) - double(cum_float_add) << "\n";
        std::cout << "  diff(rec - float-fma) = " << double(C_rec[idx]) - double(cum_float_fma) << "\n";

        std::cout << "\n  Top contributing k (abs(term),k,term):\n";
        for (int t = 0; t < 10 && t < (int)term_abs_idx.size(); ++t) {
            int k = term_abs_idx[t].second;
            std::cout << "    " << t << ": " << term_abs_idx[t].first << " , k=" << k << " , term=" << terms[k] << "\n";
        }

        std::cout << "\n  Cumulative samples (k,term,cum_double,cum_float_add,cum_float_fma):\n";
        double cdb = 0.0; float cfa = 0.0f; float cff = 0.0f;
        for (int k = 0; k < K; ++k) {
            cdb += terms[k];
            float af = float(A[ii * K + k]);
            float bf = float(B[k * N + jj]);
            cfa += af * bf;
            cff = std::fmaf(af, bf, cff);
            if (k < 10 || k % 64 == 63) {
                std::cout << "    k=" << k << ": term=" << terms[k] << ", dbl=" << cdb << ", fadd=" << cfa << ", ffma=" << cff << "\n";
            }
        }
    }

    if (frob < 1e-4) {
        std::cout << u8"結果は一致 (許容誤差内)\n";
    } else {
        std::cout << u8"結果に差分あり: 要デバッグ\n";
    }

    std::cout << "Speedup: " << (naive_ms / rec_ms) << "x\n";
    return 0;
}
