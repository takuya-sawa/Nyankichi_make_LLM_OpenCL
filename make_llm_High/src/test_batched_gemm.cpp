#include "../include/dense_tile.h"
#include <vector>
#include <iostream>
#include <cmath>

using namespace make_llm_high;

bool approx_equal(float a, float b, float tol = 1e-5f) {
    float diff = std::fabs(a - b);
    float denom = std::max(std::fabs(a), std::fabs(b));
    if (denom < 1e-6f) return diff < tol;
    return diff / denom < tol;
}

int main() {
    int batch = 4;
    int M = 16, N = 16, K = 32;
    std::vector<float> A(batch * M * K), B(batch * K * N), C(batch * M * N), Cref(batch * M * N);

    // fill
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < M*K; ++i) A[b*M*K + i] = float((b+1) * (i % 13 + 1)) / 17.0f;
        for (int i = 0; i < K*N; ++i) B[b*K*N + i] = float((b+2) * (i % 7 + 1)) / 23.0f;
    }

    std::fill(C.begin(), C.end(), 0.0f);
    std::fill(Cref.begin(), Cref.end(), 0.0f);

    batched_gemm_strided(A.data(), B.data(), C.data(), batch, M, N, K, K, N, N, M*K, K*N, M*N);

    // reference per-batch
    for (int b = 0; b < batch; ++b) {
        naive_gemm(A.data() + b*M*K, B.data() + b*K*N, Cref.data() + b*M*N, M, N, K, K, N, N);
    }

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < M*N; ++i) {
            float a = C[b*M*N + i];
            float r = Cref[b*M*N + i];
            if (!approx_equal(a, r, 1e-4f)) {
                std::cerr << "Mismatch at batch " << b << " idx " << i << " got=" << a << " ref=" << r << "\n";
                return 1;
            }
        }
    }
    std::cout << "Batched GEMM PoC test passed" << std::endl;
    return 0;
}
