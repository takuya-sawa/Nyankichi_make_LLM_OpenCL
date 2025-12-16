#include "../include/dense_tile.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace make_llm_high;

void softmax_rowwise(std::vector<float>& mat, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float maxv = -1e30f;
        for (int c = 0; c < cols; ++c) maxv = std::max(maxv, mat[r*cols + c]);
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) { mat[r*cols + c] = std::exp(mat[r*cols + c] - maxv); sum += mat[r*cols + c]; }
        for (int c = 0; c < cols; ++c) mat[r*cols + c] /= (sum + 1e-9f);
    }
}

int main() {
    int seq = 8;
    int D = 32;
    std::vector<float> x(seq * D), Wq(D*D), Wk(D*D), Wv(D*D), Wo(D*D);
    for (int i = 0; i < seq*D; ++i) x[i] = float((i%13)-6) * 0.031f;
    for (int i = 0; i < D*D; ++i) Wq[i] = float((i%7)-3) * 0.02f, Wk[i]=float((i%5)-2)*0.03f, Wv[i]=float((i%11)-5)*0.017f, Wo[i]=float((i%17)-8)*0.013f;

    std::vector<float> Q(seq*D), K(seq*D), V(seq*D);
    std::vector<float> K_T(D*seq);
    std::vector<float> attn_scores(seq*seq);
    std::vector<float> attn_output(seq*D);
    std::vector<float> out_proj(seq*D);

    // compute Q,K,V using rec_gemm
    rec_gemm(x.data(), Wq.data(), Q.data(), seq, D, D, D, D, D);
    rec_gemm(x.data(), Wk.data(), K.data(), seq, D, D, D, D, D);
    rec_gemm(x.data(), Wv.data(), V.data(), seq, D, D, D, D, D);

    // K_T
    for (int i = 0; i < seq; ++i) for (int j = 0; j < D; ++j) K_T[j*seq + i] = K[i*D + j];

    // attn_scores = Q @ K_T
    rec_gemm(Q.data(), K_T.data(), attn_scores.data(), seq, seq, D, D, seq, seq);
    float scale = 1.0f / std::sqrt((float)D);
    for (auto &v: attn_scores) v *= scale;
    softmax_rowwise(attn_scores, seq, seq);

    // attn_output = attn_scores @ V
    rec_gemm(attn_scores.data(), V.data(), attn_output.data(), seq, D, seq, seq, D, D);

    // out_proj = attn_output @ Wo
    rec_gemm(attn_output.data(), Wo.data(), out_proj.data(), seq, D, D, D, D, D);

    // reference using double
    std::vector<double> Qd(seq*D), Kd(seq*D), Vd(seq*D), K_Td(D*seq), asd(seq*seq), aod(seq*D), opd(seq*D);
    auto matmul_double = [&](const float* A, const float* B, double* C, int M, int N, int K, int lda, int ldb, int ldc){
        for (int i=0;i<M;i++) for (int j=0;j<N;j++) { double s=0; for (int k=0;k<K;k++) s+=double(A[i*lda+k])*double(B[k*ldb+j]); C[i*ldc+j]=s; }
    };
    matmul_double(x.data(), Wq.data(), Qd.data(), seq, D, D, D, D, D);
    matmul_double(x.data(), Wk.data(), Kd.data(), seq, D, D, D, D, D);
    matmul_double(x.data(), Wv.data(), Vd.data(), seq, D, D, D, D, D);
    for (int i=0;i<seq;i++) for (int j=0;j<D;j++) K_Td[j*seq + i] = Kd[i*D + j];
    matmul_double(Qd.data(), K_Td.data(), asd.data(), seq, seq, D, D, seq, seq);
    for (int i=0;i<seq*seq;i++) asd[i]*=scale;
    // softmax rowwise double
    for (int r=0;r<seq;r++){
        double maxv=-1e300; for (int c=0;c<seq;c++) maxv = std::max(maxv, asd[r*seq+c]);
        double sum=0; for (int c=0;c<seq;c++){ asd[r*seq+c]=std::exp(asd[r*seq+c]-maxv); sum+=asd[r*seq+c]; }
        for (int c=0;c<seq;c++) asd[r*seq+c]/=(sum+1e-12);
    }
    matmul_double(asd.data(), Vd.data(), aod.data(), seq, D, seq, seq, D, D);
    matmul_double(aod.data(), (const double*)Wo.data(), opd.data(), seq, D, D, D, D, D); // Wo is float, cast ok

    // compare
    double max_abs=0, max_rel=0;
    for (int i=0;i<seq*D;i++){
        double a = opd[i];
        double b = out_proj[i];
        double ad = std::fabs(a-b);
        double rel = (std::fabs(a)>1e-8)? ad/std::fabs(a) : ad;
        max_abs = std::max(max_abs, ad);
        max_rel = std::max(max_rel, rel);
    }
    std::cout << "Transformer PoC: max_abs=" << max_abs << " max_rel=" << max_rel << std::endl;
    if (max_rel > 1e-3) { std::cerr << "FAILED" << std::endl; return 1; }
    std::cout << "PASS" << std::endl;
    return 0;
}
