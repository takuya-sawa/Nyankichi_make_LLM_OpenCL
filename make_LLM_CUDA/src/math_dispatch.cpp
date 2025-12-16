#include "math_dispatch.h"
#include "math_opencl.cpp" // reuse matmul_opencl implementation
#include "../../make_llm_High/include/dense_tile.h"

void matmul_dispatch(Tensor& C, const Tensor& A, const Tensor& B) {
    int m = A.shape[0];
    int k = A.shape[1];
    int bk = B.shape[0];
    int n = B.shape[1];
    if (k != bk) { std::cerr << "matmul_dispatch: shape mismatch" << std::endl; return; }

    C.shape = {m, n};
    C.size = (size_t)m * n;
    C.h_data.assign(C.size, 0.0f);

    if (cpu_has_avx2_available()) {
        // use optimized CPU path
        make_llm_high::rec_gemm(A.h_data.data(), B.h_data.data(), C.h_data.data(), m, n, k, k, n, n);
    } else {
        // use OpenCL path (may use GPU or CPU fallback)
        matmul_opencl(C, (Tensor&)A, (Tensor&)B);
    }
}
