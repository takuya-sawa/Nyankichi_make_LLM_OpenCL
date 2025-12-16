#pragma once
#include "../include/tensor_opencl.h"

// Dispatch matmul to best available backend: AVX2 rec_gemm on CPU when available, otherwise matmul_opencl (which may use GPU or CPU fallback)
void matmul_dispatch(Tensor& C, const Tensor& A, const Tensor& B);
