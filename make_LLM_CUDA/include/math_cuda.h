#ifndef MATH_CUDA_H
#define MATH_CUDA_H

#include "tensor_cuda.h"
#include <cublas_v2.h>

/// ===================================================================
/// CUDA 数学演算：GPU 最適化された NN 基本関数
/// 
/// cuBLAS を活用した高速行列演算と
/// カスタム CUDA カーネルによる活性化関数実装
/// ===================================================================

// cuBLAS ハンドル（グローバル）
extern cublasHandle_t g_cublas_handle;

// CUDA カーネル宣言
__global__ void kernel_relu(float* data, int size);
__global__ void kernel_relu_backward(float* dx, const float* dy, const float* x, int size);
__global__ void kernel_softmax(float* data, int batch_size, int vocab_size);
__global__ void kernel_cross_entropy_loss(const float* pred, const float* target, 
                                         float* loss_out, int batch_size, int vocab_size);

// CPU インターフェース関数
void InitCublas();
void DestroyCublas();
void matmul_cuda(Tensor& C, const Tensor& A, const Tensor& B);
void relu_cuda(Tensor& x);
void relu_backward_cuda(Tensor& dx, const Tensor& dy, const Tensor& x);
void softmax_cuda(Tensor& x);
float cross_entropy_loss_cuda(const Tensor& predictions, const Tensor& targets);
void cross_entropy_backward_cuda(Tensor& dz, const Tensor& predictions, const Tensor& targets);
void layernorm_cuda(Tensor& output, const Tensor& input, 
                   const Tensor& gamma, const Tensor& beta, float eps = 1e-5f);

#endif // MATH_CUDA_H
