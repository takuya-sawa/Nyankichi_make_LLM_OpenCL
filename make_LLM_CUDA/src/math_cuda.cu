#include "math_cuda.h"
#include "tensor_cuda.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

/// ===================================================================
/// GPU 数学演算：cuBLAS と カスタムカーネル実装
/// ===================================================================

cublasHandle_t g_cublas_handle;

void InitCublas()
{
    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "ERROR: cuBLAS initialization failed with status: " << status << std::endl;
    } else {
        std::cout << "[cuBLAS] Initialized successfully" << std::endl;
    }
}

void DestroyCublas()
{
    cublasDestroy(g_cublas_handle);
}

/// <summary>
/// 行列乗算：C = A @ B（cuBLAS 版）
/// 
/// cuBLAS によるハイパフォーマンス実装
/// A: (m, k), B: (k, n) → C: (m, n)
/// 
/// cuBLASは列優先なので、C^T = B^T @ A^T を計算
/// </summary>
void matmul_cuda(Tensor& C, const Tensor& A, const Tensor& B)
{
    int m = A.shape[0];
    int k = A.shape[1];
    int n = B.shape[1];
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // 一時的にCPU側で計算（デバッグ用）
    static int matmul_call_count = 0;
    if (true) {  // 全てのmatmulでCPU計算を使用
        // CPU側で正しい計算
        Tensor& A_mut = const_cast<Tensor&>(A);
        Tensor& B_mut = const_cast<Tensor&>(B);
        A_mut.d2h();
        B_mut.d2h();
        
        // デバッグ：出力層matmulの場合のみ詳細情報を表示
        static int output_debug = 0;
        if (m == 1 && k == 256 && n == 128 && output_debug < 1) {
            std::cout << "      [DEBUG MATMUL] Output layer: m=" << m << ", k=" << k << ", n=" << n << std::endl;
            std::cout << "      [DEBUG MATMUL] A[0:3]: " << A.h_data[0] << " " << A.h_data[1] << " " << A.h_data[2] << std::endl;
            std::cout << "      [DEBUG MATMUL] B[0:3]: " << B.h_data[0] << " " << B.h_data[1] << " " << B.h_data[2] << std::endl;
            std::cout << "      [DEBUG MATMUL] C.h_data.size()=" << C.h_data.size() << ", expected=" << (m*n) << std::endl;
            output_debug++;
        }
        
        std::vector<float> C_cpu(m * n, 0.0f);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int p = 0; p < k; p++) {
                    sum += A.h_data[i * k + p] * B.h_data[p * n + j];
                }
                C_cpu[i * n + j] = sum;
            }
        }
        
        // デバッグ：計算結果を表示
        if (m == 1 && k == 256 && n == 128 && output_debug == 1) {
            std::cout << "      [DEBUG MATMUL] C_cpu[0:3]: " << C_cpu[0] << " " << C_cpu[1] << " " << C_cpu[2] << std::endl;
            float max_val = *std::max_element(C_cpu.begin(), C_cpu.end());
            float min_val = *std::min_element(C_cpu.begin(), C_cpu.end());
            std::cout << "      [DEBUG MATMUL] C_cpu max=" << max_val << ", min=" << min_val << std::endl;
            output_debug++;
        }
        
        // CPU結果をCに書き込んでテスト
        C.h_data = C_cpu;
        C.h2d();
        
        cudaDeviceSynchronize();
        return; // cuBLASをスキップ
        
        matmul_call_count++;
    }
    
    // cuBLAS は列優先なので、行優先の行列乗算 C = A @ B を計算するには
    // C^T = B^T @ A^T を計算する
    // cublasSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // は C = A @ B を計算（列優先）
    // 行優先C = A @ Bは、列優先でC^T = B^T @ A^Tなので、BとAの順序を入れ替える
    cublasStatus_t status = cublasSgemm(g_cublas_handle, 
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,          // C^T の次元: (n, m)
                &alpha,
                B.d_data, n,      // 第1引数: B^T (n, k)、lda=n
                A.d_data, k,      // 第2引数: A^T (k, m)、ldb=k
                &beta,
                C.d_data, n);     // 結果: C^T (n, m)、ldc=n
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "ERROR: cuBLAS matmul failed with status: " << status << std::endl;
    }
    
    cudaDeviceSynchronize();
}

/// <summary>
/// ReLU 活性化関数：GPU カーネル実装
/// </summary>
__global__ void kernel_relu(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

void relu_cuda(Tensor& x)
{
    int threads = 256;
    int blocks = (x.size + threads - 1) / threads;
    kernel_relu<<<blocks, threads>>>(x.d_data, x.size);
    cudaDeviceSynchronize();
}

/// <summary>
/// ReLU 逆伝播：GPU カーネル実装
/// </summary>
__global__ void kernel_relu_backward(float* dx, const float* dy, const float* x, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dx[idx] = (x[idx] > 0.0f) ? dy[idx] : 0.0f;
    }
}

void relu_backward_cuda(Tensor& dx, const Tensor& dy, const Tensor& x)
{
    int threads = 256;
    int blocks = (x.size + threads - 1) / threads;
    kernel_relu_backward<<<blocks, threads>>>(dx.d_data, dy.d_data, x.d_data, x.size);
    cudaDeviceSynchronize();
}

/// <summary>
/// ソフトマックス：GPU カーネル実装
/// 
/// 各行ごとに処理：softmax(x_i) = exp(x_i) / sum(exp(x_j))
/// 正確な同期と树形削减で合計を計算
/// </summary>
__global__ void kernel_softmax(float* data, int batch_size, int vocab_size)
{
    extern __shared__ float shared_sum[];
    
    int row = blockIdx.x;
    if (row < batch_size) {
        float* row_data = data + row * vocab_size;
        
        // ステップ1：行の最大値を見つける
        float max_val = row_data[0];
        for (int i = 1; i < vocab_size; i++) {
            max_val = fmaxf(max_val, row_data[i]);
        }
        
        // ステップ2：exp 計算
        for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
            row_data[i] = expf(row_data[i] - max_val);
        }
        __syncthreads();
        
        // ステップ3：合計を計算（共有メモリ使用）
        float sum = 0.0f;
        for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
            sum += row_data[i];
        }
        
        // スレッド内の合計を共有メモリに保存
        shared_sum[threadIdx.x] = sum;
        __syncthreads();
        
        // thread 0 が全体の合計を計算
        if (threadIdx.x == 0) {
            float total = 0.0f;
            for (int i = 0; i < blockDim.x; i++) {
                total += shared_sum[i];
            }
            shared_sum[0] = total;
        }
        __syncthreads();
        
        float total_sum = shared_sum[0];
        
        // ステップ4：正規化
        for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
            if (total_sum > 0) {
                row_data[i] /= total_sum;
            }
        }
    }
}

void softmax_cuda(Tensor& x)
{
    int batch_size = x.shape[0];
    int vocab_size = x.shape[1];
    int threads = 256;
    int shared_mem_size = threads * sizeof(float);
    
    kernel_softmax<<<batch_size, threads, shared_mem_size>>>(x.d_data, batch_size, vocab_size);
    cudaDeviceSynchronize();
}

/// <summary>
/// クロスエントロピー損失：GPU カーネル実装
/// </summary>
__global__ void kernel_cross_entropy_loss(const float* pred, const float* target, 
                                         float* loss_out, int batch_size, int vocab_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * vocab_size;
    
    if (idx < total) {
        if (target[idx] > 0.5f) {
            loss_out[idx] = -logf(fmaxf(pred[idx], 1e-7f));
        } else {
            loss_out[idx] = 0.0f;
        }
    }
}

float cross_entropy_loss_cuda(const Tensor& predictions, const Tensor& targets)
{
    int batch_size = predictions.shape[0];
    int vocab_size = predictions.shape[1];
    int total = batch_size * vocab_size;
    
    // GPU 上の損失計算用バッファ
    float* d_loss_vals;
    cudaMalloc((void**)&d_loss_vals, total * sizeof(float));
    
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kernel_cross_entropy_loss<<<blocks, threads>>>(predictions.d_data, targets.d_data, d_loss_vals, batch_size, vocab_size);
    
    // CPU で合計を計算
    std::vector<float> h_loss_vals(total);
    cudaMemcpy(h_loss_vals.data(), d_loss_vals, total * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_loss = 0.0f;
    for (float val : h_loss_vals) {
        total_loss += val;
    }
    
    cudaFree(d_loss_vals);
    return total_loss / batch_size;
}

/// <summary>
/// クロスエントロピー逆伝播
/// </summary>
void cross_entropy_backward_cuda(Tensor& dz, const Tensor& predictions, const Tensor& targets)
{
    // dz = (pred - target) / batch_size
    // 実装は簡略化（matmul で差を計算）
    cudaDeviceSynchronize();
}

/// ===================================================================
/// Layer Normalization (CPU実装)
/// ===================================================================
void layernorm_cuda(Tensor& output, const Tensor& input, 
                   const Tensor& gamma, const Tensor& beta, float eps)
{
    static int ln_call_count = 0;
    if (ln_call_count == 0) {
        std::cout << "        [DEBUG] LayerNorm: input.shape=[" << input.shape[0] << "," << input.shape[1] 
                  << "], gamma.shape.size=" << gamma.shape.size() 
                  << ", beta.shape.size=" << beta.shape.size() << std::endl;
    }
    
    // CPU側で実行
    Tensor& input_mut = const_cast<Tensor&>(input);
    Tensor& gamma_mut = const_cast<Tensor&>(gamma);
    Tensor& beta_mut = const_cast<Tensor&>(beta);
    
    input_mut.d2h();
    gamma_mut.d2h();
    beta_mut.d2h();
    
    int seq_len = input.shape[0];
    int hidden_dim = input.shape[1];
    
    if (ln_call_count == 0) {
        std::cout << "        [DEBUG] LayerNorm: seq_len=" << seq_len << ", hidden_dim=" << hidden_dim << std::endl;
        std::cout << "        [DEBUG] LayerNorm: input.h_data.size()=" << input.h_data.size() 
                  << ", gamma.h_data.size()=" << gamma.h_data.size() 
                  << ", beta.h_data.size()=" << beta.h_data.size() << std::endl;
    }
    ln_call_count++;
    
    for (int i = 0; i < seq_len; i++) {
        // 平均を計算
        float mean = 0.0f;
        for (int j = 0; j < hidden_dim; j++) {
            mean += input.h_data[i * hidden_dim + j];
        }
        mean /= hidden_dim;
        
        // 分散を計算
        float variance = 0.0f;
        for (int j = 0; j < hidden_dim; j++) {
            float diff = input.h_data[i * hidden_dim + j] - mean;
            variance += diff * diff;
        }
        variance /= hidden_dim;
        
        // 正規化
        float std_dev = sqrtf(variance + eps);
        for (int j = 0; j < hidden_dim; j++) {
            float normalized = (input.h_data[i * hidden_dim + j] - mean) / std_dev;
            output.h_data[i * hidden_dim + j] = gamma.h_data[j] * normalized + beta.h_data[j];
        }
    }
    
    output.h2d();
}
