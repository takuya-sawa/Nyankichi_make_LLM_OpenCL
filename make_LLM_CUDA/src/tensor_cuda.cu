#include "tensor_cuda.h"
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

/// ===================================================================
/// GPU テンソル実装：CUDA による高速メモリ管理
/// ===================================================================

Tensor::Tensor() : size(0), d_data(nullptr) {}

Tensor::Tensor(const std::vector<int>& shape_) : shape(shape_), d_data(nullptr)
{
    // 形状から総要素数を計算
    size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    h_data.resize(size, 0.0f);
    allocate();
    // GPUメモリもゼロで初期化
    if (d_data != nullptr) {
        cudaMemset(d_data, 0, size * sizeof(float));
    }
}

Tensor::Tensor(const Tensor& other) : shape(other.shape), size(other.size), d_data(nullptr)
{
    printf("[TRACE] Copy constructor called: other.size=%zu, other.d_data=%p\n", other.size, other.d_data);
    h_data = other.h_data;
    allocate();
    printf("[TRACE] After allocate: this.size=%zu, this.d_data=%p\n", size, d_data);
    
    // h_dataに有効なデータがある場合は、それを使ってGPUを初期化（returnでd2h()された場合）
    // そうでなければ、other.d_dataから直接コピー
    if (!h_data.empty() && d_data != nullptr) {
        cudaError_t err = cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("[ERROR] Copy constructor h2d failed: %s\n", cudaGetErrorString(err));
        } else {
            printf("[TRACE] Copy constructor: used h_data to initialize GPU (%zu bytes)\n", size * sizeof(float));
        }
    } else if (other.d_data != nullptr && d_data != nullptr) {
        cudaError_t err = cudaMemcpy(d_data, other.d_data, size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            printf("[ERROR] Copy constructor d2d failed: %s\n", cudaGetErrorString(err));
        } else {
            printf("[TRACE] Copy constructor: used other.d_data to copy GPU (%zu bytes)\n", size * sizeof(float));
        }
    } else {
        printf("[WARNING] Copy constructor: no valid source data\n");
    }
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if (this != &other) {
        printf("[TRACE] Assignment operator called: other.size=%zu, other.d_data=%p\n", other.size, other.d_data);
        deallocate();
        shape = other.shape;
        size = other.size;
        h_data = other.h_data;
        allocate();
        printf("[TRACE] After allocate: this.size=%zu, this.d_data=%p\n", size, d_data);
        
        // h_dataに有効なデータがある場合は、それを使ってGPUを初期化（returnでd2h()された場合）
        // そうでなければ、other.d_dataから直接コピー
        if (!h_data.empty() && d_data != nullptr) {
            cudaError_t err = cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                printf("[ERROR] Assignment operator h2d failed: %s\n", cudaGetErrorString(err));
            } else {
                printf("[TRACE] Assignment operator: used h_data to initialize GPU (%zu bytes)\n", size * sizeof(float));
            }
        } else if (other.d_data != nullptr && d_data != nullptr) {
            cudaError_t err = cudaMemcpy(d_data, other.d_data, size * sizeof(float), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                printf("[ERROR] Assignment operator d2d failed: %s\n", cudaGetErrorString(err));
            } else {
                printf("[TRACE] Assignment operator: used other.d_data to copy GPU (%zu bytes)\n", size * sizeof(float));
            }
        } else {
            printf("[WARNING] Assignment operator: no valid source data\n");
        }
    } else {
        printf("[TRACE] Assignment operator: self-assignment\n");
    }
    return *this;
}

Tensor::~Tensor()
{
    if (d_data != nullptr) {
        printf("[TRACE] Destructor called: size=%zu, d_data=%p (will cudaFree)\n", size, d_data);
    }
    deallocate();
}

void Tensor::allocate()
{
    if (size > 0 && d_data == nullptr) {
        cudaMalloc((void**)&d_data, size * sizeof(float));
    }
}

void Tensor::deallocate()
{
    if (d_data != nullptr) {
        cudaFree(d_data);
        d_data = nullptr;
    }
}

void Tensor::h2d()
{
    // CPU から GPU へメモリコピー
    if (d_data != nullptr && !h_data.empty()) {
        cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    }
}

void Tensor::d2h()
{
    // GPU から CPU へメモリコピー
    if (d_data != nullptr && size > 0) {
        // h_dataが空の場合は適切なサイズにリサイズ
        if (h_data.size() != size) {
            h_data.resize(size);
        }
        cudaMemcpy(h_data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

// ランダム初期化用CUDA カーネル
__global__ void kernel_randomInit(float* data, int size, curandState* states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState localState = states[idx];
        data[idx] = (curand_normal(&localState) * 0.01f);
        states[idx] = localState;
    }
}

void Tensor::randomInit()
{
    // Xavier/He初期化: std = sqrt(2.0 / (fan_in + fan_out))
    float fan_in = (shape.size() >= 2) ? shape[0] : size;
    float fan_out = (shape.size() >= 2) ? shape[1] : size;
    float std_dev = sqrtf(2.0f / (fan_in + fan_out));
    
    // CPU側で初期化（Box-Muller変換の簡易版）
    for (size_t i = 0; i < h_data.size(); i += 2) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        float z0 = std_dev * sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * 3.14159265f * u2);
        float z1 = std_dev * sqrtf(-2.0f * logf(u1 + 1e-10f)) * sinf(2.0f * 3.14159265f * u2);
        h_data[i] = z0;
        if (i + 1 < h_data.size()) {
            h_data[i + 1] = z1;
        }
    }
    h2d();
}

void Tensor::zero()
{
    // ゼロ初期化（GPU）
    cudaMemset(d_data, 0, size * sizeof(float));
    std::fill(h_data.begin(), h_data.end(), 0.0f);
}

void Tensor::setValue(int idx, float value)
{
    if (idx >= 0 && idx < (int)size) {
        h_data[idx] = value;
        cudaMemcpy(d_data + idx, &value, sizeof(float), cudaMemcpyHostToDevice);
    }
}
