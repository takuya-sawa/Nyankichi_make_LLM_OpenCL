#include "../include/tensor_opencl.h"
#include "../include/math_opencl.h"
#include <random>

Tensor::Tensor() : size(0), d_data(nullptr) {}

Tensor::Tensor(const std::vector<int>& shape_) : shape(shape_), d_data(nullptr) {
    size = 1;
    for (int dim : shape) size *= dim;
    h_data.resize(size, 0.0f);
}

Tensor::Tensor(const Tensor& other) : shape(other.shape), size(other.size), h_data(other.h_data), d_data(nullptr) {}

Tensor::~Tensor() {
    deallocate();
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        deallocate();
        shape = other.shape;
        size = other.size;
        h_data = other.h_data;
        d_data = nullptr;
    }
    return *this;
}

void Tensor::allocate() {
    // Allocate device buffer if OpenCL context exists
    cl_context ctx = GetOpenCLContext();
    if (!ctx) return; // no-op in CPU-only mode

    // If already allocated, reallocate to match current size (simple and safe)
    if (d_data) {
        clReleaseMemObject(d_data);
        d_data = nullptr;
    }

    cl_int err = 0;
    d_data = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * size, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Tensor::allocate: clCreateBuffer failed (" << err << ")" << std::endl;
        d_data = nullptr;
    }
}

void Tensor::deallocate() {
    if (d_data) {
        clReleaseMemObject(d_data);
        d_data = nullptr;
    }
}

void Tensor::h2d() {
    cl_context ctx = GetOpenCLContext();
    cl_command_queue q = GetOpenCLQueue();
    if (!ctx || !q) return; // no-op
    if (!d_data) allocate();
    if (!d_data) return; // allocation failed

    cl_int err = clEnqueueWriteBuffer(q, d_data, CL_TRUE, 0, sizeof(float) * h_data.size(), h_data.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Tensor::h2d: clEnqueueWriteBuffer failed (" << err << ")" << std::endl;
    }
}

void Tensor::d2h() {
    cl_context ctx = GetOpenCLContext();
    cl_command_queue q = GetOpenCLQueue();
    if (!ctx || !q || !d_data) return; // no-op

    cl_int err = clEnqueueReadBuffer(q, d_data, CL_TRUE, 0, sizeof(float) * h_data.size(), h_data.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Tensor::d2h: clEnqueueReadBuffer failed (" << err << ")" << std::endl;
    }
}
void Tensor::randomInit() {
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 0.01f);
    for (size_t i = 0; i < h_data.size(); ++i) h_data[i] = dist(rng);
}

void Tensor::zero() {
    std::fill(h_data.begin(), h_data.end(), 0.0f);
}

void Tensor::setValue(int idx, float value) {
    if (idx >= 0 && (size_t)idx < h_data.size()) {
        h_data[idx] = value;
    }
}
