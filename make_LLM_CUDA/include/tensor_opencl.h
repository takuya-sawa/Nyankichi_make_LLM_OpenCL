#ifndef TENSOR_OPENCL_H
#define TENSOR_OPENCL_H

#include <vector>
#include <iostream>
#include <CL/cl.h>

class Tensor {
public:
    std::vector<int> shape;
    size_t size;
    std::vector<float> h_data;
    cl_mem d_data; // OpenCL buffer handle (may be null if not allocated)


    Tensor();
    Tensor(const std::vector<int>& shape_);
    Tensor(const Tensor& other);
    ~Tensor();

    Tensor& operator=(const Tensor& other);

    void allocate();
    void deallocate();

    // Host/GPU sync (no-op for initial CPU-backed stub)
    void h2d();
    void d2h();

    void randomInit();
    void zero();
    void setValue(int idx, float value);
};

#endif // TENSOR_OPENCL_H