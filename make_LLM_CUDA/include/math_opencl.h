#ifndef MATH_OPENCL_H
#define MATH_OPENCL_H

#include "tensor_opencl.h"

void matmul_opencl(Tensor& C, Tensor& A, Tensor& B);
void softmax_opencl(Tensor& t);
void layernorm_opencl(Tensor& out, Tensor& x, Tensor& gamma, Tensor& beta);
void relu_opencl(Tensor& x);
void TransposeMatrixOpenCL(Tensor& dst, const Tensor& src);

#include <CL/cl.h>

// Init / Cleanup and device management
int ListOpenCLDevices(); // prints devices and returns count
bool SelectOpenCLDevice(int index); // select device by index (0-based)
int GetSelectedOpenCLDeviceIndex();

// Accessors for other modules
cl_context GetOpenCLContext();
cl_command_queue GetOpenCLQueue();
cl_device_id GetOpenCLDevice();

void InitOpenCL();
void DestroyOpenCL();

#endif // MATH_OPENCL_H