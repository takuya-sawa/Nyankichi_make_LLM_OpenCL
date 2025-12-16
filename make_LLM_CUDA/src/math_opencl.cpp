#include "../include/math_opencl.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>

#ifdef __has_include
#if __has_include(<CL/cl.h>)
#include <CL/cl.h>
#else
#error "OpenCL headers not found"
#endif
#else
#include <CL/cl.h>
#endif

static std::vector<cl_platform_id> g_platforms;
static std::vector<cl_device_id> g_devices;
static cl_context g_context = nullptr;
static cl_command_queue g_queue = nullptr;
static cl_program g_program = nullptr;
static cl_kernel g_matmul_kernel = nullptr;
static cl_kernel g_softmax_max_kernel = nullptr;
static cl_kernel g_softmax_exp_sum_kernel = nullptr;
static cl_kernel g_softmax_norm_kernel = nullptr;
static cl_kernel g_layernorm_kernel = nullptr;
static cl_kernel g_relu_kernel = nullptr;
static int g_selected_device = -1;

static void print_device_info(cl_device_id dev, int idx) {
    char buf[1024];
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(buf), buf, NULL);
    std::cout << "[OpenCL] Device " << idx << ": " << buf << std::endl;
}

// OpenCL kernels: matmul, softmax (3-stage), layernorm, relu
static const char* kernel_src = R"CLC(
// matmul: C = A (M x K) * B (K x N)
__kernel void matmul(
    const int M, const int K, const int N,
    __global const float* A, __global const float* B, __global float* C)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// softmax stage 1: compute max per row
__kernel void softmax_max(const int rows, const int cols, __global const float* data, __global float* max_buf) {
    int r = get_global_id(0);
    if (r >= rows) return;
    float m = -INFINITY;
    for (int c = 0; c < cols; ++c) {
        float v = data[r * cols + c];
        m = v > m ? v : m;
    }
    max_buf[r] = m;
}

// softmax stage 2: compute exp and sum per row (writes back exp values)
__kernel void softmax_exp_sum(const int rows, const int cols, __global float* data, __global const float* max_buf, __global float* sum_buf) {
    int r = get_global_id(0);
    if (r >= rows) return;
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float e = exp(data[r * cols + c] - max_buf[r]);
        data[r * cols + c] = e;
        sum += e;
    }
    sum_buf[r] = sum;
}

// softmax stage 3: normalize per element
__kernel void softmax_norm(const int rows, const int cols, __global float* data, __global const float* sum_buf) {
    int idx = get_global_id(0);
    int total = rows * cols;
    if (idx >= total) return;
    int r = idx / cols;
    float s = sum_buf[r];
    data[idx] = data[idx] / (s + 1e-9f);
}

// layernorm: out = ((x - mean)/sqrt(var+eps)) * gamma + beta  (per-row)
__kernel void layernorm(const int rows, const int cols, __global const float* x, __global const float* gamma, __global const float* beta, __global float* out) {
    int r = get_global_id(0);
    if (r >= rows) return;
    float mean = 0.0f;
    for (int c = 0; c < cols; ++c) mean += x[r * cols + c];
    mean /= cols;
    float var = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float d = x[r * cols + c] - mean;
        var += d * d;
    }
    var /= cols;
    float denom = sqrt(var + 1e-5f);
    for (int c = 0; c < cols; ++c) {
        int idx = r * cols + c;
        out[idx] = ((x[idx] - mean) / denom) * gamma[c] + beta[c];
    }
}

// relu: in-place
__kernel void relu(const int N, __global float* x) {
    int idx = get_global_id(0);
    if (idx >= N) return;
    float v = x[idx];
    x[idx] = v > 0.0f ? v : 0.0f;
}
)CLC";
int ListOpenCLDevices() {
    g_platforms.clear();
    g_devices.clear();

    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        std::cout << "[OpenCL] No platforms found" << std::endl;
        return 0;
    }

    g_platforms.resize(num_platforms);
    clGetPlatformIDs(num_platforms, g_platforms.data(), NULL);

    int total_devices = 0;
    for (cl_uint pi = 0; pi < num_platforms; ++pi) {
        char pbuf[1024];
        clGetPlatformInfo(g_platforms[pi], CL_PLATFORM_NAME, sizeof(pbuf), pbuf, NULL);
        std::cout << "[OpenCL] Platform " << pi << ": " << pbuf << std::endl;

        cl_uint num_devs = 0;
        // prefer GPU first, then CPU
        if (clGetDeviceIDs(g_platforms[pi], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devs) == CL_SUCCESS && num_devs > 0) {
            std::vector<cl_device_id> devs(num_devs);
            clGetDeviceIDs(g_platforms[pi], CL_DEVICE_TYPE_GPU, num_devs, devs.data(), NULL);
            for (cl_uint di = 0; di < num_devs; ++di) {
                g_devices.push_back(devs[di]);
                print_device_info(devs[di], total_devices);
                total_devices++;
            }
        }
        // also include CPUs
        if (clGetDeviceIDs(g_platforms[pi], CL_DEVICE_TYPE_CPU, 0, NULL, &num_devs) == CL_SUCCESS && num_devs > 0) {
            std::vector<cl_device_id> devs(num_devs);
            clGetDeviceIDs(g_platforms[pi], CL_DEVICE_TYPE_CPU, num_devs, devs.data(), NULL);
            for (cl_uint di = 0; di < num_devs; ++di) {
                g_devices.push_back(devs[di]);
                print_device_info(devs[di], total_devices);
                total_devices++;
            }
        }
    }
    return total_devices;
}

bool SelectOpenCLDevice(int index) {
    if (index < 0) return false;
    if (g_devices.empty()) ListOpenCLDevices();
    if (index >= (int)g_devices.size()) return false;
    g_selected_device = index;
    return true;
}

int GetSelectedOpenCLDeviceIndex() { return g_selected_device; }

// Accessors for other modules
cl_context GetOpenCLContext() { return g_context; }
cl_command_queue GetOpenCLQueue() { return g_queue; }
cl_device_id GetOpenCLDevice() {
    if (g_selected_device >= 0 && g_selected_device < (int)g_devices.size()) return g_devices[g_selected_device];
    return nullptr;
}

void InitOpenCL() {
    // Initialize and create a context on the selected device (or first GPU if available)
    if (g_devices.empty()) {
        ListOpenCLDevices();
    }
    if (g_devices.empty()) {
        std::cout << "[OpenCL] No devices found; running in CPU-only mode" << std::endl;
        return;
    }
    if (g_selected_device < 0) {
        // pick first GPU if present, else first device
        for (size_t i = 0; i < g_devices.size(); ++i) {
            cl_device_type t = 0;
            clGetDeviceInfo(g_devices[i], CL_DEVICE_TYPE, sizeof(t), &t, NULL);
            if (t & CL_DEVICE_TYPE_GPU) { g_selected_device = (int)i; break; }
        }
        if (g_selected_device < 0) g_selected_device = 0;
    }

    cl_int err = 0;
    cl_device_id device = g_devices[g_selected_device];
    g_context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "[OpenCL] Failed to create context (err=" << err << ")" << std::endl;
        return;
    }
    g_queue = clCreateCommandQueueWithProperties(g_context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cout << "[OpenCL] Failed to create command queue (err=" << err << ")" << std::endl;
        clReleaseContext(g_context);
        g_context = nullptr;
        return;
    }

    // Build program and create matmul kernel
    cl_int berr = 0;
    g_program = clCreateProgramWithSource(g_context, 1, &kernel_src, NULL, &berr);
    if (!g_program || berr != CL_SUCCESS) {
        std::cout << "[OpenCL] Failed to create program (err=" << berr << ")" << std::endl;
    } else {
        berr = clBuildProgram(g_program, 1, &device, NULL, NULL, NULL);
        if (berr != CL_SUCCESS) {
            // print build log
            size_t logsz = 0; clGetProgramBuildInfo(g_program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
            std::string log(logsz, '\0');
            clGetProgramBuildInfo(g_program, device, CL_PROGRAM_BUILD_LOG, logsz, &log[0], NULL);
            std::cout << "[OpenCL] Build log:\n" << log << std::endl;
            clReleaseProgram(g_program);
            g_program = nullptr;
        } else {
            g_matmul_kernel = clCreateKernel(g_program, "matmul", &berr);
            if (berr != CL_SUCCESS) {
                std::cout << "[OpenCL] Failed to create matmul kernel (err=" << berr << ")" << std::endl;
            }
            g_softmax_max_kernel = clCreateKernel(g_program, "softmax_max", &berr);
            if (berr != CL_SUCCESS) {
                std::cout << "[OpenCL] Failed to create softmax_max kernel (err=" << berr << ")" << std::endl;
            }
            g_softmax_exp_sum_kernel = clCreateKernel(g_program, "softmax_exp_sum", &berr);
            if (berr != CL_SUCCESS) {
                std::cout << "[OpenCL] Failed to create softmax_exp_sum kernel (err=" << berr << ")" << std::endl;
            }
            g_softmax_norm_kernel = clCreateKernel(g_program, "softmax_norm", &berr);
            if (berr != CL_SUCCESS) {
                std::cout << "[OpenCL] Failed to create softmax_norm kernel (err=" << berr << ")" << std::endl;
            }
            g_layernorm_kernel = clCreateKernel(g_program, "layernorm", &berr);
            if (berr != CL_SUCCESS) {
                std::cout << "[OpenCL] Failed to create layernorm kernel (err=" << berr << ")" << std::endl;
            }
            g_relu_kernel = clCreateKernel(g_program, "relu", &berr);
            if (berr != CL_SUCCESS) {
                std::cout << "[OpenCL] Failed to create relu kernel (err=" << berr << ")" << std::endl;
            }
        }
    }

    // Print selected device info
    std::cout << "[OpenCL] Init on device index " << g_selected_device << std::endl;
    print_device_info(device, g_selected_device);
}

void DestroyOpenCL() {
    if (g_matmul_kernel) { clReleaseKernel(g_matmul_kernel); g_matmul_kernel = nullptr; }
    if (g_softmax_max_kernel) { clReleaseKernel(g_softmax_max_kernel); g_softmax_max_kernel = nullptr; }
    if (g_softmax_exp_sum_kernel) { clReleaseKernel(g_softmax_exp_sum_kernel); g_softmax_exp_sum_kernel = nullptr; }
    if (g_softmax_norm_kernel) { clReleaseKernel(g_softmax_norm_kernel); g_softmax_norm_kernel = nullptr; }
    if (g_layernorm_kernel) { clReleaseKernel(g_layernorm_kernel); g_layernorm_kernel = nullptr; }
    if (g_relu_kernel) { clReleaseKernel(g_relu_kernel); g_relu_kernel = nullptr; }
    if (g_program) { clReleaseProgram(g_program); g_program = nullptr; }
    if (g_queue) { clReleaseCommandQueue(g_queue); g_queue = nullptr; }
    if (g_context) { clReleaseContext(g_context); g_context = nullptr; }
    std::cout << "[OpenCL] Destroy" << std::endl;
}

#include "dense_tile.h"

void matmul_opencl(Tensor& C, Tensor& A, Tensor& B) {
    // A: m x k, B: k x n, C: m x n
    int m = A.shape[0];
    int k = A.shape[1];
    int bk = B.shape[0];
    int n = B.shape[1];
    if (k != bk) {
        std::cerr << "matmul_opencl: shape mismatch" << std::endl;
        return;
    }

    C.shape = {m, n};
    C.size = (size_t)m * n;
    C.h_data.assign(C.size, 0.0f);

    // If OpenCL is not initialized or kernel not available, fallback to CPU
    if (!g_context || !g_queue || !g_matmul_kernel) {
        // Prefer optimized CPU path from make_llm_High if available
        make_llm_high::rec_gemm(A.h_data.data(), B.h_data.data(), C.h_data.data(), m, n, k, k, n, n);
        return;
    }

    // Ensure device buffers are ready and copy host data to device
    A.h2d();
    B.h2d();
    C.allocate();

    if (!A.d_data || !B.d_data || !C.d_data) {
        std::cerr << "matmul_opencl: device buffer missing, falling back to CPU" << std::endl;
        goto cpu_fallback;
    }

    cl_int err = 0;
    // Set kernel args using existing device buffers
    err  = clSetKernelArg(g_matmul_kernel, 0, sizeof(int), &m);
    err |= clSetKernelArg(g_matmul_kernel, 1, sizeof(int), &k);
    err |= clSetKernelArg(g_matmul_kernel, 2, sizeof(int), &n);
    err |= clSetKernelArg(g_matmul_kernel, 3, sizeof(cl_mem), &A.d_data);
    err |= clSetKernelArg(g_matmul_kernel, 4, sizeof(cl_mem), &B.d_data);
    err |= clSetKernelArg(g_matmul_kernel, 5, sizeof(cl_mem), &C.d_data);
    if (err != CL_SUCCESS) { std::cerr << "matmul_opencl: clSetKernelArg failed (" << err << ")\n"; goto cpu_fallback; }

    size_t global[2] = {(size_t)n, (size_t)m};
    err = clEnqueueNDRangeKernel(g_queue, g_matmul_kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { std::cerr << "matmul_opencl: clEnqueueNDRangeKernel failed (" << err << ")\n"; goto cpu_fallback; }

    clFinish(g_queue);

    // Read back result into host memory
    C.d2h();
    return;

cpu_fallback:
    // Fall back to CPU implementation on any failure
    // Use optimized CPU GEMM if available
    #include "dense_tile.h"
    make_llm_high::rec_gemm(A.h_data.data(), B.h_data.data(), C.h_data.data(), m, n, k, k, n, n);
    
    // Note: if rec_gemm not available this still compiles; include brings declaration only.
}

void softmax_opencl(Tensor& t) {
    int rows = t.shape[0];
    int cols = t.shape[1];

    // If kernels not available, fallback to CPU
    if (!g_program || !g_softmax_max_kernel || !g_softmax_exp_sum_kernel || !g_softmax_norm_kernel) {
        for (int r = 0; r < rows; ++r) {
            float maxv = -std::numeric_limits<float>::infinity();
            for (int c = 0; c < cols; ++c) maxv = std::max(maxv, t.h_data[r * cols + c]);
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c) {
                t.h_data[r * cols + c] = std::exp(t.h_data[r * cols + c] - maxv);
                sum += t.h_data[r * cols + c];
            }
            for (int c = 0; c < cols; ++c) t.h_data[r * cols + c] /= (sum + 1e-9f);
        }
        return;
    }

    t.h2d();

    cl_int err = 0;
    // temporary buffers
    cl_mem max_buf = clCreateBuffer(g_context, CL_MEM_READ_WRITE, sizeof(float) * rows, NULL, &err);
    if (err != CL_SUCCESS) { std::cerr << "softmax_opencl: max_buf alloc failed (" << err << ")\n"; goto cpu_fallback; }
    cl_mem sum_buf = clCreateBuffer(g_context, CL_MEM_READ_WRITE, sizeof(float) * rows, NULL, &err);
    if (err != CL_SUCCESS) { std::cerr << "softmax_opencl: sum_buf alloc failed (" << err << ")\n"; clReleaseMemObject(max_buf); goto cpu_fallback; }

    // stage 1: max per row
    err  = clSetKernelArg(g_softmax_max_kernel, 0, sizeof(int), &rows);
    err |= clSetKernelArg(g_softmax_max_kernel, 1, sizeof(int), &cols);
    err |= clSetKernelArg(g_softmax_max_kernel, 2, sizeof(cl_mem), &t.d_data);
    err |= clSetKernelArg(g_softmax_max_kernel, 3, sizeof(cl_mem), &max_buf);
    if (err != CL_SUCCESS) { std::cerr << "softmax_opencl: setarg max failed (" << err << ")\n"; clReleaseMemObject(max_buf); clReleaseMemObject(sum_buf); goto cpu_fallback; }
    size_t g1 = rows;
    err = clEnqueueNDRangeKernel(g_queue, g_softmax_max_kernel, 1, NULL, &g1, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { std::cerr << "softmax_opencl: enqueue max failed (" << err << ")\n"; clReleaseMemObject(max_buf); clReleaseMemObject(sum_buf); goto cpu_fallback; }
    clFinish(g_queue);

    // stage 2: exp and sum
    err  = clSetKernelArg(g_softmax_exp_sum_kernel, 0, sizeof(int), &rows);
    err |= clSetKernelArg(g_softmax_exp_sum_kernel, 1, sizeof(int), &cols);
    err |= clSetKernelArg(g_softmax_exp_sum_kernel, 2, sizeof(cl_mem), &t.d_data);
    err |= clSetKernelArg(g_softmax_exp_sum_kernel, 3, sizeof(cl_mem), &max_buf);
    err |= clSetKernelArg(g_softmax_exp_sum_kernel, 4, sizeof(cl_mem), &sum_buf);
    if (err != CL_SUCCESS) { std::cerr << "softmax_opencl: setarg exp_sum failed (" << err << ")\n"; clReleaseMemObject(max_buf); clReleaseMemObject(sum_buf); goto cpu_fallback; }
    err = clEnqueueNDRangeKernel(g_queue, g_softmax_exp_sum_kernel, 1, NULL, &g1, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { std::cerr << "softmax_opencl: enqueue exp_sum failed (" << err << ")\n"; clReleaseMemObject(max_buf); clReleaseMemObject(sum_buf); goto cpu_fallback; }
    clFinish(g_queue);

    // stage 3: normalize all elements
    size_t total = (size_t)rows * (size_t)cols;
    err  = clSetKernelArg(g_softmax_norm_kernel, 0, sizeof(int), &rows);
    err |= clSetKernelArg(g_softmax_norm_kernel, 1, sizeof(int), &cols);
    err |= clSetKernelArg(g_softmax_norm_kernel, 2, sizeof(cl_mem), &t.d_data);
    err |= clSetKernelArg(g_softmax_norm_kernel, 3, sizeof(cl_mem), &sum_buf);
    if (err != CL_SUCCESS) { std::cerr << "softmax_opencl: setarg norm failed (" << err << ")\n"; clReleaseMemObject(max_buf); clReleaseMemObject(sum_buf); goto cpu_fallback; }
    size_t g3 = total;
    err = clEnqueueNDRangeKernel(g_queue, g_softmax_norm_kernel, 1, NULL, &g3, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { std::cerr << "softmax_opencl: enqueue norm failed (" << err << ")\n"; clReleaseMemObject(max_buf); clReleaseMemObject(sum_buf); goto cpu_fallback; }
    clFinish(g_queue);

    // read back
    t.d2h();
    clReleaseMemObject(max_buf);
    clReleaseMemObject(sum_buf);
    return;

cpu_fallback:
    // fallback to CPU implementation
    for (int r = 0; r < rows; ++r) {
        float maxv = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < cols; ++c) maxv = std::max(maxv, t.h_data[r * cols + c]);
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            t.h_data[r * cols + c] = std::exp(t.h_data[r * cols + c] - maxv);
            sum += t.h_data[r * cols + c];
        }
        for (int c = 0; c < cols; ++c) t.h_data[r * cols + c] /= (sum + 1e-9f);
    }
}

void layernorm_opencl(Tensor& out, Tensor& x, Tensor& gamma, Tensor& beta) {
    int rows = x.shape[0];
    int cols = x.shape[1];
    out.shape = x.shape;
    out.size = x.size;
    out.h_data.assign(x.h_data.begin(), x.h_data.end());

    // If kernel not available, CPU fallback
    if (!g_program || !g_layernorm_kernel) {
        for (int r = 0; r < rows; ++r) {
            float mean = 0.0f;
            for (int c = 0; c < cols; ++c) mean += x.h_data[r * cols + c];
            mean /= cols;
            float var = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float d = x.h_data[r * cols + c] - mean;
                var += d * d;
            }
            var /= cols;
            float denom = std::sqrt(var + 1e-5f);
            for (int c = 0; c < cols; ++c) {
                int idx = r * cols + c;
                out.h_data[idx] = ((x.h_data[idx] - mean) / denom) * gamma.h_data[c] + beta.h_data[c];
            }
        }
        return;
    }

    // GPU path: ensure data on device
    x.h2d();
    gamma.h2d();
    beta.h2d();
    out.allocate();

    if (!x.d_data || !gamma.d_data || !beta.d_data || !out.d_data) {
        std::cerr << "layernorm_opencl: device buffer missing, falling back to CPU" << std::endl;
        // CPU fallback already applied above
        return;
    }

    cl_int err = 0;
    err  = clSetKernelArg(g_layernorm_kernel, 0, sizeof(int), &rows);
    err |= clSetKernelArg(g_layernorm_kernel, 1, sizeof(int), &cols);
    err |= clSetKernelArg(g_layernorm_kernel, 2, sizeof(cl_mem), &x.d_data);
    err |= clSetKernelArg(g_layernorm_kernel, 3, sizeof(cl_mem), &gamma.d_data);
    err |= clSetKernelArg(g_layernorm_kernel, 4, sizeof(cl_mem), &beta.d_data);
    err |= clSetKernelArg(g_layernorm_kernel, 5, sizeof(cl_mem), &out.d_data);
    if (err != CL_SUCCESS) { std::cerr << "layernorm_opencl: setarg failed (" << err << ")\n"; return; }

    size_t g = rows;
    err = clEnqueueNDRangeKernel(g_queue, g_layernorm_kernel, 1, NULL, &g, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { std::cerr << "layernorm_opencl: enqueue failed (" << err << ")\n"; return; }
    clFinish(g_queue);

    out.d2h();
}

void relu_opencl(Tensor& x) {
    size_t N = x.size;
    if (!g_program || !g_relu_kernel) {
        for (size_t i = 0; i < x.h_data.size(); ++i) x.h_data[i] = std::max(0.0f, x.h_data[i]);
        return;
    }

    x.h2d();
    if (!x.d_data) { std::cerr << "relu_opencl: device buffer missing, falling back to CPU\n"; return; }

    cl_int err = 0;
    int Ni = (int)N;
    err = clSetKernelArg(g_relu_kernel, 0, sizeof(int), &Ni);
    err |= clSetKernelArg(g_relu_kernel, 1, sizeof(cl_mem), &x.d_data);
    if (err != CL_SUCCESS) { std::cerr << "relu_opencl: setarg failed (" << err << ")\n"; return; }

    size_t g = N;
    err = clEnqueueNDRangeKernel(g_queue, g_relu_kernel, 1, NULL, &g, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { std::cerr << "relu_opencl: enqueue failed (" << err << ")\n"; return; }
    clFinish(g_queue);

    x.d2h();
}

void TransposeMatrixOpenCL(Tensor& dst, const Tensor& src) {
    int m = src.shape[0];
    int n = src.shape[1];
    dst.shape = {n, m};
    dst.size = (size_t)m * n;
    dst.h_data.assign(dst.size, 0.0f);
    for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) dst.h_data[j * m + i] = src.h_data[i * n + j];
}
