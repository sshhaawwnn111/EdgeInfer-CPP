
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>

#define CHECK(call) do {                                  \
    cudaError_t err = (call);                             \
    if (err != cudaSuccess) {                             \
        std::cerr << "CUDA error: "                        \
                  << cudaGetErrorString(err) << "\n";      \
        std::exit(EXIT_FAILURE);                          \
    }                                                     \
} while (0)

// CPU Reference：正確性比較基準
void rmsnorm_cpu(
    const float* x,
    const float* gamma,
    float* y,
    int rows,
    int hidden,
    float eps
) {
    for (int row = 0; row < rows; ++row) {
        const float* input = x + row * hidden;
        float* output = y + row * hidden;

        double sum = 0.0;
        for (int j = 0; j < hidden; ++j) {
            double v = input[j];
            sum += v * v;
        }

        double inv_rms =
            1.0 / std::sqrt(sum / hidden + eps);

        for (int j = 0; j < hidden; ++j) {
            output[j] = static_cast<float>(
                input[j] * inv_rms * gamma[j]
            );
        }
    }
}

// GPU Kernel：每個 Block 處理一列
__global__ void rmsnorm_kernel(
    const float* x,
    const float* gamma,
    float* y,
    int hidden,
    float eps
) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* input = x + row * hidden;
    float* output = y + row * hidden;

    // Step 1: 每個 Thread 計算局部平方和
    float sum = 0.0f;

    for (int j = tid; j < hidden; j += blockDim.x) {
        float v = input[j];
        sum += v * v;
    }

    shared[tid] = sum;
    __syncthreads();

    // Step 2: Shared Memory Parallel Reduction
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    // Step 3: Normalization
    float inv_rms =
        rsqrtf(shared[0] / hidden + eps);

    for (int j = tid; j < hidden; j += blockDim.x) {
        output[j] = input[j] * inv_rms * gamma[j];
    }
}

int main() {
    const int rows = 4;
    const int hidden = 1024;
    const int N = rows * hidden;
    const float eps = 1e-6f;

    const size_t data_bytes = N * sizeof(float);
    const size_t gamma_bytes = hidden * sizeof(float);

    std::vector<float> x(N);
    std::vector<float> gamma(hidden);
    std::vector<float> cpu_result(N);
    std::vector<float> gpu_result(N);

    // 使用非固定值，避免過於簡單的測試
    for (int i = 0; i < N; ++i) {
        x[i] = std::sin(i * 0.01f);
    }

    for (int j = 0; j < hidden; ++j) {
        gamma[j] = 0.5f + (j % 17) * 0.03f;
    }

    rmsnorm_cpu(
        x.data(), gamma.data(), cpu_result.data(),
        rows, hidden, eps
    );

    float *d_x = nullptr;
    float *d_gamma = nullptr;
    float *d_y = nullptr;

    CHECK(cudaMalloc(&d_x, data_bytes));
    CHECK(cudaMalloc(&d_gamma, gamma_bytes));
    CHECK(cudaMalloc(&d_y, data_bytes));

    CHECK(cudaMemcpy(
        d_x, x.data(), data_bytes,
        cudaMemcpyHostToDevice
    ));

    CHECK(cudaMemcpy(
        d_gamma, gamma.data(), gamma_bytes,
        cudaMemcpyHostToDevice
    ));

    const int threads = 256;
    const int blocks = rows;
    const size_t shared_bytes = threads * sizeof(float);

    rmsnorm_kernel<<<blocks, threads, shared_bytes>>>(
        d_x, d_gamma, d_y, hidden, eps
    );

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(
        gpu_result.data(), d_y, data_bytes,
        cudaMemcpyDeviceToHost
    ));

    float max_error = 0.0f;
    bool passed = true;

    for (int i = 0; i < N; ++i) {
        float error = std::abs(
            cpu_result[i] - gpu_result[i]
        );

        max_error = std::max(max_error, error);

        float tolerance =
            1e-5f + 1e-5f * std::abs(cpu_result[i]);

        if (!std::isfinite(gpu_result[i]) ||
            error > tolerance) {
            passed = false;
        }
    }

    std::cout << "RMSNorm CPU vs GPU\n";
    std::cout << "Rows: " << rows << "\n";
    std::cout << "Hidden size: " << hidden << "\n";
    std::cout << "Max absolute error: "
              << max_error << "\n";
    std::cout << (passed ? "PASS" : "FAIL") << "\n";

    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_gamma));
    CHECK(cudaFree(d_y));

    return passed ? 0 : 1;
}
