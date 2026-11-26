
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <iomanip>

#define CHECK(call) do {                                  \
    cudaError_t err = (call);                             \
    if (err != cudaSuccess) {                             \
        std::cerr << "CUDA error: "                        \
                  << cudaGetErrorString(err)              \
                  << " at line " << __LINE__ << "\n";       \
        std::exit(EXIT_FAILURE);                          \
    }                                                     \
} while (0)

// Layout: [tokens, heads, head_dim]
// RoPE: split-half rotation
// Each thread processes one pair of elements.
__global__ void rope_kernel(
    const float* input,
    float* output,
    const int* positions,
    int heads,
    int head_dim,
    float theta_base
) {
    int token = blockIdx.x;
    int head = blockIdx.y;

    int half = head_dim / 2;
    int j = blockIdx.z * blockDim.x + threadIdx.x;

    if (j >= half) return;

    int offset = (token * heads + head) * head_dim;

    float x0 = input[offset + j];
    float x1 = input[offset + j + half];

    float exponent = -2.0f * j / head_dim;
    float angle = positions[token] *
                  powf(theta_base, exponent);

    float s, c;
    sincosf(angle, &s, &c);

    output[offset + j] =
        x0 * c - x1 * s;

    output[offset + j + half] =
        x1 * c + x0 * s;
}

// Independent CPU reference using double precision
void rope_cpu(
    const std::vector<float>& input,
    std::vector<float>& output,
    const std::vector<int>& positions,
    int heads,
    int head_dim,
    double theta_base
) {
    int tokens = static_cast<int>(positions.size());
    int half = head_dim / 2;

    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            int offset = (token * heads + head) * head_dim;

            for (int j = 0; j < half; ++j) {
                double angle = positions[token] *
                    std::pow(theta_base,
                             -2.0 * j / head_dim);

                double c = std::cos(angle);
                double s = std::sin(angle);

                double x0 = input[offset + j];
                double x1 = input[offset + j + half];

                output[offset + j] =
                    static_cast<float>(x0 * c - x1 * s);

                output[offset + j + half] =
                    static_cast<float>(x1 * c + x0 * s);
            }
        }
    }
}

// Test one tensor (Q or K)
bool test_tensor(
    const char* name,
    int tokens,
    int heads,
    int head_dim,
    int start_position,
    float theta_base
) {
    int count = tokens * heads * head_dim;
    size_t bytes = static_cast<size_t>(count) * sizeof(float);

    std::vector<float> input(count);
    std::vector<float> cpu_output(count);
    std::vector<float> gpu_output(count);
    std::vector<int> positions(tokens);

    for (int i = 0; i < count; ++i) {
        input[i] = std::sin(i * 0.017f) +
                   0.2f * std::cos(i * 0.031f);
    }

    for (int t = 0; t < tokens; ++t) {
        positions[t] = start_position + t;
    }

    rope_cpu(
        input, cpu_output, positions,
        heads, head_dim, theta_base
    );

    float *d_input = nullptr;
    float *d_output = nullptr;
    int* d_positions = nullptr;

    CHECK(cudaMalloc(&d_input, bytes));
    CHECK(cudaMalloc(&d_output, bytes));
    CHECK(cudaMalloc(
        &d_positions, tokens * sizeof(int)
    ));

    CHECK(cudaMemcpy(
        d_input, input.data(), bytes,
        cudaMemcpyHostToDevice
    ));

    CHECK(cudaMemcpy(
        d_positions, positions.data(),
        tokens * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    const int threads = 128;
    const int half = head_dim / 2;

    dim3 grid(
        tokens,
        heads,
        (half + threads - 1) / threads
    );

    rope_kernel<<<grid, threads>>>(
        d_input, d_output, d_positions,
        heads, head_dim, theta_base
    );

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(
        gpu_output.data(), d_output, bytes,
        cudaMemcpyDeviceToHost
    ));

    float max_error = 0.0f;
    int failures = 0;

    for (int i = 0; i < count; ++i) {
        float error = std::abs(
            cpu_output[i] - gpu_output[i]
        );

        max_error = std::max(max_error, error);

        float tolerance =
            1e-4f + 1e-4f * std::abs(cpu_output[i]);

        if (!std::isfinite(gpu_output[i]) ||
            error > tolerance) {
            ++failures;
        }
    }

    bool passed = (failures == 0);

    std::cout << name
              << " tokens=" << tokens
              << " heads=" << heads
              << " dim=" << head_dim
              << " start_pos=" << start_position
              << " max_error=" << std::scientific
              << std::setprecision(6) << max_error
              << " failures=" << failures
              << " " << (passed ? "PASS" : "FAIL")
              << "\n";

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_positions));

    return passed;
}

int main() {
    // Qwen2.5-0.5B-Instruct configuration
    const int q_heads = 14;
    const int kv_heads = 2;
    const int head_dim = 64;
    const float theta_base = 1000000.0f;

    std::cout << "EdgeInfer-CPP RoPE Test\n";
    std::cout << "Qwen2.5-0.5B configuration\n\n";

    bool all_passed = true;

    for (int tokens : {1, 4, 17}) {
        for (int start_pos : {0, 7, 1024}) {
            bool q_ok = test_tensor(
                "Q", tokens, q_heads, head_dim,
                start_pos, theta_base
            );

            bool k_ok = test_tensor(
                "K", tokens, kv_heads, head_dim,
                start_pos, theta_base
            );

            all_passed = q_ok && k_ok && all_passed;
        }
    }

    std::cout << "\nFinal result: "
              << (all_passed ? "ALL PASS" : "FAIL")
              << "\n";

    return all_passed ? 0 : 1;
}
