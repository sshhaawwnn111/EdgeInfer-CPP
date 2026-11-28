
#include "edgeinfer/rope.h"
#include "edgeinfer/cuda_check.h"

#include <cmath>
#include <stdexcept>

namespace edgeinfer {

__global__ void rope_kernel(
    const float* input,
    float* output,
    int head_dim,
    int position,
    float theta_base
) {
    int head = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    int half = head_dim / 2;

    if (j >= half) return;

    int offset = head * head_dim;

    float angle = static_cast<float>(position) *
        powf(theta_base, -2.0f * j / head_dim);

    float s, c;
    sincosf(angle, &s, &c);

    float a = input[offset + j];
    float b = input[offset + j + half];

    output[offset + j] = a * c - b * s;
    output[offset + j + half] = b * c + a * s;
}

void apply_rope(
    const float* input,
    float* output,
    int heads,
    int head_dim,
    int position,
    float theta_base,
    cudaStream_t stream
) {
    if (!input || !output ||
        heads <= 0 || head_dim <= 0 ||
        head_dim % 2 != 0 ||
        position < 0 ||
        !std::isfinite(theta_base) ||
        theta_base <= 0.0f) {
        throw std::invalid_argument(
            "Invalid RoPE arguments");
    }

    constexpr int threads = 128;
    int half = head_dim / 2;

    dim3 grid(
        heads,
        (half + threads - 1) / threads
    );

    rope_kernel<<<grid, threads, 0, stream>>>(
        input, output, head_dim,
        position, theta_base);

    CUDA_CHECK(cudaGetLastError());
}

}
