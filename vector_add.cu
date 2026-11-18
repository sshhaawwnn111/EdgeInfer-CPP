
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define CHECK(call) do {                                  \
    cudaError_t err = (call);                             \
    if (err != cudaSuccess) {                             \
        std::cerr << cudaGetErrorString(err) << "\n";      \
        std::exit(EXIT_FAILURE);                          \
    }                                                     \
} while (0)

__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    std::vector<float> c(N);

    float *d_a, *d_b, *d_c;

    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));

    CHECK(cudaMemcpy(d_a, a.data(), bytes,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b.data(), bytes,
                     cudaMemcpyHostToDevice));

    vector_add<<<4, 256>>>(d_a, d_b, d_c, N);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(c.data(), d_c, bytes,
                     cudaMemcpyDeviceToHost));

    bool correct = true;

    for (int i = 0; i < N; i++) {
        if (c[i] != 3.0f) {
            correct = false;
            break;
        }
    }

    std::cout << (correct ? "PASS" : "FAIL") << "\n";

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    return correct ? 0 : 1;
}
