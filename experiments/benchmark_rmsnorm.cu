
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <algorithm>

#define CHECK(call) do {                              \
    cudaError_t e = (call);                           \
    if (e != cudaSuccess) {                           \
        std::cerr << cudaGetErrorString(e) << "\n";    \
        std::exit(1);                                 \
    }                                                 \
} while (0)

// 與之前相同的 RMSNorm Kernel
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

    float sum = 0.0f;

    for (int j = tid; j < hidden; j += blockDim.x) {
        float v = input[j];
        sum += v * v;
    }

    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];

        __syncthreads();
    }

    float inv_rms = rsqrtf(shared[0] / hidden + eps);

    for (int j = tid; j < hidden; j += blockDim.x) {
        output[j] = input[j] * inv_rms * gamma[j];
    }
}

void benchmark(int rows, int hidden, int threads) {
    const int N = rows * hidden;
    const size_t bytes = N * sizeof(float);
    const size_t gamma_bytes = hidden * sizeof(float);

    std::vector<float> x(N, 0.5f);
    std::vector<float> gamma(hidden, 1.0f);

    float *dx = nullptr, *dg = nullptr, *dy = nullptr;

    CHECK(cudaMalloc(&dx, bytes));
    CHECK(cudaMalloc(&dg, gamma_bytes));
    CHECK(cudaMalloc(&dy, bytes));

    CHECK(cudaMemcpy(dx, x.data(), bytes,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dg, gamma.data(), gamma_bytes,
                     cudaMemcpyHostToDevice));


    const int warmup = 50;
    const int kernels_per_graph = 100;
    const int graph_repeats = 20;
    const size_t shared_bytes = threads * sizeof(float);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Warm-up
    for (int i = 0; i < warmup; ++i) {
        rmsnorm_kernel<<<rows, threads, shared_bytes, stream>>>(
            dx, dg, dy, hidden, 1e-6f);
    }

    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(stream));

    // Capture 100 sequential kernels into one CUDA Graph
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    CHECK(cudaStreamBeginCapture(
        stream, cudaStreamCaptureModeGlobal));

    for (int i = 0; i < kernels_per_graph; ++i) {
        rmsnorm_kernel<<<rows, threads, shared_bytes, stream>>>(
            dx, dg, dy, hidden, 1e-6f);
    }

    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&graph_exec, graph, 0));

    // Warm-up the instantiated graph
    for (int i = 0; i < 3; ++i) {
        CHECK(cudaGraphLaunch(graph_exec, stream));
    }
    CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < graph_repeats; ++i) {
        CHECK(cudaGraphLaunch(graph_exec, stream));
    }

    CHECK(cudaEventRecord(stop, stream));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    double us_per_kernel =
        static_cast<double>(ms) * 1000.0 /
        (kernels_per_graph * graph_repeats);

    std::cout << std::setw(8) << rows
              << std::setw(10) << hidden
              << std::setw(10) << threads
              << std::setw(17) << std::fixed
              << std::setprecision(3) << us_per_kernel
              << "\n";

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaGraphExecDestroy(graph_exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));


    CHECK(cudaFree(dx));
    CHECK(cudaFree(dg));
    CHECK(cudaFree(dy));
}

int main() {
    std::cout << "RMSNorm CUDA Benchmark\n\n";

    std::cout << std::setw(8) << "Rows"
              << std::setw(10) << "Hidden"
              << std::setw(10) << "Threads"
              << std::setw(17) << "Latency(us)"
              << "\n";

    for (int rows : {4, 256}) {
        for (int hidden : {1024, 4096}) {
            for (int threads : {128, 256, 512}) {
                benchmark(rows, hidden, threads);
            }
        }
    }

    return 0;
}
