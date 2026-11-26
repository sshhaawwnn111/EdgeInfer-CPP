
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK(call) do {                                      \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA Error: "                            \
                  << cudaGetErrorString(err) << "\n";          \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)

// Each thread copies one FP32 element.
// Source layout: [new_tokens, kv_heads, head_dim]
// Cache layout:  [capacity, kv_heads, head_dim]
__global__ void append_kv_kernel(
    const float* src_k,
    const float* src_v,
    float* cache_k,
    float* cache_v,
    int elements,
    int destination_offset
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < elements) {
        cache_k[destination_offset + i] = src_k[i];
        cache_v[destination_offset + i] = src_v[i];
    }
}

class KVCache {
public:
    KVCache(int capacity, int heads, int dim)
        : capacity_(capacity),
          heads_(heads),
          dim_(dim),
          length_(0) {

        if (capacity <= 0 || heads <= 0 || dim <= 0) {
            std::cerr << "Invalid KV Cache shape\n";
            std::exit(EXIT_FAILURE);
        }

        elements_per_token_ = heads_ * dim_;

        size_t bytes =
            static_cast<size_t>(capacity_) *
            elements_per_token_ * sizeof(float);

        CHECK(cudaMalloc(&cache_k_, bytes));
        CHECK(cudaMalloc(&cache_v_, bytes));

        CHECK(cudaMemset(cache_k_, 0, bytes));
        CHECK(cudaMemset(cache_v_, 0, bytes));
    }

    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    ~KVCache() {
        if (cache_k_) cudaFree(cache_k_);
        if (cache_v_) cudaFree(cache_v_);
    }

    bool append(
        const float* src_k,
        const float* src_v,
        int new_tokens
    ) {
        if (new_tokens <= 0 ||
            new_tokens > capacity_ - length_) {
            return false;
        }

        int elements = new_tokens * elements_per_token_;
        int offset = length_ * elements_per_token_;

        int threads = 256;
        int blocks = (elements + threads - 1) / threads;

        append_kv_kernel<<<blocks, threads>>>(
            src_k, src_v, cache_k_, cache_v_,
            elements, offset
        );

        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        length_ += new_tokens;
        return true;
    }

    void reset() {
        // Logical reset: old contents remain in memory,
        // but are no longer considered valid.
        CHECK(cudaDeviceSynchronize());
        length_ = 0;
    }

    int length() const { return length_; }
    int capacity() const { return capacity_; }

    const float* keys() const { return cache_k_; }
    const float* values() const { return cache_v_; }

private:
    int capacity_;
    int heads_;
    int dim_;
    int elements_per_token_;
    int length_;

    float* cache_k_ = nullptr;
    float* cache_v_ = nullptr;


};

bool verify_cache(
    const KVCache& cache,
    const std::vector<float>& expected_k,
    const std::vector<float>& expected_v,
    int elements_per_token
) {
    int count = cache.length() * elements_per_token;

    std::vector<float> actual_k(count);
    std::vector<float> actual_v(count);

    size_t bytes = static_cast<size_t>(count) * sizeof(float);

    CHECK(cudaMemcpy(actual_k.data(), cache.keys(),
                     bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(actual_v.data(), cache.values(),
                     bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(actual_k[i]) ||
            !std::isfinite(actual_v[i]) ||
            actual_k[i] != expected_k[i] ||
            actual_v[i] != expected_v[i]) {
            std::cerr << "Mismatch at index " << i << "\n";
            return false;
        }
    }

    return true;
}

int main() {
    constexpr int KV_HEADS = 2;
    constexpr int HEAD_DIM = 64;
    constexpr int CAPACITY = 128;
    constexpr int ELEMENTS = KV_HEADS * HEAD_DIM;

    KVCache cache(CAPACITY, KV_HEADS, HEAD_DIM);

    std::vector<float> expected_k;
    std::vector<float> expected_v;

    auto run_append = [&](int tokens, int seed) -> bool {
        int count = tokens * ELEMENTS;
        size_t bytes = static_cast<size_t>(count)
                       * sizeof(float);

        std::vector<float> host_k(count);
        std::vector<float> host_v(count);

        for (int i = 0; i < count; ++i) {
            host_k[i] = seed + i * 0.001f;
            host_v[i] = -seed - i * 0.002f;
        }

        float* d_k = nullptr;
        float* d_v = nullptr;

        CHECK(cudaMalloc(&d_k, bytes));
        CHECK(cudaMalloc(&d_v, bytes));

        CHECK(cudaMemcpy(d_k, host_k.data(), bytes,
                         cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_v, host_v.data(), bytes,
                         cudaMemcpyHostToDevice));

        bool appended = cache.append(d_k, d_v, tokens);

        CHECK(cudaFree(d_k));
        CHECK(cudaFree(d_v));

        if (!appended) return false;

        expected_k.insert(
            expected_k.end(), host_k.begin(), host_k.end());
        expected_v.insert(
            expected_v.end(), host_v.begin(), host_v.end());

        return verify_cache(
            cache, expected_k, expected_v, ELEMENTS);
    };

    bool ok = true;

    // Prefill: append 4 tokens at once.
    bool prefill = run_append(4, 100);
    ok = ok && prefill;
    std::cout << "Prefill: "
              << (prefill ? "PASS" : "FAIL") << "\n";

    // Decode: append one token each step.
    for (int step = 0; step < 3; ++step) {
        bool valid = run_append(1, 200 + step);
        ok = ok && valid;

        std::cout << "Decode step " << step
                  << ": " << (valid ? "PASS" : "FAIL")
                  << " length=" << cache.length() << "\n";
    }

    // Verify no overflow occurs.
    bool rejected = !cache.append(
        nullptr, nullptr, CAPACITY);

    ok = ok && rejected;

    std::cout << "Capacity check: "
              << (rejected ? "PASS" : "FAIL") << "\n";

    // Reset and verify that new data starts at index 0.
    cache.reset();
    expected_k.clear();
    expected_v.clear();

    bool reset_ok = (cache.length() == 0) &&
                    run_append(2, 500);

    ok = ok && reset_ok;

    std::cout << "Reset test: "
              << (reset_ok ? "PASS" : "FAIL") << "\n";

    std::cout << "\nFinal result: "
              << (ok ? "ALL PASS" : "FAIL") << "\n";

    return ok ? 0 : 1;
}
