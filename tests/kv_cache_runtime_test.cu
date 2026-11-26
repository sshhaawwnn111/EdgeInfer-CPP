
#include "edgeinfer/kv_cache.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <utility>

constexpr int HEADS = 2;
constexpr int DIM = 64;
constexpr int CAPACITY = 8;
constexpr int ELEMENTS = HEADS * DIM;

bool verify(
    const KVCache& cache,
    const std::vector<float>& expected_k,
    const std::vector<float>& expected_v
) {
    size_t n = static_cast<size_t>(cache.length())
               * ELEMENTS;

    if (expected_k.size() != n ||
        expected_v.size() != n) return false;

    std::vector<float> actual_k(n), actual_v(n);
    size_t bytes = n * sizeof(float);

    CUDA_CHECK(cudaMemcpy(
        actual_k.data(), cache.keys(),
        bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        actual_v.data(), cache.values(),
        bytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(actual_k[i]) ||
            !std::isfinite(actual_v[i]) ||
            actual_k[i] != expected_k[i] ||
            actual_v[i] != expected_v[i]) {
            return false;
        }
    }
    return true;
}

void append_test_data(
    KVCache& cache,
    int tokens,
    int seed,
    std::vector<float>& expected_k,
    std::vector<float>& expected_v
) {
    size_t n = static_cast<size_t>(tokens) * ELEMENTS;
    size_t bytes = n * sizeof(float);

    std::vector<float> k(n), v(n);

    for (size_t i = 0; i < n; ++i) {
        k[i] = seed + static_cast<float>(i) * 0.001f;
        v[i] = -seed - static_cast<float>(i) * 0.002f;
    }

    CudaBuffer device_k(bytes);
    CudaBuffer device_v(bytes);

    device_k.copy_from_host(k.data(), bytes);
    device_v.copy_from_host(v.data(), bytes);

    cache.append(
        device_k.as<float>(),
        device_v.as<float>(),
        tokens);

    expected_k.insert(expected_k.end(), k.begin(), k.end());
    expected_v.insert(expected_v.end(), v.begin(), v.end());
}

int main() {
    try {
        // Basic CudaBuffer move test
        CudaBuffer original(1024);
        void* original_ptr = original.data();
        CudaBuffer moved(std::move(original));

        bool move_ok =
            moved.data() == original_ptr &&
            original.data() == nullptr &&
            original.size_bytes() == 0;

        std::cout << "CudaBuffer move: "
                  << (move_ok ? "PASS" : "FAIL") << "\n";

        KVCache cache(CAPACITY, HEADS, DIM);
        const float* original_k = cache.keys();
        const float* original_v = cache.values();

        std::vector<float> expected_k, expected_v;
        bool ok = move_ok;

        // Prefill 4 tokens
        append_test_data(
            cache, 4, 100, expected_k, expected_v);

        bool prefill_ok =
            cache.length() == 4 &&
            verify(cache, expected_k, expected_v);

        ok &= prefill_ok;

        std::cout << "Prefill: "
                  << (prefill_ok ? "PASS" : "FAIL")
                  << "\n";

        // Decode 3 tokens, one at a time
        for (int step = 0; step < 3; ++step) {
            append_test_data(
                cache, 1, 200 + step,
                expected_k, expected_v);

            bool step_ok =
                cache.length() == 5 + step &&
                verify(cache, expected_k, expected_v);

            ok &= step_ok;

            std::cout << "Decode " << step << ": "
                      << (step_ok ? "PASS" : "FAIL")
                      << "\n";
        }

        // Verify capacity protection
        bool overflow_rejected = false;
        try {
            cache.append(nullptr, nullptr, 2);
        } catch (const std::out_of_range&) {
            overflow_rejected = true;
        }

        bool capacity_ok =
            overflow_rejected && cache.length() == 7 &&
            verify(cache, expected_k, expected_v);

        ok &= capacity_ok;

        std::cout << "Capacity check: "
                  << (capacity_ok ? "PASS" : "FAIL")
                  << "\n";

        // Reset and reuse
        cache.reset();
        expected_k.clear();
        expected_v.clear();

        append_test_data(
            cache, 2, 500, expected_k, expected_v);

        bool reset_ok =
            cache.length() == 2 &&
            verify(cache, expected_k, expected_v);

        ok &= reset_ok;

        std::cout << "Reset: "
                  << (reset_ok ? "PASS" : "FAIL")
                  << "\n";

        // No cache reallocation during append/reset
        bool address_ok =
            cache.keys() == original_k &&
            cache.values() == original_v;

        ok &= address_ok;

        std::cout << "Cache address stable: "
                  << (address_ok ? "PASS" : "FAIL")
                  << "\n";

        std::cout << "\nFinal result: "
                  << (ok ? "ALL PASS" : "FAIL")
                  << "\n";

        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}

