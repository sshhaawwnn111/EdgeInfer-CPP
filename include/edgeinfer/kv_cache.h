
#pragma once

#include "edgeinfer/cuda_buffer.h"
#include <cstddef>
#include <limits>
#include <stdexcept>

class KVCache {
public:
    KVCache(int capacity, int kv_heads, int head_dim)
        : capacity_(capacity),
          kv_heads_(kv_heads),
          head_dim_(head_dim),
          keys_(allocation_bytes(capacity, kv_heads, head_dim)),
          values_(allocation_bytes(capacity, kv_heads, head_dim)) {

        elements_per_token_ =
            static_cast<size_t>(kv_heads_) * head_dim_;
    }

    // Input must already reside in GPU memory.
    // Layout: [tokens, kv_heads, head_dim]
    void append(
        const float* device_k,
        const float* device_v,
        int tokens
    ) {
        if (tokens <= 0 || tokens > capacity_ - length_) {
            throw std::out_of_range(
                "KVCache append exceeds remaining capacity");
        }

        if (!device_k || !device_v) {
            throw std::invalid_argument(
                "KVCache received null input");
        }

        const size_t offset =
            static_cast<size_t>(length_) * elements_per_token_;
        const size_t bytes =
            static_cast<size_t>(tokens) *
            elements_per_token_ * sizeof(float);

        float* dst_k = keys_.as<float>() + offset;
        float* dst_v = values_.as<float>() + offset;

        CUDA_CHECK(cudaMemcpy(
            dst_k, device_k, bytes,
            cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaMemcpy(
            dst_v, device_v, bytes,
            cudaMemcpyDeviceToDevice));

        length_ += tokens;
    }

    // Logical reset, no memory allocation or clearing.
    void reset() {
        CUDA_CHECK(cudaDeviceSynchronize());
        length_ = 0;
    }

    int length() const { return length_; }
    int capacity() const { return capacity_; }
    int kv_heads() const { return kv_heads_; }
    int head_dim() const { return head_dim_; }

    const float* keys() const {
        return keys_.as<float>();
    }

    const float* values() const {
        return values_.as<float>();
    }

private:
    static size_t allocation_bytes(
        int capacity, int heads, int dim
    ) {
        if (capacity <= 0 || heads <= 0 || dim <= 0) {
            throw std::invalid_argument(
                "Invalid KVCache dimensions");
        }

        const size_t c = static_cast<size_t>(capacity);
        const size_t h = static_cast<size_t>(heads);
        const size_t d = static_cast<size_t>(dim);
        const size_t max =
            std::numeric_limits<size_t>::max();

        if (c > max / h ||
            c * h > max / d ||
            c * h * d > max / sizeof(float)) {
            throw std::overflow_error(
                "KVCache allocation size overflow");
        }

        return c * h * d * sizeof(float);
    }

    int capacity_;
    int kv_heads_;
    int head_dim_;
    int length_ = 0;
    size_t elements_per_token_ = 0;

    CudaBuffer keys_;
    CudaBuffer values_;
};
