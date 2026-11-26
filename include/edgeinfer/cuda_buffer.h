
#pragma once

#include "edgeinfer/cuda_check.h"
#include <cstddef>
#include <utility>

// Owns a contiguous GPU memory allocation.
// Move-only: prevents accidental double free.
class CudaBuffer {
public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t bytes)
        : bytes_(bytes) {
        if (bytes_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, bytes_));
        }
    }

    ~CudaBuffer() noexcept {
        release();
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept {
        move_from(other);
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            release();
            move_from(other);
        }
        return *this;
    }

    void* data() { return ptr_; }
    const void* data() const { return ptr_; }

    size_t size_bytes() const { return bytes_; }

    template <typename T>
    T* as() {
        return static_cast<T*>(ptr_);
    }

    template <typename T>
    const T* as() const {
        return static_cast<const T*>(ptr_);
    }

    void copy_from_host(const void* src, size_t bytes) {
        if (bytes > bytes_) {
            throw std::runtime_error(
                "Host copy exceeds CudaBuffer capacity");
        }
        if (bytes == 0) return;
        if (!src) {
            throw std::runtime_error("Null source");
        }

        CUDA_CHECK(cudaMemcpy(
            ptr_, src, bytes, cudaMemcpyHostToDevice));
    }

    void copy_to_host(void* dst, size_t bytes) const {
        if (bytes > bytes_) {
            throw std::runtime_error(
                "Device copy exceeds CudaBuffer capacity");
        }
        if (bytes == 0) return;
        if (!dst) {
            throw std::runtime_error("Null destination");
        }

        CUDA_CHECK(cudaMemcpy(
            dst, ptr_, bytes, cudaMemcpyDeviceToHost));
    }

private:
    void release() noexcept {
        if (ptr_) {
            // A destructor must not throw.
            cudaError_t err = cudaFree(ptr_);
            (void)err;
        }
        ptr_ = nullptr;
        bytes_ = 0;
    }

    void move_from(CudaBuffer& other) noexcept {
        ptr_ = std::exchange(other.ptr_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
    }

    void* ptr_ = nullptr;
    size_t bytes_ = 0;
};
