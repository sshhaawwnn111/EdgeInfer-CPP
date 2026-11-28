
#include "edgeinfer/attention.h"
#include "edgeinfer/cuda_check.h"

#include <cmath>
#include <stdexcept>

namespace edgeinfer {

constexpr int MAX_CONTEXT = 16;

// One block processes one query head.
// Each thread handles one dimension.
__global__ void gqa_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int kv_heads,
    int head_dim,
    int group_size,
    int length
) {
    int head = blockIdx.x;
    int d = threadIdx.x;
    int kv_head = head / group_size;


    // extern keyword is used to declare a dynamic shared memory array. 
    // The size of this dynamic memory space is determined at kernel launch time by the third argument.
    extern __shared__ float reduction[]; // cross product ready for reduction.
    __shared__ float scores[MAX_CONTEXT]; // scores for each token t.
    __shared__ float denom; // denominator for softmax normalization.

    const float* query = q + head * head_dim;


    // Compute K^T * Q for each token t, then reduce the result to get the score for each token t.
    for (int t = 0; t < length; ++t) {
        int offset = (t * kv_heads + kv_head)
                     * head_dim;

        reduction[d] =
            query[d] * k[offset + d];

        __syncthreads();

        for (int stride = head_dim / 2;
             stride > 0; stride >>= 1) {
            if (d < stride) {
                reduction[d] +=
                    reduction[d + stride];
            }
            __syncthreads();
        }

        if (d == 0) {
            scores[t] = reduction[0] /
                        sqrtf(float(head_dim));
        }
        __syncthreads();
    }

    if (d == 0) {
        float maximum = -INFINITY;

        for (int t = 0; t < length; ++t) {
            maximum = fmaxf(maximum, scores[t]);
        }

        float sum = 0.0f;

        for (int t = 0; t < length; ++t) {
            // Subtract the maximum from each score to prevent overflow in the exponential function.
            scores[t] = expf(scores[t] - maximum);
            sum += scores[t];
        }

        denom = sum;
    }

    __syncthreads();

    float result = 0.0f;

    for (int t = 0; t < length; ++t) {
        int offset = (t * kv_heads + kv_head)
                     * head_dim;

        result += (scores[t] / denom) *
                  v[offset + d];
    }

    output[head * head_dim + d] = result;
}

void gqa_decode(
    const float* q,
    const float* cache_k,
    const float* cache_v,
    float* output,
    int q_heads,
    int kv_heads,
    int head_dim,
    int cache_length,
    cudaStream_t stream
) {
    if (!q || !cache_k || !cache_v || !output ||
        q_heads <= 0 ||
        kv_heads <= 0 ||
        q_heads % kv_heads != 0 ||
        head_dim <= 0 ||
        head_dim > 1024 ||
        (head_dim & (head_dim - 1)) != 0 ||
        cache_length < 1 ||
        cache_length > MAX_CONTEXT) {
        throw std::invalid_argument(
            "Invalid GQA arguments");
    }

    int group = q_heads / kv_heads;
    size_t shared_bytes =
        head_dim * sizeof(float);

    gqa_kernel<<<q_heads, head_dim,
                 shared_bytes, stream>>>(
        q, cache_k, cache_v, output,
        kv_heads, head_dim,
        group, cache_length);

    CUDA_CHECK(cudaGetLastError());
}

}
