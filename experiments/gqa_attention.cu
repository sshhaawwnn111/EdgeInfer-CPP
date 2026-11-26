
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK(call) do {                              \
    cudaError_t e = (call);                           \
    if (e != cudaSuccess) {                           \
        std::cerr << cudaGetErrorString(e)            \
                  << " line " << __LINE__ << "\n";      \
        std::exit(1);                                 \
    }                                                 \
} while (0)

constexpr int QH = 14;
constexpr int KVH = 2;
constexpr int D = 64;
constexpr int CAPACITY = 16;
constexpr int GROUP = QH / KVH;

// Q layout: [query_tokens, 14, 64]
// KV layout: [cache_tokens, 2, 64]
// Output:    [query_tokens, 14, 64]
__global__ void gqa_kernel(
    const float* q,
    const float* cache_k,
    const float* cache_v,
    float* output,
    int query_start,
    int query_tokens,
    int cache_length
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;

    if (t >= query_tokens || h >= QH) return;

    int kv_head = h / GROUP;
    int position = query_start + t;
    int valid_length = min(cache_length, position + 1);

    __shared__ float partial[D];
    __shared__ float scores[CAPACITY];
    __shared__ float denominator;

    const float* query = q + (t * QH + h) * D;

    // Compute attention scores: Q dot K / sqrt(D)
    for (int s = 0; s < valid_length; ++s) {
        int offset = (s * KVH + kv_head) * D;

        partial[d] = query[d] * cache_k[offset + d];
        __syncthreads();

        for (int stride = D / 2; stride > 0; stride >>= 1) {
            if (d < stride)
                partial[d] += partial[d + stride];

            __syncthreads();
        }

        if (d == 0)
            scores[s] = partial[0] / sqrtf((float)D);

        __syncthreads();
    }

    // Numerically stable softmax
    if (d == 0) {
        float maximum = -INFINITY;

        for (int s = 0; s < valid_length; ++s)
            maximum = fmaxf(maximum, scores[s]);

        float sum = 0.0f;

        for (int s = 0; s < valid_length; ++s) {
            scores[s] = expf(scores[s] - maximum);
            sum += scores[s];
        }

        denominator = sum;
    }

    __syncthreads();

    // Weighted sum of V
    float result = 0.0f;

    for (int s = 0; s < valid_length; ++s) {
        int offset = (s * KVH + kv_head) * D;
        result += (scores[s] / denominator) *
                  cache_v[offset + d];
    }

    output[(t * QH + h) * D + d] = result;
}

// Independent CPU reference
void gqa_cpu(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    std::vector<float>& output,
    int query_start,
    int query_tokens,
    int cache_length
) {
    for (int t = 0; t < query_tokens; ++t) {
        for (int h = 0; h < QH; ++h) {
            int kv_head = h / GROUP;
            int valid = std::min(cache_length,
                                 query_start + t + 1);

            std::vector<double> scores(valid);
            double maximum = -INFINITY;

            for (int s = 0; s < valid; ++s) {
                double dot = 0.0;

                for (int d = 0; d < D; ++d) {
                    dot += (double)q[(t * QH + h) * D + d] *
                           k[(s * KVH + kv_head) * D + d];
                }

                scores[s] = dot / std::sqrt((double)D);
                maximum = std::max(maximum, scores[s]);
            }

            double sum = 0.0;

            for (double& score : scores) {
                score = std::exp(score - maximum);
                sum += score;
            }

            for (int d = 0; d < D; ++d) {
                double result = 0.0;

                for (int s = 0; s < valid; ++s) {
                    result += scores[s] / sum *
                              v[(s * KVH + kv_head) * D + d];
                }

                output[(t * QH + h) * D + d] =
                    (float)result;
            }
        }
    }
}

std::vector<float> make_data(int count, float phase) {
    std::vector<float> data(count);

    for (int i = 0; i < count; ++i)
        data[i] = std::sin(i * 0.017f + phase) +
                  0.3f * std::cos(i * 0.029f - phase);

    return data;
}

bool test_attention(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const float* d_k,
    const float* d_v,
    int query_start,
    int query_tokens,
    int cache_length
) {
    int count = query_tokens * QH * D;
    size_t bytes = count * sizeof(float);

    std::vector<float> reference(count);
    std::vector<float> result(count);

    gqa_cpu(q, k, v, reference,
            query_start, query_tokens, cache_length);

    float *d_q = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_q, bytes));
    CHECK(cudaMalloc(&d_out, bytes));

    CHECK(cudaMemcpy(d_q, q.data(), bytes,
                     cudaMemcpyHostToDevice));

    dim3 grid(query_tokens, QH);
    gqa_kernel<<<grid, D>>>(
        d_q, d_k, d_v, d_out,
        query_start, query_tokens, cache_length);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(result.data(), d_out, bytes,
                     cudaMemcpyDeviceToHost));

    float max_error = 0.0f;
    int failures = 0;

    for (int i = 0; i < count; ++i) {
        float error = std::abs(result[i] - reference[i]);
        max_error = std::max(max_error, error);

        float tolerance =
            1e-5f + 1e-5f * std::abs(reference[i]);

        if (!std::isfinite(result[i]) ||
            error > tolerance)
            ++failures;
    }

    std::cout << (query_start == 0 ? "Prefill" : "Decode")
              << ": max_error=" << max_error
              << " failures=" << failures
              << " " << (failures == 0 ? "PASS" : "FAIL")
              << "\n";

    CHECK(cudaFree(d_q));
    CHECK(cudaFree(d_out));

    return failures == 0;
}

int main() {
    std::cout << "EdgeInfer-CPP GQA Attention\n\n";

    const size_t cache_bytes =
        CAPACITY * KVH * D * sizeof(float);

    float *d_k = nullptr, *d_v = nullptr;
    CHECK(cudaMalloc(&d_k, cache_bytes));
    CHECK(cudaMalloc(&d_v, cache_bytes));

    // Prefill: positions 0, 1, 2, 3
    auto q0 = make_data(4 * QH * D, 0.1f);
    auto k0 = make_data(4 * KVH * D, 0.2f);
    auto v0 = make_data(4 * KVH * D, 0.3f);

    size_t prefill_bytes = k0.size() * sizeof(float);

    CHECK(cudaMemcpy(d_k, k0.data(), prefill_bytes,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_v, v0.data(), prefill_bytes,
                     cudaMemcpyHostToDevice));

    bool prefill_ok = test_attention(
        q0, k0, v0, d_k, d_v, 0, 4, 4);

    // Decode: append token at position 4
    auto q1 = make_data(QH * D, 0.4f);
    auto k1 = make_data(KVH * D, 0.5f);
    auto v1 = make_data(KVH * D, 0.6f);

    size_t token_bytes = KVH * D * sizeof(float);

    CHECK(cudaMemcpy(d_k + 4 * KVH * D,
                     k1.data(), token_bytes,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_v + 4 * KVH * D,
                     v1.data(), token_bytes,
                     cudaMemcpyHostToDevice));

    std::vector<float> full_k = k0;
    std::vector<float> full_v = v0;

    full_k.insert(full_k.end(), k1.begin(), k1.end());
    full_v.insert(full_v.end(), v1.begin(), v1.end());

    bool decode_ok = test_attention(
        q1, full_k, full_v, d_k, d_v, 4, 1, 5);

    CHECK(cudaFree(d_k));
    CHECK(cudaFree(d_v));

    bool passed = prefill_ok && decode_ok;

    std::cout << "\nFinal result: "
              << (passed ? "ALL PASS" : "FAIL") << "\n";

    return passed ? 0 : 1;
}
