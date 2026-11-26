
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::cerr << cudaGetErrorString(e) \
                  << " line " << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while (0)

constexpr int QH = 14;
constexpr int KVH = 2;
constexpr int D = 64;
constexpr int GROUP = QH / KVH;
constexpr int CAP = 16;
constexpr float BASE = 1000000.0f;

// GPU RoPE: split-half rotation
__global__ void rope(
    const float* x, float* y,
    int heads, int position
) {
    int h = blockIdx.x;
    int j = threadIdx.x;
    if (j >= D / 2) return;

    int offset = h * D;
    float angle = position *
        powf(BASE, -2.0f * j / D);

    float s, c;
    sincosf(angle, &s, &c);

    float a = x[offset + j];
    float b = x[offset + j + D / 2];

    y[offset + j] = a * c - b * s;
    y[offset + j + D / 2] = b * c + a * s;
}

// Append one token into preallocated KV cache
__global__ void append_kv(
    const float* k, const float* v,
    float* cache_k, float* cache_v,
    int position
) {
    int i = threadIdx.x;
    if (i >= KVH * D) return;

    int offset = position * KVH * D + i;
    cache_k[offset] = k[i];
    cache_v[offset] = v[i];
}

// One block per query head, for one token
__global__ void attention(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int length
) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    int kvh = h / GROUP;

    __shared__ float reduction[D];
    __shared__ float scores[CAP];
    __shared__ float denominator;

    for (int t = 0; t < length; ++t) {
        int offset = (t * KVH + kvh) * D;

        reduction[d] =
            q[h * D + d] * k[offset + d];
        __syncthreads();

        for (int stride = D / 2;
             stride > 0; stride >>= 1) {
            if (d < stride)
                reduction[d] += reduction[d + stride];
            __syncthreads();
        }

        if (d == 0)
            scores[t] = reduction[0] / sqrtf(float(D));

        __syncthreads();
    }

    if (d == 0) {
        float maximum = -INFINITY;
        for (int t = 0; t < length; ++t)
            maximum = fmaxf(maximum, scores[t]);

        float sum = 0.0f;
        for (int t = 0; t < length; ++t) {
            scores[t] = expf(scores[t] - maximum);
            sum += scores[t];
        }
        denominator = sum;
    }
    __syncthreads();

    float result = 0.0f;
    for (int t = 0; t < length; ++t) {
        int offset = (t * KVH + kvh) * D;
        result += scores[t] / denominator *
                  v[offset + d];
    }

    out[h * D + d] = result;
}

std::vector<float> make_data(int n, float phase) {
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i)
        x[i] = sinf(i * 0.017f + phase);
    return x;
}

// Independent CPU RoPE reference
void cpu_rope(std::vector<float>& x,
              int heads, int position) {
    for (int h = 0; h < heads; ++h) {
        for (int j = 0; j < D / 2; ++j) {
            double angle = position *
                std::pow(double(BASE), -2.0 * j / D);
            double c = std::cos(angle);
            double s = std::sin(angle);

            int a = h * D + j;
            int b = a + D / 2;
            double x0 = x[a], x1 = x[b];

            x[a] = float(x0 * c - x1 * s);
            x[b] = float(x1 * c + x0 * s);
        }
    }
}

// CPU Attention reference
std::vector<float> cpu_attention(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    int length
) {
    std::vector<float> out(QH * D);

    for (int h = 0; h < QH; ++h) {
        int kvh = h / GROUP;
        std::vector<double> scores(length);
        double maximum = -INFINITY;

        for (int t = 0; t < length; ++t) {
            double dot = 0;
            for (int d = 0; d < D; ++d)
                dot += double(q[h * D + d]) *
                    k[(t * KVH + kvh) * D + d];

            scores[t] = dot / std::sqrt(double(D));
            maximum = std::max(maximum, scores[t]);
        }

        double sum = 0;
        for (auto& score : scores) {
            score = std::exp(score - maximum);
            sum += score;
        }

        for (int d = 0; d < D; ++d) {
            double result = 0;
            for (int t = 0; t < length; ++t)
                result += scores[t] / sum *
                    v[(t * KVH + kvh) * D + d];

            out[h * D + d] = float(result);
        }
    }
    return out;
}

int main() {
    float *dq, *dk, *dv;
    float *rq, *rk, *ck, *cv, *dout;

    CHECK(cudaMalloc(&dq, QH * D * sizeof(float)));
    CHECK(cudaMalloc(&dk, KVH * D * sizeof(float)));
    CHECK(cudaMalloc(&dv, KVH * D * sizeof(float)));
    CHECK(cudaMalloc(&rq, QH * D * sizeof(float)));
    CHECK(cudaMalloc(&rk, KVH * D * sizeof(float)));
    CHECK(cudaMalloc(&ck, CAP * KVH * D * sizeof(float)));
    CHECK(cudaMalloc(&cv, CAP * KVH * D * sizeof(float)));
    CHECK(cudaMalloc(&dout, QH * D * sizeof(float)));

    std::vector<float> cpu_k, cpu_v;
    bool all_pass = true;

    // First 4 tokens act as prefill.
    // Fifth token acts as decode.
    for (int pos = 0; pos < 5; ++pos) {
        auto q = make_data(QH * D, 0.1f + pos);
        auto k = make_data(KVH * D, 0.2f + pos);
        auto v = make_data(KVH * D, 0.3f + pos);

        auto cpu_q = q;
        auto cpu_key = k;

        cpu_rope(cpu_q, QH, pos);
        cpu_rope(cpu_key, KVH, pos);

        cpu_k.insert(cpu_k.end(),
                     cpu_key.begin(), cpu_key.end());
        cpu_v.insert(cpu_v.end(),
                     v.begin(), v.end());

        auto expected = cpu_attention(
            cpu_q, cpu_k, cpu_v, pos + 1);

        CHECK(cudaMemcpy(
            dq, q.data(), QH * D * sizeof(float),
            cudaMemcpyHostToDevice));

        CHECK(cudaMemcpy(
            dk, k.data(), KVH * D * sizeof(float),
            cudaMemcpyHostToDevice));

        CHECK(cudaMemcpy(
            dv, v.data(), KVH * D * sizeof(float),
            cudaMemcpyHostToDevice));

        rope<<<QH, 32>>>(dq, rq, QH, pos);
        rope<<<KVH, 32>>>(dk, rk, KVH, pos);
        CHECK(cudaGetLastError());

        append_kv<<<1, 128>>>(
            rk, dv, ck, cv, pos);
        CHECK(cudaGetLastError());

        attention<<<QH, D>>>(
            rq, ck, cv, dout, pos + 1);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        std::vector<float> actual(QH * D);
        CHECK(cudaMemcpy(
            actual.data(), dout,
            QH * D * sizeof(float),
            cudaMemcpyDeviceToHost));

        float max_error = 0;
        int failures = 0;

        for (size_t i = 0; i < actual.size(); ++i) {
            float error = std::abs(actual[i] - expected[i]);
            max_error = std::max(max_error, error);

            float tolerance =
                1e-4f + 1e-4f * std::abs(expected[i]);

            if (!std::isfinite(actual[i]) ||
                error > tolerance)
                ++failures;
        }

        bool passed = failures == 0;
        all_pass &= passed;

        std::cout
            << (pos < 4 ? "Prefill" : "Decode")
            << " position=" << pos
            << " max_error=" << max_error
            << " failures=" << failures
            << " " << (passed ? "PASS" : "FAIL")
            << "\n";
    }

    CHECK(cudaFree(dq));
    CHECK(cudaFree(dk));
    CHECK(cudaFree(dv));
    CHECK(cudaFree(rq));
    CHECK(cudaFree(rk));
    CHECK(cudaFree(ck));
    CHECK(cudaFree(cv));
    CHECK(cudaFree(dout));

    std::cout << "\nFinal result: "
              << (all_pass ? "ALL PASS" : "FAIL")
              << "\n";

    return all_pass ? 0 : 1;
}
