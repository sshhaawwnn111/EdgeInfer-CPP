
#pragma once
#include <cuda_runtime.h>

namespace edgeinfer {

// Single-token GQA Decode.
// Q: [q_heads, head_dim]
// KV: [cache_length, kv_heads, head_dim]
// Out: [q_heads, head_dim]
//
// Initial implementation limits cache_length <= 16.
void gqa_decode(
    const float* q,
    const float* cache_k,
    const float* cache_v,
    float* output,
    int q_heads,
    int kv_heads,
    int head_dim,
    int cache_length,
    cudaStream_t stream = nullptr
);

}
