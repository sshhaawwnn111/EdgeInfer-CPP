
#pragma once
#include <cuda_runtime.h>

namespace edgeinfer {

// One token, layout [heads, head_dim].
// Input and output must be GPU pointers.
// Qwen2.5 split-half RoPE.
void apply_rope(
    const float* input,
    float* output,
    int heads,
    int head_dim,
    int position,
    float theta_base = 1000000.0f,
    cudaStream_t stream = nullptr
);

}
