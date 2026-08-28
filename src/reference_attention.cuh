#pragma once

#include <cuda_fp16.h>

// Naive full-precision reference attention: Q*K^T / sqrt(d), softmax, * V.
// No optimizations. Uses fp32 accumulation internally for numerical accuracy.
// Serves as the correctness oracle for binary and compensated attention.
//
// Q   - [batch, n_heads, head_dim] (fp16)
// K   - [batch, n_heads, seq_len, head_dim] (fp16)
// V   - [batch, n_heads, seq_len, head_dim] (fp16)
// out - [batch, n_heads, head_dim] (fp16)
void launch_reference_attention(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half*       out,
    int           batch,
    int           n_heads,
    int           seq_len,
    int           head_dim,
    cudaStream_t  stream = 0
);
