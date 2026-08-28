#pragma once

#include "binary_encode.cuh"
#include "residual.cuh"
#include <cuda_fp16.h>

// Fused single-token encode kernel: takes one new K or V vector per (batch, head),
// binary-encodes it, computes INT4 residuals, and writes both directly into the
// ring-buffer caches at position current_len (mod max_seq).
//
// kv           - new token vectors [batch * n_heads, head_dim] fp16
// binary_cache - ring buffer of BinaryVec,   shape [batch * n_heads * max_seq]
// residual_cache - ring buffer of ResidualVec, shape [batch * n_heads * max_seq]
// batch, n_heads, head_dim, current_len, max_seq - layout parameters
void launch_online_encode_token(
    const __half* kv,
    BinaryVec*    binary_cache,
    ResidualVec*  residual_cache,
    int           batch,
    int           n_heads,
    int           head_dim,
    int           current_len,
    int           max_seq,
    cudaStream_t  stream = 0
);
