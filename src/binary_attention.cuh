#pragma once

#include "binary_encode.cuh"
#include <cuda_fp16.h>

// Binary-only attention: Q is full fp16, K and V are binary-encoded.
// Computes approximate attention output using sign-masked scoring.
//
// Q      - query vectors [batch, n_heads, head_dim] (fp16)
// K_bin  - binary-encoded key cache [batch, n_heads, seq_len] (BinaryVec)
// V_bin  - binary-encoded value cache [batch, n_heads, seq_len] (BinaryVec)
// out    - output [batch, n_heads, head_dim] (fp16)
// batch, n_heads, seq_len, head_dim - tensor dimensions
// head_dim must be a multiple of 32 and <= MAX_HEAD_DIM
void launch_binary_attention(
    const __half*    Q,
    const BinaryVec* K_bin,
    const BinaryVec* V_bin,
    __half*          out,
    int              batch,
    int              n_heads,
    int              seq_len,
    int              head_dim,
    cudaStream_t     stream = 0
);
