#pragma once

#include "binary_encode.cuh"
#include "residual.cuh"
#include <cuda_fp16.h>

// Residual-compensated attention: binary scores corrected by INT4 residual dot product.
// Output uses binary V reconstruction plus INT4 V residual weighted sum.
//
// Q       - query vectors [batch, n_heads, head_dim] (fp16)
// K_bin   - binary-encoded key cache [batch, n_heads, seq_len]
// K_res   - INT4 key residuals [batch, n_heads, seq_len]
// V_bin   - binary-encoded value cache [batch, n_heads, seq_len]
// V_res   - INT4 value residuals [batch, n_heads, seq_len]
// out     - output [batch, n_heads, head_dim] (fp16)
void launch_compensated_attention(
    const __half*       Q,
    const BinaryVec*    K_bin,
    const ResidualVec*  K_res,
    const BinaryVec*    V_bin,
    const ResidualVec*  V_res,
    __half*             out,
    int                 batch,
    int                 n_heads,
    int                 seq_len,
    int                 head_dim,
    cudaStream_t        stream = 0
);
