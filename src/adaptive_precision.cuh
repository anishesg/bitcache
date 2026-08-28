#pragma once

#include "binary_encode.cuh"
#include <cuda_fp16.h>

// Compute attention entropy per (batch, head) from binary attention scores.
// High entropy = flat attention distribution (binary is sufficient).
// Low entropy  = peaked distribution (residual correction improves quality).
//
// Q           - query vectors [batch * n_heads, head_dim] fp16
// K_bin       - binary key cache [batch * n_heads * seq_len] BinaryVec
// entropy_out - output entropy [batch * n_heads] float32
// batch, n_heads, seq_len, head_dim - dimensions
void launch_compute_head_entropy(
    const __half*    Q,
    const BinaryVec* K_bin,
    float*           entropy_out,
    int              batch,
    int              n_heads,
    int              seq_len,
    int              head_dim,
    cudaStream_t     stream = 0
);

// Per-head precision mode determined by adaptive_select().
// BINARY: binary-only attention (fast, more memory efficient)
// COMPENSATED: residual-corrected attention (slower, higher quality)
enum class PrecisionMode : uint8_t {
    BINARY      = 0,
    COMPENSATED = 1,
};

// Select per-head precision mode based on entropy values.
// Heads with entropy above the threshold are set to BINARY.
// Heads with entropy below the threshold are set to COMPENSATED.
// Returns a host vector of PrecisionMode per (batch * n_heads) slot.
//
// entropy          - [batch * n_heads] float32 device tensor (from launch_compute_head_entropy)
// batch, n_heads   - dimensions
// entropy_threshold - empirical threshold; calibrated from the first 128 tokens.
//                    Typical value: log(seq_len) * 0.5 (increases with sequence length).
void launch_adaptive_select(
    const float*   entropy,
    PrecisionMode* modes_out,
    int            batch,
    int            n_heads,
    float          entropy_threshold,
    cudaStream_t   stream = 0
);
