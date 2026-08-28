#pragma once

#include "binary_encode.cuh"
#include <cuda_fp16.h>
#include <cstdint>

// INT4 residual for one K or V vector.
// Two INT4 values are packed per byte: low nibble = element 2i, high nibble = element 2i+1.
// Each nibble is in [-7, 7] (we reserve -8 to avoid sign issues).
// Reconstruction: element i = (nibble_i / 7.0f) * scale
struct ResidualVec {
    // packed INT4 residuals: head_dim/2 bytes
    uint8_t nibbles[MAX_HEAD_DIM / 2];
    // per-vector absmax used during quantization
    __half scale;
};

// Compute residuals between original fp16 vectors and their binary reconstructions,
// then quantize to INT4 and pack two per byte.
//
// src      - original fp16 vectors [n_vecs, head_dim]
// binary   - corresponding BinaryVec array [n_vecs]
// out      - output ResidualVec array [n_vecs]
// n_vecs   - number of vectors
// head_dim - must be a multiple of 32
void launch_residual_encode(
    const __half*    src,
    const BinaryVec* binary,
    ResidualVec*     out,
    int              n_vecs,
    int              head_dim,
    cudaStream_t     stream = 0
);

// Device helper: dequantize a single INT4 nibble to float.
// nibble is a 4-bit signed value in [-7, 7].
__device__ __forceinline__ float dequant_nibble(int8_t nibble, float scale) {
    return (nibble / 7.0f) * scale;
}

// Device helper: extract low nibble (signed) from a packed byte.
__device__ __forceinline__ int8_t nibble_lo(uint8_t byte) {
    int8_t raw = (int8_t)(byte & 0x0F);
    // sign-extend from 4 bits
    return raw >= 8 ? raw - 16 : raw;
}

// Device helper: extract high nibble (signed) from a packed byte.
__device__ __forceinline__ int8_t nibble_hi(uint8_t byte) {
    int8_t raw = (int8_t)((byte >> 4) & 0x0F);
    return raw >= 8 ? raw - 16 : raw;
}
