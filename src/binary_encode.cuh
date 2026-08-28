#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// Number of uint32 words needed to pack head_dim sign bits
#define WORDS_PER_VEC(head_dim) (((head_dim) + 31) / 32)

// Maximum supported head_dim
#define MAX_HEAD_DIM 256

// Packed binary representation of one K or V vector
struct BinaryVec {
    // sign bits: bit i of word[i/32] is set if element i >= zero_threshold
    uint32_t words[MAX_HEAD_DIM / 32];
    // mean absolute value of the original fp16 vector, used for reconstruction
    __half scale;
};

// Launch configuration: one thread per output vector, blockDim.x = 128
// Inputs:
//   src      - fp16 input [batch * seq_len, head_dim]
//   out      - packed output [batch * seq_len] BinaryVec
//   n_vecs   - total number of vectors (batch * seq_len)
//   head_dim - must be a multiple of 32
void launch_binary_encode(
    const __half* src,
    BinaryVec*    out,
    int           n_vecs,
    int           head_dim,
    cudaStream_t  stream = 0
);

// Device helper: extract sign bit with configurable zero threshold.
// Returns 1 if val >= threshold, 0 otherwise.
__device__ __forceinline__ int sign_bit(float val, float threshold = 0.0f) {
    return val >= threshold ? 1 : 0;
}
