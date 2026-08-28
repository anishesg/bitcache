#pragma once

#include "binary_encode.cuh"
#include <cuda_fp16.h>
#include <cstdint>

// Compute approximate dot product between a full-precision query and a binary key.
// Returns scale_q * scale_k * (2 * popcount(q_sign XNOR k_sign) - head_dim)
//
// q_sign  - packed sign bits of Q vector (head_dim/32 words)
// k       - packed BinaryVec for the K vector
// scale_q - mean absolute value of the Q vector
// head_dim - must be a multiple of 32
__device__ __forceinline__ float xnor_dot(
    const uint32_t* __restrict__ q_sign,
    const BinaryVec* __restrict__ k,
    float  scale_q,
    int    head_dim
) {
    const int n_words = head_dim / 32;
    int total_popcount = 0;
#pragma unroll 4
    for (int w = 0; w < n_words; ++w) {
        // XNOR: bits that agree (both 0 or both 1) are set in result
        uint32_t xnor_word = ~(q_sign[w] ^ k->words[w]);
        total_popcount += __popc(xnor_word);
    }
    float scale_k = __half2float(k->scale);
    // Map from popcount in [0, head_dim] to approximate dot product
    return scale_q * scale_k * (float)(2 * total_popcount - head_dim);
}

// Pack sign bits of a float array into uint32 words.
// Operates on a single thread for a contiguous float array of length head_dim.
// Used at query time (Q is never stored in binary form in the KV-cache).
__device__ __forceinline__ void pack_sign_bits(
    const float* __restrict__ vals,
    uint32_t* __restrict__    words,
    int                       head_dim
) {
    const int n_words = head_dim / 32;
    for (int w = 0; w < n_words; ++w) {
        uint32_t packed = 0u;
#pragma unroll
        for (int b = 0; b < 32; ++b) {
            if (vals[w * 32 + b] >= 0.0f) {
                packed |= (1u << b);
            }
        }
        words[w] = packed;
    }
}

// Pack sign bits from __half array.
__device__ __forceinline__ void pack_sign_bits_half(
    const __half* __restrict__ vals,
    uint32_t* __restrict__     words,
    int                        head_dim
) {
    const int n_words = head_dim / 32;
    for (int w = 0; w < n_words; ++w) {
        uint32_t packed = 0u;
#pragma unroll
        for (int b = 0; b < 32; ++b) {
            if (__half2float(vals[w * 32 + b]) >= 0.0f) {
                packed |= (1u << b);
            }
        }
        words[w] = packed;
    }
}

// Compute mean absolute value of a __half array (single thread, full vector).
__device__ __forceinline__ float mean_abs_half(
    const __half* __restrict__ vals,
    int                        head_dim
) {
    float sum = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        sum += fabsf(__half2float(vals[i]));
    }
    return sum / (float)head_dim;
}
