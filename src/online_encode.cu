#include "online_encode.cuh"
#include <cuda_fp16.h>
#include <cassert>

// Fused single-token encode kernel.
//
// One CUDA block per (batch, head) pair.
// Each block:
//   1. Reads head_dim fp16 elements for this (batch, head) slot.
//   2. Packs sign bits into uint32 words and computes mean absolute value (BinaryVec).
//   3. Computes residual = original - binary_reconstruction, finds absmax.
//   4. Quantizes residuals to INT4 and packs nibbles (ResidualVec).
//   5. Writes BinaryVec and ResidualVec to the ring-buffer slot at current_len % max_seq.
//
// All operations are fused into a single kernel launch, eliminating two separate
// binary_encode and residual_encode launches per decode step.
__global__ void online_encode_token_kernel(
    const __half* __restrict__ kv,
    BinaryVec* __restrict__    binary_cache,
    ResidualVec* __restrict__  residual_cache,
    int                        head_dim,
    int                        write_slot,     // current_len % max_seq
    int                        max_seq,
    int                        n_slots         // batch * n_heads
) {
    const int slot_idx = blockIdx.x;
    if (slot_idx >= n_slots) return;

    const __half* vec = kv + (long long)slot_idx * head_dim;

    // Destination in ring buffers: (slot_idx * max_seq + write_slot)
    const int cache_idx       = slot_idx * max_seq + write_slot;
    BinaryVec*   bvec         = &binary_cache[cache_idx];
    ResidualVec* rvec         = &residual_cache[cache_idx];

    const int n_words = head_dim / 32;

    // Shared memory layout:
    //   [0..blockDim.x-1]    : partial abs-sum for binary scale reduction
    //   [blockDim.x..blockDim.x+n_words-1] : packed sign words (uint32)
    //   [blockDim.x+n_words..blockDim.x+n_words+head_dim-1] : float residuals
    //   [blockDim.x+n_words+head_dim..+blockDim.x-1]        : partial absmax
    extern __shared__ float smem[];

    float*    partial_abs  = smem;
    uint32_t* packed_words = (uint32_t*)(smem + blockDim.x);
    float*    residuals    = (float*)(packed_words + n_words);
    float*    partial_max  = residuals + head_dim;

    // --- Phase 1: binary encode ---
    for (int w = threadIdx.x; w < n_words; w += blockDim.x) {
        packed_words[w] = 0u;
    }
    __syncthreads();

    float local_abs_sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float val = __half2float(vec[i]);
        local_abs_sum += fabsf(val);
        if (val >= 0.0f) {
            atomicOr(&packed_words[i / 32], 1u << (i % 32));
        }
    }
    partial_abs[threadIdx.x] = local_abs_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_abs[threadIdx.x] += partial_abs[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float scale_bin = partial_abs[0] / (float)head_dim;

    if (threadIdx.x == 0) {
        bvec->scale = __float2half(scale_bin);
        for (int w = 0; w < n_words; w++) {
            bvec->words[w] = packed_words[w];
        }
    }
    __syncthreads();

    // --- Phase 2: residual encode ---
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        int bit = (packed_words[i / 32] >> (i % 32)) & 1u;
        float reconstructed = bit ? scale_bin : -scale_bin;
        float orig = __half2float(vec[i]);
        float res  = orig - reconstructed;
        residuals[i] = res;
        local_max = fmaxf(local_max, fabsf(res));
    }
    partial_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_max[threadIdx.x] = fmaxf(partial_max[threadIdx.x],
                                              partial_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float absmax     = partial_max[0];
    float inv_absmax = (absmax > 0.0f) ? (7.0f / absmax) : 0.0f;

    if (threadIdx.x == 0) {
        rvec->scale = __float2half(absmax);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < head_dim / 2; i += blockDim.x) {
        int q0 = __float2int_rn(residuals[2 * i]     * inv_absmax);
        int q1 = __float2int_rn(residuals[2 * i + 1] * inv_absmax);
        q0 = max(-7, min(7, q0));
        q1 = max(-7, min(7, q1));
        rvec->nibbles[i] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }
}

void launch_online_encode_token(
    const __half* kv,
    BinaryVec*    binary_cache,
    ResidualVec*  residual_cache,
    int           batch,
    int           n_heads,
    int           head_dim,
    int           current_len,
    int           max_seq,
    cudaStream_t  stream
) {
    assert(head_dim % 32 == 0);
    assert(head_dim <= MAX_HEAD_DIM);

    const int n_slots  = batch * n_heads;
    const int threads  = 128;
    const int n_words  = head_dim / 32;
    const int write_slot = current_len % max_seq;

    // Shared: partial_abs[threads] + packed_words[n_words] + residuals[head_dim] + partial_max[threads]
    size_t smem_bytes = (size_t)(2 * threads + head_dim) * sizeof(float)
                      + (size_t)n_words * sizeof(uint32_t);

    online_encode_token_kernel<<<n_slots, threads, smem_bytes, stream>>>(
        kv, binary_cache, residual_cache,
        head_dim, write_slot, max_seq, n_slots
    );
}
