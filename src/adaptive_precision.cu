#include "adaptive_precision.cuh"
#include "xnor_dot.cuh"
#include <cuda_fp16.h>
#include <cassert>
#include <cmath>

// One block per (batch, head) pair.
// Computes softmax attention weights from binary scores, then calculates
// Shannon entropy H = -sum(p * log(p)) as a measure of attention sharpness.
//
// Sharp attention (few large weights) -> low entropy -> needs residual correction.
// Flat attention (uniform weights)    -> high entropy -> binary approximation is fine.
__global__ void compute_head_entropy_kernel(
    const __half* __restrict__    Q,
    const BinaryVec* __restrict__ K_bin,
    float* __restrict__           entropy_out,
    int                           seq_len,
    int                           head_dim,
    int                           n_slots    // batch * n_heads
) {
    const int slot = blockIdx.x;
    if (slot >= n_slots) return;

    const __half* q = Q + (long long)slot * head_dim;
    const BinaryVec* k_cache = K_bin + (long long)slot * seq_len;

    // Shared memory: partial sums for reductions
    extern __shared__ float smem[];
    float* partials = smem;  // [blockDim.x]

    // Pack Q sign bits and compute Q scale on the fly (single thread sections)
    // Registers for Q sign bits (up to MAX_HEAD_DIM / 32 = 8 words)
    uint32_t q_sign[MAX_HEAD_DIM / 32];
    float scale_q = 0.0f;

    // Each thread accumulates partial abs-sum and writes sign bits via shared atomics
    float* packed_f = smem + blockDim.x;  // reuse smem for packed words (uint32 overlay)
    uint32_t* packed_words = (uint32_t*)packed_f;
    const int n_words = head_dim / 32;

    for (int w = threadIdx.x; w < n_words; w += blockDim.x) {
        packed_words[w] = 0u;
    }
    __syncthreads();

    float local_abs = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float val = __half2float(q[i]);
        local_abs += fabsf(val);
        if (val >= 0.0f) atomicOr(&packed_words[i / 32], 1u << (i % 32));
    }
    partials[threadIdx.x] = local_abs;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    scale_q = partials[0] / (float)head_dim;

    // Copy packed words to registers for fast per-thread xnor_dot
    for (int w = 0; w < n_words; w++) q_sign[w] = packed_words[w];
    __syncthreads();

    // --- Online softmax over binary scores ---
    // Two-pass: first find max, then accumulate exp sum and entropy.
    // Pass 1: each thread finds its local max score
    float local_max = -1e30f;
    for (int s = threadIdx.x; s < seq_len; s += blockDim.x) {
        int popcnt = 0;
        for (int w = 0; w < n_words; w++) {
            popcnt += __popc(~(q_sign[w] ^ k_cache[s].words[w]));
        }
        float score = scale_q * __half2float(k_cache[s].scale)
                      * (float)(2 * popcnt - head_dim);
        local_max = fmaxf(local_max, score);
    }
    partials[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] = fmaxf(partials[threadIdx.x], partials[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float global_max = partials[0];

    // Pass 2: accumulate exp sum and entropy numerator -sum(p_i * score_i) + log(Z)
    float local_exp_sum = 0.0f;
    float local_wsum    = 0.0f;  // sum of exp(s - max) * s for entropy via E[score]
    for (int s = threadIdx.x; s < seq_len; s += blockDim.x) {
        int popcnt = 0;
        for (int w = 0; w < n_words; w++) {
            popcnt += __popc(~(q_sign[w] ^ k_cache[s].words[w]));
        }
        float score = scale_q * __half2float(k_cache[s].scale)
                      * (float)(2 * popcnt - head_dim);
        float e = expf(score - global_max);
        local_exp_sum += e;
        local_wsum    += e * score;
    }
    partials[threadIdx.x] = local_exp_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    float exp_sum = partials[0];

    partials[threadIdx.x] = local_wsum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    float wsum = partials[0];

    if (threadIdx.x == 0) {
        // H = log(Z) + max - E[score] under softmax
        // This is the standard log-sum-exp entropy identity:
        // H = -sum p_i * log(p_i) = log(Z) - (1/Z) * sum exp(s_i - max) * (s_i - max) - 0
        // = log(Z) - wsum/exp_sum + global_max
        float log_Z = logf(exp_sum) + global_max;
        float entropy = log_Z - wsum / fmaxf(exp_sum, 1e-12f);
        entropy_out[slot] = entropy;
    }
}

// One thread per slot: compare entropy to threshold and write mode.
__global__ void adaptive_select_kernel(
    const float*    entropy,
    PrecisionMode*  modes_out,
    int             n_slots,
    float           threshold
) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n_slots) return;
    modes_out[slot] = (entropy[slot] >= threshold)
                    ? PrecisionMode::BINARY
                    : PrecisionMode::COMPENSATED;
}

void launch_compute_head_entropy(
    const __half*    Q,
    const BinaryVec* K_bin,
    float*           entropy_out,
    int              batch,
    int              n_heads,
    int              seq_len,
    int              head_dim,
    cudaStream_t     stream
) {
    assert(head_dim % 32 == 0);
    assert(head_dim <= MAX_HEAD_DIM);

    const int n_slots = batch * n_heads;
    const int threads = 128;
    const int n_words = head_dim / 32;
    // smem: partials[threads] + packed_words[n_words] (uint32, occupying n_words * 4 bytes)
    size_t smem_bytes = (size_t)threads * sizeof(float)
                      + (size_t)n_words * sizeof(uint32_t);

    compute_head_entropy_kernel<<<n_slots, threads, smem_bytes, stream>>>(
        Q, K_bin, entropy_out, seq_len, head_dim, n_slots
    );
}

void launch_adaptive_select(
    const float*   entropy,
    PrecisionMode* modes_out,
    int            batch,
    int            n_heads,
    float          entropy_threshold,
    cudaStream_t   stream
) {
    const int n_slots = batch * n_heads;
    const int threads = 128;
    const int blocks  = (n_slots + threads - 1) / threads;

    adaptive_select_kernel<<<blocks, threads, 0, stream>>>(
        entropy, modes_out, n_slots, entropy_threshold
    );
}
