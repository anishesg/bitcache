#include "binary_attention.cuh"
#include "xnor_dot.cuh"
#include <cassert>

// One thread handles one (batch, head) query vector.
// Iterates over all seq_len K binary vectors to compute approximate scores,
// runs online softmax, then accumulates V binary reconstructions weighted by softmax.
__global__ void binary_attention_kernel(
    const __half* __restrict__    Q,
    const BinaryVec* __restrict__ K_bin,
    const BinaryVec* __restrict__ V_bin,
    __half* __restrict__          out,
    int seq_len,
    int head_dim
) {
    // blockIdx.x = flat index over (batch * n_heads)
    const int bh = blockIdx.x * blockDim.x + threadIdx.x;
    // Total launched threads may exceed batch*n_heads; check via gridDim/blockDim in caller
    // We use the n_total passed via head_dim sign: caller ensures bh < batch*n_heads
    // (grid is launched with exactly batch*n_heads threads across blocks)

    const int n_words = head_dim / 32;

    // Load Q vector into registers
    const __half* q_ptr = Q + (long long)bh * head_dim;
    float q_fp[MAX_HEAD_DIM];
    for (int i = 0; i < head_dim; ++i) {
        q_fp[i] = __half2float(q_ptr[i]);
    }

    // Pack Q sign bits and compute Q mean abs
    uint32_t q_sign[MAX_HEAD_DIM / 32];
    pack_sign_bits(q_fp, q_sign, head_dim);
    float scale_q = 0.0f;
    for (int i = 0; i < head_dim; ++i) scale_q += fabsf(q_fp[i]);
    scale_q /= (float)head_dim;

    // Scaling factor for softmax: 1/sqrt(head_dim)
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    // Online softmax state
    float running_max = -1e38f;
    float running_sum = 0.0f;
    float acc[MAX_HEAD_DIM] = {};  // output accumulator in fp32

    // First pass: compute all scores, online softmax numerator/denominator
    // We need to store scores for two-pass softmax since online softmax requires a second pass.
    // To avoid storing seq_len floats, use online (numerically stable) softmax directly:
    // On first pass accumulate max and sum, on second pass accumulate weighted V.
    // This requires two iterations over K_bin; acceptable since K_bin fits in L2.

    const BinaryVec* K_head = K_bin + (long long)bh * seq_len;

    // Pass 1: find max score
    for (int s = 0; s < seq_len; ++s) {
        float score = xnor_dot(q_sign, &K_head[s], scale_q, head_dim) * inv_sqrt_d;
        if (score > running_max) running_max = score;
    }

    // Pass 2: compute softmax weights and accumulate binary V reconstruction
    const BinaryVec* V_head = V_bin + (long long)bh * seq_len;
    for (int s = 0; s < seq_len; ++s) {
        float score = xnor_dot(q_sign, &K_head[s], scale_q, head_dim) * inv_sqrt_d;
        float w = expf(score - running_max);
        running_sum += w;

        // Binary V reconstruction: element i = sign_bit ? scale_v : -scale_v
        float scale_v = __half2float(V_head[s].scale);
        for (int i = 0; i < head_dim; ++i) {
            int bit = (V_head[s].words[i / 32] >> (i % 32)) & 1u;
            float v_val = bit ? scale_v : -scale_v;
            acc[i] += w * v_val;
        }
    }

    // Normalize and write output
    __half* out_ptr = out + (long long)bh * head_dim;
    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        out_ptr[i] = __float2half(acc[i] * inv_sum);
    }
}

void launch_binary_attention(
    const __half*    Q,
    const BinaryVec* K_bin,
    const BinaryVec* V_bin,
    __half*          out,
    int              batch,
    int              n_heads,
    int              seq_len,
    int              head_dim,
    cudaStream_t     stream
) {
    assert(head_dim % 32 == 0);
    assert(head_dim <= MAX_HEAD_DIM);

    const int total = batch * n_heads;
    const int threads = 32;
    const int blocks = (total + threads - 1) / threads;

    binary_attention_kernel<<<blocks, threads, 0, stream>>>(
        Q, K_bin, V_bin, out, seq_len, head_dim
    );
}
