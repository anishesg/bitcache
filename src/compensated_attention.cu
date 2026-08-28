#include "compensated_attention.cuh"
#include "xnor_dot.cuh"
#include <cassert>

// Dequantize INT4 residual vector into float accumulator (additive).
__device__ __forceinline__ void dequant_residual_add(
    const ResidualVec* __restrict__ res,
    float* __restrict__             acc,
    float                           weight,
    int                             head_dim
) {
    float scale = __half2float(res->scale);
    for (int i = 0; i < head_dim / 2; ++i) {
        uint8_t byte = res->nibbles[i];
        float r0 = dequant_nibble(nibble_lo(byte), scale);
        float r1 = dequant_nibble(nibble_hi(byte), scale);
        acc[2 * i]     += weight * r0;
        acc[2 * i + 1] += weight * r1;
    }
}

// Dot product of Q (float array) against dequantized INT4 residual K vector.
__device__ __forceinline__ float residual_dot(
    const float* __restrict__       q_fp,
    const ResidualVec* __restrict__ k_res,
    int                             head_dim
) {
    float scale = __half2float(k_res->scale);
    float dot = 0.0f;
    for (int i = 0; i < head_dim / 2; ++i) {
        uint8_t byte = k_res->nibbles[i];
        float r0 = dequant_nibble(nibble_lo(byte), scale);
        float r1 = dequant_nibble(nibble_hi(byte), scale);
        dot += q_fp[2 * i] * r0 + q_fp[2 * i + 1] * r1;
    }
    return dot;
}

// One thread per (batch, head) query.
// Score = binary_dot(Q, K) + residual_dot(Q, K_res), then online softmax.
// Output = sum_s( softmax(s) * (binary_V(s) + dequant_residual_V(s)) )
__global__ void compensated_attention_kernel(
    const __half* __restrict__       Q,
    const BinaryVec* __restrict__    K_bin,
    const ResidualVec* __restrict__  K_res,
    const BinaryVec* __restrict__    V_bin,
    const ResidualVec* __restrict__  V_res,
    __half* __restrict__             out,
    int seq_len,
    int head_dim
) {
    const int bh = blockIdx.x * blockDim.x + threadIdx.x;

    const int n_words = head_dim / 32;

    const __half* q_ptr = Q + (long long)bh * head_dim;
    float q_fp[MAX_HEAD_DIM];
    for (int i = 0; i < head_dim; ++i) {
        q_fp[i] = __half2float(q_ptr[i]);
    }

    uint32_t q_sign[MAX_HEAD_DIM / 32];
    pack_sign_bits(q_fp, q_sign, head_dim);

    float scale_q = 0.0f;
    for (int i = 0; i < head_dim; ++i) scale_q += fabsf(q_fp[i]);
    scale_q /= (float)head_dim;

    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    const BinaryVec*   K_bin_head = K_bin + (long long)bh * seq_len;
    const ResidualVec* K_res_head = K_res + (long long)bh * seq_len;
    const BinaryVec*   V_bin_head = V_bin + (long long)bh * seq_len;
    const ResidualVec* V_res_head = V_res + (long long)bh * seq_len;

    // Pass 1: find max compensated score
    float running_max = -1e38f;
    for (int s = 0; s < seq_len; ++s) {
        float score = (xnor_dot(q_sign, &K_bin_head[s], scale_q, head_dim)
                       + residual_dot(q_fp, &K_res_head[s], head_dim)) * inv_sqrt_d;
        if (score > running_max) running_max = score;
    }

    // Pass 2: softmax-weighted sum of corrected V
    float running_sum = 0.0f;
    float acc[MAX_HEAD_DIM] = {};
    for (int s = 0; s < seq_len; ++s) {
        float score = (xnor_dot(q_sign, &K_bin_head[s], scale_q, head_dim)
                       + residual_dot(q_fp, &K_res_head[s], head_dim)) * inv_sqrt_d;
        float w = expf(score - running_max);
        running_sum += w;

        // Binary V base
        float scale_v = __half2float(V_bin_head[s].scale);
        for (int i = 0; i < head_dim; ++i) {
            int bit = (V_bin_head[s].words[i / 32] >> (i % 32)) & 1u;
            acc[i] += w * (bit ? scale_v : -scale_v);
        }

        // Additive INT4 V residual
        dequant_residual_add(&V_res_head[s], acc, w, head_dim);
    }

    __half* out_ptr = out + (long long)bh * head_dim;
    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        out_ptr[i] = __float2half(acc[i] * inv_sum);
    }
}

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
    cudaStream_t        stream
) {
    assert(head_dim % 32 == 0);
    assert(head_dim <= MAX_HEAD_DIM);

    const int total   = batch * n_heads;
    const int threads = 32;
    const int blocks  = (total + threads - 1) / threads;

    compensated_attention_kernel<<<blocks, threads, 0, stream>>>(
        Q, K_bin, K_res, V_bin, V_res, out, seq_len, head_dim
    );
}
