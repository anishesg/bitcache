#include "reference_attention.cuh"
#include <cassert>

// One thread per (batch, head) query.
// Computes exact attention in fp32 accumulation.
__global__ void reference_attention_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__       out,
    int seq_len,
    int head_dim
) {
    const int bh = blockIdx.x * blockDim.x + threadIdx.x;

    const __half* q_ptr = Q + (long long)bh * head_dim;
    // K layout: [batch*n_heads, seq_len, head_dim]
    const __half* k_base = K + (long long)bh * seq_len * head_dim;
    const __half* v_base = V + (long long)bh * seq_len * head_dim;

    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    // Load Q into registers
    float q_fp[256];
    for (int i = 0; i < head_dim; ++i) {
        q_fp[i] = __half2float(q_ptr[i]);
    }

    // Pass 1: compute all scores and find max (for numerical stability)
    float score_max = -1e38f;
    // Use local array on stack; seq_len can be large but we allocate dynamically via dynamic parallelism
    // For correctness testing, seq_len <= 8192; stack usage is acceptable here.
    // For large seq_len in production, use global memory scratch -- but this is a reference kernel.
    //
    // We recompute scores in pass 2 to avoid storing a seq_len float array per thread.
    for (int s = 0; s < seq_len; ++s) {
        const __half* k_ptr = k_base + (long long)s * head_dim;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            dot += q_fp[i] * __half2float(k_ptr[i]);
        }
        float score = dot * inv_sqrt_d;
        if (score > score_max) score_max = score;
    }

    // Pass 2: softmax-weighted sum
    float acc[256] = {};
    float sum_exp = 0.0f;
    for (int s = 0; s < seq_len; ++s) {
        const __half* k_ptr = k_base + (long long)s * head_dim;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            dot += q_fp[i] * __half2float(k_ptr[i]);
        }
        float w = expf(dot * inv_sqrt_d - score_max);
        sum_exp += w;

        const __half* v_ptr = v_base + (long long)s * head_dim;
        for (int i = 0; i < head_dim; ++i) {
            acc[i] += w * __half2float(v_ptr[i]);
        }
    }

    __half* out_ptr = out + (long long)bh * head_dim;
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        out_ptr[i] = __float2half(acc[i] * inv_sum);
    }
}

void launch_reference_attention(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half*       out,
    int           batch,
    int           n_heads,
    int           seq_len,
    int           head_dim,
    cudaStream_t  stream
) {
    assert(head_dim <= 256);

    const int total   = batch * n_heads;
    const int threads = 32;
    const int blocks  = (total + threads - 1) / threads;

    reference_attention_kernel<<<blocks, threads, 0, stream>>>(
        Q, K, V, out, seq_len, head_dim
    );
}
