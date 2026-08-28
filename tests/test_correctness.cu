#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

#include "binary_encode.cuh"
#include "xnor_dot.cuh"
#include "residual.cuh"
#include "binary_attention.cuh"
#include "compensated_attention.cuh"
#include "reference_attention.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// Fill device fp16 buffer with uniform random in [-1, 1]
static void fill_random_fp16(std::vector<__half>& h_buf, int n, float scale = 1.0f) {
    for (int i = 0; i < n; ++i) {
        float v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        h_buf[i] = __float2half(v * scale);
    }
}

// Cosine similarity between two fp16 host vectors
static float cosine_sim(const std::vector<__half>& a, const std::vector<__half>& b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        float ai = __half2float(a[i]);
        float bi = __half2float(b[i]);
        dot += ai * bi;
        na  += ai * ai;
        nb  += bi * bi;
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

// Max absolute error between two fp16 host vectors
static float max_abs_err(const std::vector<__half>& a, const std::vector<__half>& b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; ++i) {
        mx = fmaxf(mx, fabsf(__half2float(a[i]) - __half2float(b[i])));
    }
    return mx;
}

// Kendall tau rank correlation between two float vectors (attention weights for one query)
// O(n^2) is fine for seq_len <= 8192 in a test context.
static float kendall_tau(const std::vector<float>& a, const std::vector<float>& b, int n) {
    long long concordant = 0, discordant = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            float da = a[i] - a[j];
            float db = b[i] - b[j];
            if (da * db > 0) ++concordant;
            else if (da * db < 0) ++discordant;
        }
    }
    long long total = (long long)n * (n - 1) / 2;
    if (total == 0) return 1.0f;
    return (float)(concordant - discordant) / (float)total;
}

struct Config {
    int seq_len;
    int head_dim;
};

static void run_config(const Config& cfg) {
    const int batch   = 1;
    const int n_heads = 1;
    const int S = cfg.seq_len;
    const int D = cfg.head_dim;

    printf("\n=== seq_len=%d, head_dim=%d ===\n", S, D);

    // Host buffers
    std::vector<__half> h_Q(batch * n_heads * D);
    std::vector<__half> h_K(batch * n_heads * S * D);
    std::vector<__half> h_V(batch * n_heads * S * D);

    srand(42 + S + D);
    fill_random_fp16(h_Q, h_Q.size(), 0.5f);
    fill_random_fp16(h_K, h_K.size(), 0.5f);
    fill_random_fp16(h_V, h_V.size(), 0.5f);

    // Device buffers
    __half *d_Q, *d_K, *d_V;
    CUDA_CHECK(cudaMalloc(&d_Q, h_Q.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_K, h_K.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_V, h_V.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), h_K.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), h_V.size() * sizeof(__half), cudaMemcpyHostToDevice));

    // Encode K and V to binary
    BinaryVec *d_K_bin, *d_V_bin;
    CUDA_CHECK(cudaMalloc(&d_K_bin, batch * n_heads * S * sizeof(BinaryVec)));
    CUDA_CHECK(cudaMalloc(&d_V_bin, batch * n_heads * S * sizeof(BinaryVec)));
    launch_binary_encode(d_K, d_K_bin, batch * n_heads * S, D);
    launch_binary_encode(d_V, d_V_bin, batch * n_heads * S, D);

    // Encode K and V residuals
    ResidualVec *d_K_res, *d_V_res;
    CUDA_CHECK(cudaMalloc(&d_K_res, batch * n_heads * S * sizeof(ResidualVec)));
    CUDA_CHECK(cudaMalloc(&d_V_res, batch * n_heads * S * sizeof(ResidualVec)));
    launch_residual_encode(d_K, d_K_bin, d_K_res, batch * n_heads * S, D);
    launch_residual_encode(d_V, d_V_bin, d_V_res, batch * n_heads * S, D);

    // Run reference attention (K and V in [batch*n_heads, seq_len, head_dim] layout)
    __half *d_out_ref, *d_out_bin, *d_out_comp;
    CUDA_CHECK(cudaMalloc(&d_out_ref,  batch * n_heads * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out_bin,  batch * n_heads * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out_comp, batch * n_heads * D * sizeof(__half)));

    launch_reference_attention(d_Q, d_K, d_V, d_out_ref, batch, n_heads, S, D);
    launch_binary_attention(d_Q, d_K_bin, d_V_bin, d_out_bin, batch, n_heads, S, D);
    launch_compensated_attention(d_Q, d_K_bin, d_K_res, d_V_bin, d_V_res, d_out_comp,
                                 batch, n_heads, S, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy outputs back
    std::vector<__half> h_ref(batch * n_heads * D);
    std::vector<__half> h_bin(batch * n_heads * D);
    std::vector<__half> h_comp(batch * n_heads * D);
    CUDA_CHECK(cudaMemcpy(h_ref.data(),  d_out_ref,  h_ref.size()  * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bin.data(),  d_out_bin,  h_bin.size()  * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_out_comp, h_comp.size() * sizeof(__half), cudaMemcpyDeviceToHost));

    // Output quality metrics
    float cos_bin  = cosine_sim(h_ref, h_bin,  D);
    float cos_comp = cosine_sim(h_ref, h_comp, D);
    float mae_bin  = max_abs_err(h_ref, h_bin,  D);
    float mae_comp = max_abs_err(h_ref, h_comp, D);

    printf("  binary    output: cosine_sim=%.4f, max_abs_err=%.4f\n", cos_bin,  mae_bin);
    printf("  compensated output: cosine_sim=%.4f, max_abs_err=%.4f\n", cos_comp, mae_comp);

    // Compute K attention scores for rank correlation (raw dot products, no softmax)
    // We compute them on the host for the first query only.
    // Reference scores: Q * K[s] / sqrt(D)
    std::vector<float> ref_scores(S), bin_scores(S), comp_scores(S);
    const float inv_sqrt_d = 1.0f / sqrtf((float)D);

    // Pack Q sign bits on host for binary scoring
    float q_fp[MAX_HEAD_DIM];
    for (int i = 0; i < D; ++i) q_fp[i] = __half2float(h_Q[i]);
    float scale_q = 0.0f;
    for (int i = 0; i < D; ++i) scale_q += fabsf(q_fp[i]);
    scale_q /= (float)D;
    uint32_t q_sign[MAX_HEAD_DIM / 32] = {};
    for (int w = 0; w < D / 32; ++w) {
        for (int b = 0; b < 32; ++b) {
            if (q_fp[w * 32 + b] >= 0.0f) q_sign[w] |= (1u << b);
        }
    }

    // Copy binary K to host for CPU scoring
    std::vector<BinaryVec> h_K_bin(S);
    std::vector<ResidualVec> h_K_res(S);
    CUDA_CHECK(cudaMemcpy(h_K_bin.data(), d_K_bin, S * sizeof(BinaryVec), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_K_res.data(), d_K_res, S * sizeof(ResidualVec), cudaMemcpyDeviceToHost));

    for (int s = 0; s < S; ++s) {
        // Reference
        float dot = 0.0f;
        for (int i = 0; i < D; ++i) dot += q_fp[i] * __half2float(h_K[s * D + i]);
        ref_scores[s] = dot * inv_sqrt_d;

        // Binary
        int popcnt = 0;
        for (int w = 0; w < D / 32; ++w) {
            popcnt += __builtin_popcount(~(q_sign[w] ^ h_K_bin[s].words[w]));
        }
        bin_scores[s] = scale_q * __half2float(h_K_bin[s].scale)
                        * (float)(2 * popcnt - D) * inv_sqrt_d;

        // Compensated
        float res_dot = 0.0f;
        float k_res_scale = __half2float(h_K_res[s].scale);
        for (int i = 0; i < D / 2; ++i) {
            uint8_t byte = h_K_res[s].nibbles[i];
            int8_t nlo = (int8_t)(byte & 0x0F); if (nlo >= 8) nlo -= 16;
            int8_t nhi = (int8_t)((byte >> 4) & 0x0F); if (nhi >= 8) nhi -= 16;
            res_dot += q_fp[2*i]   * (nlo / 7.0f) * k_res_scale;
            res_dot += q_fp[2*i+1] * (nhi / 7.0f) * k_res_scale;
        }
        comp_scores[s] = bin_scores[s] + res_dot * inv_sqrt_d;
    }

    // Kendall tau on a subset for large seq_len to keep test time reasonable
    int tau_n = std::min(S, 512);
    float tau_bin  = kendall_tau(ref_scores, bin_scores,  tau_n);
    float tau_comp = kendall_tau(ref_scores, comp_scores, tau_n);
    printf("  score rank correlation (Kendall tau, n=%d): binary=%.4f, compensated=%.4f\n",
           tau_n, tau_bin, tau_comp);

    // Cleanup
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_K_bin); cudaFree(d_V_bin);
    cudaFree(d_K_res); cudaFree(d_V_res);
    cudaFree(d_out_ref); cudaFree(d_out_bin); cudaFree(d_out_comp);
}

int main() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    Config configs[] = {
        {512,  64},
        {512,  128},
        {2048, 64},
        {2048, 128},
        {8192, 128},
    };

    for (auto& cfg : configs) {
        run_config(cfg);
    }

    printf("\nAll correctness tests complete.\n");
    return 0;
}
