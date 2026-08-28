#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#include "binary_encode.cuh"
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

static double time_kernel_ms(cudaEvent_t start, cudaEvent_t stop, int n_iters,
                               cudaStream_t stream,
                               void (*fn)(cudaStream_t)) {
    // Warmup
    fn(stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < n_iters; ++i) fn(stream);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return (double)ms / n_iters;
}

struct BenchState {
    int batch, n_heads, seq_len, head_dim;
    __half*      d_Q;
    __half*      d_K;
    __half*      d_V;
    BinaryVec*   d_K_bin;
    BinaryVec*   d_V_bin;
    ResidualVec* d_K_res;
    ResidualVec* d_V_res;
    __half*      d_out;
};

static void setup_bench(BenchState& s) {
    int total_kv = s.batch * s.n_heads * s.seq_len;
    int D = s.head_dim;

    CUDA_CHECK(cudaMalloc(&s.d_Q, s.batch * s.n_heads * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&s.d_K, total_kv * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&s.d_V, total_kv * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&s.d_K_bin, total_kv * sizeof(BinaryVec)));
    CUDA_CHECK(cudaMalloc(&s.d_V_bin, total_kv * sizeof(BinaryVec)));
    CUDA_CHECK(cudaMalloc(&s.d_K_res, total_kv * sizeof(ResidualVec)));
    CUDA_CHECK(cudaMalloc(&s.d_V_res, total_kv * sizeof(ResidualVec)));
    CUDA_CHECK(cudaMalloc(&s.d_out, s.batch * s.n_heads * D * sizeof(__half)));

    // Fill with random half data
    std::vector<__half> tmp(total_kv * D);
    for (auto& x : tmp) x = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);

    CUDA_CHECK(cudaMemcpy(s.d_K, tmp.data(), tmp.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_V, tmp.data(), tmp.size() * sizeof(__half), cudaMemcpyHostToDevice));
    std::vector<__half> qbuf(s.batch * s.n_heads * D);
    for (auto& x : qbuf) x = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
    CUDA_CHECK(cudaMemcpy(s.d_Q, qbuf.data(), qbuf.size() * sizeof(__half), cudaMemcpyHostToDevice));

    // Pre-compute binary and residual encodings (not timed)
    launch_binary_encode(s.d_K, s.d_K_bin, total_kv, D);
    launch_binary_encode(s.d_V, s.d_V_bin, total_kv, D);
    launch_residual_encode(s.d_K, s.d_K_bin, s.d_K_res, total_kv, D);
    launch_residual_encode(s.d_V, s.d_V_bin, s.d_V_res, total_kv, D);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void teardown_bench(BenchState& s) {
    cudaFree(s.d_Q); cudaFree(s.d_K); cudaFree(s.d_V);
    cudaFree(s.d_K_bin); cudaFree(s.d_V_bin);
    cudaFree(s.d_K_res); cudaFree(s.d_V_res);
    cudaFree(s.d_out);
}

// Memory bytes read per query for each kernel variant
// head_dim=128, batch=1, n_heads=1
static double bytes_ref(int seq_len, int head_dim) {
    // Reads full fp16 K and V for all tokens
    return (double)seq_len * 2 * head_dim * sizeof(__half);
}

static double bytes_binary(int seq_len, int head_dim) {
    // Reads BinaryVec for K and V: (head_dim/8 + 2) bytes each
    double bvec_bytes = (double)(head_dim / 8 + sizeof(__half));
    return seq_len * 2 * bvec_bytes;
}

static double bytes_compensated(int seq_len, int head_dim) {
    // BinaryVec + ResidualVec per K and V token
    double bvec_bytes = (double)(head_dim / 8 + sizeof(__half));
    double rvec_bytes = (double)(head_dim / 2 + sizeof(__half));
    return seq_len * 2 * (bvec_bytes + rvec_bytes);
}

int main() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s  HBM bandwidth: %.0f GB/s\n",
           prop.name, (double)prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2e-6);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    const int head_dim = 128;
    const int batch    = 1;
    const int n_heads  = 1;
    const int n_iters  = 20;

    int seq_lens[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536};
    int n_configs  = sizeof(seq_lens) / sizeof(seq_lens[0]);

    printf("\n%-10s  %-12s  %-12s  %-12s  %-10s  %-10s  %-12s\n",
           "seq_len", "ref_ms", "binary_ms", "comp_ms",
           "ref_BW", "bin_BW", "bytes_saved");
    printf("%s\n", std::string(90, '-').c_str());

    for (int ci = 0; ci < n_configs; ++ci) {
        int S = seq_lens[ci];

        BenchState state;
        state.batch    = batch;
        state.n_heads  = n_heads;
        state.seq_len  = S;
        state.head_dim = head_dim;
        setup_bench(state);

        // Reference
        auto fn_ref = [&](cudaStream_t st) {
            launch_reference_attention(state.d_Q, state.d_K, state.d_V, state.d_out,
                                       batch, n_heads, S, head_dim, st);
        };
        double ms_ref = time_kernel_ms(ev_start, ev_stop, n_iters, 0, fn_ref);

        // Binary
        auto fn_bin = [&](cudaStream_t st) {
            launch_binary_attention(state.d_Q, state.d_K_bin, state.d_V_bin, state.d_out,
                                    batch, n_heads, S, head_dim, st);
        };
        double ms_bin = time_kernel_ms(ev_start, ev_stop, n_iters, 0, fn_bin);

        // Compensated
        auto fn_comp = [&](cudaStream_t st) {
            launch_compensated_attention(state.d_Q, state.d_K_bin, state.d_K_res,
                                         state.d_V_bin, state.d_V_res, state.d_out,
                                         batch, n_heads, S, head_dim, st);
        };
        double ms_comp = time_kernel_ms(ev_start, ev_stop, n_iters, 0, fn_comp);

        double b_ref  = bytes_ref(S, head_dim);
        double b_bin  = bytes_binary(S, head_dim);
        double b_comp = bytes_compensated(S, head_dim);

        // Effective bandwidth in GB/s
        double bw_ref  = b_ref  / (ms_ref  * 1e-3) / 1e9;
        double bw_bin  = b_bin  / (ms_bin  * 1e-3) / 1e9;

        double ratio_bin  = b_ref / b_bin;
        double ratio_comp = b_ref / b_comp;

        printf("%-10d  %-12.3f  %-12.3f  %-12.3f  %-10.2f  %-10.2f  "
               "bin=%.1fx comp=%.1fx\n",
               S, ms_ref, ms_bin, ms_comp,
               bw_ref, bw_bin,
               ratio_bin, ratio_comp);

        teardown_bench(state);
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    printf("\nBytes loaded per query at seq_len=8192, head_dim=128:\n");
    printf("  fp16 reference:    %.0f KB\n", bytes_ref(8192, 128) / 1024.0);
    printf("  binary only:       %.0f KB  (%.1fx compression)\n",
           bytes_binary(8192, 128) / 1024.0, bytes_ref(8192, 128) / bytes_binary(8192, 128));
    printf("  binary + residual: %.0f KB  (%.1fx compression)\n",
           bytes_compensated(8192, 128) / 1024.0,
           bytes_ref(8192, 128) / bytes_compensated(8192, 128));

    return 0;
}
