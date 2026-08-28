#include "residual.cuh"
#include <cuda_fp16.h>
#include <cassert>
#include <cmath>

// Each block processes one vector.
// Phase 1: compute residual = original - sign_reconstruction, find absmax.
// Phase 2: quantize to INT4 and pack nibbles.
__global__ void residual_encode_kernel(
    const __half* __restrict__    src,
    const BinaryVec* __restrict__ binary,
    ResidualVec* __restrict__     out,
    int                           n_vecs,
    int                           head_dim
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_vecs) return;

    const __half* vec      = src + (long long)vec_idx * head_dim;
    const BinaryVec* bvec  = &binary[vec_idx];
    ResidualVec* rvec      = &out[vec_idx];

    // Shared: residuals (float) + partial absmax reduction
    extern __shared__ float smem[];
    float* residuals = smem;                    // [head_dim]
    float* partial_max = smem + head_dim;       // [blockDim.x]

    float scale_bin = __half2float(bvec->scale);

    // Compute residual for each element owned by this thread
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        // Binary reconstruction: sign_bit * scale
        int bit = (bvec->words[i / 32] >> (i % 32)) & 1u;
        float reconstructed = (bit ? scale_bin : -scale_bin);
        float orig = __half2float(vec[i]);
        float res  = orig - reconstructed;
        residuals[i] = res;
        local_max = fmaxf(local_max, fabsf(res));
    }
    partial_max[threadIdx.x] = local_max;
    __syncthreads();

    // Reduce to find absmax
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_max[threadIdx.x] = fmaxf(partial_max[threadIdx.x],
                                              partial_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float absmax = partial_max[0];
    float inv_absmax = (absmax > 0.0f) ? (7.0f / absmax) : 0.0f;

    if (threadIdx.x == 0) {
        rvec->scale = __float2half(absmax);
    }
    __syncthreads();

    // Quantize pairs of elements and pack into nibbles
    // Element 2i -> low nibble, element 2i+1 -> high nibble
    for (int i = threadIdx.x; i < head_dim / 2; i += blockDim.x) {
        int e0 = 2 * i;
        int e1 = 2 * i + 1;

        int q0 = __float2int_rn(residuals[e0] * inv_absmax);
        int q1 = __float2int_rn(residuals[e1] * inv_absmax);

        // Clamp to [-7, 7]
        q0 = max(-7, min(7, q0));
        q1 = max(-7, min(7, q1));

        // Pack: low nibble = q0 & 0xF, high nibble = (q1 & 0xF) << 4
        uint8_t packed = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        rvec->nibbles[i] = packed;
    }
}

void launch_residual_encode(
    const __half*    src,
    const BinaryVec* binary,
    ResidualVec*     out,
    int              n_vecs,
    int              head_dim,
    cudaStream_t     stream
) {
    assert(head_dim % 32 == 0);
    assert(head_dim <= MAX_HEAD_DIM);

    const int threads = 128;
    // Shared: head_dim floats (residuals) + threads floats (partial_max)
    size_t smem_bytes = (head_dim + threads) * sizeof(float);

    residual_encode_kernel<<<n_vecs, threads, smem_bytes, stream>>>(
        src, binary, out, n_vecs, head_dim
    );
}
