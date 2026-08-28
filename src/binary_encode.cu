#include "binary_encode.cuh"
#include <cuda_fp16.h>
#include <cassert>

// Each block handles one vector.
// Thread i accumulates element i's sign bit into a shared uint32 word.
// After all elements are processed, one thread writes the packed words and scale.
__global__ void binary_encode_kernel(
    const __half* __restrict__ src,
    BinaryVec* __restrict__    out,
    int                        n_vecs,
    int                        head_dim
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_vecs) return;

    const int n_words = (head_dim + 31) / 32;
    const __half* vec = src + (long long)vec_idx * head_dim;

    // Shared: packed words + partial sum for scale computation
    extern __shared__ float smem[];
    float* partial_abs = smem;                          // [blockDim.x] for scale reduction
    uint32_t* packed   = (uint32_t*)(smem + blockDim.x); // [n_words]

    // Initialize packed words to zero
    for (int w = threadIdx.x; w < n_words; w += blockDim.x) {
        packed[w] = 0u;
    }
    __syncthreads();

    // Each thread handles elements strided across the vector
    float local_abs_sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float val = __half2float(vec[i]);
        local_abs_sum += fabsf(val);
        if (val >= 0.0f) {
            // Set bit i in word i/32
            atomicOr(&packed[i / 32], 1u << (i % 32));
        }
    }
    partial_abs[threadIdx.x] = local_abs_sum;
    __syncthreads();

    // Parallel reduction for mean absolute value
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_abs[threadIdx.x] += partial_abs[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean_abs = partial_abs[0] / (float)head_dim;
        out[vec_idx].scale = __float2half(mean_abs);
        // Copy packed words
        for (int w = 0; w < n_words; w++) {
            out[vec_idx].words[w] = packed[w];
        }
    }
}

void launch_binary_encode(
    const __half* src,
    BinaryVec*    out,
    int           n_vecs,
    int           head_dim,
    cudaStream_t  stream
) {
    assert(head_dim % 32 == 0);
    assert(head_dim <= MAX_HEAD_DIM);

    const int threads = 128;
    const int n_words = head_dim / 32;
    // Shared memory: threads floats for reduction + n_words uint32 for packing
    size_t smem_bytes = threads * sizeof(float) + n_words * sizeof(uint32_t);

    binary_encode_kernel<<<n_vecs, threads, smem_bytes, stream>>>(
        src, out, n_vecs, head_dim
    );
}
