#include <torch/extension.h>
#include "binary_encode.cuh"
#include "residual.cuh"
#include "binary_attention.cuh"
#include "compensated_attention.cuh"
#include "online_encode.cuh"
#include "adaptive_precision.cuh"

#define CHECK_CUDA(x)       TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x)       TORCH_CHECK((x).dtype() == torch::kFloat16, #x " must be float16")
#define CHECK_INPUT(x)      CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_HALF(x)

// binary_encode: fp16 tensor [n_vecs, head_dim] -> opaque byte tensor for BinaryVec array
// Returns a uint8 tensor of shape [n_vecs * sizeof(BinaryVec)] that carries packed sign bits
// and scale values. Callers treat it as an opaque buffer.
torch::Tensor binary_encode(torch::Tensor src) {
    CHECK_INPUT(src);
    TORCH_CHECK(src.dim() == 2, "src must be 2D [n_vecs, head_dim]");
    int n_vecs   = src.size(0);
    int head_dim = src.size(1);
    TORCH_CHECK(head_dim % 32 == 0, "head_dim must be a multiple of 32");
    TORCH_CHECK(head_dim <= MAX_HEAD_DIM, "head_dim exceeds MAX_HEAD_DIM=", MAX_HEAD_DIM);

    auto out = torch::empty(
        {(int64_t)(n_vecs * sizeof(BinaryVec))},
        torch::TensorOptions().dtype(torch::kUInt8).device(src.device())
    );

    launch_binary_encode(
        reinterpret_cast<const __half*>(src.data_ptr<at::Half>()),
        reinterpret_cast<BinaryVec*>(out.data_ptr<uint8_t>()),
        n_vecs, head_dim,
        at::cuda::getCurrentCUDAStream()
    );
    return out;
}

// residual_encode: original fp16 [n_vecs, head_dim] + binary blob -> ResidualVec blob
torch::Tensor residual_encode(torch::Tensor src, torch::Tensor binary_blob) {
    CHECK_INPUT(src);
    CHECK_CUDA(binary_blob); CHECK_CONTIGUOUS(binary_blob);
    int n_vecs   = src.size(0);
    int head_dim = src.size(1);
    TORCH_CHECK(head_dim % 32 == 0, "head_dim must be a multiple of 32");
    TORCH_CHECK(head_dim <= MAX_HEAD_DIM, "head_dim exceeds MAX_HEAD_DIM=", MAX_HEAD_DIM);
    TORCH_CHECK(
        binary_blob.numel() == (int64_t)(n_vecs * sizeof(BinaryVec)),
        "binary_blob size mismatch"
    );

    auto out = torch::empty(
        {(int64_t)(n_vecs * sizeof(ResidualVec))},
        torch::TensorOptions().dtype(torch::kUInt8).device(src.device())
    );

    launch_residual_encode(
        reinterpret_cast<const __half*>(src.data_ptr<at::Half>()),
        reinterpret_cast<const BinaryVec*>(binary_blob.data_ptr<uint8_t>()),
        reinterpret_cast<ResidualVec*>(out.data_ptr<uint8_t>()),
        n_vecs, head_dim,
        at::cuda::getCurrentCUDAStream()
    );
    return out;
}

// binary_attention: Q [batch, n_heads, head_dim], K_bin blob, V_bin blob -> [batch, n_heads, head_dim]
torch::Tensor binary_attention(
    torch::Tensor Q,
    torch::Tensor K_bin,
    torch::Tensor V_bin,
    int batch, int n_heads, int seq_len, int head_dim
) {
    CHECK_INPUT(Q);
    CHECK_CUDA(K_bin); CHECK_CONTIGUOUS(K_bin);
    CHECK_CUDA(V_bin); CHECK_CONTIGUOUS(V_bin);

    auto out = torch::empty_like(Q);

    launch_binary_attention(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const BinaryVec*>(K_bin.data_ptr<uint8_t>()),
        reinterpret_cast<const BinaryVec*>(V_bin.data_ptr<uint8_t>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        batch, n_heads, seq_len, head_dim,
        at::cuda::getCurrentCUDAStream()
    );
    return out;
}

// compensated_attention: adds INT4 residual correction to binary attention
torch::Tensor compensated_attention(
    torch::Tensor Q,
    torch::Tensor K_bin, torch::Tensor K_res,
    torch::Tensor V_bin, torch::Tensor V_res,
    int batch, int n_heads, int seq_len, int head_dim
) {
    CHECK_INPUT(Q);
    CHECK_CUDA(K_bin); CHECK_CONTIGUOUS(K_bin);
    CHECK_CUDA(K_res); CHECK_CONTIGUOUS(K_res);
    CHECK_CUDA(V_bin); CHECK_CONTIGUOUS(V_bin);
    CHECK_CUDA(V_res); CHECK_CONTIGUOUS(V_res);

    auto out = torch::empty_like(Q);

    launch_compensated_attention(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const BinaryVec*>(K_bin.data_ptr<uint8_t>()),
        reinterpret_cast<const ResidualVec*>(K_res.data_ptr<uint8_t>()),
        reinterpret_cast<const BinaryVec*>(V_bin.data_ptr<uint8_t>()),
        reinterpret_cast<const ResidualVec*>(V_res.data_ptr<uint8_t>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        batch, n_heads, seq_len, head_dim,
        at::cuda::getCurrentCUDAStream()
    );
    return out;
}

// online_encode_token: encode a single new token [batch, n_heads, head_dim] and append
// to existing caches. Mutates binary_cache and residual_cache in-place.
void online_encode_token(
    torch::Tensor kv,          // [batch, n_heads, head_dim] fp16
    torch::Tensor binary_cache, // [batch * n_heads * max_seq * sizeof(BinaryVec)] uint8
    torch::Tensor residual_cache, // [batch * n_heads * max_seq * sizeof(ResidualVec)] uint8
    int batch, int n_heads, int head_dim, int current_len, int max_seq
) {
    CHECK_INPUT(kv);
    CHECK_CUDA(binary_cache);  CHECK_CONTIGUOUS(binary_cache);
    CHECK_CUDA(residual_cache); CHECK_CONTIGUOUS(residual_cache);

    launch_online_encode_token(
        reinterpret_cast<const __half*>(kv.data_ptr<at::Half>()),
        reinterpret_cast<BinaryVec*>(binary_cache.data_ptr<uint8_t>()),
        reinterpret_cast<ResidualVec*>(residual_cache.data_ptr<uint8_t>()),
        batch, n_heads, head_dim, current_len, max_seq,
        at::cuda::getCurrentCUDAStream()
    );
}

// compute_head_entropy: returns per-head entropy from binary attention scores [batch, n_heads]
torch::Tensor compute_head_entropy(
    torch::Tensor Q,
    torch::Tensor K_bin,
    int batch, int n_heads, int seq_len, int head_dim
) {
    CHECK_INPUT(Q);
    CHECK_CUDA(K_bin); CHECK_CONTIGUOUS(K_bin);

    auto entropy = torch::empty(
        {batch, n_heads},
        torch::TensorOptions().dtype(torch::kFloat32).device(Q.device())
    );

    launch_compute_head_entropy(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const BinaryVec*>(K_bin.data_ptr<uint8_t>()),
        entropy.data_ptr<float>(),
        batch, n_heads, seq_len, head_dim,
        at::cuda::getCurrentCUDAStream()
    );
    return entropy;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("binary_encode",          &binary_encode,          "Encode fp16 vectors to binary (sign bits + scale)");
    m.def("residual_encode",        &residual_encode,        "Compute INT4 residuals from fp16 and binary encoding");
    m.def("binary_attention",       &binary_attention,       "Binary-only approximate attention");
    m.def("compensated_attention",  &compensated_attention,  "Residual-compensated binary attention");
    m.def("online_encode_token",    &online_encode_token,    "Fused single-token encode and cache append");
    m.def("compute_head_entropy",   &compute_head_entropy,   "Compute per-head attention entropy for adaptive precision");
}
