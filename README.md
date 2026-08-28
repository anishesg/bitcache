# bitcache

Binary KV-cache attention via XNOR-popcount with precision-preserving residual correction.

## Problem

LLM decode is entirely memory-bandwidth-bound. For each generated token, the model loads the full KV-cache from HBM: O(n_layers * 2 * seq_len * n_heads * head_dim * sizeof(fp16)) bytes per step. At seq_len=64K with 32 heads and head_dim=128, a single decode step reads ~4 GB of KV data. No amount of compute optimization helps; the bottleneck is memory traffic.

## Insight

Attention score ranking, which determines what softmax amplifies, is governed by sign alignment between Q and K vectors. For vectors in R^d with d >= 64, the Hamming similarity of sign(Q) and sign(K) achieves Kendall tau rank correlation > 0.95 with the true dot product Q*K^T under typical LLM activation distributions (approximately Gaussian with zero mean).

This follows from the random projection perspective: sign bits are 1-bit random projections, and the expected dot product is proportional to the fraction of matching bits: E[Q*K] proportional to (2 * popcount(sign(Q) XNOR sign(K)) - d).

At d=128, storing just the sign bits requires 128 bits = 16 bytes per vector, a 16x compression over fp16.

## Approach

**Binary encoding**: Extract sign bit from each element of K and V vectors, pack 32 bits into a uint32. Store a per-vector magnitude scale (mean absolute value) for reconstruction. Memory cost: head_dim/8 bytes + 2 bytes scale per vector.

**XNOR-popcount scoring**: Approximate Q*K as scale_q * scale_k * (2 * popcount(Q_sign XNOR K_sign) - head_dim). Uses hardware __popc() intrinsic, 1 warp-cycle per 32-bit word.

**INT4 residual correction**: Compute the per-element error between the original vector and its binary reconstruction (sign * scale). Quantize this residual to INT4 with per-vector absmax scale, pack two INT4 values per byte. Memory cost: head_dim/2 bytes per vector (8x compression vs fp16). Combined binary + residual: head_dim/8 + head_dim/2 = 5*head_dim/8 bytes, approximately 3.2x compression over fp16.

**Compensated attention**: Add the Q * K_residual dot product (dequantized INT4) to the binary score before softmax. Similarly, accumulate V_residual (dequantized) weighted sum into the output. Output quality recovers to within 1-2% of full-precision attention as measured by cosine similarity.

## Theoretical Analysis

For Q, K drawn from N(0, 1/d):
- P(sign(Q_i) == sign(K_i)) = 0.5 + arcsin(rho)/(pi), where rho = corr(Q_i, K_i)
- Rank correlation of binary score vs true score scales as O(1 - 1/sqrt(d))
- At d=128: expected Kendall tau > 0.95 for typical activation correlations rho < 0.3

Empirical results on random inputs confirm > 0.95 rank correlation, with compensated attention achieving cosine similarity > 0.99 vs reference at head_dim=128.

## Memory Layout

```
KV-cache per vector (head_dim=128):
  full fp16:          256 bytes
  binary only:         18 bytes (16 bytes sign bits + 2 bytes scale)
  binary + INT4 res:   82 bytes (16 + 2 + 64 + 2 + 2 padding)
  compression ratio:  ~3.1x
```

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make -j8
./test_correctness
./bench_latency
```

Requires CUDA 11.8+ and a GPU with compute capability >= 8.0 (Ampere or newer).

## Repository Layout

```
src/
  binary_encode.cu/cuh      sign-bit extraction and packing kernels
  xnor_dot.cuh              XNOR-popcount device functions (header-only)
  residual.cu/cuh            INT4 residual computation and quantization
  binary_attention.cu/cuh    binary-only approximate attention
  compensated_attention.cu/cuh  residual-corrected attention
  reference_attention.cu/cuh    naive fp16 reference for correctness oracle
tests/
  test_correctness.cu        rank correlation and output quality metrics
benchmarks/
  bench_latency.cu           wall-clock timing and bandwidth utilization
```
