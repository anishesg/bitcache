"""
Low-level functional API for direct kernel access.

Each function is a thin wrapper around the corresponding C extension entry point,
with input validation and docstrings. Use these when you need fine-grained control
over which kernels to invoke, or when building your own cache management layer.
"""

from __future__ import annotations

import torch
from torch import Tensor

from bitcache import _C, _require_extension


def encode(src: Tensor) -> Tensor:
    """
    Binary-encode a batch of fp16 vectors.

    Extracts sign bits and computes a per-vector mean-absolute-value scale.
    The output is an opaque uint8 buffer holding an array of BinaryVec structs.

    Args:
        src: [n_vecs, head_dim] float16 on CUDA, head_dim must be multiple of 32

    Returns:
        binary_blob: [n_vecs * sizeof(BinaryVec)] uint8 on CUDA
    """
    _require_extension()
    assert src.is_cuda, "src must be on CUDA"
    assert src.is_contiguous(), "src must be contiguous"
    assert src.dtype == torch.float16, "src must be float16"
    assert src.dim() == 2, "src must be 2D [n_vecs, head_dim]"
    return _C.binary_encode(src)


def compute_residual(src: Tensor, binary_blob: Tensor) -> Tensor:
    """
    Compute INT4 residuals between original fp16 vectors and their binary reconstructions.

    Residual = original - sign_reconstruction. The residuals are quantized to INT4
    with per-vector absmax scaling and packed two nibbles per byte.

    Args:
        src:         [n_vecs, head_dim] float16 on CUDA
        binary_blob: [n_vecs * sizeof(BinaryVec)] uint8 on CUDA (from encode())

    Returns:
        residual_blob: [n_vecs * sizeof(ResidualVec)] uint8 on CUDA
    """
    _require_extension()
    assert src.is_cuda and binary_blob.is_cuda, "tensors must be on CUDA"
    assert src.is_contiguous() and binary_blob.is_contiguous(), "tensors must be contiguous"
    assert src.dtype == torch.float16, "src must be float16"
    return _C.residual_encode(src, binary_blob)


def binary_attention(
    Q: Tensor,
    K_bin: Tensor,
    V_bin: Tensor,
    batch: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
) -> Tensor:
    """
    Binary-only approximate attention.

    Scores are computed via XNOR-popcount between packed Q and K sign bits.
    V reconstruction uses the binary sign pattern scaled by the stored mean magnitude.
    No residual correction is applied.

    Args:
        Q:        [batch, n_heads, head_dim] float16 on CUDA
        K_bin:    [batch * n_heads * seq_len * sizeof(BinaryVec)] uint8 on CUDA
        V_bin:    [batch * n_heads * seq_len * sizeof(BinaryVec)] uint8 on CUDA
        batch, n_heads, seq_len, head_dim: tensor dimensions

    Returns:
        output: [batch, n_heads, head_dim] float16
    """
    _require_extension()
    assert Q.is_cuda, "Q must be on CUDA"
    assert Q.is_contiguous(), "Q must be contiguous"
    assert Q.dtype == torch.float16, "Q must be float16"
    return _C.binary_attention(Q, K_bin, V_bin, batch, n_heads, seq_len, head_dim)


def compensated_attention(
    Q: Tensor,
    K_bin: Tensor,
    K_res: Tensor,
    V_bin: Tensor,
    V_res: Tensor,
    batch: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
) -> Tensor:
    """
    Residual-compensated binary attention.

    Attention scores = binary_dot(Q, K) + residual_dot(Q, K_res).
    Output = binary_V_reconstruction + residual_V_correction.
    The INT4 residual correction recovers 1-2% output quality over binary-only.

    Args:
        Q:     [batch, n_heads, head_dim] float16 on CUDA
        K_bin: [batch * n_heads * seq_len * sizeof(BinaryVec)] uint8 on CUDA
        K_res: [batch * n_heads * seq_len * sizeof(ResidualVec)] uint8 on CUDA
        V_bin: [batch * n_heads * seq_len * sizeof(BinaryVec)] uint8 on CUDA
        V_res: [batch * n_heads * seq_len * sizeof(ResidualVec)] uint8 on CUDA
        batch, n_heads, seq_len, head_dim: tensor dimensions

    Returns:
        output: [batch, n_heads, head_dim] float16
    """
    _require_extension()
    assert Q.is_cuda, "Q must be on CUDA"
    assert Q.is_contiguous(), "Q must be contiguous"
    assert Q.dtype == torch.float16, "Q must be float16"
    return _C.compensated_attention(
        Q, K_bin, K_res, V_bin, V_res,
        batch, n_heads, seq_len, head_dim,
    )


def sizeof_binary_vec() -> int:
    """Return the size in bytes of one BinaryVec struct on the current build."""
    _require_extension()
    return _C.sizeof_binary_vec()


def sizeof_residual_vec() -> int:
    """Return the size in bytes of one ResidualVec struct on the current build."""
    _require_extension()
    return _C.sizeof_residual_vec()
