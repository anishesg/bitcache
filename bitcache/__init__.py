"""
bitcache: binary KV-cache with XNOR-popcount scoring and INT4 residual correction.

The BitCache class manages the full encode/store/attend lifecycle with ring-buffer
semantics for streaming autoregressive generation.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Optional

try:
    from bitcache import _C
    _HAS_EXTENSION = True
except ImportError:
    _HAS_EXTENSION = False
    _C = None  # type: ignore[assignment]


def _require_extension() -> None:
    if not _HAS_EXTENSION:
        raise RuntimeError(
            "bitcache C extension not found. Run: pip install -e . or python setup.py build_ext --inplace"
        )


class BitCache:
    """
    Binary KV-cache for a single transformer layer.

    Encodes incoming K/V vectors to binary (sign bits + scale) on append,
    optionally stores INT4 residuals for compensated attention, and dispatches
    to the appropriate CUDA kernel on attend().

    Ring-buffer semantics: once max_seq is reached, old entries are overwritten
    starting from the oldest (FIFO eviction), so the total allocated memory is
    fixed regardless of generation length.

    Args:
        batch:       batch size
        n_heads:     number of attention heads
        head_dim:    dimension per head (must be multiple of 32, <= 256)
        max_seq:     maximum cached sequence length (ring buffer capacity)
        device:      torch device string, e.g. "cuda:0"
        compensated: if True, store INT4 residuals and use compensated attention
    """

    def __init__(
        self,
        batch: int,
        n_heads: int,
        head_dim: int,
        max_seq: int,
        device: str = "cuda",
        compensated: bool = True,
    ) -> None:
        _require_extension()
        assert head_dim % 32 == 0, "head_dim must be a multiple of 32"
        assert head_dim <= 256, "head_dim must be <= 256"

        self.batch       = batch
        self.n_heads     = n_heads
        self.head_dim    = head_dim
        self.max_seq     = max_seq
        self.device      = torch.device(device)
        self.compensated = compensated

        self._bv_size = _C.sizeof_binary_vec()
        self._rv_size = _C.sizeof_residual_vec()

        n_slots = batch * n_heads * max_seq

        # Binary caches for K and V: opaque uint8 buffers holding BinaryVec structs
        self._k_bin = torch.zeros(n_slots * self._bv_size, dtype=torch.uint8, device=self.device)
        self._v_bin = torch.zeros(n_slots * self._bv_size, dtype=torch.uint8, device=self.device)

        # INT4 residual caches (only allocated when compensated=True)
        if self.compensated:
            self._k_res = torch.zeros(n_slots * self._rv_size, dtype=torch.uint8, device=self.device)
            self._v_res = torch.zeros(n_slots * self._rv_size, dtype=torch.uint8, device=self.device)
        else:
            self._k_res = None
            self._v_res = None

        # Ring buffer state
        self._write_pos = 0    # next slot to write (in [0, max_seq))
        self._filled    = 0    # number of valid positions (saturates at max_seq)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def append(self, K: Tensor, V: Tensor) -> None:
        """
        Encode a new set of K/V vectors and store them in the ring buffer.

        Args:
            K: [batch, n_heads, head_dim] float16
            V: [batch, n_heads, head_dim] float16
        """
        self._validate_kv(K, V)
        _C.online_encode_token(
            K, self._k_bin, self._k_res if self.compensated else self._k_bin,
            self.batch, self.n_heads, self.head_dim, self._write_pos, self.max_seq,
        )
        _C.online_encode_token(
            V, self._v_bin, self._v_res if self.compensated else self._v_bin,
            self.batch, self.n_heads, self.head_dim, self._write_pos, self.max_seq,
        )
        self._write_pos = (self._write_pos + 1) % self.max_seq
        self._filled    = min(self._filled + 1, self.max_seq)

    def attend(self, Q: Tensor) -> Tensor:
        """
        Run attention over the current cache contents.

        Args:
            Q: [batch, n_heads, head_dim] float16

        Returns:
            output: [batch, n_heads, head_dim] float16
        """
        assert self._filled > 0, "cache is empty, call append() first"
        seq_len  = self._filled
        batch    = self.batch
        n_heads  = self.n_heads
        head_dim = self.head_dim

        # Slice to the valid portion of the ring buffer.
        # For a not-yet-full buffer this is always the prefix [0, seq_len).
        # For a full buffer the valid range wraps; we use contiguous prefix for now.
        k_bin_slice = self._k_bin[: batch * n_heads * seq_len * self._bv_size]
        v_bin_slice = self._v_bin[: batch * n_heads * seq_len * self._bv_size]

        if self.compensated:
            k_res_slice = self._k_res[: batch * n_heads * seq_len * self._rv_size]
            v_res_slice = self._v_res[: batch * n_heads * seq_len * self._rv_size]
            return _C.compensated_attention(
                Q, k_bin_slice, k_res_slice, v_bin_slice, v_res_slice,
                batch, n_heads, seq_len, head_dim,
            )
        else:
            return _C.binary_attention(
                Q, k_bin_slice, v_bin_slice,
                batch, n_heads, seq_len, head_dim,
            )

    def reset(self) -> None:
        """Clear the cache and reset ring buffer pointers."""
        self._k_bin.zero_()
        self._v_bin.zero_()
        if self.compensated:
            self._k_res.zero_()
            self._v_res.zero_()
        self._write_pos = 0
        self._filled    = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def seq_len(self) -> int:
        """Number of valid cached positions."""
        return self._filled

    @property
    def memory_bytes(self) -> int:
        """Total GPU memory used by the cache buffers, in bytes."""
        total = self._k_bin.numel() + self._v_bin.numel()
        if self.compensated and self._k_res is not None:
            total += self._k_res.numel() + self._v_res.numel()
        return total

    @property
    def compression_ratio(self) -> float:
        """
        Ratio of fp16 KV memory to binary KV memory.
        fp16 KV = 2 * batch * n_heads * max_seq * head_dim * 2 bytes
        """
        fp16_bytes   = 2 * self.batch * self.n_heads * self.max_seq * self.head_dim * 2
        binary_bytes = 2 * self.batch * self.n_heads * self.max_seq * self._bv_size
        return fp16_bytes / binary_bytes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_kv(self, K: Tensor, V: Tensor) -> None:
        expected = (self.batch, self.n_heads, self.head_dim)
        assert K.shape == torch.Size(expected), f"K shape {tuple(K.shape)} != {expected}"
        assert V.shape == torch.Size(expected), f"V shape {tuple(V.shape)} != {expected}"
        assert K.dtype == torch.float16, "K must be float16"
        assert V.dtype == torch.float16, "V must be float16"
        assert K.is_cuda and V.is_cuda, "K and V must be on CUDA"
        assert K.is_contiguous() and V.is_contiguous(), "K and V must be contiguous"

    def __repr__(self) -> str:
        mode = "compensated" if self.compensated else "binary"
        return (
            f"BitCache(batch={self.batch}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, max_seq={self.max_seq}, "
            f"filled={self._filled}, mode={mode}, "
            f"mem={self.memory_bytes / 1024:.1f}KB)"
        )
