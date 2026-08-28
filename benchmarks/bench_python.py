"""
Python latency benchmark: BitCache.attend() vs torch sdpa across seq_len 1K-64K.

Reports wall-clock time and speedup per seq_len, plus effective compression ratio.
Run with: python benchmarks/bench_python.py
"""

import math
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from bitcache import BitCache

WARMUP  = 10
ITERS   = 50
BATCH   = 1
N_HEADS = 8
HEAD_DIM = 128


def time_fn(fn, warmup: int, iters: int) -> float:
    """Returns median wall-clock time in milliseconds over iters runs."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)

    times.sort()
    return times[len(times) // 2]


def bench_seq_len(seq_len: int, device: str) -> None:
    torch.manual_seed(0)

    Q = torch.randn(BATCH, N_HEADS, HEAD_DIM, device=device, dtype=torch.float16) * 0.5
    K = torch.randn(BATCH, N_HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float16) * 0.5
    V = torch.randn(BATCH, N_HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float16) * 0.5

    # Build BitCache from pre-populated tensors by appending token by token
    # For the benchmark we pre-build the cache once, then time only attend()
    cache = BitCache(
        batch=BATCH, n_heads=N_HEADS, head_dim=HEAD_DIM,
        max_seq=seq_len + 8, device=device, compensated=True,
    )
    for step in range(seq_len):
        cache.append(
            K[:, :, step, :].contiguous(),
            V[:, :, step, :].contiguous(),
        )

    Q_cont = Q.contiguous()

    # Reference: torch sdpa (full fp16)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    K_fp  = K.float()
    V_fp  = V.float()

    def sdpa_fn():
        return F.scaled_dot_product_attention(
            Q.float().unsqueeze(2), K_fp, V_fp, scale=scale
        )

    def bitcache_fn():
        return cache.attend(Q_cont)

    ms_sdpa     = time_fn(sdpa_fn,     WARMUP, ITERS)
    ms_bitcache = time_fn(bitcache_fn, WARMUP, ITERS)

    speedup = ms_sdpa / ms_bitcache
    fp16_bytes   = 2 * BATCH * N_HEADS * seq_len * HEAD_DIM * 2  # K and V
    binary_bytes = cache.memory_bytes
    compression  = fp16_bytes / binary_bytes

    print(
        f"  seq={seq_len:6d}  sdpa={ms_sdpa:7.3f}ms  "
        f"bitcache={ms_bitcache:7.3f}ms  "
        f"speedup={speedup:5.2f}x  "
        f"compression={compression:.2f}x"
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        return

    device = "cuda"
    name   = torch.cuda.get_device_name(0)
    print(f"Latency benchmark on {name}")
    print(f"batch={BATCH}, n_heads={N_HEADS}, head_dim={HEAD_DIM}, "
          f"warmup={WARMUP}, iters={ITERS}\n")
    print(f"  {'seq_len':>8}  {'sdpa':>12}  {'bitcache':>12}  {'speedup':>9}  compression")
    print("  " + "-" * 65)

    seq_lens = [1024, 2048, 4096, 8192, 16384, 32768, 64000]
    for seq_len in seq_lens:
        try:
            bench_seq_len(seq_len, device)
        except torch.cuda.OutOfMemoryError:
            print(f"  seq={seq_len:6d}  OOM (full-precision reference)")
            break


if __name__ == "__main__":
    main()
