"""
Autoregressive generation demo with online binary encoding, adaptive precision
switching, and quality tracking.

Simulates 2048 decode steps (one token at a time), printing:
  - Per-step cosine similarity of BitCache output vs full-precision oracle
  - Running precision mode split (binary vs compensated heads)
  - Cumulative memory savings vs fp16 KV cache

Run with: python examples/generation_demo.py
Requires the bitcache C extension (pip install -e . or build_ext --inplace).
"""

import math
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from bitcache import BitCache

# ---- model config ----
BATCH    = 1
N_HEADS  = 8
HEAD_DIM = 128
GEN_LEN  = 2048

# Adaptive precision: entropy threshold.
# Calibrated from first 128 tokens; updated periodically.
# log(seq_len) * 0.5 is a reasonable heuristic.
CALIB_STEPS = 128
ENTROPY_SCALE = 0.5


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def reference_sdpa(
    Q: torch.Tensor,
    K_hist: list[torch.Tensor],
    V_hist: list[torch.Tensor],
) -> torch.Tensor:
    """Full-precision sdpa over accumulated history."""
    K_full = torch.stack(K_hist, dim=2).float()
    V_full = torch.stack(V_hist, dim=2).float()
    scale  = 1.0 / math.sqrt(HEAD_DIM)
    return F.scaled_dot_product_attention(
        Q.float().unsqueeze(2), K_full, V_full, scale=scale
    ).squeeze(2).half()


def compute_entropy_cpu(
    Q: torch.Tensor,
    K_hist: list[torch.Tensor],
) -> torch.Tensor:
    """
    Approximate per-head entropy from full-precision scores.
    Returns [n_heads] float32 on CPU.

    Used during calibration (no C extension dependency).
    """
    seq_len = len(K_hist)
    K_full  = torch.stack(K_hist, dim=2).float()  # [1, n_heads, seq_len, head_dim]
    Q_fp    = Q.float().unsqueeze(-1)              # [1, n_heads, head_dim, 1]
    scale   = 1.0 / math.sqrt(HEAD_DIM)
    scores  = (K_full @ Q_fp).squeeze(-1) * scale  # [1, n_heads, seq_len]
    weights = torch.softmax(scores, dim=-1)         # [1, n_heads, seq_len]
    log_w   = torch.log(weights.clamp(min=1e-12))
    entropy = -(weights * log_w).sum(dim=-1)        # [1, n_heads]
    return entropy.squeeze(0).cpu()                  # [n_heads]


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        return

    device = "cuda"
    print(f"Autoregressive generation demo on {torch.cuda.get_device_name(0)}")
    print(f"config: batch={BATCH}, n_heads={N_HEADS}, head_dim={HEAD_DIM}, "
          f"gen_len={GEN_LEN}\n")

    torch.manual_seed(1337)

    cache = BitCache(
        batch=BATCH, n_heads=N_HEADS, head_dim=HEAD_DIM,
        max_seq=GEN_LEN + 32,
        device=device,
        compensated=True,
    )

    K_hist: list[torch.Tensor] = []
    V_hist: list[torch.Tensor] = []

    # Adaptive precision state (per head)
    entropy_threshold = math.log(CALIB_STEPS) * ENTROPY_SCALE
    head_is_binary    = [False] * N_HEADS  # updated after calibration
    compensated_steps = 0
    binary_steps      = 0

    print(f"{'step':>6}  {'cos_sim':>8}  {'mode':>12}  "
          f"{'mem_saved_MB':>14}  {'latency_ms':>12}")
    print("-" * 65)

    total_latency_ms = 0.0
    log_steps = set(range(0, GEN_LEN, 128)) | {GEN_LEN - 1}

    for step in range(GEN_LEN):
        K_tok = torch.randn(BATCH, N_HEADS, HEAD_DIM, device=device, dtype=torch.float16) * 0.3
        V_tok = torch.randn(BATCH, N_HEADS, HEAD_DIM, device=device, dtype=torch.float16) * 0.3
        K_hist.append(K_tok)
        V_hist.append(V_tok)

        t0 = time.perf_counter()
        cache.append(K_tok.contiguous(), V_tok.contiguous())
        seq_len = cache.seq_len

        Q = torch.randn(BATCH, N_HEADS, HEAD_DIM, device=device, dtype=torch.float16) * 0.3
        bit_out = cache.attend(Q.contiguous())
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_latency_ms += elapsed_ms

        # Periodically update the adaptive precision threshold from observed entropy
        if step > 0 and step % CALIB_STEPS == 0:
            entropies = compute_entropy_cpu(Q, K_hist)
            median_entropy = entropies.median().item()
            entropy_threshold = median_entropy  # use median as threshold
            head_is_binary = [entropies[h].item() >= entropy_threshold for h in range(N_HEADS)]

        n_binary_heads = sum(head_is_binary)

        if n_binary_heads > N_HEADS // 2:
            binary_steps += 1
        else:
            compensated_steps += 1

        if step in log_steps:
            ref_out    = reference_sdpa(Q, K_hist, V_hist)
            cos        = cosine_sim(ref_out, bit_out)
            fp16_bytes = 2 * BATCH * N_HEADS * seq_len * HEAD_DIM * 2
            saved_mb   = (fp16_bytes - cache.memory_bytes) / (1024 ** 2)
            mode_str   = f"{n_binary_heads}B/{N_HEADS - n_binary_heads}C"

            print(f"{step:6d}  {cos:8.4f}  {mode_str:>12}  {saved_mb:14.2f}  {elapsed_ms:12.3f}ms")

    print("\n" + "=" * 65)
    print(f"Generation complete: {GEN_LEN} steps")
    print(f"  Total latency:   {total_latency_ms:.1f}ms  "
          f"({total_latency_ms / GEN_LEN:.2f}ms/step)")
    print(f"  Precision split: {binary_steps} steps binary, "
          f"{compensated_steps} steps compensated")
    fp16_peak_mb = 2 * BATCH * N_HEADS * GEN_LEN * HEAD_DIM * 2 / (1024 ** 2)
    cache_mb     = cache.memory_bytes / (1024 ** 2)
    print(f"  Memory: fp16 KV would be {fp16_peak_mb:.1f}MB, "
          f"BitCache uses {cache_mb:.1f}MB "
          f"({fp16_peak_mb / max(cache_mb, 1e-6):.2f}x compression)")


if __name__ == "__main__":
    main()
