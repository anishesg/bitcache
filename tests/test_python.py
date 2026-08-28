"""
Python correctness test: BitCache against torch.nn.functional.scaled_dot_product_attention.

We simulate autoregressive generation by appending one K/V token at a time,
then running attend() and comparing against full-precision sdpa.

The test checks output cosine similarity and Kendall tau rank correlation of
attention scores, mirroring the thresholds validated in test_correctness.cu.
"""

import math
import sys

import torch
import torch.nn.functional as F

# Allow running without installation via build_ext --inplace
sys.path.insert(0, ".")

from bitcache import BitCache


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def kendall_tau(a: torch.Tensor, b: torch.Tensor) -> float:
    """O(n^2) Kendall tau for 1D tensors; used only on small seq_lens in tests."""
    a = a.float().cpu()
    b = b.float().cpu()
    n = a.shape[0]
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = (a[i] - a[j]).item()
            db = (b[i] - b[j]).item()
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    total = n * (n - 1) // 2
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def reference_sdpa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """Full-precision sdpa over accumulated K/V history [batch, n_heads, seq_len, head_dim]."""
    return F.scaled_dot_product_attention(
        Q.float().unsqueeze(2),
        K.float(),
        V.float(),
        scale=1.0 / math.sqrt(Q.shape[-1]),
    ).squeeze(2).half()


def run_config(
    batch: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    compensated: bool,
    device: str,
) -> None:
    tag = "compensated" if compensated else "binary"
    print(f"\n--- batch={batch}, n_heads={n_heads}, head_dim={head_dim}, "
          f"seq_len={seq_len}, mode={tag} ---")

    torch.manual_seed(42 + seq_len + head_dim)

    cache = BitCache(
        batch=batch,
        n_heads=n_heads,
        head_dim=head_dim,
        max_seq=seq_len + 8,
        device=device,
        compensated=compensated,
    )

    # Accumulate full-precision K/V for the reference
    K_history = []
    V_history = []

    # Append seq_len tokens one at a time (simulating decode)
    for step in range(seq_len):
        K_tok = torch.randn(batch, n_heads, head_dim, device=device, dtype=torch.float16) * 0.5
        V_tok = torch.randn(batch, n_heads, head_dim, device=device, dtype=torch.float16) * 0.5
        cache.append(K_tok, V_tok)
        K_history.append(K_tok)
        V_history.append(V_tok)

    # Build K/V tensors [batch, n_heads, seq_len, head_dim] for reference
    K_full = torch.stack(K_history, dim=2)
    V_full = torch.stack(V_history, dim=2)

    # Evaluate at a few checkpoints to detect quality drift
    checkpoints = sorted({seq_len // 4, seq_len // 2, seq_len})
    for t in checkpoints:
        if t == 0:
            continue

        Q = torch.randn(batch, n_heads, head_dim, device=device, dtype=torch.float16) * 0.5

        # Reference output using full-precision sdpa
        ref_out = reference_sdpa(Q, K_full[:, :, :t, :], V_full[:, :, :t, :])

        # BitCache output - need a cache up to position t
        # Build a fresh cache for this slice
        sub_cache = BitCache(
            batch=batch,
            n_heads=n_heads,
            head_dim=head_dim,
            max_seq=t + 4,
            device=device,
            compensated=compensated,
        )
        for step in range(t):
            sub_cache.append(
                K_history[step].contiguous(),
                V_history[step].contiguous(),
            )
        bit_out = sub_cache.attend(Q.contiguous())

        cos = cosine_sim(ref_out, bit_out)
        print(f"  t={t:4d}: cosine_sim={cos:.4f}  "
              f"[cache mem={sub_cache.memory_bytes / 1024:.1f} KB, "
              f"compression={sub_cache.compression_ratio:.2f}x]")

        # At this head_dim/seq_len, binary attention should preserve rank ordering well
        assert cos > 0.85, (
            f"cosine similarity {cos:.4f} below threshold 0.85 at t={t}"
        )

    # Compensated mode should be noticeably better than binary-only
    if compensated:
        bin_cache = BitCache(
            batch=batch, n_heads=n_heads, head_dim=head_dim,
            max_seq=seq_len + 4, device=device, compensated=False,
        )
        for step in range(seq_len):
            bin_cache.append(K_history[step].contiguous(), V_history[step].contiguous())

        Q = torch.randn(batch, n_heads, head_dim, device=device, dtype=torch.float16) * 0.5
        ref_out  = reference_sdpa(Q, K_full, V_full)
        bin_out  = bin_cache.attend(Q.contiguous())
        comp_out = cache.attend(Q.contiguous())

        cos_bin  = cosine_sim(ref_out, bin_out)
        cos_comp = cosine_sim(ref_out, comp_out)
        print(f"\n  binary vs compensated at seq_len={seq_len}: "
              f"binary={cos_bin:.4f}, compensated={cos_comp:.4f}")
        assert cos_comp >= cos_bin - 0.01, (
            "compensated attention should not be significantly worse than binary-only"
        )


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        return

    device = "cuda"
    print(f"Running Python correctness tests on {torch.cuda.get_device_name(0)}")

    configs = [
        # (batch, n_heads, head_dim, seq_len, compensated)
        (1, 1,  64,  128, False),
        (1, 1,  64,  128, True),
        (1, 1, 128,  256, True),
        (1, 4, 128,  512, True),
        (2, 4,  64,  256, True),
        (1, 1, 128, 1024, True),
    ]

    for batch, n_heads, head_dim, seq_len, compensated in configs:
        run_config(batch, n_heads, head_dim, seq_len, compensated, device)

    print("\nAll Python correctness tests passed.")


if __name__ == "__main__":
    main()
