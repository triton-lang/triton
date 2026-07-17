"""Benchmark: GQA Flash-Attention vs. PyTorch SDPA.

Requires a CUDA GPU (performance numbers are meaningless under the CPU
interpreter). Reports TFLOP/s for the forward pass across sequence lengths.

    python bench_flash_attn.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flash_attn import flash_attention  # noqa: E402

try:
    import triton
except ImportError:
    triton = None


def flops(Z, H, N, D, causal):
    # 2 matmuls (QK, PV), each 2*N*N*D, times Z*H; halve for causal
    f = 2 * 2.0 * Z * H * N * N * D
    return f * 0.5 if causal else f


def run():
    assert torch.cuda.is_available(), "benchmark needs a CUDA GPU"
    Z, H, H_KV, D = 4, 32, 8, 128  # GQA 4:1, Llama-3-8B-ish head config
    causal = True
    dtype = torch.float16
    print(f"{'N_CTX':>8} {'flash TFLOP/s':>14} {'sdpa TFLOP/s':>14} {'speedup':>8}")
    for N in [512, 1024, 2048, 4096, 8192]:
        q = torch.randn(Z, H, N, D, device="cuda", dtype=dtype)
        k = torch.randn(Z, H_KV, N, D, device="cuda", dtype=dtype)
        v = torch.randn(Z, H_KV, N, D, device="cuda", dtype=dtype)
        scale = 1.0 / math.sqrt(D)

        def flash():
            return flash_attention(q, k, v, causal=causal, sm_scale=scale, block_m=128, block_n=64)

        kk = k.repeat_interleave(H // H_KV, dim=1)
        vv = v.repeat_interleave(H // H_KV, dim=1)

        def sdpa():
            return torch.nn.functional.scaled_dot_product_attention(q, kk, vv, is_causal=causal, scale=scale)

        tf = triton.testing.do_bench(flash)
        ts = triton.testing.do_bench(sdpa)
        fl = flops(Z, H, N, D, causal) / 1e12
        print(f"{N:>8} {fl / (tf * 1e-3):>14.1f} {fl / (ts * 1e-3):>14.1f} "
              f"{ts / tf:>7.2f}x")


if __name__ == "__main__":
    run()
