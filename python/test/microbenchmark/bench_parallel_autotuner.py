"""Benchmark for parallel autotuner compilation.

Measures autotuning time across worker counts and overlap modes.
Each case uses a unique scale factor (derived from --key) to force fresh compilations.

Usage:
    python bench_parallel_autotuner.py              # starts at key=1
    python bench_parallel_autotuner.py --key 100    # starts at key=100

Each case uses key*num_cases+i as its scale factor, so consecutive keys never
overlap. Increment --key between runs to force all-new compilations.
"""

import os
import time

import torch

import triton
import triton.language as tl

CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk})
    for bm in [32, 64, 128]
    for bn in [32, 64, 128]
    for bk in [32, 64]
]

CASES = [
    ("workers=1 (sequential)", 1, True),
    ("workers=0 (auto)", 0, True),
    ("workers=2, overlap=off", 2, False),
    ("workers=4, overlap=off", 4, False),
    ("workers=8, overlap=off", 8, False),
    ("workers=12, overlap=off", 12, False),
    ("workers=2, overlap=on", 2, True),
    ("workers=4, overlap=on", 4, True),
    ("workers=8, overlap=on", 8, True),
    ("workers=12, overlap=on", 12, True),
]


def _run_matmul(scale):
    """Define a fresh autotuned matmul kernel, run it, verify correctness, return elapsed time."""

    @triton.autotune(configs=list(CONFIGS), key=["M", "N", "K", "SCALE"])
    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        acc = acc * SCALE
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    M, N, K = 512, 512, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))

    start = time.perf_counter()
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scale,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ref = torch.matmul(a.float(), b.float()).half() * scale
    max_diff = (c - ref).abs().max().item()
    if max_diff > 1.0 * scale:
        raise RuntimeError(f"max_diff={max_diff:.4f} exceeds tolerance for scale={scale}")

    return elapsed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark parallel autotuner")
    parser.add_argument(
        "--key", type=int, default=1, help="Starting scale factor (1-1000). Each case gets "
        "a unique scale derived from key * num_cases + i, so consecutive keys never "
        "overlap. Increment between runs for fresh compilations. (default: 1)")
    args = parser.parse_args()

    if args.key < 1 or args.key > 1000:
        parser.error("--key must be between 1 and 1000")

    print(f"Parallel autotuner benchmark - {len(CONFIGS)} configs, key={args.key}")
    print()
    print(f"  {'Case':30s}  {'Time':>7s}  {'vs seq':>7s}")
    print(f"  {'-'*48}")

    baseline = None
    for i, (label, workers, overlap) in enumerate(CASES):
        scale = args.key * len(CASES) + i
        os.environ["TRITON_AUTOTUNING_COMPILE_WORKERS"] = str(workers)
        os.environ["TRITON_AUTOTUNING_OVERLAP_BENCH"] = "1" if overlap else "0"
        try:
            elapsed = _run_matmul(scale)
        except Exception as e:
            print(f"  {label:30s}  FAILED (scale={scale})")
            for line in str(e).split("\n")[-3:]:
                print(f"    {line}")
            continue
        if baseline is None:
            baseline = elapsed
        speedup = baseline / elapsed if elapsed > 0 else float("inf")
        print(f"  {label:30s}  {elapsed:6.2f}s  {speedup:5.2f}x")


if __name__ == "__main__":
    main()
