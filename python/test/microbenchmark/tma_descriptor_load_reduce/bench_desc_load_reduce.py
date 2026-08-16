#!/usr/bin/env python3
import argparse
import json
import platform
import statistics
import time

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def desc_load_reduce_kernel(
    x_desc,
    out_ptr,
    stride_out_0,
    stride_out_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BLOCKS_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // NUM_BLOCKS_N
    pid_n = pid % NUM_BLOCKS_N

    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Descriptor/TMA load -> [1, BLOCK_M, BLOCK_N].
    x = x_desc.load([0, off_m, off_n])
    x = x.to(tl.float32)

    # Collapse BLOCK_M rows and leave BLOCK_N partial-reduction results.
    x_max = tl.max(x, axis=1)

    offs_n = off_n + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + pid_m * stride_out_0 + offs_n * stride_out_1
    tl.store(out_ptrs, tl.reshape(x_max, [BLOCK_N]))


def parse_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def check_output(x, out, block_m: int, block_n: int):
    num_blocks_m = x.shape[1] // block_m
    num_blocks_n = x.shape[2] // block_n
    tiles = {
        (0, 0),
        (num_blocks_m // 2, num_blocks_n // 2),
        (num_blocks_m - 1, num_blocks_n - 1),
    }
    for tile_m, tile_n in tiles:
        off_m = tile_m * block_m
        off_n = tile_n * block_n
        ref = x[0, off_m : off_m + block_m, off_n : off_n + block_n].to(torch.float32).amax(dim=0)
        actual = out[tile_m, off_n : off_n + block_n]
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)


def bench_case(args, block_m: int, block_n: int, dtype_name: str, num_warps: int):
    dtype = parse_dtype(dtype_name)
    m = args.m
    n = args.n

    if m % block_m != 0 or n % block_n != 0:
        return {
            "status": "skipped",
            "reason": "M/N not divisible by BLOCK_M/BLOCK_N",
            "M": m,
            "N": n,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "dtype": dtype_name,
            "num_warps": num_warps,
        }

    torch.manual_seed(args.seed)
    x = torch.randn(1, m, n, dtype=dtype, device="cuda")
    out = torch.empty(m // block_m, n, dtype=torch.float32, device="cuda")
    x_desc = TensorDescriptor(x, list(x.shape), list(x.stride()), [1, block_m, block_n])

    grid = ((m // block_m) * (n // block_n),)
    num_blocks_n = n // block_n

    def fn():
        desc_load_reduce_kernel[grid](
            x_desc,
            out,
            out.stride(0),
            out.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            NUM_BLOCKS_N=num_blocks_n,
            num_warps=num_warps,
        )

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()

    if args.check:
        check_output(x, out, block_m, block_n)

    samples = [float(triton.testing.do_bench_cudagraph(fn)) for _ in range(args.repeat)]
    median_ms = statistics.median(samples)
    bytes_moved = x.numel() * x.element_size() + out.numel() * out.element_size()
    bw_gbs = bytes_moved / 1e9 / (median_ms / 1e3)

    return {
        "status": "ok",
        "triton": triton.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "python": platform.python_version(),
        "M": m,
        "N": n,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "dtype": dtype_name,
        "num_warps": num_warps,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "check": args.check,
        "median_ms": median_ms,
        "samples_ms": samples,
        "bw_gbs": bw_gbs,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark descriptor load followed by a partial max reduction")
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--shapes", default="32x128")
    parser.add_argument("--dtypes", default="bf16")
    parser.add_argument("--warps", default="4")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    print(
        json.dumps(
            {
                "event": "env",
                "triton": triton.__version__,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "capability": torch.cuda.get_device_capability(0),
                "M": args.m,
                "N": args.n,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "check": args.check,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    shapes = []
    for item in args.shapes.split(","):
        block_m, block_n = item.lower().split("x", 1)
        shapes.append((int(block_m), int(block_n)))

    dtypes = [item.strip() for item in args.dtypes.split(",") if item.strip()]
    warps = [int(item.strip()) for item in args.warps.split(",") if item.strip()]

    started = time.time()
    for block_m, block_n in shapes:
        for dtype_name in dtypes:
            for num_warps in warps:
                result = bench_case(args, block_m, block_n, dtype_name, num_warps)
                result["elapsed_s"] = round(time.time() - started, 3)
                print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
