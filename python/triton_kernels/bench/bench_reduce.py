import argparse
import statistics

import torch

from triton_kernels.reduce import reduce, _select_reduce_forward_config


def _csv_ints(s):
    return [int(x) for x in s.split(",") if x]


def _flush_cache(cache_killer):
    if cache_killer is not None:
        cache_killer.add_(1.0)


def bench_reduce(k, s0, s1, iters, cache_killer):
    x = torch.randn((k, s0, s1), device="cuda", dtype=torch.float32)
    for _ in range(10):
        _flush_cache(cache_killer)
        reduce(x, dim=0, y_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    times_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        _flush_cache(cache_killer)
        start.record()
        reduce(x, dim=0, y_dtype=torch.bfloat16)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    times_ms.sort()
    return statistics.median(times_ms), statistics.mean(times_ms), times_ms[int(0.9 * (iters - 1))]


def main():
    parser = argparse.ArgumentParser(description="Benchmark wide-S1 reduce_forward shapes.")
    parser.add_argument("--ks", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--s0s", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--s1s", default="1024,2048,4096,8192,16384,32768")
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument(
        "--flush-mb",
        type=int,
        default=512,
        help="Touch this many MiB before each measured reduce. Set to 0 to benchmark hot-cache repeats.",
    )
    args = parser.parse_args()

    cache_killer = None
    if args.flush_mb > 0:
        n_elements = args.flush_mb * 1024 * 1024 // torch.empty((), dtype=torch.float32).element_size()
        cache_killer = torch.empty(n_elements, device="cuda", dtype=torch.float32)
        cache_killer.zero_()

    print("K,S0,Y_S1,BLOCK_S0,BLOCK_S1,median_ms,mean_ms,p90_ms", flush=True)
    for s1 in _csv_ints(args.s1s):
        for k in _csv_ints(args.ks):
            for s0 in _csv_ints(args.s0s):
                opt_flags = _select_reduce_forward_config(s0, s1, 1, k, False)
                median_ms, mean_ms, p90_ms = bench_reduce(k, s0, s1, args.iters, cache_killer)
                print(
                    f"{k},{s0},{s1},{opt_flags.block_s0},{opt_flags.block_x_s1},{median_ms:.6f},{mean_ms:.6f},{p90_ms:.6f}",
                    flush=True)


if __name__ == "__main__":
    main()
