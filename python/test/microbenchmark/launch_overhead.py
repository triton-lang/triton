"""
Original code by @bertmaher; profiling added by @apgoucher
"""

import cProfile
import pstats
import time

import numpy as np
import torch

import triton
import triton.language as tl


@triton.jit
def nop_args(
    t1,
    t2,
    t3,
    t4,
    t5,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    i8,
    i9,
    c1: tl.constexpr,
    c2: tl.constexpr,
    c3: tl.constexpr,
    c4: tl.constexpr,
    c5: tl.constexpr,
):
    pass


def do_bench_walltime(fn):
    print("Compiling...")
    fn()
    torch.cuda.synchronize()

    for _ in range(1000):
        fn()
    torch.cuda.synchronize()

    n_repeat = 10000

    mses = []

    for _ in range(25):
        print("Running %d benchmarking iterations..." % n_repeat)
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(n_repeat):
            fn()
        torch.cuda.synchronize()
        end_time = time.time()
        wall_time_ms = (end_time - start_time) * 1e3 / n_repeat
        mses.append(wall_time_ms)

    mses = np.array(mses)

    print("Running profiler...")
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(n_repeat):
        fn()
    torch.cuda.synchronize()
    profile.disable()
    stats = pstats.Stats(profile)
    stats.sort_stats("time")
    stats.print_stats()
    return mses


def main():
    targs = [torch.zeros(1, device="cuda") for _ in range(5)]
    iargs = [1 for _ in range(9)]
    cargs = [32 for _ in range(5)]

    usecs = do_bench_walltime(lambda: nop_args[
        1,
    ](*targs, *iargs, *cargs)) * 1000.0

    print(usecs)
    print(sorted(usecs)[len(usecs) >> 1])


if __name__ == "__main__":
    main()
