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
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def nop_args(
    t1,
    t2,
    t3,
    t4,
    t5,
    nc1,
    nc2,
    nc3,
    nc4,
    nc5,
    nc6,
    nc7,
    nc8,
    nc9,
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


def main(use_tensor_desc: bool):
    if use_tensor_desc:
        targs = [TensorDescriptor.from_tensor(torch.zeros(1, 16, device="cuda"), block_shape=[1, 16]) for _ in range(5)]
    else:
        targs = [torch.zeros(1, device="cuda") for _ in range(5)]
    ncargs = [0, 1, 1024, 2**31 - 1, 2**64 - 1, False, True, None, (16, 16)]
    cargs = [32, False, True, 0, 64]

    usecs = do_bench_walltime(lambda: nop_args[
        1,
    ](*targs, *ncargs, *cargs)) * 1000.0

    print(usecs)
    print(sorted(usecs)[len(usecs) >> 1])


if __name__ == "__main__":
    print("launch overhead of kernel with Tensor inputs")
    main(use_tensor_desc=False)
    print("launch overhead of kernel with TensorDescriptor inputs")
    main(use_tensor_desc=True)
