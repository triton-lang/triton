"""Cost of specializing kernel arguments, which is paid on every launch.

Reports two numbers:

  * the per-argument cost of `native_specialize_impl`, isolated from everything
    else a launch does;
  * launch wall time for a trivial kernel taking 1, 4 and 8 tensor arguments,
    whose slope is the marginal cost of one more argument.

The slope is the number to watch: the fixed part of a launch is dominated by the
driver's own submission cost, which nothing here can change.
"""

import argparse
import statistics
import timeit

import torch
import triton
from triton._C.libtriton import native_specialize_impl
from triton.runtime import driver


@triton.jit
def nop_1(a0):
    pass


@triton.jit
def nop_4(a0, a1, a2, a3):
    pass


@triton.jit
def nop_8(a0, a1, a2, a3, a4, a5, a6, a7):
    pass


KERNELS = {1: nop_1, 4: nop_4, 8: nop_8}


def bench_ns(fn, number, repeat):
    """Median of `repeat` batches, minus the cost of calling an empty lambda.

    Median rather than mean because the noise is one-sided -- a descheduled
    batch can only read high -- and a single outlier moves the mean by several
    percent. Median rather than minimum because the minimum drifts downwards as
    `repeat` grows, which would make runs with different `repeat` values
    incomparable.
    """
    overhead = statistics.median(timeit.repeat(lambda: None, number=number, repeat=repeat)) / number
    elapsed = statistics.median(timeit.repeat(fn, number=number, repeat=repeat)) / number
    return (elapsed - overhead) * 1e9


def main(quick: bool):
    number = 20000 if quick else 200000
    repeat = 3 if quick else 7

    backend = triton.compiler.compiler.make_backend(driver.active.get_current_target())
    tensor = torch.empty(1024, dtype=torch.float32)
    mode = backend.supports_native_tensor_specialization
    per_arg = bench_ns(lambda: native_specialize_impl(backend, tensor, False, True, True), number, repeat)
    print(f"specialize one tensor argument ({type(backend).__name__}, mode {mode}): {per_arg:.1f} ns")

    launch_us = {}
    for nargs, kernel in KERNELS.items():
        args = [torch.empty(1024, dtype=torch.float32, device="cuda") for _ in range(nargs)]
        kernel[(1, )](*args)
        torch.cuda.synchronize()
        launch = lambda k=kernel, a=args: k[(1, )](*a)
        launch_us[nargs] = bench_ns(launch, number // 10, repeat) / 1e3
        torch.cuda.synchronize()
        print(f"launch with {nargs} tensor argument(s): {launch_us[nargs]:.3f} us")

    lo, hi = min(KERNELS), max(KERNELS)
    marginal = (launch_us[hi] - launch_us[lo]) / (hi - lo)
    print(f"marginal cost per tensor argument: {marginal:.3f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer iterations, for use as a pre-commit check")
    args = parser.parse_args()
    main(args.quick)
