import logging
import os
from typing import Callable, Dict, Tuple

import pandas as pd
import torch
import triton
import triton.language as tl
from packaging import version
from tqdm import tqdm

try:
    import coloredlogs

    coloredlogs.install(level="INFO")
except:
    pass

if version.parse(triton.__version__) < version.parse("3.0"):
    # 989adb9a29496c22a36ef82ca69cad5dad536b9c
    from triton_21.jit import jit
else:
    from triton_30.jit import jit

IS_TRITON_VERSION_OLD = version.parse(triton.__version__) < version.parse("3.0")
TRITON_MAJOR = version.parse(triton.__version__).major

triton.jit = jit
triton.runtime.jit.jit = jit

logger = logging.getLogger(__name__)

pd.options.display.float_format = "{:,.2f}".format


def _get_kernels(backward=False):
    # y = a * x
    @triton.jit
    def fwd_kernel_short(
        x_ptr,
        a,
        output_ptr,
        N,
        M_STRIDE,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        x_ptr += pid * M_STRIDE
        output_ptr += pid * M_STRIDE
        offsets = tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x * a, mask=mask)

    @triton.jit
    def fwd_kernel_long(
        x_ptr,
        a,
        b,
        c,
        d,
        e,
        f,
        g,
        h,
        output_ptr,
        N,
        M_STRIDE,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        mul = a * b * c * d * e * f * g * h
        # fwd_kernel_short(x_ptr, mul, output_ptr, N, M_STRIDE, BLOCK_SIZE_N)
        pid = tl.program_id(axis=0)
        x_ptr += pid * M_STRIDE
        output_ptr += pid * M_STRIDE
        offsets = tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x * mul, mask=mask)

    if not backward:
        return (
            fwd_kernel_short,
            fwd_kernel_long,
        )

    # dx = dy / a
    @triton.jit
    def bwd_kernel_short(
        dy_ptr,
        a,
        output_ptr,
        N,
        M_STRIDE,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        dy_ptr += pid * M_STRIDE
        output_ptr += pid * M_STRIDE
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        dy = tl.load(dy_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, dy / a, mask=mask)

    @triton.jit
    def bwd_kernel_long(
        dy_ptr,
        a,
        b,
        c,
        d,
        e,
        f,
        g,
        h,
        output_ptr,
        N,
        M_STRIDE,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        dy_ptr += pid * M_STRIDE
        output_ptr += pid * M_STRIDE
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        mul = a * b * c * d * e * f * g * h
        dy = tl.load(dy_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, dy / mul, mask=mask)

    return (
        fwd_kernel_short,
        bwd_kernel_short,
        fwd_kernel_long,
        bwd_kernel_long,
    )


def get_kernels(backward=False) -> Dict[str, Tuple[triton.runtime.JITFunction, ...]]:
    """Build kernels with default JITFuntion, optimized python JITFunction
    and optimized CPP JITFunction"""

    kernels = dict()

    os.environ["TRITON_EXPERIMENTAL_JIT_FUNCTION_PYTHON"] = "0"
    os.environ["TRITON_EXPERIMENTAL_JIT_FUNCTION_CPP"] = "0"
    kernels["default"] = _get_kernels(backward=backward)

    if TRITON_MAJOR < 3:
        from triton_21.low_latency_jit_python_generated import (
            LowLatencyJITFunctionPythonGeneratedShort,
            LowLatencyJITFunctionPythonGeneratedLong,
        )
        _tmp = _get_kernels(backward=backward)
        kernels["python_generated"] = (
            LowLatencyJITFunctionPythonGeneratedShort(_tmp[0]),
            LowLatencyJITFunctionPythonGeneratedLong(_tmp[1]),
        )
        from triton_21.low_latency_jit_python_generated_with_type import (
            LowLatencyJITFunctionPythonGeneratedWithTypeShort,
            LowLatencyJITFunctionPythonGeneratedWithTypeLong,
        )
        kernels["python_generated_with_type"] = (
            LowLatencyJITFunctionPythonGeneratedWithTypeShort(_tmp[0]),
            LowLatencyJITFunctionPythonGeneratedWithTypeLong(_tmp[1]),
        )

        os.environ["TRITON_EXPERIMENTAL_JIT_FUNCTION_PYTHON"] = "1"
        kernels["python"] = _get_kernels(backward=backward)
        os.environ["TRITON_EXPERIMENTAL_JIT_FUNCTION_PYTHON"] = "0"

        # os.environ["TRITON_EXPERIMENTAL_JIT_FUNCTION_CPP"] = "1"
        # kernels["cpp"] = _get_kernels(backward=backward)
        # os.environ["TRITON_EXPERIMENTAL_JIT_FUNCTION_CPP"] = "0"
    else:
        os.environ["TRITON_OLD_JIT_FUNCTION"] = "1"
        kernels["old"] = _get_kernels(backward=backward)
        os.environ["TRITON_OLD_JIT_FUNCTION"] = "0"

    return kernels


def do_bench(f, rep, warmup) -> float:
    """Run time in milliseconds"""

    torch.cuda.synchronize()
    for _ in range(warmup):
        f()
    being = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    being.record()
    for _ in range(rep):
        f()
    end.record()
    torch.cuda.synchronize()
    return being.elapsed_time(end) / rep


def run_benchmarks(
    benchmark: Callable[[], Dict[str, float]], num_runs: int, name: str
) -> pd.DataFrame:
    res = []
    for _ in tqdm(range(num_runs)):
        res.append(benchmark())

    df = pd.DataFrame(res)
    df *= 1000

    df.to_csv(f"{name}_triton_{TRITON_MAJOR}.csv", index=False)
    print(df.mean().T)

    return df
