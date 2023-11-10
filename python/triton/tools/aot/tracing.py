from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction

from .compiler import AOTCompilationResult
from .linker import AOTLinkerResult


# Tracing Tools #
@dataclass
class AOTTraceResult:
    kernel_name: str
    # Path to compiled kernels
    kernel_path: str | Path
    # JITFunction used to compile the kernel
    jit_fn: JITFunction
    # Args / Kwargs passed to `triton.compiler.compiler.compile`
    jit_args: dict
    compilation_result: AOTCompilationResult
    linker_result: AOTLinkerResult


@dataclass
class TraceGridConfig(dict):
    """Purpose of this class is to enable dynamic grids during tracing

    In the typical JIT workflow, the grid is set at compile time, and the kernel is specialized for that grid.
    However, when tracing, we want to be able to trace the kernel for different grids.
    This class allows us to do that by specifying a grid for the JIT compiler and a grid for the tracer.

    The jit grid is the same as the grid used in the typical JIT workflow.
    The trace grid is the grid used for tracing and is a 3-tuple of strings that can reference args in the kernel.
    E.g., for a matrix multiplication kernel with matrix dims `M, N, K`, the trace grid could be `M / 16, N / 16, 1`.

    """

    jit_grid: Union[callable, tuple[int, int, int]]
    trace_grid: tuple[str, str, str]

    def __post_init__(self):
        self.update(self.__dict__)


@dataclass
class TraceConfig(dict):
    """Kwargs passed to `JITFunction.run`"""

    # Launch params
    num_warps: Optional[int] = None
    num_stages: Optional[int] = None
    num_ctas: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = True

    # JIT options
    do_not_specialize: List[str] = None
    debug: bool = False
    noinline: bool = False

    # Additional options
    extern_libs: List[str] = None
    stream: Optional[int] = None
    warmup: bool = False
    device: Optional[int] = None
    device_type: Optional[str] = None
    # Trace options
    trace: bool = True
    trace_dir: Optional[Path] = None
    trace_grid: TraceGridConfig = None

    def __post_init__(self):
        self.update(self.__dict__)


# Kernel-specific configs


@dataclass
class MatMulConfig(dict):
    dtype_in: torch.dtype = torch.float16
    dtype_out: torch.dtype = torch.float32
    M: int = 16
    N: int = 16
    K: int = 16
    BLOCK_M: tl.constexpr = 16
    BLOCK_N: tl.constexpr = 16
    BLOCK_K: tl.constexpr = 16
    seed: torch.seed = 0

    def __post_init__(self):
        self.update(self.__dict__)


class KernelTracer(ABC):
    KERNEL: str

    @abstractmethod
    def build_args(self, config):
        """Set args for the kernel as an OrderedDict"""
        ...

    @abstractmethod
    def build_constants(self, config):
        """Set constants for the kernel as a dict"""
        ...

    @abstractmethod
    def build_grid(self, config):
        """Set grid for the kernel as a callable or a 3-tuple of ints"""
        ...

    def check_specializations(self, params, expected_specializations):
        no_specs = [p.name for p in params if p.do_not_specialize]

        assert set(no_specs) == set(
            expected_specializations
        ), f"Incorrect specializations, expected {expected_specializations}"

    def check_args(self, args, expected_args):
        assert len(args) == len(
            expected_args
        ), f"Incorrect number of args, expected {expected_args}"
        for i, (expected_key, actual_key) in enumerate(zip(expected_args, args.keys())):
            assert (
                expected_key == actual_key
            ), f"Incorrect arg name at position {i}, expected {expected_key}, got {actual_key}"

    def check_constants(self, kernel_constants, expected_constants):
        assert set(kernel_constants.keys()) == set(
            expected_constants
        ), f"Incorrect constants, expected {expected_constants}"

    def trace(self, kernel_config, trace_config: TraceConfig):
        """Trace a kernel with the given args and constants

        Args:
            additional_jit_kwargs: number of warps, specializations, etc. -- see `triton.runtime.jit.JITFunction.run` special args.
        """

        do_not_specialize = trace_config.pop("do_not_specialize")
        debug = trace_config.pop("debug")
        noinline = trace_config.pop("noinline")

        jitted_fn = JITFunction(
            self.kernel,
            do_not_specialize=do_not_specialize,
            debug=debug,
            noinline=noinline,
        )
        if do_not_specialize:
            self.check_specializations(jitted_fn.params, do_not_specialize)

        expected_constants = [p.name for p in jitted_fn.params if p.is_constexpr]
        expected_args = [p.name for p in jitted_fn.params if not p.is_constexpr]
        # Check do not specialize

        args = self.build_args(kernel_config)
        self.check_args(args, expected_args)

        constants = self.build_constants(kernel_config)
        self.check_constants(constants, expected_constants)

        grid = self.build_grid(kernel_config)

        trace_artifact: AOTCompilationResult = jitted_fn[grid](
            *args.values(),
            **constants,
            **trace_config,
        )
        return trace_artifact


class MatMulKernelTracer(KernelTracer):
    KERNEL = "matmul_kernel.py"

    def __init__(self, kernel_dir):
        self.kernel_dir = Path(kernel_dir).absolute()
        self.kernel = (self.kernel_dir / self.KERNEL).absolute()

    def trace(
        self, kernel_configs: List[MatMulConfig], trace_configs: List[TraceConfig]
    ):
        outputs = []
        checks = []
        traces = []
        for kconfig, tconfig in zip(kernel_configs, trace_configs):
            trace = super().trace(kconfig, tconfig)
            traces.append(trace)
            checks.append(self.CHECK)
            outputs.append(self.C.detach().cpu())
        return traces, outputs, checks

    def build_args(self, config: MatMulConfig):
        # Set up matrices
        torch.manual_seed(config.seed)

        A_shape = config.M, config.K
        B_shape = config.K, config.N
        C_shape = config.M, config.N
        A = torch.randn(A_shape, dtype=config.dtype_in, device="cuda").contiguous()
        B = torch.randn(B_shape, dtype=config.dtype_in, device="cuda").contiguous()
        C = torch.empty(C_shape, dtype=config.dtype_out, device="cuda")

        # Save for verification
        self.CHECK = torch.matmul(A, B).to(config.dtype_out).detach().cpu()
        self.C = C
        M, K = A.shape
        _, N = B.shape

        stride_cm = C.stride(0)
        stride_cn = C.stride(1)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)

        args = OrderedDict(
            C=C,
            A=A,
            B=B,
            M=M,
            N=N,
            K=K,
            stride_cm=stride_cm,
            stride_cn=stride_cn,
            stride_am=stride_am,
            stride_ak=stride_ak,
            stride_bk=stride_bk,
            stride_bn=stride_bn,
        )

        return args

    def build_constants(self, config: MatMulConfig):
        return {
            "BLOCK_M": config.BLOCK_M,
            "BLOCK_N": config.BLOCK_N,
            "BLOCK_K": config.BLOCK_K,
        }

    def build_grid(self, config: MatMulConfig) -> TraceGridConfig:
        jit_grid = lambda META: (
            (
                triton.cdiv(config.M, META["BLOCK_M"]),
                triton.cdiv(config.N, META["BLOCK_N"]),
                1,
            )
        )
        trace_grid = (f"M / {config.BLOCK_M}", f"N / {config.BLOCK_N}", "1")
        return TraceGridConfig(jit_grid, trace_grid)
