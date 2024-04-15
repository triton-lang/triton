import functools
import logging

from triton.compiler import CompiledKernel
from triton.compiler import get_arch_default_num_stages as _get_arch_default_num_stages
from triton.compiler import get_arch_default_num_warps as _get_arch_default_num_warps
from triton.runtime.jit import (
    get_backend,
    get_cuda_version_key,
    get_current_device,
)


try:
    from torch._C import _cuda_getCurrentRawStream

    def get_cuda_stream(idx):
        return _cuda_getCurrentRawStream(idx)

except ImportError:
    import torch

    def get_cuda_stream(idx):
        return torch.cuda.current_stream(idx).cuda_stream


logger = logging.getLogger(__name__)

DEFAULT_NUM_WARPS = _get_arch_default_num_warps("cuda")
DEFAULT_NUM_STAGES = _get_arch_default_num_stages("cuda")


class LowLatencyJITFunctionPythonGeneratedWithTypeShort:
    def __getitem__(self, grid):
        return functools.partial(self.run, grid=grid)

    def __init__(self, jit_function):
        self.debug = jit_function.debug
        self._original_jit_function = jit_function
        self.cache = {0: dict()}

    def run(
        self,
        dy,  # tl.tensor
        a,  # tl.float32
        output,  # tl.tensor
        N,  # tl.uint32
        M_STRIDE,  # tl.uint32
        BLOCK_SIZE,  # tl.uint32
        *,
        grid=None,
        num_warps=DEFAULT_NUM_WARPS,
        num_ctas=1,
        num_stages=DEFAULT_NUM_STAGES,
        enable_warp_specialization=False,
        enable_fp_fusion=True,
        extern_libs=None,
        stream=None,
        warmup=False,
        device=None,
        device_type="cuda",
    ):
        assert grid is not None
        dy_ptr = dy.data_ptr()
        output_ptr = output.data_ptr()

        if device_type != "cuda":
            device_backend = get_backend(device_type)
            if device_backend is None:
                raise ValueError("Cannot find backend for " + device_type)
        if device_type == "cuda":
            version_key = get_cuda_version_key()
        else:
            version_key = device_backend.get_version_key()

        key = (
            version_key,
            (
                dy.dtype,
                "fp32",
                output.dtype,
                "i32",
                "i32",
            ),
            (BLOCK_SIZE,),
            (
                (dy_ptr % 16 == 0,),
                (False,),
                (output_ptr % 16 == 0,),
                (
                    N % 16 == 0,
                    N % 8 == 0,
                    N == 1,
                ),
                (
                    M_STRIDE % 16 == 0,
                    M_STRIDE % 8 == 0,
                    M_STRIDE == 1,
                ),
                (
                    BLOCK_SIZE % 16 == 0,
                    BLOCK_SIZE % 8 == 0,
                    BLOCK_SIZE == 1,
                ),
            ),
            num_warps,
            num_ctas,
            num_stages,
            enable_warp_specialization,
            enable_fp_fusion,
            self.debug,
        )

        if extern_libs is not None:
            key = (key, tuple(extern_libs.items()))

        if device is None:
            if device_type == "cuda":
                device = get_current_device()
                # set_current_device(device)
            else:
                device = device_backend.get_current_device()
                device_backend.set_current_device(device)

        if stream is None and not warmup:
            if device_type == "cuda":
                stream = get_cuda_stream(device)
            else:
                stream = device_backend.get_stream()

        # Kernel is not cached; we have to compile.
        if key not in self.cache[device]:
            self._original_jit_function[grid](
                dy,
                a,
                output,
                N,
                M_STRIDE,
                BLOCK_SIZE,
                num_warps=num_warps,
                num_ctas=num_ctas,
                num_stages=num_stages,
                enable_warp_specialization=enable_warp_specialization,
                enable_fp_fusion=enable_fp_fusion,
                extern_libs=extern_libs,
                stream=stream,
                warmup=warmup,
                device=device,
                device_type=device_type,
            )
            self.cache[device][key] = self._original_jit_function.cache[device][key]
            # logger.error(self._original_jit_function.cache[device][key])

        bin = self.cache[device][key]
        if not warmup:
            if callable(grid):
                grid = grid((dy, a, output, N, M_STRIDE, BLOCK_SIZE))
            grid_size = len(grid)
            bin.c_wrapper(
                grid[0],
                grid[1] if grid_size > 1 else 1,
                grid[2] if grid_size > 2 else 1,
                bin.num_warps,
                bin.num_ctas,
                *bin.clusterDims,
                bin.shared,
                stream,
                bin.cu_function,
                CompiledKernel.launch_enter_hook,
                CompiledKernel.launch_exit_hook,
                bin,
                dy_ptr,
                a,
                output_ptr,
                N,
                M_STRIDE,
            )
        return bin

    def __repr__(self):
        return f"LowLatencyJITFunctionPythonGeneratedWithType({self.module}:{self.fn.__name__})"


class LowLatencyJITFunctionPythonGeneratedWithTypeLong(
    LowLatencyJITFunctionPythonGeneratedWithTypeShort
):

    def run(
        self,
        dy,
        a,
        b,
        c,
        d,
        e,
        f,
        g,
        h,
        output,
        N,
        M_STRIDE,
        BLOCK_SIZE,
        *,
        grid=None,
        num_warps=DEFAULT_NUM_WARPS,
        num_ctas=1,
        num_stages=DEFAULT_NUM_STAGES,
        enable_warp_specialization=False,
        enable_fp_fusion=True,
        extern_libs=None,
        stream=None,
        warmup=False,
        device=None,
        device_type="cuda",
    ):
        assert grid is not None
        dy_ptr = dy.data_ptr()
        output_ptr = output.data_ptr()

        if device_type != "cuda":
            device_backend = get_backend(device_type)
            if device_backend is None:
                raise ValueError("Cannot find backend for " + device_type)
        if device_type == "cuda":
            version_key = get_cuda_version_key()
        else:
            version_key = device_backend.get_version_key()

        key = (
            version_key,
            (
                dy.dtype,
                "fp32",
                "fp32",
                "fp32",
                "fp32",
                "fp32",
                "fp32",
                "fp32",
                "fp32",
                output.dtype,
                "i32",
                "i32",
            ),
            (BLOCK_SIZE,),
            (
                (dy_ptr % 16 == 0,),
                (False,),
                (False,),
                (False,),
                (False,),
                (False,),
                (False,),
                (False,),
                (False,),
                (output_ptr % 16 == 0,),
                (
                    N % 16 == 0,
                    N % 8 == 0,
                    N == 1,
                ),
                (
                    M_STRIDE % 16 == 0,
                    M_STRIDE % 8 == 0,
                    M_STRIDE == 1,
                ),
                (
                    BLOCK_SIZE % 16 == 0,
                    BLOCK_SIZE % 8 == 0,
                    BLOCK_SIZE == 1,
                ),
            ),
            num_warps,
            num_ctas,
            num_stages,
            enable_warp_specialization,
            enable_fp_fusion,
            self.debug,
        )

        if extern_libs is not None:
            key = (key, tuple(extern_libs.items()))

        if device is None:
            if device_type == "cuda":
                device = get_current_device()
                # set_current_device(device)
            else:
                device = device_backend.get_current_device()
                device_backend.set_current_device(device)

        if stream is None and not warmup:
            if device_type == "cuda":
                stream = get_cuda_stream(device)
            else:
                stream = device_backend.get_stream()

        # Kernel is not cached; we have to compile.
        if key not in self.cache[device]:
            self._original_jit_function[grid](
                dy,
                a,
                b,
                c,
                d,
                e,
                f,
                g,
                h,
                output,
                N,
                M_STRIDE,
                BLOCK_SIZE,
                num_warps=num_warps,
                num_ctas=num_ctas,
                num_stages=num_stages,
                enable_warp_specialization=enable_warp_specialization,
                enable_fp_fusion=enable_fp_fusion,
                extern_libs=extern_libs,
                stream=stream,
                warmup=warmup,
                device=device,
                device_type=device_type,
            )
            self.cache[device][key] = self._original_jit_function.cache[device][key]
            # logger.error(self._original_jit_function.cache[device][key])

        bin = self.cache[device][key]
        if not warmup:
            if callable(grid):
                grid = grid((dy, a, output, N, M_STRIDE, BLOCK_SIZE))

            grid_size = len(grid)

            bin.c_wrapper(
                grid[0],
                grid[1] if grid_size > 1 else 1,
                grid[2] if grid_size > 2 else 1,
                bin.num_warps,
                bin.num_ctas,
                *bin.clusterDims,
                bin.shared,
                stream,
                bin.cu_function,
                CompiledKernel.launch_enter_hook,
                CompiledKernel.launch_exit_hook,
                bin,
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
            )
        return bin
