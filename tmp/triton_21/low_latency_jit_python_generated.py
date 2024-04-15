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


def _key_of(arg):
    if isinstance(arg, int):
        # if -(2**31) <= arg and arg <= 2**31 - 1:
        if -2147483648 <= arg and arg <= 2147483647:
            return "i32"
        elif 0 <= arg:
            return "u64"
        else:
            return "i64"
    elif isinstance(arg, float):
        return "fp32"
    elif isinstance(arg, bool):
        return "i1"
    elif arg is None:
        return None
    raise TypeError(f"Unsupported type {type(arg)} for {arg}")


def _prime_key(value):
    if isinstance(value, int):
        return (
            value % 16 == 0,
            value % 8 == 0,
            value == 1,
        )
    return (False,)


class LowLatencyJITFunctionPythonGeneratedShort:
    def __getitem__(self, grid):
        return functools.partial(self.run, grid=grid)

    def __init__(self, jit_function):
        self.debug = jit_function.debug
        self._original_jit_function = jit_function
        self.cache = {0: dict()}

    def run(
        self,
        dy,
        a,
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

        # sig_key = (
        #     dy.dtype,
        #     _key_of(a),
        #     output.dtype,
        #     _key_of(N),
        #     _key_of(M_STRIDE),
        # )

        # spec_key = (
        #     (dy_ptr % 16 == 0,),
        #     _prime_key(a),
        #     (output_ptr % 16 == 0,),
        #     _prime_key(N),
        #     _prime_key(M_STRIDE),
        #     _prime_key(BLOCK_SIZE),
        # )
        # constexpr_key = (BLOCK_SIZE,)

        if device_type != "cuda":
            device_backend = get_backend(device_type)
            if device_backend is None:
                raise ValueError("Cannot find backend for " + device_type)
        if device_type == "cuda":
            version_key = get_cuda_version_key()
        else:
            version_key = device_backend.get_version_key()

        # key = (
        #     version_key,
        #     sig_key,
        #     constexpr_key,
        #     spec_key,
        #     num_warps,
        #     num_ctas,
        #     num_stages,
        #     enable_warp_specialization,
        #     enable_fp_fusion,
        #     self.debug,
        # )

        key = (
            version_key,
            (
                dy.dtype,
                _key_of(a),
                output.dtype,
                _key_of(N),
                _key_of(M_STRIDE),
            ),
            (BLOCK_SIZE,),
            (
                (dy_ptr % 16 == 0,),
                _prime_key(a),
                (output_ptr % 16 == 0,),
                _prime_key(N),
                _prime_key(M_STRIDE),
                _prime_key(BLOCK_SIZE),
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
        return f"LowLatencyJITFunctionGenerated({self.module}:{self.fn.__name__})"


class LowLatencyJITFunctionPythonGeneratedLong(
    LowLatencyJITFunctionPythonGeneratedShort
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

        # sig_key = (
        #     dy.dtype,
        #     _key_of(a),
        #     _key_of(b),
        #     _key_of(c),
        #     _key_of(d),
        #     _key_of(e),
        #     _key_of(f),
        #     _key_of(g),
        #     _key_of(h),
        #     output.dtype,
        #     _key_of(N),
        #     _key_of(M_STRIDE),
        # )

        # spec_key = (
        #     (dy_ptr % 16 == 0,),
        #     _prime_key(a),
        #     _prime_key(b),
        #     _prime_key(c),
        #     _prime_key(d),
        #     _prime_key(e),
        #     _prime_key(f),
        #     _prime_key(g),
        #     _prime_key(h),
        #     (output_ptr % 16 == 0,),
        #     _prime_key(N),
        #     _prime_key(M_STRIDE),
        #     _prime_key(BLOCK_SIZE),
        # )
        # constexpr_key = (BLOCK_SIZE,)

        if device_type != "cuda":
            device_backend = get_backend(device_type)
            if device_backend is None:
                raise ValueError("Cannot find backend for " + device_type)
        if device_type == "cuda":
            version_key = get_cuda_version_key()
        else:
            version_key = device_backend.get_version_key()

        # key = (
        #     version_key,
        #     sig_key,
        #     constexpr_key,
        #     spec_key,
        #     num_warps,
        #     num_ctas,
        #     num_stages,
        #     enable_warp_specialization,
        #     enable_fp_fusion,
        #     self.debug,
        # )

        key = (
            version_key,
            (
                dy.dtype,
                _key_of(a),
                _key_of(b),
                _key_of(c),
                _key_of(d),
                _key_of(e),
                _key_of(f),
                _key_of(g),
                _key_of(h),
                output.dtype,
                _key_of(N),
                _key_of(M_STRIDE),
            ),
            (BLOCK_SIZE,),
            (
                (dy_ptr % 16 == 0,),
                _prime_key(a),
                _prime_key(b),
                _prime_key(c),
                _prime_key(d),
                _prime_key(e),
                _prime_key(f),
                _prime_key(g),
                _prime_key(h),
                (output_ptr % 16 == 0,),
                _prime_key(N),
                _prime_key(M_STRIDE),
                _prime_key(BLOCK_SIZE),
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
