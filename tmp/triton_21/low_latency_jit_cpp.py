import functools
import inspect
import logging
import os
import textwrap
from collections import defaultdict

from torch.utils.cpp_extension import load
from triton.compiler import CompiledKernel, compile
from triton.compiler import get_arch_default_num_stages as _get_arch_default_num_stages
from triton.compiler import get_arch_default_num_warps as _get_arch_default_num_warps
from triton.runtime.jit import JITFunction as _JITFunction
from triton.runtime.jit import (
    KernelArg,
    KernelParam,
    TMAInfos,
    get_backend,
    get_cuda_stream,
    get_cuda_version_key,
    get_current_device,
)

logger = logging.getLogger(__name__)

import pathlib
import torch

path = pathlib.Path(__file__).parent.resolve()

m = load(
    name="jit_function",
    sources=(path / "jit_function.cpp").as_posix(),
    extra_cflags=["-O3"],
    # with_cuda=True,
)
logger.warning("Loaded low_latency_jit_function")


@functools.cache
def get_arch_default_num_warps(device_type):
    return _get_arch_default_num_warps(device_type)


@functools.cache
def get_arch_default_num_stages(device_type, capability=None):
    return _get_arch_default_num_stages(device_type, capability)


@functools.cache
def get_supported_device_type():
    import torch

    return "hip" if torch.version.hip else "cuda"


class LowLatencyJITFunctionCPP(_JITFunction, m.LowLatencyJITFunction):
    @staticmethod
    def _pinned_memory_of(arg):
        if hasattr(arg, "is_pinned"):
            return arg.is_pinned()
        return False

    def run(
        self,
        *args,
        grid=None,
        num_warps=None,
        num_ctas=1,
        num_stages=None,
        enable_warp_specialization=False,
        enable_fp_fusion=True,
        extern_libs=None,
        stream=None,
        warmup=False,
        device=None,
        device_type=None,
        **kwargs,
    ):
        # logger.error(args)
        keys, non_constexpr_arg_values = self.get_call_params_tuple(args, kwargs)
        # device, stream, keys, non_constexpr_arg_values = self.get_call_params_tuple(args, kwargs)
        # logger.error(keys)

        assert num_ctas > 0
        assert grid is not None
        device_type = device_type if device_type else get_supported_device_type()

        device_backend = None
        if device_type != "cuda":
            device_backend = get_backend(device_type)
            if device_backend is None:
                raise ValueError("Cannot find backend for " + device_type)

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

        if num_warps is None:
            num_warps = get_arch_default_num_warps(device_type)
        if num_stages is None:
            num_stages = get_arch_default_num_stages(device_type)

        key = (
            device,
            keys,
            extern_libs,
            num_warps,
            num_ctas,
            num_stages,
            enable_warp_specialization,
            enable_fp_fusion,
            self.debug,
        )
        # if extern_libs is not None:
        #     key = (key, tuple(extern_libs.items()))

        # Kernel is not cached; we have to compile.
        if key not in self.cache:
            logger.info(f"Compiling {self.name}")
            bound_args = self.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            args = [
                KernelArg(arg_value, param)
                for (_, arg_value), param in zip(
                    bound_args.arguments.items(), self.params
                )
            ]
            configs = (self._get_config(*[arg.value for arg in args]),)
            constants = {
                arg.param.num: arg.value
                for arg in args
                if arg.param.is_constexpr
                or arg.param.num in configs[0].equal_to_1
                or arg.value is None
            }
            for i, arg in constants.items():
                if callable(arg):
                    raise TypeError(f"Callable constexpr at index {i} is not supported")

            # Build kernel signature -- doesn't include constexpr arguments.
            signature = {
                arg.param.num: self._type_of(self._key_of(arg.value))
                for arg in args
                if not arg.param.is_constexpr
            }

            if self._call_hook(
                key,
                signature,
                device,
                constants,
                num_warps,
                num_ctas,
                num_stages,
                enable_warp_specialization,
                enable_fp_fusion,
                extern_libs,
                configs,
            ):
                return None

            self.cache[key] = compile(
                self,
                signature=signature,
                device=device,
                constants=constants,
                num_warps=num_warps,
                num_ctas=num_ctas,
                num_stages=num_stages,
                enable_warp_specialization=enable_warp_specialization,
                enable_fp_fusion=enable_fp_fusion,
                extern_libs=extern_libs,
                configs=configs,
                debug=self.debug,
                device_type=device_type,
            )

        bin = self.cache[key]
        if not warmup:
            grid_size = len(grid)
            bin.c_wrapper(
                grid[0],
                grid[1] if grid_size > 1 else 1,
                grid[2] if grid_size > 2 else 1,
                bin.num_warps,
                bin.num_ctas,
                bin.clusterDims[0],
                bin.clusterDims[1],
                bin.clusterDims[2],
                bin.shared,
                stream,
                bin.cu_function,
                # CompiledKernel.launch_enter_hook,
                # CompiledKernel.launch_exit_hook,
                # bin,
                # *bin.assemble_tensormap_to_arg(non_constexpr_arg_values),
                *(_.data_ptr() if torch.is_tensor(_) else _ for _ in bin.assemble_tensormap_to_arg(non_constexpr_arg_values)),
            )
        return bin

    def __init__(
        self, fn, version=None, do_not_specialize=None, debug=None, noinline=None
    ):
        self.name = fn.__name__
        logger.info(f"Crating LowLatencyJITFunction for {self.name}")
        do_not_specialize = do_not_specialize if do_not_specialize else []
        self.fn = fn
        self.module = fn.__module__
        self.version = version
        self.signature = inspect.signature(fn)
        self.do_not_specialize = do_not_specialize
        self.starting_line_number = inspect.getsourcelines(fn)[1]

        self.params = []
        for i, param in enumerate(self.signature.parameters.values()):
            dns = len(do_not_specialize) > 0 and (
                i in do_not_specialize or param.name in do_not_specialize
            )
            self.params.append(KernelParam(i, param, dns))

        m.LowLatencyJITFunction.__init__(
            self,
            self.divisibility,
            self.divisibility_8,
            params=self.params,
            cuda_version_key=get_cuda_version_key(),
        )

        # function source code (without decorators)
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find("def") :]
        # cache of just-in-time compiled kernels
        self.cache = dict()
        self.hash = None
        # JITFunction can be instantiated as kernel
        # when called with a grid using __getitem__
        self.kernel = None
        self.debug = True if os.environ.get("TRITON_DEBUG", "0") == "1" else debug
        self.noinline = noinline

        # tma info
        self.tensormaps_info = TMAInfos()

        # TODO(jlebar): Remove uses of these fields outside this file, then
        # remove the fields here.
        self.arg_names = [p.name for p in self.params]
        self.constexprs = [p.num for p in self.params if p.is_constexpr]

        # re-use docs of wrapped function
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__

    def __repr__(self):
        return f"LowLatencyJITFunctionCPP({self.module}:{self.fn.__name__})"
