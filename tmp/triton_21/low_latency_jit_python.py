import functools
import inspect
import logging
import os
import textwrap
from collections import defaultdict
from typing import Set
import torch

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
    set_current_device,
)

logger = logging.getLogger(__name__)


@functools.cache
def get_arch_default_num_warps(device_type):
    return _get_arch_default_num_warps(device_type)


@functools.cache
def get_arch_default_num_stages(device_type, capability=None):
    return _get_arch_default_num_stages(device_type, capability)


@functools.cache
def get_cuda_device_type_if_cuda():
    import torch

    return "hip" if torch.version.hip else "cuda"


@functools.cache
def get_supported_device_type():
    import torch

    return "hip" if torch.version.hip else "cuda"


def get_device_type(args):
    is_cpu = False
    not_cuda_types = set()
    for arg in args:
        if hasattr(arg, "device"):
            device_type = arg.device.type
            if device_type == "cuda":
                return get_cuda_device_type_if_cuda()
            elif device_type == "cpu":
                is_cpu = True
            else:
                not_cuda_types.add(device_type)

    if is_cpu:
        for arg in args:
            if hasattr(arg, "is_pinned") and arg.device.type == "cuda":
                return get_cuda_device_type_if_cuda()
    return not_cuda_types.pop() if len(not_cuda_types) > 0 else "cuda"


class LowLatencyJITFunctionPython(_JITFunction):
    def _conclude_device_type(
        self, device_types: Set[str], is_pinned_memory: bool
    ) -> str:
        # device_types = [device_type for device_type in device_types if device_type != ""]
        # Return cuda if one of the input tensors is cuda
        if "cuda" in device_types:
            # import torch

            # return "hip" if torch.version.hip else "cuda"
            return get_cuda_device_type_if_cuda()

        is_cpu = "cpu" in device_types
        # is_pinned_memory = any(pinned_memory_flag for pinned_memory_flag in pinned_memory_flags)
        # Return cuda if all the input tensors are cpu while the memory is pinned
        if is_cpu and is_pinned_memory:
            return "cuda"

        return device_types.pop() if len(device_types) > 0 else "cuda"

    def specialization_key(self):
        assert not self.param.do_not_specialize

        if hasattr(self, "value"):
            return (self.value.data_ptr() % _JITFunction.divisibility == 0,)
        if isinstance(self.value, int):
            # bool is a subclass of int, so we don't check explicitly above.
            return (
                self.value % _JITFunction.divisibility == 0,
                self.value % _JITFunction.divisibility_8 == 0,
                self.value == 1,
            )

        return (False,)

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
        # Bind the remaining arguments to `fn`.
        # bound_args = self.signature.bind(*args, **kwargs)
        # bound_args.apply_defaults()
        # assert len(bound_args.arguments) == len(self.params)

        bound_args2 = {
            param.name: value for param, value in zip(self.positional_or_kw_args, args)
        }
        if len(args) < len(self.positional_or_kw_args):
            for param in self.positional_or_kw_args[len(args) :]:
                bound_args2[param.name] = kwargs[param.name]
        for param in self.kw_only_args:
            if param.name in kwargs:
                bound_args2[param.name] = kwargs[param.name]
            else:
                bound_args2[param.name] = param.default
        # logger.error(bound_args)
        # logger.error(bound_args2)
        bound_args = bound_args2

        # args = [KernelArg(arg_value, param) for (_, arg_value), param in zip(bound_args.arguments.items(), self.params)]
        args = [
            KernelArg(arg_value, param)
            for arg_value, param in zip(bound_args.values(), self.params)
        ]

        non_constexpr_arg_values = [
            args[idx].value for idx in self.non_constexpr_indices
        ]
        # non_constexpr_arg_values = [arg.value for arg in args if not arg.param.is_constexpr]

        # sig_key = tuple(arg.signature_key() for arg in args if not arg.param.is_constexpr)
        # spec_key = tuple(arg.specialization_key() for arg in args if not arg.param.do_not_specialize)
        sig_key = tuple(args[idx].signature_key() for idx in self.constexpr_indices)
        spec_key = tuple(
            args[idx].specialization_key() for idx in self.do_not_specialize_indices
        )
        constexpr_key = tuple(args[idx].value for idx in self.constexpr_indices)

        assert num_ctas > 0
        assert grid is not None
        if callable(grid):
            # Arguments are passed as a dict to `grid`, by contract.
            # TODO(jlebar): In the new launch API, pass the compiler flags as a
            # second parameter to `grid`.
            grid = grid(bound_args.arguments)
            # grid = grid(dict(bound_args.arguments))
        grid_size = len(grid)
        grid_0 = grid[0]
        grid_1 = grid[1] if grid_size > 1 else 1
        grid_2 = grid[2] if grid_size > 2 else 1
        device_type = device_type if device_type else get_supported_device_type()
        # device_types = {arg.device.type for arg in non_constexpr_arg_values if hasattr(arg, "device")}
        # device_type = "cuda"
        # device_type = self._conclude_device_type(device_types,
        #                                          any(arg.is_pinned() for arg in non_constexpr_arg_values if hasattr(arg, "is_pinned"))
        # )
        # device_type = get_device_type(non_constexpr_arg_values)

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

        if device_type == "cuda":
            version_key = get_cuda_version_key()
        else:
            version_key = device_backend.get_version_key()
        key = (
            device,
            extern_libs,
            version_key,
            sig_key,
            constexpr_key,
            spec_key,
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
            bin.c_wrapper(
                grid_0,
                grid_1,
                grid_2,
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

        self.positional_or_kw_args = [
            p
            for p in self.signature.parameters.values()
            if p.kind in {p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD}
        ]
        self.kw_only_args = [
            p for p in self.signature.parameters.values() if p.kind == p.KEYWORD_ONLY
        ]

        self.params = []
        for i, param in enumerate(self.signature.parameters.values()):
            dns = len(do_not_specialize) > 0 and (
                i in do_not_specialize or param.name in do_not_specialize
            )
            self.params.append(KernelParam(i, param, dns))

        self.non_constexpr_indices = [
            idx for idx, p in enumerate(self.params) if not p.is_constexpr
        ]
        self.constexpr_indices = [
            idx for idx, p in enumerate(self.params) if p.is_constexpr
        ]
        self.do_not_specialize_indices = [
            idx for idx, p in enumerate(self.params) if p.do_not_specialize
        ]

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
        return f"LowLatencyJITFunctionPython({self.module}:{self.fn.__name__})"
