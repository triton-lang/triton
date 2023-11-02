from __future__ import annotations, division

import ast
import hashlib
import inspect
import os
import textwrap
from collections import defaultdict, namedtuple
from functools import cached_property
from typing import Any, Callable, Generic, Optional, TypeVar, Union, cast, overload, Sequence, Tuple, Protocol
import warnings

from .._C.libtriton.triton import TMAInfos
from ..common.backend import get_backend, get_cuda_version_key
from .interpreter import InterpretedFunction

# A callable function.  In practice, this is a function wrapped by @jit.
KernelT = TypeVar("KernelT", bound=Callable[..., Any])

GridT = Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]]


def get_cuda_stream(idx=None):
    if idx is None:
        idx = get_current_device()
    try:
        from torch._C import _cuda_getCurrentRawStream

        return _cuda_getCurrentRawStream(idx)
    except ImportError:
        import torch

        return torch.cuda.current_stream(idx).cuda_stream


def get_current_device():
    import torch

    return torch.cuda.current_device()


def set_current_device(idx):
    import torch

    torch.cuda.set_device(idx)


def get_device_capability(idx):
    import torch

    return torch.cuda.get_device_capability(idx)


def _extract_flags(kernel_kwargs: dict[str, Any]):
    """Removes Triton flag args from kernel_kwargs.

    Returns a tuple of (flags, kernel args without flags).
    """
    # Don't overwrite the user's copy of kernel_kwargs.
    kernel_kwargs = dict(kernel_kwargs)

    flags = {}

    def extract(name):
        if name in kernel_kwargs:
            flags[name] = kernel_kwargs[name]
            del kernel_kwargs[name]

    extract("num_warps")
    extract("num_ctas")
    extract("num_stages")
    extract("enable_warp_specialization")
    extract("enable_fp_fusion")
    extract("extern_libs")
    extract("stream")
    extract("warmup")
    extract("device")
    extract("device_type")

    if flags:
        flags_str = ", ".join(f"{f}=foo" for f in flags.keys())
        warnings.warn(
            textwrap.dedent(
                f"""\
            Passing Triton compiler flags as kernel arguments is deprecated and will be removed.

            Change:
              kernel_fn[grid](kernel_arg1, kernel_arg2, ..., {flags_str}),
            to:
              kernel_fn.with_flags({flags_str})[grid](kernel_arg1, kernel_arg2, ...).
            """
            ),
            DeprecationWarning,
        )

    return flags, kernel_kwargs


def _reify_grid(grid: GridT, flags, kernel_bound_args, has_explicit_flags) -> Tuple[int, int, int]:
    assert grid is not None
    if callable(grid):
        # Arguments are passed as a dict to `grid`, by contract.
        if not has_explicit_flags:
            # In the old API, the grid function gets all of the flags as
            # a single dict.
            grid = grid(dict(**kernel_bound_args, **flags))
        else:
            # In the new API, the grid function gets the kernel args as
            # its first argument, and the compiler flags as its second
            # argument (if the grid fn takes two args).
            grid_fn_sig = inspect.signature(grid)
            if len(grid_fn_sig.parameters) == 1:
                grid = grid(dict(**kernel_bound_args))
            elif len(grid_fn_sig.parameters) == 2:
                grid = grid(dict(**kernel_bound_args), flags)
            else:
                assert False, "Grid function must take 1 or 2 arguments, namely kernel_params, compiler_flags"

    grid = cast(GridT, grid)
    grid_size = len(grid)
    return (
        grid[0],
        grid[1] if grid_size > 1 else 1,
        grid[2] if grid_size > 2 else 1,
    )


# -----------------------------------------------------------------------------
# Dependencies Finder
# -----------------------------------------------------------------------------


class DependenciesFinder(ast.NodeVisitor):
    """
    This AST visitor is used to find dependencies of a JITFunction. This can
    be used to invalidate a JITFunction's hash when its source code -- or
    that of its dependencies -- changes.
    """

    def __init__(self, globals, src) -> None:
        super().__init__()
        self.ret = hashlib.sha1(src.encode("utf-8")).hexdigest()
        self.globals = globals

    def visit_Name(self, node):
        return self.globals.get(node.id, None)

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        while isinstance(lhs, ast.Attribute):
            lhs = self.visit(lhs.value)
        if lhs is None or (
            getattr(lhs, "__name__", "") == "triton" or getattr(lhs, "__name__", "").endswith(".triton")
        ):
            return None
        return getattr(lhs, node.attr)

    def visit_Call(self, node):
        func = self.visit(node.func)
        if func is None:
            return
        if inspect.isbuiltin(func):
            return
        if func.__module__ and (func.__module__.startswith("triton.") or ".triton." in func.__module__):
            return
        assert isinstance(
            func, JITFunction
        ), f'Function "{func.__name__}" is being called from a Triton function but is not a Triton function itself. Decorate it with @triton.jit to fix this'
        if func.hash is None:
            tree = ast.parse(func.src)
            finder = DependenciesFinder(func.__globals__, func.src)
            finder.visit(tree)
            func.hash = finder.ret
        noinline = str(getattr(func, "noinline", False))
        self.ret = (self.ret + func.hash + noinline).encode("utf-8")
        self.ret = hashlib.sha1(self.ret).hexdigest()


# -----------------------------------------------------------------------------
# JITFunction
# -----------------------------------------------------------------------------


def _normalize_ty(ty) -> str:
    if isinstance(ty, type):
        return ty.__name__
    elif isinstance(ty, str):
        return ty
    return repr(ty)


class KernelParam:
    """Represents a parameter to a @jit'ed function.

    A parameter is just the name plus metadata; a parameter plus a value is a
    KernelArg.
    """

    def __init__(self, num: int, param: inspect.Parameter, do_not_specialize: bool):
        self.num = num
        self._param = param
        self.do_not_specialize = do_not_specialize

    @cached_property
    def name(self):
        return self._param.name

    @cached_property
    def annotation(self):
        if not self._param.annotation or self._param.annotation == inspect.Parameter.empty:
            return ""
        return _normalize_ty(self._param.annotation)

    @cached_property
    def is_constexpr(self):
        return "constexpr" in self.annotation

    @property
    def default(self):
        return self._param.default

    @property
    def has_default(self):
        return self._param.default != inspect.Parameter.empty


class KernelArg:
    """Represents an argument to a @jit'ed function.

    An argument is a parameter plus a value.
    """

    def __init__(self, value, param):
        self.value = value
        self.param = param

    @property
    def name(self):
        return self.param.name

    def signature_key(self):
        annotation = self.param.annotation
        if "Tensor" in annotation:
            return self.value.dtype
        elif annotation == "bool":
            return "i1"
        elif annotation == "float":
            return "fp32"
        else:
            return JITFunction._key_of(self.value)

    def specialization_key(self):
        assert not self.param.do_not_specialize

        try:
            return (self.value.data_ptr() % JITFunction.divisibility == 0,)
        except AttributeError:
            pass

        if isinstance(self.value, int):
            # bool is a subclass of int, so we don't check explicitly above.
            return (
                self.value % JITFunction.divisibility == 0,
                self.value % JITFunction.divisibility_8 == 0,
                self.value == 1,
            )

        return (False,)


class KernelInterface(Generic[KernelT]):
    """Abstract class implementing the public API for a Triton kernel.

    This class implements the syntactic sugar of:
      - invoking a kernel as kernel[grid](args),
      - doing everything-but-invoking a kernel using the warmup() function, and
      - the with_flags() function, in e.g. kernel.with_flags(...)[](args).

    It forwards the actual running of the kernel to
    self.__run_with_flags and self.__warmup_with_flags, which are passed to the
    the constructor.
    """

    # Type interface of self.__run_with_flags.
    class RunWithFlagsT(Protocol):
        def __call__(
            self,
            grid: Tuple[int, int, int],
            flags: dict[str, Any],
            kernel_args: dict[str, Any],  # not including compiler flags like `num_warps`
        ):
            ...

    # Type interface of self.__warmup_with_flags.
    class WarmupWithFlagsT(Protocol):
        def __call__(self, flags: dict[str, Any], kernel_args: dict[str, Any]):
            ...

    def __init__(
        self,
        # Signature of the user's kernel function.
        signature: inspect.Signature,
        run_with_flags: KernelInterface.RunWithFlagsT,
        warmup_with_flags: KernelInterface.WarmupWithFlagsT | None,
        flags: dict[str, Any] | None = None,
    ):
        self.signature = signature
        self.__run_with_flags = run_with_flags
        self.__warmup_with_flags = warmup_with_flags
        self.__flags = flags

    def __getitem__(self, grid: GridT) -> KernelT:
        def wrapper(*args, **kwargs):
            # With the old API (self.__flags is None), the compiler flags are
            # passed in kwargs.
            if self.__flags is not None:
                flags = self.__flags
                kernel_kwargs = kwargs
            else:
                flags, kernel_kwargs = _extract_flags(kwargs)

            # Bind the kernel args to the user's kernel function declaration.
            bound_args = self.signature.bind(*args, **kernel_kwargs)
            bound_args.apply_defaults()

            reified_grid = _reify_grid(
                grid,
                flags,
                bound_args.arguments,
                has_explicit_flags=(self.__flags is not None),
            )
            return self.__run_with_flags(reified_grid, flags, bound_args.arguments)

        return cast(KernelT, wrapper)

    def warmup(self, *args, **kwargs):
        def wrapper(*args, **kwargs):
            if self.__flags is not None:
                flags = self.__flags
                kernel_kwargs = kwargs
            else:
                flags, kernel_kwargs = _extract_flags(kwargs)

            bound_args = self.signature.bind(*args, **kernel_kwargs)
            bound_args.apply_defaults()

            assert self.__warmup_with_flags
            return self.__warmup_with_flags(flags, bound_args.arguments)

        return cast(KernelT, wrapper)

    def with_flags(
        self,
        num_warps: int | None = None,
        num_ctas: int = 1,
        num_stages: int | None = None,
        enable_warp_specialization: bool = False,
        enable_fp_fusion: bool = True,
        extern_libs: dict[str, str] | None = None,
        stream: int | None = None,
        warmup: bool = False,
        device: int | None = None,
        device_type: str | None = None,
    ) -> KernelInterface[KernelT]:
        assert self.__flags is None, "Cannot call with_flags twice"

        flags = {}

        def add_to_flags(name, value):
            if value is not None:
                flags[name] = value

        add_to_flags("num_warps", num_warps)
        add_to_flags("num_ctas", num_ctas)
        add_to_flags("num_stages", num_stages)
        add_to_flags("enable_warp_specialization", enable_warp_specialization)
        add_to_flags("enable_fp_fusion", enable_fp_fusion)
        add_to_flags("extern_libs", extern_libs)
        add_to_flags("stream", stream)
        add_to_flags("warmup", warmup)
        add_to_flags("device", device)
        add_to_flags("device_type", device_type)

        return KernelInterface(self.signature, self.__run_with_flags, self.__warmup_with_flags, flags=flags)


class JITFunction(KernelInterface[KernelT]):
    # Hook for inspecting compiled functions and modules
    cache_hook = None
    divisibility = 16
    # As Hopper TMA load and store primitive requires the tensor stride to be 16-byte aligned.
    # And we only support WGMMA with float16 dtype on Hopper for now.
    # So whether the LoadOp and StoreOp will lowering into TMA copy depend on whether the tensor stride is divisible by 8.
    # TODO: Make it more reasonable to handle multiple dtypes.
    divisibility_8 = 8

    @staticmethod
    def _key_of(arg):
        if hasattr(arg, "dtype"):
            return arg.dtype
        elif isinstance(arg, bool):
            return "i1"
        elif isinstance(arg, int):
            if -(2**31) <= arg and arg <= 2**31 - 1:
                return "i32"
            elif 2**63 <= arg and arg <= 2**64 - 1:
                return "u64"
            else:
                return "i64"
        elif isinstance(arg, float):
            return "fp32"
        elif arg is None:
            return None
        else:
            raise TypeError(f"Unsupported type {type(arg)} for {arg}")

    @staticmethod
    def _device_of(arg):
        try:
            return arg.device.type
        except AttributeError:
            return ""

    @staticmethod
    def _pinned_memory_of(arg):
        try:
            return arg.is_pinned()
        except (AttributeError, TypeError):
            return False

    @staticmethod
    def _spec_of(arg):
        if hasattr(arg, "data_ptr"):
            return arg.data_ptr() % JITFunction.divisibility == 0
        elif isinstance(arg, int):
            return (arg % 16 == 0, arg == 1)
        return (arg is None,)

    # TODO(jlebar): Fold this into the KernelArg class.
    def _get_config(self, *args):
        def is_divisible_by_16(x):
            if hasattr(x, "data_ptr"):
                return x.data_ptr() % JITFunction.divisibility == 0
            elif isinstance(x, int):
                return x % JITFunction.divisibility == 0
            if x is None:
                return True
            return False

        def is_divisible_by_8(x):
            if isinstance(x, int):
                return x % JITFunction.divisibility_8 == 0
            if x is None:
                return True
            return False

        divisible_by_16 = {
            param.num
            for param, arg in zip(self.params, args)
            if is_divisible_by_16(arg) and not param.do_not_specialize
        }
        divisible_by_8 = {
            param.num for param, arg in zip(self.params, args) if is_divisible_by_8(arg) and not param.do_not_specialize
        }
        equal_to_1 = {
            param.num
            for param, arg in zip(self.params, args)
            if isinstance(arg, int) and not isinstance(arg, bool) and arg == 1 and not param.do_not_specialize
        }
        # folded equal_to_1 and None
        # TODO: method to collect all folded args
        none_args = {param.num for param, arg in zip(self.params, args) if arg is None and not param.do_not_specialize}
        ids_of_folded_args = equal_to_1 | none_args
        return namedtuple(
            "instance_descriptor", ["divisible_by_16", "equal_to_1", "ids_of_folded_args", "divisible_by_8"]
        )(tuple(divisible_by_16), tuple(equal_to_1), tuple(ids_of_folded_args), tuple(divisible_by_8))
        # return _triton.code_gen.instance_descriptor(divisible_by_16,
        # equal_to_1)

    @staticmethod
    def _type_of(key):
        # `None` is nullptr.  Implicitly convert to *i8.
        if key is None:
            return "*i8"
        dtype_str = str(key).split(".")[-1]
        tys = {
            "bool": "i1",
            "float8e4nv": "fp8e4nv",
            "float8e5": "fp8e5",
            "float8e4b15": "fp8e4b15",
            "float8e4b15x4": "fp8e4b15x4",
            "float8_e4m3fn": "fp8e4nv",
            "float8_e5m2": "fp8e5",
            "float16": "fp16",
            "bfloat16": "bf16",
            "float32": "fp32",
            "float64": "fp64",
            "int8": "i8",
            "int16": "i16",
            "int32": "i32",
            "int64": "i64",
            "uint8": "u8",
            "uint16": "u16",
            "uint32": "u32",
            "uint64": "u64",
        }
        # reinterpret can create triton type
        for v in list(tys.values()):
            tys[v] = v
        return key if isinstance(key, str) else f"*{tys[dtype_str]}"

    def _make_constants(self, constexpr_key):
        constants = dict(zip(self.constexprs, constexpr_key))
        return constants

    def _call_hook(
        self,
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
        if JITFunction.cache_hook is None:
            return False

        name = self.fn.__name__
        module = self.fn.__module__
        arg_reprs = ", ".join([f"{param.name}: {ty}" for param, ty in zip(self.params, key[1])])
        repr = f"{name}[num_warps={num_warps}, num_ctas={num_ctas}, num_stages={num_stages}, enable_warp_specialization={enable_warp_specialization}, enable_fp_fusion={enable_fp_fusion}]({arg_reprs})"
        key = str(key)

        class LegacyCompiler:
            def __init__(self, module, name):
                self.module = module
                self.name = name
                pass

        kwargs = dict(
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
        )

        return JITFunction.cache_hook(
            key=key,
            repr=repr,
            fn=LegacyCompiler(module, name),
            compile={"key": key, **kwargs},
            is_manual_warmup=False,
            already_compiled=False,
        )

    def _conclude_device_type(self, device_types: list[str], pinned_memory_flags: list[bool]) -> str:
        device_types = [device_type for device_type in device_types if device_type != ""]
        # Return cuda if one of the input tensors is cuda
        if "cuda" in device_types:
            import torch

            return "hip" if torch.version.hip else "cuda"

        is_cpu = all(device_type == "cpu" for device_type in device_types)
        is_pinned_memory = any(pinned_memory_flag for pinned_memory_flag in pinned_memory_flags)
        # Return cuda if all the input tensors are cpu while the memory is pinned
        if is_cpu and is_pinned_memory:
            return "cuda"

        return device_types[0] if len(device_types) > 0 else "cuda"

    def run_with_flags(
        self,
        grid: Tuple[int, int, int] | None,
        flags: dict[str, Any],
        # kernel_args must not include compiler flags.  Filter them into `flags`
        # before calling this function.
        kernel_args: dict[str, Any],
    ):
        from ..compiler import CompiledKernel, compile, get_arch_default_num_stages, get_arch_default_num_warps

        if not flags:
            flags = {}

        # TODO: Raise an error if we receive an unknown flag name.
        num_warps = flags.get("num_warps", None)
        num_ctas = flags.get("num_ctas", 1)
        num_stages = flags.get("num_stages", None)
        enable_warp_specialization = flags.get("enable_warp_specialization", False)
        enable_fp_fusion = flags.get("enable_fp_fusion", True)
        extern_libs = flags.get("extern_libs")
        stream = flags.get("stream")
        warmup = flags.get("warmup", False)
        device = flags.get("device")
        device_type = flags.get("device_type")

        assert len(kernel_args) == len(self.params)
        args = [KernelArg(arg_value, param) for (_, arg_value), param in zip(kernel_args.items(), self.params)]

        non_constexpr_arg_values = [arg.value for arg in args if not arg.param.is_constexpr]

        sig_key = tuple(arg.signature_key() for arg in args if not arg.param.is_constexpr)
        spec_key = tuple(arg.specialization_key() for arg in args if not arg.param.do_not_specialize)
        constexpr_key = tuple(arg.value for arg in args if arg.param.is_constexpr)

        assert num_ctas > 0
        if device_type is None:
            device_types = [self._device_of(arg) for arg in non_constexpr_arg_values]
            device_types = [_device_type for _device_type in device_types if _device_type != ""]
            device_type = self._conclude_device_type(
                device_types, [self._pinned_memory_of(arg) for arg in non_constexpr_arg_values]
            )

        device_backend = None
        if device_type not in ["cuda"]:
            device_backend = get_backend(device_type)
            if device_backend is None:
                raise ValueError("Cannot find backend for " + device_type)

        if device is None:
            if device_type in ["cuda"]:
                device = get_current_device()
                set_current_device(device)
            else:
                device = device_backend.get_current_device()
                device_backend.set_current_device(device)
        if stream is None and not warmup:
            if device_type in ["cuda"]:
                stream = get_cuda_stream(device)
            else:
                stream = device_backend.get_stream()

        if num_warps is None:
            num_warps = get_arch_default_num_warps(device_type)
        if num_stages is None:
            num_stages = get_arch_default_num_stages(device_type)

        if device_type in ["cuda"]:
            version_key = get_cuda_version_key()
        else:
            version_key = device_backend.get_version_key()
        key = (
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
        if extern_libs is not None:
            key = (key, tuple(extern_libs.items()))

        # Kernel is not cached; we have to compile.
        if key not in self.cache[device]:
            configs = (self._get_config(*[arg.value for arg in args]),)
            constants = {
                arg.param.num: arg.value
                for arg in args
                if arg.param.is_constexpr or arg.param.num in configs[0].equal_to_1 or arg.value is None
            }
            for i, arg in constants.items():
                if callable(arg):
                    raise TypeError(f"Callable constexpr at index {i} is not supported")

            # Build kernel signature -- doesn't include constexpr arguments.
            signature = {
                arg.param.num: self._type_of(self._key_of(arg.value)) for arg in args if not arg.param.is_constexpr
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

            self.cache[device][key] = compile(
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

        bin = self.cache[device][key]
        if not warmup:
            assert grid is not None
            bin.c_wrapper(
                grid[0],
                grid[1],
                grid[2],
                bin.num_warps,
                bin.num_ctas,
                bin.clusterDims[0],
                bin.clusterDims[1],
                bin.clusterDims[2],
                bin.shared,
                stream,
                bin.cu_function,
                CompiledKernel.launch_enter_hook,
                CompiledKernel.launch_exit_hook,
                bin,
                *bin.assemble_tensormap_to_arg(non_constexpr_arg_values),
            )
        return bin

    def __init__(self, fn, version=None, do_not_specialize=None, debug=None, noinline=None):
        super().__init__(inspect.signature(fn), self.run_with_flags, self.warmup_with_flags)
        do_not_specialize = do_not_specialize if do_not_specialize else []

        self.fn = fn
        self.module = fn.__module__
        self.version = version
        self.do_not_specialize = do_not_specialize

        self.params = []
        for i, param in enumerate(self.signature.parameters.values()):
            dns = do_not_specialize and (i in do_not_specialize or param.name in do_not_specialize)
            self.params.append(KernelParam(i, param, dns))

        # function source code (without decorators)
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find("def") :]
        # cache of just-in-time compiled kernels
        self.cache = defaultdict(dict)
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

    @property
    def cache_key(self):
        # TODO : hash should be attribute of `self`
        if self.hash is None:
            dependencies_finder = DependenciesFinder(globals=self.__globals__, src=self.src)
            dependencies_finder.visit(self.parse())
            self.hash = dependencies_finder.ret
        return self.hash

    def warmup_with_flags(self, flags: dict[str, Any], kernel_args):
        # TODO(jlebar): Why are we not wrapping dtype in kwargs?
        assert flags.get("warmup", True), "Cannot call warmup_with_flags with warmup=False; that's a contradiction."
        return self.run_with_flags(
            grid=None,
            flags={**flags, "warmup": True},
            kernel_args=kernel_args,
        )

    # we do not parse `src` in the constructor because
    # the user might want to monkey-patch self.src dynamically.
    # Our unit tests do this, for example.
    def parse(self):
        tree = ast.parse(self.src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Cannot call @triton.jit'd outside of the scope of a kernel")

    def __setattr__(self, name, value):
        super(JITFunction, self).__setattr__(name, value)
        # - when `.src` attribute is set, cache path needs
        #   to be reinitialized
        if name == "src":
            self.hash = None

    def __repr__(self):
        return f"JITFunction({self.module}:{self.fn.__name__})"


# -----------------------------------------------------------------------------
# `jit` decorator
# -----------------------------------------------------------------------------


@overload
def jit(fn: KernelT) -> JITFunction[KernelT]:
    ...


@overload
def jit(
    *,
    version=None,
    do_not_specialize: Optional[Sequence[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[KernelT], JITFunction[KernelT]]:
    ...


def jit(
    fn: Optional[KernelT] = None,
    *,
    version=None,
    do_not_specialize: Optional[Sequence[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[JITFunction[KernelT], Callable[[KernelT], JITFunction[KernelT]]]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

    def decorator(fn: KernelT) -> JITFunction[KernelT] | InterpretedFunction[KernelT]:
        assert callable(fn)
        if os.getenv("TRITON_INTERPRET", "0") == "1":
            return InterpretedFunction(fn)
        else:
            return JITFunction(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                debug=debug,
                noinline=noinline,
            )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator


# -----------------------------------------------------------------------------
# Utilities for mocking tensors
# -----------------------------------------------------------------------------


class MockTensor:
    """
    Can be used in place of real tensors when calling:
        kernel.warmup(MockTensor(torch.float32), ...)
    """

    @staticmethod
    def wrap_dtype(arg):
        if arg.__class__.__name__ == "dtype" and arg.__module__ == "torch":
            return MockTensor(arg)
        return arg

    def __init__(self, dtype):
        self.dtype = dtype

    @staticmethod
    def data_ptr():
        return 0  # optimistically assumes multiple of 16


class TensorWrapper:
    def __init__(self, base, dtype):
        self.dtype = dtype
        self.base = base
        self.is_cuda = base.is_cuda
        self.device = base.device
        self.shape = self.base.shape

    def data_ptr(self):
        return self.base.data_ptr()

    def stride(self, i):
        return self.base.stride(i)

    def __str__(self) -> str:
        return f"TensorWrapper[{self.dtype}]({self.base})"

    def element_size(self):
        return self.base.element_size()


def reinterpret(tensor, dtype):
    if isinstance(tensor, TensorWrapper):
        if dtype == tensor.base.dtype:
            # Reinterpreting to the original interpretation; return the base.
            return tensor.base
        else:
            # Reinterpreting a wrapped tensor to a different type.
            return TensorWrapper(tensor.base, dtype)
    elif hasattr(tensor, "data_ptr"):
        # A new wrapper is needed around an unwrapped tensor.
        return TensorWrapper(tensor, dtype)
    else:
        raise TypeError(f"Cannot reinterpret a {type(tensor)}.")
