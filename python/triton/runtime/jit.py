from __future__ import annotations, division

import ast
import functools
import hashlib
import inspect
import os
import textwrap
from collections import defaultdict, namedtuple
from functools import cached_property
from typing import Callable, Generic, Iterable, List, Optional, TypeVar, Union, cast, overload

from .._C.libtriton.triton import TMAInfos
from ..common.backend import get_backend, get_cuda_version_key
from .interpreter import InterpretedFunction
from ..runtime.driver import driver

T = TypeVar("T")

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
        if lhs is None or (getattr(lhs, "__name__", "") == "triton"
                           or getattr(lhs, "__name__", "").endswith(".triton")):
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
        func_cache_key = func.cache_key
        noinline = str(getattr(func, "noinline", False))
        self.ret = (self.ret + func_cache_key + noinline).encode("utf-8")
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
            return (self.value.data_ptr() % JITFunction.divisibility == 0, )
        except AttributeError:
            pass

        if isinstance(self.value, int):
            # bool is a subclass of int, so we don't check explicitly above.
            return (
                self.value % JITFunction.divisibility == 0,
                self.value % JITFunction.divisibility_8 == 0,
                self.value == 1,
            )

        return (False, )


class KernelInterface(Generic[T]):
    run: T

    def __getitem__(self, grid) -> T:
        """
        A JIT function is launched with: fn[grid](*args, **kwargs).
        Hence JITFunction.__getitem__ returns a callable proxy that
        memorizes the grid.
        """
        return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
        # return cast(T, functools.partial(cast(Callable, self.run), grid=grid))


class JITFunction(KernelInterface[T]):
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
        return (arg is None, )

    # TODO(jlebar): Fold this into the KernelArg class.
    def _get_config(self, *args):
        from ..compiler import AttrsDescriptor

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
            param.num
            for param, arg in zip(self.params, args)
            if is_divisible_by_8(arg) and not param.do_not_specialize
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
        return AttrsDescriptor(tuple(divisible_by_16), tuple(equal_to_1), tuple(ids_of_folded_args),
                               tuple(divisible_by_8))
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

    def run(self, *args, grid, warmup, **kwargs):
        from ..compiler import CompiledKernel, compile, ASTSource
        from ..compiler.backends.cuda import CUDABackend
        # deprecated arguments
        assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
        assert "device" not in kwargs, "device option is deprecated; current device will be used"
        assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"
        # parse options
        device = driver.get_current_device()
        stream = driver.get_current_stream(device)
        target = driver.get_current_target()
        backend = CUDABackend(target)
        kwargs["debug"] = self.debug
        options = backend.parse_options(kwargs)
        # bind non-reserved keyword args and set defaults
        kwargs = {k: v for k, v in kwargs.items() if not k in options.__dict__}
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        assert len(bound_args.arguments) == len(self.params)
        # canonicalize grid
        assert grid is not None
        if callable(grid):
            # Arguments are passed as a dict to `grid`, by contract.
            # TODO(jlebar): In the new launch API, pass the compiler flags as a
            # second parameter to `grid`.
            grid = grid(dict(bound_args.arguments))
        grid_size = len(grid)
        grid_0 = grid[0]
        grid_1 = grid[1] if grid_size > 1 else 1
        grid_2 = grid[2] if grid_size > 2 else 1
        # compute cache key
        args = [KernelArg(arg_value, param) for (_, arg_value), param in zip(bound_args.arguments.items(), self.params)]
        sig_key = tuple(arg.signature_key() for arg in args if not arg.param.is_constexpr)
        spec_key = tuple(arg.specialization_key() for arg in args if not arg.param.do_not_specialize)
        constexpr_key = tuple(arg.value for arg in args if arg.param.is_constexpr)
        key = (get_cuda_version_key(), sig_key, constexpr_key, spec_key, options)
        # Kernel is not cached; we have to compile.
        if key not in self.cache[device]:
            configs = (self._get_config(*[arg.value for arg in args]), )
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
                arg.param.num: self._type_of(self._key_of(arg.value))
                for arg in args
                if not arg.param.is_constexpr
            }

            if self._call_hook(key, signature, device, constants, options.num_warps, options.num_ctas,
                               options.num_stages, options.enable_warp_specialization, options.enable_fp_fusion,
                               options.extern_libs, configs):
                return None
            # compile the kernel
            src = ASTSource(self, signature, constants, configs[0])
            self.cache[device][key] = compile(
                src,
                target=target,
                options=options.__dict__,
            )

        kernel = self.cache[device][key]
        if not warmup:
            args = [arg.value for arg in args if not arg.param.is_constexpr]
            kernel.run(grid_0, grid_1, grid_2, kernel.num_warps, kernel.num_ctas,  # number of warps/ctas per instance
                       kernel.cluster_dims[0], kernel.cluster_dims[1], kernel.cluster_dims[2],  # cluster
                       kernel.shared, stream, kernel.function, CompiledKernel.launch_enter_hook,
                       CompiledKernel.launch_exit_hook, kernel,
                       *driver.assemble_tensormap_to_arg(kernel.metadata["tensormaps_info"], args))
        return kernel

    def __init__(self, fn, version=None, do_not_specialize=None, debug=None, noinline=None):
        do_not_specialize = do_not_specialize if do_not_specialize else []

        self.fn = fn
        self.module = fn.__module__
        self.version = version
        self.signature = inspect.signature(fn)
        self.do_not_specialize = do_not_specialize
        self.starting_line_number = inspect.getsourcelines(fn)[1]

        self.params = []
        for i, param in enumerate(self.signature.parameters.values()):
            dns = do_not_specialize and (i in do_not_specialize or param.name in do_not_specialize)
            self.params.append(KernelParam(i, param, dns))

        # function source code (without decorators)
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find("def"):]
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
            self.hash = dependencies_finder.ret + str(self.starting_line_number)
        return self.hash

    def warmup(self, *args, grid, **kwargs):
        return self.run(grid=grid, warmup=True, *map(MockTensor.wrap_dtype, args), **kwargs)

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
def jit(fn: T) -> JITFunction[T]:
    ...


@overload
def jit(
    *,
    version=None,
    do_not_specialize: Optional[Iterable[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[T], JITFunction[T]]:
    ...


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    do_not_specialize: Optional[Iterable[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[JITFunction[T], Callable[[T], JITFunction[T]]]:
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

    def decorator(fn: T) -> JITFunction[T]:
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
