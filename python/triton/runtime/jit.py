from __future__ import annotations, division

import ast
import functools
import hashlib
import inspect
import os
import subprocess
import textwrap
from collections import defaultdict, namedtuple
from typing import (Callable, Generic, Iterable, List, Optional, TypeVar, Union, cast,
                    overload)

from .._C.libtriton.triton import TMAInfos
from ..common.backend import get_backend, path_to_ptxas
from ..language.core import dtype
from .interpreter import InterpretedFunction

TRITON_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRITON_VERSION = "2.1.0"


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


T = TypeVar('T')

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
        if lhs is None or (getattr(lhs, "__name__", "") == "triton" or getattr(lhs, "__name__", "").endswith(".triton")):
            return None
        return getattr(lhs, node.attr)

    def visit_Call(self, node):
        func = self.visit(node.func)
        if func is None:
            return
        if inspect.isbuiltin(func):
            return
        if func.__module__ and (func.__module__.startswith('triton.') or '.triton.' in func.__module__):
            return
        assert isinstance(func, JITFunction), f"Function \"{func.__name__}\" is being called from a Triton function but is not a Triton function itself. Decorate it with @triton.jit to fix this"
        if func.hash is None:
            tree = ast.parse(func.src)
            finder = DependenciesFinder(func.__globals__, func.src)
            finder.visit(tree)
            func.hash = finder.ret
        noinline = str(getattr(func, 'noinline', False))
        self.ret = (self.ret + func.hash + noinline).encode("utf-8")
        self.ret = hashlib.sha1(self.ret).hexdigest()

# -----------------------------------------------------------------------------
# JITFunction
# -----------------------------------------------------------------------------


@functools.lru_cache()
def version_key():
    import pkgutil
    contents = []
    # frontend
    with open(__file__, "rb") as f:
        contents += [hashlib.sha1(f.read()).hexdigest()]
    # compiler
    compiler_path = os.path.join(TRITON_PATH, 'compiler')
    for lib in pkgutil.iter_modules([compiler_path]):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
            contents += [hashlib.sha1(f.read()).hexdigest()]
    # backend
    libtriton_hash = hashlib.sha1()
    with open(os.path.join(TRITON_PATH, "_C/libtriton.so"), "rb") as f:
        while True:
            chunk = f.read(1024 ** 2)
            if not chunk:
                break
            libtriton_hash.update(chunk)
    contents.append(libtriton_hash.hexdigest())
    # language
    language_path = os.path.join(TRITON_PATH, 'language')
    for lib in pkgutil.iter_modules([language_path]):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
            contents += [hashlib.sha1(f.read()).hexdigest()]
    # ptxas version
    ptxas = path_to_ptxas()[0]
    ptxas_version = hashlib.sha1(subprocess.check_output([ptxas, "--version"])).hexdigest()
    return '-'.join(TRITON_VERSION) + '-' + ptxas_version + '-' + '-'.join(contents)


def _normalize_ty(ty) -> str:
    if isinstance(ty, type):
        return ty.__name__
    elif isinstance(ty, str):
        return ty
    return repr(ty)


class KernelInterface(Generic[T]):
    run: T

    def __getitem__(self, grid) -> T:
        """
        A JIT function is launched with: fn[grid](*args, **kwargs).
        Hence JITFunction.__getitem__ returns a callable proxy that
        memorizes the grid.
        """
        return cast(T, functools.partial(cast(Callable, self.run), grid=grid))


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
            if -2**31 <= arg and arg <= 2**31 - 1:
                return "i32"
            elif 2**63 <= arg and arg <= 2**64 - 1:
                return "u64"
            else:
                return "i64"
        elif isinstance(arg, float):
            return 'fp32'
        elif arg is None:
            return None
        else:
            raise TypeError(f'Unsupported type {type(arg)} for {arg}')

    @staticmethod
    def _device_of(arg):
        if hasattr(arg, "device"):
            if hasattr(arg.device, 'type'):
                return arg.device.type

        return ''

    @staticmethod
    def _pinned_memory_of(arg):
        if hasattr(arg, "is_pinned"):
            if isinstance(arg.is_pinned, Callable):
                return arg.is_pinned()

        return False

    @staticmethod
    def _spec_of(arg):
        if hasattr(arg, "data_ptr"):
            return (arg.data_ptr() % JITFunction.divisibility == 0)
        elif isinstance(arg, int):
            return (arg % 16 == 0, arg == 1)
        return (arg is None, )

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
        divisible_by_16 = {i for i, arg in enumerate(
            args) if is_divisible_by_16(arg) and i not in self.do_not_specialize}
        divisible_by_8 = {i for i, arg in enumerate(
            args) if is_divisible_by_8(arg) and i not in self.do_not_specialize}
        equal_to_1 = {
            i for i, arg in enumerate(args) if isinstance(
                arg, int) and not isinstance(
                arg, bool) and arg == 1 and i not in self.do_not_specialize}
        # folded equal_to_1 and None
        # TODO: method to collect all folded args
        none_args = {i for i, arg in enumerate(args) if arg is None and i not in self.do_not_specialize}
        ids_of_folded_args = equal_to_1 | none_args
        return namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1", "ids_of_folded_args", "divisible_by_8"])(
            tuple(divisible_by_16), tuple(equal_to_1), tuple(ids_of_folded_args), tuple(divisible_by_8))
        # return _triton.code_gen.instance_descriptor(divisible_by_16,
        # equal_to_1)

    @staticmethod
    def _type_of(key):
        # None are nullptr -- implicitly converted to *i8
        if key is None:
            return '*i8'
        dtype_str = str(key).split(".")[-1]
        tys = {
            "bool": "i1",
            "float8e4nv": "fp8e4nv",
            "float8_e4m3fn": "fp8e4nv",
            "float8e4b8": "fp8e4b8",
            "float8_e4m3fnuz": "fp8e4b8",
            "float8e5": "fp8e5",
            "float8_e5m2": "fp8e5",
            "float8e5b16": "fp8e5b16",
            "float8_e5m2fnuz": "fp8e5b16",
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

<<<<<<< HEAD
    def _call_hook(self, key, signature, device, constants, num_warps, num_ctas, num_stages, waves_per_eu, matrix_instr_nonkdim, enable_warp_specialization, extern_libs, configs):
=======
    def _call_hook(self, key, signature, device, constants, num_warps, num_ctas, num_stages, enable_warp_specialization, enable_fp_fusion, extern_libs, configs):
>>>>>>> 721897fcc4f942aa97d2e9ba3787a5e213758177
        if JITFunction.cache_hook is None:
            return False
        name = self.fn.__name__
        module = self.fn.__module__
        arg_reprs = ', '.join([f'{name}: {ty}' for name, ty in zip(self.arg_names, key[1])])
<<<<<<< HEAD
        repr = f"{name}[num_warps={num_warps}, num_ctas={num_ctas}, num_stages={num_stages}, waves_per_eu={waves_per_eu}, matrix_instr_nonkdim={matrix_instr_nonkdim}, enable_warp_specialization={enable_warp_specialization}]({arg_reprs})"
=======
        repr = f"{name}[num_warps={num_warps}, num_ctas={num_ctas}, num_stages={num_stages}, enable_warp_specialization={enable_warp_specialization}, enable_fp_fusion={enable_fp_fusion}]({arg_reprs})"
>>>>>>> 721897fcc4f942aa97d2e9ba3787a5e213758177
        key = str(key)

        class LegacyCompiler:
            def __init__(self, module, name):
                self.module = module
                self.name = name
                pass

        kwargs = dict(signature=signature, device=device, constants=constants,
<<<<<<< HEAD
                      num_warps=num_warps, num_ctas=num_ctas, num_stages=num_stages, waves_per_eu=waves_per_eu, enable_warp_specialization=enable_warp_specialization, extern_libs=extern_libs,
=======
                      num_warps=num_warps, num_ctas=num_ctas, num_stages=num_stages, enable_warp_specialization=enable_warp_specialization, enable_fp_fusion=enable_fp_fusion, extern_libs=extern_libs,
>>>>>>> 721897fcc4f942aa97d2e9ba3787a5e213758177
                      configs=configs)

        return JITFunction.cache_hook(key=key, repr=repr, fn=LegacyCompiler(module, name), compile={
                                      "key": key, **kwargs}, is_manual_warmup=False, already_compiled=False)

    def _get_arg_specialization_key(self, arg_name, arg):
        arg_annotation = self.__annotations__.get(arg_name, '')
        if arg_annotation == '':
            return (arg.data_ptr() % JITFunction.divisibility == 0) if hasattr(arg, "data_ptr") \
                else (arg % JITFunction.divisibility == 0, arg % JITFunction.divisibility_8 == 0, arg == 1) if isinstance(arg, int) \
                else (False,)
        elif 'Tensor' in arg_annotation:
            return (arg.data_ptr() % JITFunction.divisibility == 0)
        elif 'int' in arg_annotation or 'bool' in arg_annotation:
            return (arg % JITFunction.divisibility == 0, arg % JITFunction.divisibility_8 == 0, arg == 1)
        else:
            return (False,)

    def _get_arg_sig_key(self, arg_name, arg) -> str:
        arg_annotation = self.__annotations__.get(arg_name, '')
        if 'Tensor' in arg_annotation:
            return arg.dtype
        elif arg_annotation == 'bool':
            return "i1"
        elif arg_annotation == 'float':
            return 'fp32'
        else:
            return self._key_of(arg)

    def _conclude_device_type(self, device_types: List[str], pinned_memory_flags: List[bool]) -> str:
        device_types = [device_type for device_type in device_types if device_type != '']
        # Return cuda if one of the input tensors is cuda
        if 'cuda' in device_types:
            import torch
            return 'hip' if torch.version.hip else 'cuda'

        is_cpu = all(device_type == 'cpu' for device_type in device_types)
        is_pinned_memory = any(pinned_memory_flag for pinned_memory_flag in pinned_memory_flags)
        # Return cuda if all the input tensors are cpu while the memory is pinned
        if is_cpu and is_pinned_memory:
            return 'cuda'

        return device_types[0] if len(device_types) > 0 else 'cuda'

    def _make_launcher(self):
        regular_args = [arg for i, arg in enumerate(
            self.arg_names) if i not in self.constexprs]
        constexpr_args = [arg for i, arg in enumerate(
            self.arg_names) if i in self.constexprs]

        def regular_args_v(args_proxy):
            return [args_proxy[arg_name] for arg_name in regular_args]

<<<<<<< HEAD
        def launcher_body(args_proxy, grid, num_warps, num_ctas, num_stages, waves_per_eu, matrix_instr_nonkdim, enable_warp_specialization, extern_libs, stream, warmup, device, device_type):
=======
        def launcher_body(args_proxy, grid, num_warps, num_ctas, num_stages, enable_warp_specialization, enable_fp_fusion, extern_libs, stream, warmup, device, device_type):
>>>>>>> 721897fcc4f942aa97d2e9ba3787a5e213758177
            from ..compiler import (CompiledKernel, compile,
                                    get_arch_default_num_stages,
                                    get_arch_default_num_warps)
            sig_key = tuple([self._get_arg_sig_key(arg_name, args_proxy[arg_name]) for arg_name in regular_args])
            constexpr_key = tuple([args_proxy[arg_name] for arg_name in constexpr_args])
            specializations = []
            for i, arg_name in enumerate(regular_args):
                if i in self.do_not_specialize:
                    continue
                specializations += [self._get_arg_specialization_key(arg_name, args_proxy[arg_name])]

            spec_key = tuple(specializations)
            assert num_ctas > 0
            assert grid is not None
            if callable(grid):
                grid = grid(args_proxy)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1
            if device_type is None:
                device_types = [self._device_of(arg) for arg in regular_args_v(args_proxy)]
                device_types = [_device_type for _device_type in device_types if _device_type != '']
                device_type = self._conclude_device_type(device_types, [self._pinned_memory_of(arg) for arg in
                                                                        regular_args_v(args_proxy)])

            device_backend = None
            if device_type not in ['cuda']:
                device_backend = get_backend(device_type)
                if device_backend is None:
                    raise ValueError('Cannot find backend for ' + device_type)

            if device is None:
                if device_type in ['cuda']:
                    device = get_current_device()
                    set_current_device(device)
                else:
                    device = device_backend.get_current_device()
                    device_backend.set_current_device(device)
            if stream is None and not warmup:
                if device_type in ['cuda']:
                    stream = get_cuda_stream(device)
                else:
                    stream = device_backend.get_stream()

            if num_warps is None:
                num_warps = get_arch_default_num_warps(device_type)
            if num_stages is None:
                num_stages = get_arch_default_num_stages(device_type)

<<<<<<< HEAD
            key = (version_key, sig_key, constexpr_key, spec_key, num_warps, num_ctas, num_stages, waves_per_eu, matrix_instr_nonkdim, enable_warp_specialization, self.debug)
=======
            key = (version_key(), sig_key, constexpr_key, spec_key, num_warps, num_ctas, num_stages, enable_warp_specialization, enable_fp_fusion, self.debug)
>>>>>>> 721897fcc4f942aa97d2e9ba3787a5e213758177
            if extern_libs is not None:
                key = (key, tuple(extern_libs.items()))

            bin = self.cache[device].get(key, None)
            if bin is not None:
                # build dict of constant values
                args = regular_args_v(args_proxy)
                # Create tensormaps and append to args
                args = bin.assemble_tensormap_to_arg(args)
                if not warmup:
                    bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps, bin.num_ctas, bin.clusterDims[0], bin.clusterDims[1], bin.clusterDims[2], bin.shared, stream, bin.cu_function, CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, bin, *args)
                return bin
            # kernel not cached -- compile
            else:
                # build dict of constant values
                args = regular_args_v(args_proxy)
                all_args = tuple([args_proxy[arg_name] for arg_name in self.arg_names])
                configs = self._get_config(*all_args),
                constants = self._make_constants(constexpr_key)
                constants.update({i: None for i, arg in enumerate(all_args) if arg is None})
                constants.update({i: 1 for i in configs[0].equal_to_1})
                # build kernel signature -- doesn't include specialized arguments
                signature = {i: self._type_of(self._key_of(arg)) for i, arg in enumerate(all_args) if i not in self.constexprs}
                # build stub signature -- includes arguments that are specialized
                for i, arg in constants.items():
                    if callable(arg):
                        raise TypeError(f"Callable constexpr at index {i} is not supported")
<<<<<<< HEAD
                if not self._call_hook(key, signature, device, constants, num_warps, num_ctas, num_stages, waves_per_eu, matrix_instr_nonkdim, enable_warp_specialization, extern_libs, configs):
                    bin = compile(self, signature=signature, device=device, constants=constants, num_warps=num_warps, num_ctas=num_ctas, num_stages=num_stages, waves_per_eu=waves_per_eu, matrix_instr_nonkdim=matrix_instr_nonkdim, enable_warp_specialization=enable_warp_specialization, extern_libs=extern_libs, configs=configs, debug=self.debug, device_type=device_type)
=======
                if not self._call_hook(key, signature, device, constants, num_warps, num_ctas, num_stages, enable_warp_specialization, enable_fp_fusion, extern_libs, configs):
                    bin = compile(self, signature=signature, device=device, constants=constants, num_warps=num_warps, num_ctas=num_ctas, num_stages=num_stages, enable_warp_specialization=enable_warp_specialization, enable_fp_fusion=enable_fp_fusion, extern_libs=extern_libs, configs=configs, debug=self.debug, device_type=device_type)
>>>>>>> 721897fcc4f942aa97d2e9ba3787a5e213758177
                    # Create tensormaps and append to args
                    args = bin.assemble_tensormap_to_arg(args)
                    if not warmup:
                        bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps, bin.num_ctas, bin.clusterDims[0], bin.clusterDims[1], bin.clusterDims[2], bin.shared, stream, bin.cu_function, CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, bin, *args)
                    self.cache[device][key] = bin
                    return bin
                return None

        # create a wrapper to call launcher_body
        args_map = ','.join([f'"{arg}": {arg}' for arg in self.arg_names])
        args_signature = ', '.join(name if dflt == inspect._empty else f'{name} = triton.language.dtype(\'{dflt}\')' if dtype.is_dtype(f'{dflt}') else f'{name} = {dflt}' for name, dflt in zip(self.arg_names, self.arg_defaults))
        args_signature = args_signature + ', ' if len(args_signature) > 0 else ''
        src = f"""
import triton
<<<<<<< HEAD
def {self.fn.__name__}({args_signature}grid=None, num_warps=None, num_ctas=1, num_stages=None, waves_per_eu=0, matrix_instr_nonkdim=0, enable_warp_specialization=False, extern_libs=None, stream=None, warmup=False, device=None, device_type=None):
    return launcher_body({{{args_map}}}, grid, num_warps, num_ctas, num_stages, waves_per_eu, matrix_instr_nonkdim, enable_warp_specialization, extern_libs, stream, warmup, device, device_type)
=======
def {self.fn.__name__}({args_signature}grid=None, num_warps=None, num_ctas=1, num_stages=None, enable_warp_specialization=False, enable_fp_fusion=True, extern_libs=None, stream=None, warmup=False, device=None, device_type=None):
    return launcher_body({{{args_map}}}, grid, num_warps, num_ctas, num_stages, enable_warp_specialization, enable_fp_fusion, extern_libs, stream, warmup, device, device_type)
>>>>>>> 721897fcc4f942aa97d2e9ba3787a5e213758177
"""
        scope = {"launcher_body": launcher_body}
        exec(src, scope)
        return scope[self.fn.__name__]

    def __init__(self, fn, version=None, do_not_specialize=None, debug=None, noinline=None):
        self.fn = fn
        self.module = fn.__module__
        self.version = version
        # function signature information
        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]
        self.arg_defaults = [v.default for v in signature.parameters.values()]
        self.has_defaults = any(v != inspect._empty for v in self.arg_defaults)
        # function source code (without decorators)
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find("def"):]
        # cache of just-in-time compiled kernels
        self.cache = defaultdict(dict)
        self.hash = None
        # JITFunction can be instantiated as kernel
        # when called with a grid using __getitem__
        self.kernel_decorators = []
        self.kernel = None
        self.debug = True if os.environ.get("TRITON_DEBUG", "0") == "1" else debug
        self.noinline = noinline
        # annotations
        self.__annotations__ = {name: _normalize_ty(ty) for name, ty in fn.__annotations__.items()}
        # index of constexprs
        self.constexprs = [self.arg_names.index(name) for name, ty in self.__annotations__.items() if 'constexpr' in ty]
        # specialization hints
        regular_args = [arg for i, arg in enumerate(self.arg_names) if i not in self.constexprs]
        self.do_not_specialize = [] if do_not_specialize is None else do_not_specialize
        self.do_not_specialize = {regular_args.index(arg) if isinstance(arg, str) else arg for arg in self.do_not_specialize}
        # tma info
        self.tensormaps_info = TMAInfos()
        # launcher
        self.run = self._make_launcher()
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
            self.hash = dependencies_finder.ret + version_key()
        return self.hash

    def warmup(self, *args, **kwargs):
        return self.run(*map(MockTensor.wrap_dtype, args), **kwargs, warmup=True)

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
        # - when kernel decorators change, cached kernel
        #   needs to be cleared
        if name == 'kernel_decorators':
            self.kernel = None
        super(JITFunction, self).__setattr__(name, value)
        # - when `.src` attribute is set, cache path needs
        #   to be reinitialized
        if name == 'src':
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
        if arg.__class__.__name__ == "dtype" and\
           arg.__module__ == "torch":
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
        return f'TensorWrapper[{self.dtype}]({self.base})'

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
        raise TypeError(f'Cannot reinterpret a {type(tensor)}.')
