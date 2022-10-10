from __future__ import annotations, division

import ast
import functools
import hashlib
import inspect
import os
import subprocess
import textwrap
from collections import defaultdict, namedtuple

import torch

import triton
from triton.utils import MockTensor

try:
    from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
except ImportError:
    get_cuda_stream = lambda dev_idx: torch.cuda.current_stream(dev_idx).cuda_stream

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
        self.ret = hashlib.md5(src.encode("utf-8")).hexdigest()
        self.globals = globals

    def visit_Name(self, node):
        return self.globals.get(node.id, None)

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        while isinstance(lhs, ast.Attribute):
            lhs = self.visit(lhs.value)
        if lhs is None or lhs is triton:
            return None
        return getattr(lhs, node.attr)

    def visit_Call(self, node):
        func = self.visit(node.func)
        if func is None:
            return
        if inspect.isbuiltin(func):
            return
        if func.__module__ and func.__module__.startswith('triton.'):
            return
        assert isinstance(func, JITFunction)
        if func.hash is None:
            tree = ast.parse(func.src)
            finder = DependenciesFinder(func.__globals__, func.src)
            finder.visit(tree)
            func.hash = finder.ret
        self.ret = (self.ret + func.hash).encode("utf-8")
        self.ret = hashlib.md5(self.ret).hexdigest()

# -----------------------------------------------------------------------------
# JITFunction
# -----------------------------------------------------------------------------


@functools.lru_cache()
def version_key():
    import pkgutil
    contents = []
    # frontend
    with open(__file__, "rb") as f:
        contents += [hashlib.md5(f.read()).hexdigest()]
    with open(triton.compiler.__file__, "rb") as f:
        contents += [hashlib.md5(f.read()).hexdigest()]
    # backend
    with open(triton._C.libtriton.__file__, "rb") as f:
        contents += [hashlib.md5(f.read()).hexdigest()]
    # language
    language_path = os.path.join(*triton.__path__, 'language')
    for lib in pkgutil.iter_modules([language_path]):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
            contents += [hashlib.md5(f.read()).hexdigest()]
    # ptxas version
    try:
        ptxas_version = hashlib.md5(subprocess.check_output(["ptxas", "--version"])).hexdigest()
    except Exception:
        ptxas_version = ''
    return '-'.join(triton.__version__) + '-' + ptxas_version + '-' + '-'.join(contents)


class KernelInterface:

    def __getitem__(self, grid):
        """
        A JIT function is launched with: fn[grid](*args, **kwargs).
        Hence JITFunction.__getitem__ returns a callable proxy that
        memorizes the grid.
        """
        def launcher(*args, **kwargs):
            return self.run(*args, grid=grid, **kwargs)
        return launcher


class JITFunction(KernelInterface):

    # Hook for inspecting compiled functions and modules
    cache_hook = None
    divisibility = 16

    @staticmethod
    def _key_of(arg):
        if hasattr(arg, "dtype"):
            return arg.dtype
        elif isinstance(arg, bool):
            return "i1"
        elif isinstance(arg, int):
            if -2**31 <= arg and arg <= 2**31 - 1:
                return "i32"
            elif 2**31 <= arg and arg <= 2**32 - 1:
                return "u32"
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
        divisible_by_16 = [i for i, arg in enumerate(args) if is_divisible_by_16(arg) and i not in self.do_not_specialize]
        equal_to_1 = [i for i, arg in enumerate(args) if isinstance(arg, int) and arg == 1 and i not in self.do_not_specialize]
        return namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"])(tuple(divisible_by_16), tuple(equal_to_1))
        # return _triton.code_gen.instance_descriptor(divisible_by_16, equal_to_1)

    @staticmethod
    def _type_of(key):
        if isinstance(key, (torch.dtype, triton.language.dtype)):
            ty = {
                torch.bool: 'i1',
                torch.float16: 'fp16',
                torch.bfloat16: 'bf16',
                torch.float32: 'fp32',
                torch.float64: 'fp64',
                torch.uint8: 'u8',
                torch.int8: 'i8',
                torch.int16: 'i16',
                torch.int32: 'i32',
                torch.int64: 'i64',

                triton.language.uint8: 'u8',
                triton.language.uint16: 'u16',
                triton.language.uint32: 'u32',
                triton.language.uint64: 'u64',
                triton.language.float8: 'fp8',
            }[key]
            return f'*{ty}'
        if key is None:
            return '*i8'
        assert isinstance(key, str)
        return key

    def _make_signature(self, sig_key):
        signature = ",".join([self._type_of(k) for i, k in enumerate(sig_key)])
        return signature

    def _make_constants(self, constexpr_key):
        constants = {i: k for i, k in zip(self.constexprs, constexpr_key)}
        return constants

    def _call_hook(self, key, signature, device, constants, num_warps, num_stages, extern_libs, configs):
        if JITFunction.cache_hook is None:
            return False
        name = self.fn.__name__
        module = self.fn.__module__
        arg_reprs = ', '.join([f'{name}: {ty}' for name, ty in zip(self.arg_names, key[1])])
        repr = f"{name}[num_warps={num_warps}, num_stages={num_stages}]({arg_reprs})"
        key = str(key)

        class LegacyCompiler:
            def __init__(self, module, name):
                self.module = module
                self.name = name
                pass

        kwargs = dict(signature=signature, device=device, constants=constants,
                      num_warps=num_warps, num_stages=num_stages, extern_libs=extern_libs,
                      configs=configs)

        return JITFunction.cache_hook(key=key, repr=repr, fn=LegacyCompiler(module, name), compile={"key": key, **kwargs}, is_manual_warmup=False, already_compiled=False)

    def _make_launcher(self):
        regular_args = [f'{arg}' for i, arg in enumerate(self.arg_names) if i not in self.constexprs]
        constexpr_args = [f'{arg}' for i, arg in enumerate(self.arg_names) if i in self.constexprs]
        args = ', '.join(regular_args)
        # cache key for regular argument type
        sig_keys = ', '.join([f'_key_of({arg})' for arg in regular_args])
        # cache key for constexpr argument values
        constexpr_keys = ', '.join(constexpr_args)
        # cache key for argument specialization
        specializations = []
        for i, arg in enumerate(regular_args):
            if i in self.do_not_specialize:
                continue
            specializations += [f'({arg}.data_ptr() % {JITFunction.divisibility} == 0) if hasattr({arg}, "data_ptr") '
                                f'else ({arg} % {JITFunction.divisibility} == 0, {arg} == 1) if isinstance({arg}, int) '
                                f'else (False,)']
        spec_keys = ', '.join(specializations)
        grid_args = ','.join([f'"{arg}": {arg}' for arg in self.arg_names])

        src = f"""
def {self.fn.__name__}({', '.join(self.arg_names)}, grid, num_warps=4, num_stages=3, extern_libs=None, stream=None, warmup=False):
    sig_key =  {sig_keys},
    constexpr_key = {f'{constexpr_keys},' if len(constexpr_keys) > 0 else tuple()}
    spec_key = {f'{spec_keys},' if len(spec_keys) > 0 else tuple()}
    key = (version_key, sig_key, constexpr_key, spec_key)
    if not extern_libs is None:
      key = (key, tuple(extern_libs.items()))
    assert num_warps > 0 and (num_warps & (num_warps - 1)) == 0, "num_warps must be a power of 2"
    if callable(grid):
        grid = grid({{{grid_args}}})
    grid_size = len(grid)
    grid_0 = grid[0]
    grid_1 = grid[1] if grid_size > 1 else 1
    grid_2 = grid[2] if grid_size > 2 else 1
    device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    if stream is None and not warmup:
      stream = get_cuda_stream(device)
    try:
      bin = cache[device][key]
      if not warmup:
          bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps, bin.shared, stream, bin.cu_function, triton.compiler.CompiledKernel.launch_enter_hook, triton.compiler.CompiledKernel.launch_exit_hook, bin, {args})
      return bin
    # kernel not cached -- compile
    except KeyError:
      # build dict of constant values
      args = [{args}]
      all_args = {', '.join([f'{arg}' for arg in self.arg_names])},
      configs = self._get_config(*all_args),
      constants = self._make_constants(constexpr_key)
      constants.update({{i: None for i, arg in enumerate(all_args) if arg is None}})
      constants.update({{i: 1 for i in configs[0].equal_to_1}})
      # build kernel signature -- doesn't include specialized arguments
      signature = {{ i: self._type_of(_key_of(arg)) for i, arg in enumerate(all_args) if i not in self.constexprs }}
      # build stub signature -- includes arguments that are specialized
      for i, arg in constants.items():
        if callable(arg):
          raise TypeError(f"Callable constexpr at index {i} is not supported")
      if not self._call_hook(key, signature, device, constants, num_warps, num_stages, extern_libs, configs):
        bin = triton.compile(self, signature, device, constants, num_warps, num_stages, extern_libs=extern_libs, configs=configs)
        if not warmup:
            bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps, bin.shared, stream, bin.cu_function, triton.compiler.CompiledKernel.launch_enter_hook, triton.compiler.CompiledKernel.launch_exit_hook, bin, *args)
        self.cache[device][key] = bin
        return bin
      return None
"""
        scope = {"version_key": version_key(), "get_cuda_stream": get_cuda_stream,
                 "self": self, "_spec_of": self._spec_of, "_key_of": self._key_of,
                 "cache": self.cache, "triton": triton, "torch": torch}
        exec(src, scope)
        return scope[self.fn.__name__]

    def __init__(self, fn, version=None, do_not_specialize=None):
        self.fn = fn
        self.module = fn.__module__
        self.version = version
        # function signature information
        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]
        self.has_defaults = any([v.default != inspect._empty for v in signature.parameters.values()])
        # specialization hints
        self.do_not_specialize = [] if do_not_specialize is None else do_not_specialize
        self.do_not_specialize = set([self.arg_names.index(arg) if isinstance(arg, str) else arg for arg in self.do_not_specialize])
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
        # annotations
        self.annotations = {self.arg_names.index(name): ty for name, ty in fn.__annotations__.items()}
        self.__annotations__ = fn.__annotations__
        # index of constexprs
        self.constexprs = [self.arg_names.index(ann) for ann in self.__annotations__.keys()]
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


def jit(*args, **kwargs):
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, :code:`torch.tensor` arguments are implicitly converted to pointers using the :code:`.data_ptr()` method.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * objects within the triton.language package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """
    if args:
        assert len(args) == 1
        assert callable(args[0])
        return JITFunction(args[0], **kwargs)
    else:
        def decorator(fn):
            return JITFunction(fn, **kwargs)
        return decorator


class TensorWrapper:
    def __init__(self, base, dtype):
        self.dtype = dtype
        self.base = base
        self.is_cuda = base.is_cuda
        self.device = base.device

    def data_ptr(self):
        return self.base.data_ptr()

    def __str__(self) -> str:
        return f'TensorWrapper[{self.dtype}]({self.base})'


def reinterpret(tensor, dtype):
    if isinstance(tensor, TensorWrapper):
        if dtype == tensor.base.dtype:
            # Reinterpreting to the original interpretation; return the base.
            return tensor.base
        else:
            # Reinterpreting a wrapped tensor to a different type.
            return TensorWrapper(tensor.base, dtype)
    elif isinstance(tensor, torch.Tensor):
        # A new wrapper is needed around an unwrapped tensor.
        return TensorWrapper(tensor, dtype)
    else:
        raise TypeError(f'Cannot reinterpret a {type(tensor)}.')
