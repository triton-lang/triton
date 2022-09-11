from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import os
import subprocess
import tempfile
import textwrap
from typing import Any, Dict, List, Optional

import torch

import triton
import triton._C.libtriton.triton as _triton
from ..compiler import compile
from ..tools.disasm import extract

try:
    from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
except ImportError:
    get_cuda_stream = lambda dev_idx: torch.cuda.current_stream(dev_idx).cuda_stream

# -----------------------------------------------------------------------------
# Binary
# -----------------------------------------------------------------------------

VALID_BACKENDS: List[str] = (
    _triton.runtime.backend.CUDA,
)


class Binary:
    def __init__(self, backend: str, name: str, asm: Dict[str, str], shared_mem: int, num_warps: int):
        assert backend in VALID_BACKENDS, "backend should within [%s], but get a \"%s\"" % (', '.join(VALID_BACKENDS), backend)
        self.backend = backend
        self.name = name
        self.asm = asm
        self.shared_mem = shared_mem
        self.num_warps = num_warps


class LoadedBinary:
    def __init__(self, device: int, bin: Binary):
        module, kernel = _triton.load_binary(bin.backend,
                                             bin.name,
                                             bin.asm,
                                             bin.shared_mem,
                                             device)
        self.bin = bin
        self.asm = bin.asm
        self.sass = ''
        self.module = module
        self.kernel = kernel
        self.device = device
        self.shared_mem = bin.shared_mem

    def __call__(self, stream, args, grid_0, grid_1=1, grid_2=1):
        _triton.runtime.enqueue(self.bin.backend, stream, self.kernel,
                                grid_0, grid_1, grid_2,
                                self.bin.num_warps * 32, 1, 1,
                                args, self.bin.shared_mem)

    def get_sass(self, fun=None):
        if self.sass:
            return self.sass
        fd, path = tempfile.mkstemp()
        try:
            with open(fd, 'wb') as cubin:
                cubin.write(self.asm['cubin'])
            self.sass = extract(path, fun)
        finally:
            os.remove(path)
        self.asm['sass'] = self.sass
        return self.sass

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


class JITFunction:

    cache_hook = None

    @staticmethod
    def _key_of(arg):
        if hasattr(arg, "dtype"):
            return arg.dtype
        elif isinstance(arg, int):
            return 'i32' if arg < 2**31 else 'i64'
        elif isinstance(arg, float):
            return 'f32'
        elif arg is None:
            return None
        else:
            raise TypeError(f'Unsupported type {type(arg)} for {arg}')
    
    @staticmethod
    def _spec_of(arg):
        if hasattr(arg, "data_ptr"):
            return (arg.data_ptr() % 16 == 0)
        elif isinstance(arg, int):
            return (arg % 16 == 0, arg == 1)
        return (arg is None, )

    @staticmethod
    def _get_config(*args):
        def is_divisible_by_16(x):
            if hasattr(x, "data_ptr"):
                return x.data_ptr() % 16 == 0
            elif isinstance(x, int):
                return x % 16 == 0
            if x is None:
                return True
            return False
        divisible_by_16 = {i for i, arg in enumerate(args) if is_divisible_by_16(arg)}
        equal_to_1 = {i for i, arg in enumerate(args) if isinstance(arg, int) and arg == 1}
        return _triton.code_gen.instance_descriptor(divisible_by_16, equal_to_1)

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

    def _make_signature(self, key):
        signature = ",".join([self._type_of(k) for i, k in enumerate(key) \
                          if i not in self.constexprs])
        constants = {i: k for i, k in enumerate(key) if i in self.constexprs}
        return signature, constants

    def _make_launcher(self):
        regular_args = [f'{arg}' for i, arg in enumerate(self.arg_names) if not i in self.constexprs]
        constexpr_args = [f'{arg}' for i, arg in enumerate(self.arg_names) if i in self.constexprs]
        args = ', '.join(regular_args)
        # cache key for regular argument type
        regular_keys = ', '.join([f'_key_of({arg})' for arg in regular_args])
        # cache key for constexpr argument values
        constexpr_keys = ', '.join(constexpr_args)
        # cache key for argument specialization
        specializations = []
        for arg in regular_args:
            specializations += [f'({arg}.data_ptr() % 16 == 0) if hasattr({arg}, "data_ptr") '
                                f'else ({arg} % 16 == 0, {arg} == 1) if isinstance({arg}, int) '
                                f'else (False,)']
        spec_keys = ', '.join(specializations)
        grid_args = ','.join([f'"{arg}": {arg}' for arg in self.arg_names])

        src = f"""
def {self.fn.__name__}({', '.join(self.arg_names)}, grid, num_warps=4, num_stages=3, stream=None):
    sig_key =  ({regular_keys}, {constexpr_keys})
    spec_key = ({spec_keys})
    key = (version_key, sig_key, spec_key)
    if callable(grid):
        grid = grid({{{grid_args}}})
    grid_size = len(grid)
    grid_0 = grid[0]
    grid_1 = grid[1] if grid_size > 1 else 1
    grid_2 = grid[2] if grid_size > 2 else 1
    if stream is None:
      stream = get_cuda_stream(None)
    try:
      bin = cache[key]
      bin.c_wrapper(grid_0, grid_1, grid_2, stream, {args})
      return bin
    except KeyError:
      # kernel not cached -- compile
      args = [{args}]
      signature, constants = self._make_signature(sig_key)
      constants |= {{i: None for i, arg in enumerate(args) if arg is None}}
      configs = [self._get_config(*args)]
      device = 0
      bin = triton.compile(self, signature, device, constants, num_warps, num_stages, configs=configs)
      bin.c_wrapper(grid_0, grid_1, grid_2, stream, *args)
      self.cache[key] = bin
      return bin
"""
        scope = {"version_key": version_key(), "get_cuda_stream": get_cuda_stream, 
                 "self": self, "_spec_of": self._spec_of, "_key_of": self._key_of, 
                 "cache": self.cache, "triton": triton, "torch": torch}
        exec(src, scope)
        return scope[self.fn.__name__]

    def __init__(self, fn, version=None, do_not_specialize=None):
        self.fn = fn
        self.module = fn.__module__
        # function signature information
        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]
        self.has_defaults = any([v.default!=inspect._empty for v in signature.parameters.values()])
        # function source code (without decorators)
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find("def"):]
        # cache of just-in-time compiled kernels
        self.cache = dict()
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
    @functools.lru_cache()
    def cache_key(self):
        # TODO : hash should be attribute of `self`
        if self.hash is None:
            dependencies_finder = DependenciesFinder(globals=self.__globals__, src=self.src)
            dependencies_finder.visit(self.parse())
            self.hash = dependencies_finder.ret + version_key()
        return self.hash

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
            JITFunction.cache_key.fget.cache_clear()

    def __getitem__(self, grid):
        """
        A JIT function is launched with: fn[grid](*args, **kwargs).
        Hence JITFunction.__getitem__ returns a callable proxy that
        memorizes the grid.
        """
        def launcher(*args, **kwargs):
          return self.run(*args, grid=grid, **kwargs)
        return launcher

    def __repr__(self):
        return f"JITFunction({self.module}:{self.fn.__name__})"


def pow2_divisor(N):
    if N % 16 == 0:
        return 16
    if N % 8 == 0:
        return 8
    if N % 4 == 0:
        return 4
    if N % 2 == 0:
        return 2
    return 1


class _KernelCache:
    def __init__(self,
                 fn: JITFunction,
                 fn_type: str,
                 constants: Dict[str, Any],
                 num_warps: int = 4,
                 num_stages: int = 3):
        # hold the arguments for building a kernel
        self.fn = fn
        self.fn_type = fn_type
        self.constants = constants
        self.num_warps = num_warps
        self.num_stages = num_stages

        # kernel compilation cache
        self._binary_cache: Optional[LoadedBinary] = None

    @property
    def binary_cache(self):
        return self._binary_cache

    def set_binary_cache(self, binary: LoadedBinary):
        assert binary
        assert not self._binary_cache, "cannot set binary cache duplicately"
        self._binary_cache = binary


def build_kernel(fn: JITFunction,
                 fn_type: str,
                 constants: Dict[str, Any],
                 num_warps: int = 4,
                 num_stages: int = 3,
                 ) -> _KernelCache:
    return _KernelCache(fn, fn_type, constants, num_warps, num_stages)


torch_dtype_to_bytes = {
    torch.int8: 1,
    torch.uint8: 1,

    torch.int16: 2,
    torch.short: 2,

    torch.int: 4,
    torch.int32: 4,

    torch.long: 8,
    torch.int64: 8,

    torch.float32: 4,
    torch.float: 4,

    torch.float16: 2,
    torch.half: 2,
    torch.bfloat16: 2,
    # free to extend
}


def launch_kernel(kernel: _KernelCache, grid, device, *wargs, **kwargs):
    def is_tensor(arg):
        return hasattr(arg, 'data_ptr')  # a torch.tensor

    # prepare function args for compile
    kwargs = {kernel.fn.arg_names.index(name): value for name, value in kwargs.items()}
    wargs = list(wargs)
    for i, pos in enumerate(sorted(kwargs)):
        wargs.insert(pos + i, kwargs[pos])
    assert len(wargs) == len(kernel.fn.arg_names), "Function argument list not match, need %d but get %d args" % (len(kernel.fn.arg_names), len(wargs))

    if not kernel.binary_cache:
        # build the kernel cache
        backend = _triton.runtime.backend.CUDA

        attributes = dict()
        for i, arg in enumerate(wargs):
            if i in kernel.fn.do_not_specialize:
                continue
            if isinstance(arg, int):
                attributes[i] = pow2_divisor(arg)
            elif is_tensor(arg):
                assert arg.dtype in torch_dtype_to_bytes
                addr = arg.data_ptr()
                range_size = _triton.runtime.get_pointer_range_size(addr)
                divisibility = min(pow2_divisor(addr), pow2_divisor(range_size)) // torch_dtype_to_bytes[arg.dtype]
                attributes[i] = divisibility

        attributes_ = dict()
        for i, value in attributes.items():
            attributes_[kernel.fn.arg_names[i]] = value

        cubin, shem_size, kernel_name = compile(kernel.fn, kernel.fn_type, device=device, constants=kernel.constants, attributes=attributes_, num_warps=kernel.num_warps, num_stages=kernel.num_stages, output="cubin")
        assert cubin
        assert kernel_name

        max_shared_memory = _triton.runtime.max_shared_memory(backend, device)
        assert shem_size <= max_shared_memory, "shared memory out of resource, max size is %d, but want %s" % (max_shared_memory, shem_size)

        asm = dict(cubin=cubin)
        binary = Binary(backend, kernel_name, asm, shem_size, kernel.num_warps)
        loaded_binary = LoadedBinary(device, binary)
        kernel.set_binary_cache(loaded_binary)

    torch.cuda.set_device(device)
    stream = get_cuda_stream(device)

    _triton.runtime.launch_binary(kernel.binary_cache, wargs, kernel.fn.do_not_specialize, kernel.fn.arg_names,
                                  stream, kernel.num_warps, kernel.num_stages, grid)


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