from __future__ import annotations, division

import builtins
import time
import ast
import functools
import hashlib
import inspect
import json
import os
import subprocess
import tempfile
import textwrap
from enum import Enum
from typing import (
    Any,
    Dict,
    Set,
    Tuple,
    Union,
    List,
    Optional,
    Callable,
    Sequence,
    overload,
    TypeVar,
    Iterable,
)
from typing import cast as typecast
from collections import defaultdict, namedtuple

import torch
from filelock import FileLock

import triton

import triton._C.libtriton.triton as _triton
from triton._C.libtriton.triton import ir

try:
    from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
except ImportError:
    get_cuda_stream = lambda dev_idx: torch.cuda.current_stream(dev_idx).cuda_stream


from .tools.disasm import extract

C = TypeVar("C", bound=Callable)


# -----------------------------------------------------------------------------
# JITFunction
# -----------------------------------------------------------------------------


@functools.lru_cache()
def version_key():
    contents = []

    # frontend
    with open(__file__, "rb") as f:
        contents += [hashlib.md5(f.read()).hexdigest()]

    # TODO(crutcher): walk/hash packages
    if False:
        # backend
        with open(triton._C.libtriton.__file__, "rb") as f:
            contents += [hashlib.md5(f.read()).hexdigest()]
        # language
        language_path = os.path.join(*triton.__path__, "language")
        for lib in pkgutil.iter_modules([language_path]):
            with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
                contents += [hashlib.md5(f.read()).hexdigest()]
    # ptxas version
    try:
        ptxas_version = hashlib.md5(
            subprocess.check_output(["ptxas", "--version"])
        ).hexdigest()
    except Exception:
        ptxas_version = ""
    return "-".join(triton.__version__) + "-" + ptxas_version + "-" + "-".join(contents)


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
        if func.__module__ and func.__module__.startswith("triton."):
            return
        assert isinstance(func, JITFunction)
        if func.hash is None:
            tree = ast.parse(func.src)
            finder = DependenciesFinder(func.__globals__, func.src)
            finder.visit(tree)
            func.hash = finder.ret
        self.ret = (self.ret + func.hash).encode("utf-8")
        self.ret = hashlib.md5(self.ret).hexdigest()


class JITFunction(KernelInterface):

    # Hook for inspecting compiled functions and modules
    cache_hook = None
    divisibility = 16
    do_not_specialize: Set[Any]
    cache: Dict[torch.device, object]
    kernel_decorators: List[Any]

    @staticmethod
    def _key_of(arg):
        if hasattr(arg, "dtype"):
            return arg.dtype
        elif isinstance(arg, bool):
            return "i1"
        elif isinstance(arg, int):
            if -(2**31) <= arg and arg <= 2**31 - 1:
                return "i32"
            elif 2**31 <= arg and arg <= 2**32 - 1:
                return "u32"
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
    def _spec_of(arg):
        if hasattr(arg, "data_ptr"):
            return arg.data_ptr() % JITFunction.divisibility == 0
        elif isinstance(arg, int):
            return (arg % 16 == 0, arg == 1)
        return (arg is None,)

    def _get_config(self, *args):
        def is_divisible_by_16(x):
            if hasattr(x, "data_ptr"):
                return x.data_ptr() % JITFunction.divisibility == 0
            elif isinstance(x, int):
                return x % JITFunction.divisibility == 0
            if x is None:
                return True
            return False

        divisible_by_16 = [
            i
            for i, arg in enumerate(args)
            if is_divisible_by_16(arg) and i not in self.do_not_specialize
        ]
        equal_to_1 = [
            i
            for i, arg in enumerate(args)
            if isinstance(arg, int) and arg == 1 and i not in self.do_not_specialize
        ]
        return namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"])(
            tuple(divisible_by_16), tuple(equal_to_1)
        )
        # return _triton.code_gen.instance_descriptor(divisible_by_16, equal_to_1)

    @staticmethod
    def _type_of(key):
        if isinstance(key, (torch.dtype, dtype)):
            ty = {
                torch.bool: "i1",
                torch.float16: "fp16",
                torch.bfloat16: "bf16",
                torch.float32: "fp32",
                torch.float64: "fp64",
                torch.uint8: "u8",
                torch.int8: "i8",
                torch.int16: "i16",
                torch.int32: "i32",
                torch.int64: "i64",
                uint8: "u8",
                uint16: "u16",
                uint32: "u32",
                uint64: "u64",
                float8: "fp8",
            }[key]
            return f"*{ty}"
        if key is None:
            return "*i8"
        assert isinstance(key, str)
        return key

    def _make_signature(self, sig_key):
        signature = ",".join([self._type_of(k) for i, k in enumerate(sig_key)])
        return signature

    def _make_constants(self, constexpr_key):
        constants = {i: k for i, k in zip(self.constexprs, constexpr_key)}
        return constants

    def _call_hook(
        self,
        key,
        signature,
        device,
        constants,
        num_warps,
        num_stages,
        extern_libs,
        configs,
    ):
        if JITFunction.cache_hook is None:
            return False
        name = self.fn.__name__
        module = self.fn.__module__
        arg_reprs = ", ".join(
            [f"{name}: {ty}" for name, ty in zip(self.arg_names, key[1])]
        )
        repr = f"{name}[num_warps={num_warps}, num_stages={num_stages}]({arg_reprs})"
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
            num_stages=num_stages,
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

    def _make_launcher(self):
        regular_args = [
            f"{arg}" for i, arg in enumerate(self.arg_names) if i not in self.constexprs
        ]
        constexpr_args = [
            f"{arg}" for i, arg in enumerate(self.arg_names) if i in self.constexprs
        ]
        args = ", ".join(regular_args)
        # cache key for regular argument type
        sig_keys = ", ".join([f"_key_of({arg})" for arg in regular_args])
        # cache key for constexpr argument values
        constexpr_keys = ", ".join(constexpr_args)
        # cache key for argument specialization
        specializations = []
        for i, arg in enumerate(regular_args):
            if i in self.do_not_specialize:
                continue
            specializations += [
                f'({arg}.data_ptr() % {JITFunction.divisibility} == 0) if hasattr({arg}, "data_ptr") '
                f"else ({arg} % {JITFunction.divisibility} == 0, {arg} == 1) if isinstance({arg}, int) "
                f"else (False,)"
            ]
        spec_keys = ", ".join(specializations)
        grid_args = ",".join([f'"{arg}": {arg}' for arg in self.arg_names])

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
          bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps, bin.shared, stream, bin.cu_function, triton.CompiledKernel.launch_enter_hook, triton.CompiledKernel.launch_exit_hook, bin, {args})
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
            bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps, bin.shared, stream, bin.cu_function, triton.CompiledKernel.launch_enter_hook, triton.CompiledKernel.launch_exit_hook, bin, *args)
        self.cache[device][key] = bin
        return bin
      return None
"""
        scope = {
            "version_key": version_key(),
            "get_cuda_stream": get_cuda_stream,
            "self": self,
            "_spec_of": self._spec_of,
            "_key_of": self._key_of,
            "cache": self.cache,
            "triton": triton,
            "torch": torch,
        }
        exec(src, scope)
        return scope[self.fn.__name__]

    def __init__(
        self,
        fn,
        version=None,
        do_not_specialize: Optional[Iterable[Any]] = None,
    ):
        self.fn = fn
        self.module = fn.__module__
        self.version = version
        # function signature information
        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]
        self.has_defaults = any(
            [v.default != inspect._empty for v in signature.parameters.values()]
        )
        # specialization hints
        do_not_specialize = [] if do_not_specialize is None else do_not_specialize
        self.do_not_specialize = set(
            [
                self.arg_names.index(arg) if isinstance(arg, str) else arg
                for arg in do_not_specialize
            ]
        )
        # function source code (without decorators)
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find("def") :]
        # cache of just-in-time compiled kernels
        self.cache = defaultdict(dict)
        self.hash = None
        # JITFunction can be instantiated as kernel
        # when called with a grid using __getitem__
        self.kernel_decorators = []
        self.kernel = None
        # annotations
        self.annotations = {
            self.arg_names.index(name): ty for name, ty in fn.__annotations__.items()
        }
        self.__annotations__ = fn.__annotations__
        # index of constexprs
        self.constexprs = [
            self.arg_names.index(ann) for ann in self.__annotations__.keys()
        ]
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
            dependencies_finder = DependenciesFinder(
                globals=self.__globals__, src=self.src
            )
            dependencies_finder.visit(self.parse())
            self.hash = dependencies_finder.ret + version_key()
        return self.hash

    def warmup(self, *args, **kwargs):
        return self.run(
            *map(triton.utils.MockTensor.wrap_dtype, args), **kwargs, warmup=True
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
        # - when kernel decorators change, cached kernel
        #   needs to be cleared
        if name == "kernel_decorators":
            self.kernel = None
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
def jit(fn: Callable) -> JITFunction:
    ...


@overload
def jit(
    *,
    version=None,
    do_not_specialize: Optional[Iterable[Any]] = None,
) -> Callable[[Callable], JITFunction]:
    ...


def jit(
    fn: Optional[Callable] = None,
    *,
    version=None,
    do_not_specialize: Optional[Iterable[Any]] = None,
) -> Union[JITFunction, Callable[[Callable], JITFunction]]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, :code:`torch.tensor` arguments are implicitly converted to pointers using the :code:`.data_ptr()` method.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * objects within the package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

    def decorator(fn: Callable) -> JITFunction:
        assert callable(fn)
        return JITFunction(
            fn,
            version=version,
            do_not_specialize=do_not_specialize,
        )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator


class Autotuner(KernelInterface):
    configs: List[Config]
    key_idx: List[int]
    cache: Dict[Tuple[Any, ...], tensor]
    reset_idx: List[int]
    hook: Callable[[List[tensor]], None]
    perf_model: Optional[Callable]
    configs_top_k: Optional[List[Config]]
    early_config_prune: Optional[Callable[[List[Config], int], List[Config]]]
    fn: Callable

    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        prune_configs_by: Optional[Dict] = None,
    ):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'prune_num_stages_by'(optional): a function used to prune num_stages. It take configs:List[Config] as its input, and returns pruned configs.
        """
        if not configs:
            self.configs = [Config(dict(), num_warps=4, num_stages=2)]
        else:
            self.configs = configs
        self.key_idx = [arg_names.index(k) for k in key]
        self.cache = dict()
        # hook to reset all required tensor to zeros before relaunching a kernel
        self.hook = lambda args: None
        if reset_to_zero is not None:
            self.reset_idx = [arg_names.index(k) for k in reset_to_zero]

            def _hook(args):
                for i in self.reset_idx:
                    args[i].zero_()

            self.hook = _hook
        self.arg_names = arg_names
        # prune configs
        if prune_configs_by:
            perf_model, top_k = (
                prune_configs_by["perf_model"],
                prune_configs_by["top_k"],
            )
            if "early_config_prune" in prune_configs_by:
                early_config_prune = prune_configs_by["early_config_prune"]
        else:
            perf_model, top_k, early_config_prune = None, None, None
        self.perf_model = perf_model
        self.configs_top_k = top_k
        self.early_config_prune = early_config_prune
        self.fn = fn

    def _bench(self, *args, config, **meta):
        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols."
            )
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.kwargs)

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(self.nargs)
            self.hook(args)
            self.fn.run(
                *args,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                **current,
            )

        from triton.testing import do_bench

        return do_bench(kernel_call)

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        if len(self.configs) > 1:
            key = tuple([args[i] for i in self.key_idx])
            if key not in self.cache:
                # prune configs
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()
                timings = {
                    config: self._bench(*args, config=config, **kwargs)
                    for config in pruned_configs
                }
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                self.hook(args)
                self.configs_timings = timings
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if config.pre_hook is not None:
            config.pre_hook(self.nargs)
        return self.fn.run(
            *args,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
            **kwargs,
            **config.kwargs,
        )

    def prune_configs(self, kwargs):
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.kwargs,
                        num_stages=config.num_stages,
                        num_warps=config.num_warps,
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[
                    :top_k
                ]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        for config in self.prune_configs(kwargs):
            self.fn.warmup(
                *args,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                **kwargs,
                **config.kwargs,
            )
        self.nargs = None


class Config:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar meta: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type meta: dict[Str, Any]
    :ivar num_warps: the number of warps to use for the kernel when compiled for GPUs. For example, if
                      `num_warps=8`, then each kernel instance will be automatically parallelized to
                      cooperatively execute using `8 * 32 = 256` threads.
    :type num_warps: int
    :ivar num_stages: the number of stages that the compiler should use when software-pipelining loops.
                       Mostly useful for matrix multiplication workloads on SM80+ GPUs.
    :type num_stages: int
    :ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
                    function are args.
    """

    def __init__(self, kwargs, num_warps=4, num_stages=2, pre_hook=None):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.pre_hook = pre_hook

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        res.append(f"num_warps: {self.num_warps}")
        res.append(f"num_stages: {self.num_stages}")
        return ", ".join(res)


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton.autotune(configs=[
            triton.Config(meta={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(meta={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE']

    :note: When all the configurations are evaluated, the kernel will run multiple time.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           reset the value of the provided tensor to `zero` before running any configuration.

    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It take configs:List[Config] as its input, and returns pruned configs.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    """

    def decorator(fn):
        return Autotuner(
            fn, fn.arg_names, configs, key, reset_to_zero, prune_configs_by
        )

    return decorator


class Heuristics(KernelInterface):
    def __init__(self, fn, arg_names, values) -> None:
        self.fn = fn
        self.values = values
        self.arg_names = arg_names

    def run(self, *args, **kwargs):
        for v, heur in self.values.items():
            kwargs[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
        return self.fn.run(*args, **kwargs)


def heuristics(values):
    """
    Decorator for specifying how the values of certain meta-parameters may be computed.
    This is useful for cases where auto-tuning is prohibitevely expensive, or just not applicable.

    .. highlight:: python
    .. code-block:: python

        @triton.heuristics(values={'BLOCK_SIZE': lambda args: 2 ** int(math.ceil(math.log2(args[1])))})
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE'] # smallest power-of-two >= x_size


    .param values: a dictionary of meta-parameter names and functions that compute the value of the meta-parameter.
                   each such function takes a list of positional arguments as input.
    .type values: dict[str, Callable[[list[Any]], Any]]
    """

    def decorator(fn):
        return Heuristics(fn, fn.arg_names, values)

    return decorator


def mangle_ty(ty):
    if ty.is_ptr():
        return "P" + mangle_ty(ty.element_ty)
    if ty.is_int():
        return "i" + str(ty.int_bitwidth)
    if ty.is_fp8():
        return "fp8"
    if ty.is_fp16():
        return "fp16"
    if ty.is_bf16():
        return "bf16"
    if ty.is_fp32():
        return "fp32"
    if ty.is_fp64():
        return "fp64"
    if ty.is_void():
        return "V"
    if ty.is_block():
        elt = mangle_ty(ty.scalar)
        shape = "_".join(map(str, ty.shape))
        return f"{elt}S{shape}S"
    assert False, "Unsupported type"


def is_triton_tensor(value):
    return isinstance(value, tensor)


class CompilationError(Exception):
    def __init__(self, src, node):
        self.message = f"at {node.lineno}:{node.col_offset}:\n"
        self.message += "\n".join(src.split("\n")[: node.lineno])
        self.message += "\n" + " " * node.col_offset + "^"
        self.src = src
        self.node = node
        super().__init__(self.message)

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return (type(self), (self.src, self.node))


class OutOfResources(Exception):
    def __init__(self, required, limit, name):
        self.message = (
            f"out of resource: {name}, "
            f"Required: {required}, "
            f"Hardware limit: {limit}"
        )
        self.required = required
        self.limit = limit
        self.name = name
        super().__init__(self.message)

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return (type(self), (self.required, self.limit, self.name))


def make_ptx(mod: Any, device: int) -> Tuple[str, int]:
    """
    Translate TritonGPU module to PTX code.
    :param mod: a TritonGPU dialect module
    :return:
        - PTX code
        - shared memory alloaction size
    """
    return _triton.translate_triton_gpu_to_ptx(mod, device)


def make_cubin(ptx, device):
    """
    Compile TritonGPU module to cubin.
    :param ptx: ptx code
    :param device: CUDA device
    :return: str
    """
    return _triton.compile_ptx_to_cubin(ptx, device)


def ptx_get_kernel_name(ptx: str) -> str:
    """
    Get kernel name from PTX code.
    This Kernel name is required when launching the kernel.
    """
    # There is a name mangling in PTX codegen, so the original kernel names in Triton IR are not available in PTX/cubin.
    assert ptx
    for line in ptx.split("\n"):
        line = line.strip()
        if line.startswith("// .globl"):
            return line.split()[-1]
    raise AssertionError(f"No kernel name found in:\n{ptx}")


def generate_name_initializer(signature):
    src = "int i = 0;\n"
    tys = signature.split(",")
    for i, ty in enumerate(tys):
        src


def binary_name_to_header_name(name):
    if len(name) > 128:
        # avoid filename too long errors (filename limit is 255)
        name = "kernel_" + hashlib.sha256(name.encode("utf-8")).hexdigest()
    return f"{name}.h"


def default_cache_dir():
    return os.path.join(os.environ["HOME"], ".triton", "cache")


class CacheManager:
    def __init__(self, key):
        self.key = key
        self.lock_path = None
        # create cache directory if it doesn't exist
        self.cache_dir = os.environ.get("TRITON_CACHE_DIR", default_cache_dir())
        if self.cache_dir:
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)

    def _make_path(self, filename):
        return os.path.join(self.cache_dir, filename)

    def has_file(self, filename):
        if not self.cache_dir:
            return False
        return os.path.exists(self._make_path(filename))

    def put(self, data, filename, binary=True):
        if not self.cache_dir:
            return
        assert self.lock_path is not None
        filepath = self._make_path(filename)
        with FileLock(self.lock_path):
            # use tempfile to be robust against program interruptions
            mode = "wb" if binary else "w"
            with open(filepath + ".tmp", mode) as f:
                f.write(data)
            os.rename(filepath + ".tmp", filepath)


# utilties for generating and compiling C wrappers


class CompiledKernel:

    # Hooks for external tools to monitor the execution of triton kernels
    launch_enter_hook = None
    launch_exit_hook = None

    def __init__(self, fn_name, so_path, cache_dir, device):
        # initialize launcher
        import importlib.util

        spec = importlib.util.spec_from_file_location("launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.c_wrapper = getattr(mod, "launch")
        # initialize metadata
        with open(os.path.join(cache_dir, f"{fn_name}.json")) as f:
            metadata = json.load(f)
        self.shared = metadata["shared"]
        self.num_warps = metadata["num_warps"]
        self.num_stages = metadata["num_stages"]
        # initialize asm dict
        self.asm = dict()
        with open(os.path.join(cache_dir, f"{fn_name}.cubin"), "rb") as f:
            self.asm["cubin"] = f.read()
        with open(os.path.join(cache_dir, f"{fn_name}.ptx"), "r") as f:
            self.asm["ptx"] = f.read()
        with open(os.path.join(cache_dir, f"{fn_name}.llir"), "r") as f:
            self.asm["llir"] = f.read()
        with open(os.path.join(cache_dir, f"{fn_name}.ttir"), "r") as f:
            self.asm["ttir"] = f.read()

        mod, func, n_regs, n_spills = _triton.code_gen.load_binary(
            metadata["name"], self.asm["cubin"], self.shared, device
        )
        self.fn_name = fn_name
        self.cu_module = mod
        self.cu_function = func
        self.n_regs = n_regs
        self.n_spills = n_spills

    def __getitem__(self, grid):
        def runner(*args, stream=None):
            if stream is None:
                stream = torch.cuda.current_stream().cuda_stream
            self.c_wrapper(
                grid[0],
                grid[1],
                grid[2],
                self.num_warps,
                self.shared,
                stream,
                self.cu_function,
                CompiledKernel.launch_enter_hook,
                CompiledKernel.launch_exit_hook,
                self,
                *args,
            )

        return runner

    def get_sass(self, fun=None):
        if "sass" in self.asm:
            return self.asm["sass"]
        fd, path = tempfile.mkstemp()
        try:
            with open(fd, "wb") as cubin:
                cubin.write(self.asm["cubin"])
            self.sass = extract(path, fun)
        finally:
            os.remove(path)
        self.asm["sass"] = self.sass
        return self.sass


class TensorWrapper:
    def __init__(self, base, dtype):
        self.dtype = dtype
        self.base = base
        self.is_cuda = base.is_cuda
        self.device = base.device

    def data_ptr(self):
        return self.base.data_ptr()

    def __str__(self) -> str:
        return f"TensorWrapper[{self.dtype}]({self.base})"


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
        raise TypeError(f"Cannot reinterpret a {type(tensor)}.")


def _to_tensor(x, builder):
    if isinstance(x, bool):
        return tensor(builder.get_int1(x), int1)
    # Note: compile-time const integers are represented by unsigned values
    elif isinstance(x, int):
        if -(2**31) <= x < 2**31:
            return tensor(builder.get_int32(x), int32)
        elif 2**31 <= x < 2**32:
            return tensor(builder.get_uint32(x), uint32)
        elif -(2**63) <= x < 2**63:
            return tensor(builder.get_int64(x), int64)
        elif 2**63 <= x < 2**64:
            return tensor(builder.get_uint64(x), uint64)
        else:
            raise RuntimeError(f"Nonrepresentable integer {x}.")
    elif isinstance(x, float):
        return tensor(builder.get_float32(x), float32)
    elif isinstance(x, constexpr):
        if x.value is None:
            return None
        return _to_tensor(x.value, builder)
    elif isinstance(x, tensor):
        return x
    elif x is None:
        return None
    assert False, f"cannot convert {x} to tensor"


def builtin(fn: C) -> C:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError(
                "Did you forget to add @triton.jit ? (`_builder` argument must be provided outside of JIT functions.)"
            )
        return fn(*args, **kwargs)

    wrapper.__triton_builtin__ = True

    return typecast(C, wrapper)


class dtype:
    SINT_TYPES = ["int1", "int8", "int16", "int32", "int64"]
    UINT_TYPES = ["uint8", "uint16", "uint32", "uint64"]
    FP_TYPES = ["fp8", "fp16", "bf16", "fp32", "fp64"]
    OTHER_TYPES = ["void"]

    class SIGNEDNESS(Enum):
        SIGNED = 0
        UNSIGNED = 1

    name: str
    numel: int = 1
    shape: Tuple[int, ...] = tuple()
    int_signedness: SIGNEDNESS
    int_bitwidth: int
    primitive_bitwidth: int
    fp_mantissa_width: int = 0

    def __init__(self, name):
        self.name = name
        assert (
            name
            in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES
        ), name
        if name in dtype.SINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.SIGNED
            self.int_bitwidth = int(name.split("int")[-1])
            self.primitive_bitwidth = self.int_bitwidth
        elif name in dtype.UINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.UNSIGNED
            self.int_bitwidth = int(name.split("int")[-1])
            self.primitive_bitwidth = self.int_bitwidth
        elif name in dtype.FP_TYPES:
            if name == "fp8":
                self.fp_mantissa_width = 3
                self.primitive_bitwidth = 8
            elif name == "fp16":
                self.fp_mantissa_width = 10
                self.primitive_bitwidth = 16
            elif name == "bf16":
                self.fp_mantissa_width = 7
                self.primitive_bitwidth = 16
            elif name == "fp32":
                self.fp_mantissa_width = 23
                self.primitive_bitwidth = 32
            elif name == "fp64":
                self.fp_mantissa_width = 53
                self.primitive_bitwidth = 64
        elif name == "void":
            self.primitive_bitwidth = 0

    def is_fp8(self) -> bool:
        return self.name == "fp8"

    def is_fp16(self) -> bool:
        return self.name == "fp16"

    def is_bf16(self) -> bool:
        return self.name == "bf16"

    def is_fp32(self) -> bool:
        return self.name == "fp32"

    def is_fp64(self) -> bool:
        return self.name == "fp64"

    def is_int1(self) -> bool:
        return self.name == "int1"

    def is_int8(self) -> bool:
        return self.name == "int8"

    def is_int16(self) -> bool:
        return self.name == "int16"

    def is_int32(self) -> bool:
        return self.name == "int32"

    def is_int64(self) -> bool:
        return self.name == "int64"

    def is_uint8(self) -> bool:
        return self.name == "uint8"

    def is_uint16(self) -> bool:
        return self.name == "uint16"

    def is_uint32(self) -> bool:
        return self.name == "uint32"

    def is_uint64(self) -> bool:
        return self.name == "uint64"

    def is_floating(self) -> bool:
        return self.name in dtype.FP_TYPES

    def is_int_signed(self) -> bool:
        return self.name in dtype.SINT_TYPES

    def is_int_unsigned(self) -> bool:
        return self.name in dtype.UINT_TYPES

    def is_int(self) -> bool:
        return self.name in dtype.SINT_TYPES + dtype.UINT_TYPES

    def is_bool(self) -> bool:
        return self.is_int1()

    def is_void(self) -> bool:
        return self.name == "void"

    def is_block(self) -> bool:
        return False

    def is_ptr(self) -> bool:
        return False

    def __eq__(self, other: Any):
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __ne__(self, other: Any):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name,))

    @property
    def scalar(self) -> dtype:
        return self

    def to_ir(self, builder: ir.builder) -> ir.type:
        if self.name == "void":
            return builder.get_void_ty()
        elif self.name == "int1":
            return builder.get_int1_ty()
        elif self.name == "int8" or self.name == "uint8":
            return builder.get_int8_ty()
        elif self.name == "int16" or self.name == "uint16":
            return builder.get_int16_ty()
        elif self.name == "int32" or self.name == "uint32":
            return builder.get_int32_ty()
        elif self.name == "int64" or self.name == "uint64":
            return builder.get_int64_ty()
        elif self.name == "fp8":
            return builder.get_fp8_ty()
        elif self.name == "fp16":
            return builder.get_half_ty()
        elif self.name == "bf16":
            return builder.get_bf16_ty()
        elif self.name == "fp32":
            return builder.get_float_ty()
        elif self.name == "fp64":
            return builder.get_double_ty()
        raise ValueError(f"fail to convert {self} to ir type")

    def __str__(self):
        return self.name

    @property
    def cache_key_part(self) -> str:
        """See cache_key_part() in triton.cc."""
        return self.name

    def __repr__(self):
        return f"{self.name}"


class pointer_type(dtype):
    element_ty: dtype
    address_space: int

    def __init__(self, element_ty: dtype, address_space: int = 1):
        if not isinstance(element_ty, dtype):
            raise TypeError("element_ty is a {type(element_ty).__name__}.")
        self.element_ty = element_ty
        self.address_space = address_space

        self.name = self.__str__()

    def to_ir(self, builder: ir.builder) -> ir.pointer_type:
        return ir.type.make_ptr(self.element_ty.to_ir(builder), 1)

    def __str__(self):
        return f"pointer<{self.element_ty}>"

    def __repr__(self):
        return self.__str__()

    def is_ptr(self):
        return True

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, pointer_type):
            return False
        return (
            self.element_ty == other.element_ty
            and self.address_space == other.address_space
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def scalar(self) -> pointer_type:
        return self


class block_type(dtype):
    element_ty: dtype
    shape: Tuple[int, ...]
    numel: int

    def __init__(self, element_ty: dtype, shape: Sequence[int]):
        self.element_ty = element_ty
        # FIXME:
        # block_type's shape is a tuple of int
        # while tensor's shape is a list of constexpr
        self.numel = 1

        shape = list(shape)
        for i, s in enumerate(shape):
            if isinstance(s, constexpr):
                shape[i] = s.value
            self.numel *= shape[i]

        self.shape = tuple(shape)

        self.name = self.__str__()

    def to_ir(self, builder: ir.builder) -> ir.block_type:
        return ir.type.make_block(self.element_ty.to_ir(builder), self.shape)

    def __str__(self) -> str:
        return f"<{self.shape}, {self.element_ty}>"

    def __repr__(self) -> str:
        return self.__str__()

    def is_block(self) -> bool:
        return True

    def get_block_shapes(self) -> Tuple[int, ...]:
        return self.shape

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, block_type):
            return False
        return self.element_ty == other.element_ty and self.shape == other.shape

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def scalar(self) -> dtype:
        return self.element_ty


class function_type(dtype):
    def __init__(self, ret_type: dtype, param_types: List[dtype]) -> None:
        self.ret_type = ret_type
        self.param_types = param_types

    def __str__(self):
        return f"fn ({self.param_types}) -> {self.ret_type}"

    def to_ir(self, builder: ir.builder):
        ir_param_types = [ty.to_ir(builder) for ty in self.param_types]
        return ir.type.make_function(self.ret_type.to_ir(builder), ir_param_types)


class tuple_type(dtype):
    def __init__(self, element_types: List[dtype]) -> None:
        self.element_types = element_types

    def __str__(self):
        return f"<{self.element_types}>"

    def to_ir(self, builder: ir.builder):
        ir_element_types = [ty.to_ir(builder) for ty in self.element_types]
        return ir.struct_type.get(ir_element_types, True)


# scalar types
void = dtype("void")
int1 = dtype("int1")
int8 = dtype("int8")
int16 = dtype("int16")
int32 = dtype("int32")
int64 = dtype("int64")
uint8 = dtype("uint8")
uint16 = dtype("uint16")
uint32 = dtype("uint32")
uint64 = dtype("uint64")
float8 = dtype("fp8")
float16 = dtype("fp16")
bfloat16 = dtype("bf16")
float32 = dtype("fp32")
float64 = dtype("fp64")
# pointer types
pi32_t = pointer_type(int32)

# -----------------------
# constexpr
# -----------------------


class constexpr:
    """
    This class is used to store a value that is known at compile-time.
    """

    def __init__(self, value):
        if isinstance(value, constexpr):
            self.value = value.value
        else:
            self.value = value

    def __repr__(self) -> str:
        return f"constexpr[{self.value}]"

    def __bool__(self):
        return bool(self.value)

    def __ge__(self, other):
        other = other.value if isinstance(other, constexpr) else other
        return self.value >= other

    def __gt__(self, other):
        other = other.value if isinstance(other, constexpr) else other
        return self.value > other

    def __le__(self, other):
        other = other.value if isinstance(other, constexpr) else other
        return self.value <= other

    def __lt__(self, other):
        other = other.value if isinstance(other, constexpr) else other
        return self.value < other

    def __eq__(self, other):
        other = other.value if isinstance(other, constexpr) else other
        return self.value == other

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __call__(self, *args, **kwds):
        return self.value(*args, **kwds)

    def to(self, dtype, bitcast=False, _builder: ir.builder = None) -> constexpr:
        if dtype in [float8, float16, bfloat16]:
            raise ValueError("floating point constexpr must be float64")
        ret_ty: type
        if dtype.is_int():
            ret_ty = int
        elif dtype.is_bool():
            ret_ty = bool
        elif dtype.is_floating():
            ret_ty = float
        return constexpr(ret_ty(self.value))


dtype_t = dtype


class tensor:
    handle: ir.pointer_type
    type: dtype_t
    dtype: dtype_t
    shape: List[constexpr]
    numel: constexpr

    # infer dtype from ir type
    @staticmethod
    def _to_dtype(ir_type):
        # block type
        if ir_type.is_block():
            scalar_ty = tensor._to_dtype(ir_type.scalar)
            return block_type(scalar_ty, ir_type.get_block_shapes())
        # pointer type
        if ir_type.is_ptr():
            element_ty = tensor._to_dtype(ir_type.element)
            return pointer_type(element_ty)
        # primitive type
        if ir_type.is_void():
            return void
        if ir_type.is_int1():
            return int1
        if ir_type.is_int8():
            return int8
        if ir_type.is_int16():
            return int16
        if ir_type.is_int32():
            return int32
        if ir_type.is_int64():
            return int64
        if ir_type.is_fp8():
            return float8
        if ir_type.is_fp16():
            return float16
        if ir_type.is_bf16():
            return bfloat16
        if ir_type.is_fp32():
            return float32
        if ir_type.is_fp64():
            return float64
        raise ValueError(f"Unsupported type {ir_type.repr()}")

    def __init__(self, handle, type: dtype_t):
        # IR handle
        self.handle = handle
        # Block shape
        shape = (1,)
        if self.handle.type.is_block():
            shape = self.handle.type.shape
        numel = 1
        for s in shape:
            numel *= s
        is_pow2 = numel and (not (numel & (numel - 1)))
        if not is_pow2:
            raise ValueError(
                "Triton tensors must have a power-of-two number of elements"
            )
        self.numel = constexpr(numel)
        self.shape = [constexpr(s) for s in shape]

        self.type = type  # Tensor type (can be block_type)
        # Following the practice in pytorch, dtype is scalar type
        self.dtype = type.scalar

    def __str__(self) -> str:
        # ex. "float32[3,4]"
        return str(self.dtype) + "[" + ",".join(str(s) for s in self.shape) + "]"

    @builtin
    def __add__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _add(self, other, _builder)

    def __radd__(self, other, _builder: ir.builder = None):
        return self.__add__(other, _builder=_builder)

    @builtin
    def __sub__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _sub(self, other, _builder)

    def __rsub__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _sub(other, self, _builder)

    @builtin
    def __mul__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _mul(self, other, _builder)

    def __rmul__(self, other, _builder: ir.builder = None) -> tensor:
        return self.__mul__(other, _builder=_builder)

    @builtin
    def __truediv__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _truediv(self, other, _builder)

    def __rtruediv__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _truediv(other, self, _builder)

    @builtin
    def __floordiv__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _floordiv(self, other, _builder)

    @builtin
    def __rfloordiv__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _floordiv(other, self, _builder)

    @builtin
    def __mod__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _mod(self, other, _builder)

    @builtin
    def __rmod__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _mod(other, self, _builder)

    # unary operators
    @builtin
    def __neg__(self, _builder: ir.builder = None) -> tensor:
        return _minus(self, _builder)

    @builtin
    def __invert__(self, _builder: ir.builder = None) -> tensor:
        return _invert(self, _builder)

    # bitwise operators

    @builtin
    def __and__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _and_(self, other, _builder)

    @builtin
    def __or__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _or_(self, other, _builder)

    @builtin
    def __xor__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _xor_(self, other, _builder)

    @builtin
    def __lshift__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _shl(self, other, _builder)

    @builtin
    def __rshift__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _lshr(self, other, _builder)

    # comparison operators

    # >
    @builtin
    def __gt__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _greater_than(self, other, _builder)

    @builtin
    def __rgt__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _greater_than(other, self, _builder)

    # >=
    @builtin
    def __ge__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _greater_equal(self, other, _builder)

    @builtin
    def __rge__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _greater_equal(other, self, _builder)

    # <
    @builtin
    def __lt__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _less_than(self, other, _builder)

    @builtin
    def __rlt__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _less_than(other, self, _builder)

    # <=
    @builtin
    def __le__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _less_equal(self, other, _builder)

    @builtin
    def __rle__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _less_equal(other, self, _builder)

    # ==
    @builtin
    def __eq__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _equal(self, other, _builder)

    @builtin
    def __ne__(self, other, _builder: ir.builder = None) -> tensor:
        other = _to_tensor(other, _builder)
        return _not_equal(self, other, _builder)

    @builtin
    def __getitem__(self, slices, _builder: ir.builder = None) -> tensor:
        if isinstance(slices, slice):
            slices = [slices]
        src_shape = self.shape
        dst_shape = []
        curr = 0
        for sl in slices:
            if isinstance(sl, constexpr) and sl.value is None:
                dst_shape.append(1)
            elif sl == slice(None, None, None):
                dst_shape.append(src_shape[curr].value)
                curr += 1
        ret = _reshape(self, dst_shape, _builder)
        return ret

    @builtin
    def to(self, dtype, bitcast=False, _builder: ir.builder = None) -> tensor:
        if isinstance(bitcast, constexpr):
            bitcast = bitcast.value
        if bitcast:
            return _bitcast(self, dtype, _builder)
        return _cast(self, dtype, _builder)


# Create custom exception that prints message "hello"
class IncompatibleTypeErrorimpl(Exception):
    def __init__(self, type_a, type_b):
        self.type_a = type_a
        self.type_b = type_b
        self.message = (
            "invalid operands of type "
            + self.type_a.__repr__()
            + " and "
            + self.type_b.__repr__()
        )
        super(IncompatibleTypeErrorimpl, self).__init__(self.message)


# ===----------------------------------------------------------------------===##
# Programming Model
# ===----------------------------------------------------------------------===##


def _program_id(axis: int, builder: ir.builder) -> tensor:
    axis = _constexpr_to_value(axis)
    return tensor(builder.create_get_program_id(axis), int32)


def _num_programs(axis: int, builder: ir.builder) -> tensor:
    return tensor(builder.create_get_num_programs(axis), int32)


# ===----------------------------------------------------------------------===//
#                               Implicit Casting Utilities
# ===----------------------------------------------------------------------===//


def _integer_promote_impl(a_ty: dtype, b_ty: dtype) -> dtype:
    a_rank = a_ty.int_bitwidth
    b_rank = b_ty.int_bitwidth
    a_sn = a_ty.int_signedness
    b_sn = b_ty.int_signedness
    # Rules for signedness taken from "Usual arithmetic conversions" on
    # https://en.cppreference.com/w/c/language/conversion.
    if a_sn == b_sn:
        return a_ty if a_rank > b_rank else b_ty
    elif a_sn == dtype.SIGNEDNESS.UNSIGNED:
        return a_ty if a_rank >= b_rank else b_ty
    elif b_sn == dtype.SIGNEDNESS.UNSIGNED:
        return b_ty if b_rank >= a_rank else a_ty
    assert False


def _computation_type_impl(
    a_ty: dtype,
    b_ty: dtype,
    *,
    div_or_mod: bool,
) -> dtype:
    # 1) if one operand is double, the other is implicitly
    #    converted to double
    if a_ty.is_fp64() or b_ty.is_fp64():
        return float64
    # 2) if one operand is float, the other is implicitly
    #    converted to float
    if a_ty.is_fp32() or b_ty.is_fp32():
        return float32
    # 3 ) if one operand is half, the other is implicitly converted to half
    #     unless we're doing / or %, which do not exist natively in PTX for fp16.
    #     Supported PTX op: add, sub, mul, fma, neg, abs, min, max, tanh, ex2, setp
    if a_ty.is_fp16() or b_ty.is_fp16():
        if div_or_mod:
            return float32
        else:
            return float16
    # 4) return bf16 only if both operands are of bf16
    if a_ty.is_bf16() or b_ty.is_bf16():
        if div_or_mod:
            return float32
        if a_ty.is_bf16() and b_ty.is_bf16():
            return bfloat16
        return float32
    if not a_ty.is_int() or not b_ty.is_int():
        assert False
    # 5 ) both operands are integer and undergo
    #    integer promotion
    if div_or_mod and a_ty.int_signedness != b_ty.int_signedness:
        raise ValueError(
            "Cannot use /, #, or % with "
            + a_ty.__repr__()
            + " and "
            + b_ty.__repr__()
            + " because they have different signedness;"
            "this is unlikely to result in a useful answer. Cast them to the same signedness."
        )
    return _integer_promote_impl(a_ty, b_ty)


# ===----------------------------------------------------------------------===//
#                               Binary Operators
# ===----------------------------------------------------------------------===//


def _check_ptr_type_impl(
    type_a: dtype,
    type_b: dtype,
    *,
    allow_ptr_a: bool,
) -> None:
    if type_a.is_ptr():
        if not allow_ptr_a:
            raise IncompatibleTypeErrorimpl(type_a, type_b)
        # T* + U* with T != U
        if type_b.is_ptr() and (type_a != type_b):
            raise IncompatibleTypeErrorimpl(type_a, type_b)
        # T* + float
        if type_b.is_floating():
            raise IncompatibleTypeErrorimpl(type_a, type_b)


def _binary_op_type_checking_impl(
    lhs: tensor,
    rhs: tensor,
    *,
    builder: ir.builder,
    allow_lhs_ptr: bool = False,
    allow_rhs_ptr: bool = False,
    arithmetic_check: bool = True,
    div_or_mod: bool = False,
) -> Tuple[tensor, tensor]:
    # implicit broadcasting
    lhs, rhs = _broadcast_impl_value(lhs, rhs, builder)
    # implicit typecasting
    lhs_sca_ty = lhs.type.scalar
    rhs_sca_ty = rhs.type.scalar
    _check_ptr_type_impl(lhs_sca_ty, rhs_sca_ty, allow_ptr_a=allow_lhs_ptr)
    _check_ptr_type_impl(rhs_sca_ty, lhs_sca_ty, allow_ptr_a=allow_rhs_ptr)
    if arithmetic_check and not lhs_sca_ty.is_ptr() and not rhs_sca_ty.is_ptr():
        ret_sca_ty = _computation_type_impl(
            lhs_sca_ty, rhs_sca_ty, div_or_mod=div_or_mod
        )
        lhs = _cast(lhs, ret_sca_ty, builder)
        rhs = _cast(rhs, ret_sca_ty, builder)
    return lhs, rhs


def _add(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _binary_op_type_checking_impl(
        input,
        other,
        builder=builder,
        allow_lhs_ptr=True,
        allow_rhs_ptr=True,
    )
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar

    # offset + ptr
    # ptr + offset
    if other_scalar_ty.is_ptr() and not input_scalar_ty.is_ptr():
        input, other = other, input
    if input_scalar_ty.is_ptr():
        return tensor(builder.create_gep(input.handle, [other.handle]), input.type)
    # float + float
    elif input_scalar_ty.is_floating():
        return tensor(builder.create_fadd(input.handle, other.handle), input.type)
    # int + int
    elif input_scalar_ty.is_int():
        return tensor(builder.create_add(input.handle, other.handle), input.type)
    assert False


def _sub(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _binary_op_type_checking_impl(
        input,
        other,
        builder=builder,
        allow_lhs_ptr=True,
        allow_rhs_ptr=False,
    )
    scalar_ty = input.type.scalar
    # ptr - offset
    if scalar_ty.is_ptr():
        return tensor(
            builder.create_gep(input.handle, [_minus(other, builder).handle]),
            input.type,
        )
    # float - float
    if scalar_ty.is_floating():
        return tensor(builder.create_fsub(input.handle, other.handle), input.type)
    # int - int
    elif scalar_ty.is_int():
        return tensor(builder.create_sub(input.handle, other.handle), input.type)
    assert False


def _mul(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _binary_op_type_checking_impl(input, other, builder=builder)
    scalar_ty = input.type.scalar
    # float * float
    if scalar_ty.is_floating():
        return tensor(builder.create_fmul(input.handle, other.handle), input.type)
    # * int
    elif scalar_ty.is_int():
        return tensor(builder.create_mul(input.handle, other.handle), input.type)
    assert False


def _truediv(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _binary_op_type_checking_impl(
        input,
        other,
        builder=builder,
        allow_lhs_ptr=False,
        allow_rhs_ptr=False,
        arithmetic_check=True,
        div_or_mod=True,
    )
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    # float / int
    if input_scalar_ty.is_floating() and other_scalar_ty.is_int():
        other = _cast(other, input_scalar_ty, builder)
    # int / float
    elif input_scalar_ty.is_int() and other_scalar_ty.is_floating():
        input = _cast(input, other_scalar_ty, builder)
    # int / int (cast to float32)
    elif input_scalar_ty.is_int() and other_scalar_ty.is_int():
        input = _cast(input, float32, builder)
        other = _cast(other, float32, builder)
    # float / float (cast to highest exponent type)
    elif input_scalar_ty.is_floating() and other_scalar_ty.is_floating():
        if input_scalar_ty.fp_mantissa_width > other_scalar_ty.fp_mantissa_width:
            other = _cast(other, input_scalar_ty, builder)
        else:
            input = _cast(input, other_scalar_ty, builder)
    # unreachable
    else:
        assert False
    return tensor(builder.create_fdiv(input.handle, other.handle), input.type)


def _floordiv(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _binary_op_type_checking_impl(
        input,
        other,
        builder=builder,
        allow_lhs_ptr=False,
        allow_rhs_ptr=False,
        arithmetic_check=True,
        div_or_mod=True,
    )
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    if input_scalar_ty.is_int() and other_scalar_ty.is_int():
        ret_ty = _integer_promote_impl(input_scalar_ty, other_scalar_ty)
        input = _cast(input, ret_ty, builder)
        other = _cast(other, ret_ty, builder)
        if ret_ty.is_int_signed():
            return tensor(builder.create_sdiv(input.handle, other.handle), input.type)
        else:
            return tensor(builder.create_udiv(input.handle, other.handle), input.type)
    assert False


def _fdiv(
    input: tensor, other: tensor, ieee_rounding: bool, builder: ir.builder
) -> tensor:
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    if not input_scalar_ty.is_floating() or not other_scalar_ty.is_floating():
        raise ValueError("both operands of fdiv must have floating poscalar type")
    input, other = _binary_op_type_checking_impl(
        input,
        other,
        builder=builder,
        allow_lhs_ptr=False,
        allow_rhs_ptr=False,
        arithmetic_check=False,
        div_or_mod=True,
    )
    ret = builder.create_fdiv(input.handle, other.handle)
    ret.set_fdiv_ieee_rounding(ieee_rounding)
    return tensor(ret, input.type)


def _mod(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _binary_op_type_checking_impl(
        input,
        other,
        builder=builder,
        allow_lhs_ptr=False,
        allow_rhs_ptr=False,
        arithmetic_check=True,
        div_or_mod=True,
    )
    scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    # float % float
    if scalar_ty.is_floating():
        return tensor(builder.create_frem(input.handle, other.handle), input.type)
    # % int
    elif scalar_ty.is_int():
        if scalar_ty.int_signedness != other_scalar_ty.int_signedness:
            raise ValueError(
                "Cannot mod "
                + scalar_ty.__repr__()
                + " by "
                + other_scalar_ty.__repr__()
                + " "
                "because they have different signedness;"
                "this is unlikely to result in a useful answer. Cast them to the same signedness."
            )
        if scalar_ty.is_int_signed():
            return tensor(builder.create_srem(input.handle, other.handle), input.type)
        else:
            return tensor(builder.create_urem(input.handle, other.handle), input.type)
    assert False


##############
# bitwise ops
##############


def _bitwise_op_type_checking_impl(
    input: tensor,
    other: tensor,
    builder: ir.builder,
) -> Tuple[tensor, tensor]:
    input, other = _binary_op_type_checking_impl(
        input,
        other,
        builder=builder,
        allow_lhs_ptr=False,
        allow_rhs_ptr=False,
        arithmetic_check=False,
    )
    input_sca_ty = input.type.scalar
    other_sca_ty = other.type.scalar
    if not input_sca_ty.is_int() or not other_sca_ty.is_int():
        raise IncompatibleTypeErrorimpl(input_sca_ty, other_sca_ty)
    ret_sca_ty = _integer_promote_impl(input_sca_ty, other_sca_ty)
    if ret_sca_ty != input_sca_ty:
        input = _cast(input, ret_sca_ty, builder)
    if ret_sca_ty != other_sca_ty:
        other = _cast(other, ret_sca_ty, builder)
    return input, other


def _and_(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _bitwise_op_type_checking_impl(input, other, builder)
    return tensor(builder.create_and(input.handle, other.handle), input.type)


def _or_(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _bitwise_op_type_checking_impl(input, other, builder)
    return tensor(builder.create_or(input.handle, other.handle), input.type)


def _xor_(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _bitwise_op_type_checking_impl(input, other, builder)
    return tensor(builder.create_xor(input.handle, other.handle), input.type)


def _lshr(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _bitwise_op_type_checking_impl(input, other, builder)
    return tensor(builder.create_lshr(input.handle, other.handle), input.type)


def _shl(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _bitwise_op_type_checking_impl(input, other, builder)
    return tensor(builder.create_shl(input.handle, other.handle), input.type)


# ===----------------------------------------------------------------------===//
#                               Unary Operators
# ===----------------------------------------------------------------------===//


def _plus(input: tensor) -> tensor:
    return input


def _minus(input: tensor, builder: ir.builder) -> tensor:
    input_sca_ty = input.type.scalar
    if input_sca_ty.is_ptr():
        raise ValueError(
            "wrong type argument to unary minus (" + input_sca_ty.__repr__() + ")"
        )
    _0 = tensor(ir.constant.get_null_value(input_sca_ty.to_ir(builder)), input_sca_ty)
    return _sub(_0, input, builder)


def _invert(input: tensor, builder: tensor) -> tensor:
    input_sca_ty = input.type.scalar
    if input_sca_ty.is_ptr() or input_sca_ty.is_floating():
        raise ValueError(
            "wrong type argument to unary invert (" + input_sca_ty.__repr__() + ")"
        )
    _1 = tensor(
        ir.constant.get_all_ones_value(input_sca_ty.to_ir(builder)), input_sca_ty
    )
    return _xor_(input, _1, builder)


# ===----------------------------------------------------------------------===//
#                               Comparison Operators
# ===----------------------------------------------------------------------===//
def _bool_like(v: tensor) -> dtype:
    type = v.type
    if isinstance(type, block_type):
        shape = type.shape
        return block_type(int1, shape)

    return int1


def _greater_than(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _binary_op_type_checking_impl(input, other, builder=builder)
    scalar_ty = input.type.scalar
    # float > float
    if scalar_ty.is_floating():
        return tensor(
            builder.create_fcmpOGT(input.handle, other.handle), _bool_like(input)
        )
    # > int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tensor(
                builder.create_icmpSGT(input.handle, other.handle), _bool_like(input)
            )
        else:
            return tensor(
                builder.create_icmpUGT(input.handle, other.handle), _bool_like(input)
            )
    assert False


def _greater_equal(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _binary_op_type_checking_impl(input, other, builder=builder)
    scalar_ty = input.type.scalar
    # float >= float
    if scalar_ty.is_floating():
        return tensor(
            builder.create_fcmpOGE(input.handle, other.handle), _bool_like(input)
        )
    # >= int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tensor(
                builder.create_icmpSGE(input.handle, other.handle), _bool_like(input)
            )
        else:
            return tensor(
                builder.create_icmpUGE(input.handle, other.handle), _bool_like(input)
            )
    assert False


def _less_than(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _binary_op_type_checking_impl(input, other, builder=builder)
    scalar_ty = input.type.scalar
    # float < float
    if scalar_ty.is_floating():
        return tensor(
            builder.create_fcmpOLT(input.handle, other.handle), _bool_like(input)
        )
    # < int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tensor(
                builder.create_icmpSLT(input.handle, other.handle), _bool_like(input)
            )
        else:
            return tensor(
                builder.create_icmpULT(input.handle, other.handle), _bool_like(input)
            )
    assert False


def _less_equal(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _binary_op_type_checking_impl(input, other, builder=builder)
    scalar_ty = input.type.scalar
    # float < float
    if scalar_ty.is_floating():
        return tensor(
            builder.create_fcmpOLE(input.handle, other.handle), _bool_like(input)
        )
    # < int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tensor(
                builder.create_icmpSLE(input.handle, other.handle), _bool_like(input)
            )
        else:
            return tensor(
                builder.create_icmpULE(input.handle, other.handle), _bool_like(input)
            )
    assert False


def _equal(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _binary_op_type_checking_impl(input, other, builder=builder)
    scalar_ty = input.type.scalar
    # float == float
    if scalar_ty.is_floating():
        return tensor(
            builder.create_fcmpOEQ(input.handle, other.handle), _bool_like(input)
        )
    # == int
    elif scalar_ty.is_int():
        return tensor(
            builder.create_icmpEQ(input.handle, other.handle), _bool_like(input)
        )
    assert False


def _not_equal(input: tensor, other: tensor, builder: ir.builder) -> tensor:
    input, other = _binary_op_type_checking_impl(input, other, builder=builder)
    scalar_ty = input.type.scalar
    # float == float
    if scalar_ty.is_floating():
        return tensor(
            builder.create_fcmpUNE(input.handle, other.handle), _bool_like(input)
        )
    # == int
    elif scalar_ty.is_int():
        return tensor(
            builder.create_icmpNE(input.handle, other.handle), _bool_like(input)
        )
    assert False


# ===----------------------------------------------------------------------===//
#                               Block Creation
# ===----------------------------------------------------------------------===//


def _arange(start: int, end: int, builder: ir.builder) -> tensor:
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("arange's arguments must be of type constexpr")

    shape = [end - start]
    ret_ty = block_type(int32, shape)
    return tensor(builder.get_range(start, end), ret_ty)


def _zeros(shape: List[int], dtype: dtype, builder: ir.builder) -> tensor:
    _0 = ir.constant.get_null_value(dtype.to_ir(builder))
    ret_ty = block_type(dtype, shape)
    return tensor(builder.create_splat(_0, shape), ret_ty)


# ===----------------------------------------------------------------------===//
#                               Shape Manipulation
# ===----------------------------------------------------------------------===//


def _reshape(input: tensor, dst_shape: List[int], builder: ir.builder) -> tensor:
    numel = 1
    for s in dst_shape:
        numel *= s
    if input.type.numel != numel:
        raise ValueError("cannot reshape block of different shape")
    ret_ty = block_type(input.type.scalar, dst_shape)
    return tensor(builder.create_reshape(input.handle, dst_shape), ret_ty)


def _cat(lhs: tensor, rhs: tensor, builder: ir.builder) -> tensor:
    assert lhs.type.is_block() and rhs.type.is_block()
    assert lhs.type.shape[1:] == rhs.type.shape[1:]
    ret_shape = [lhs.type.shape[0] + rhs.type.shape[0]]
    ret_ty = block_type(lhs.type.scalar, ret_shape)
    return tensor(builder.create_cat(lhs.handle, rhs.handle), ret_ty)


def _broadcast_impl_shape(
    input: tensor, shape: List[int], builder: ir.builder
) -> tensor:
    type = input.type
    if not isinstance(type, block_type):
        ret_ty = block_type(input.type, shape)
        return tensor(builder.create_splat(input.handle, shape), ret_ty)

    src_shape = type.get_block_shapes()
    if len(src_shape) != len(shape):
        raise ValueError(f"Cannot broadcast, rank mismatch: {src_shape}, {shape}")
    if shape == src_shape:
        return input
    for i in range(len(src_shape)):
        if shape[i] != src_shape[i] and src_shape[i] != 1:
            raise ValueError(
                f"Cannot broadcast, the expanded size of the tensor ({shape[i]})"
                f" must match the existing size ({src_shape[1]}) at non-singleton dimension"
                f" {i}: {src_shape}, {shape}"
            )
    ret_ty = block_type(input.type.scalar, shape)
    return tensor(builder.create_broadcast(input.handle, shape), ret_ty)


def _broadcast_impl_value(
    lhs: tensor, rhs: tensor, builder: ir.builder
) -> Tuple[tensor, tensor]:
    lhs_ty = lhs.type
    rhs_ty = rhs.type

    # scalar, scalar
    if not isinstance(lhs_ty, block_type) and not isinstance(rhs_ty, block_type):
        return lhs, rhs

    # make_shape_compatible(block, block)
    if isinstance(lhs_ty, block_type) and isinstance(rhs_ty, block_type):
        lhs_shape = lhs_ty.get_block_shapes()
        rhs_shape = rhs_ty.get_block_shapes()
        if len(lhs_shape) != len(rhs_shape):
            raise ValueError(
                "Cannot make_shape_compatible: blocks must have the same rank"
            )
        ret_shape = []
        for i in range(len(lhs_shape)):
            left = lhs_shape[i]
            right = rhs_shape[i]
            if left == 1:
                ret_shape.append(right)
            elif right == 1:
                ret_shape.append(left)
            elif left == right:
                ret_shape.append(left)
            else:
                raise ValueError(
                    "Cannot make_shape_compatible: incompatible dimensions "
                    "at index " + str(i) + ": " + str(left) + " and " + str(right)
                )
        if lhs_shape != ret_shape:
            ret_ty = block_type(lhs_ty.scalar, ret_shape)
            lhs = tensor(builder.create_broadcast(lhs.handle, ret_shape), ret_ty)
        if rhs_shape != ret_shape:
            ret_ty = block_type(rhs_ty.scalar, ret_shape)
            rhs = tensor(builder.create_broadcast(rhs.handle, ret_shape), ret_ty)

        return lhs, rhs

    # make_shape_compatible(block, scalar)
    if isinstance(lhs_ty, block_type) and not isinstance(rhs_ty, block_type):
        rhs_ty = block_type(rhs_ty.scalar, lhs_ty.shape)
        rhs = tensor(
            builder.create_splat(rhs.handle, lhs_ty.get_block_shapes()), rhs_ty
        )

        return lhs, rhs

    # make_shape_compatible(scalar, block)
    if not isinstance(lhs_ty, block_type) and isinstance(rhs_ty, block_type):
        lhs_ty = block_type(lhs_ty.scalar, rhs_ty.shape)
        lhs = tensor(
            builder.create_splat(lhs.handle, rhs_ty.get_block_shapes()), lhs_ty
        )

    return lhs, rhs


#######
# dequantize
#######


def _dequantize(
    input: tensor,
    scale: tensor,
    shift: tensor,
    nbit: int,
    dst_ty: dtype,
    builder: ir.builder,
) -> tensor:
    input_ty = input.type
    assert isinstance(input_ty, block_type)
    assert input_ty.element_ty.is_int32() or input_ty.element_ty.is_int16()
    assert nbit in [2, 4, 8]
    assert dst_ty == float16

    shape = input_ty.get_block_shapes()
    factor = input_ty.element_ty.primitive_bitwidth // nbit
    dst_shape = shape[:-1] + (factor * shape[-1],)

    dst_ty = block_type(dst_ty, dst_shape)
    return tensor(
        builder.create_dequantize(
            input.handle, scale.handle, shift.handle, dst_ty.to_ir(builder)
        ),
        dst_ty,
    )


#######
# cast
#######


def _bitcast(input: tensor, dst_ty: dtype, builder: ir.builder) -> tensor:
    src_ty = input.type
    if isinstance(src_ty, block_type):
        dst_ty = block_type(dst_ty, src_ty.get_block_shapes())
    if src_ty == dst_ty:
        return input
    src_sca_ty = src_ty.scalar
    dst_sca_ty = dst_ty.scalar
    if src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
        return _cast(input, dst_ty, builder)
    # Bitcast
    src_bits = src_sca_ty.primitive_bitwidth
    dst_bits = dst_sca_ty.primitive_bitwidth
    if src_bits != dst_bits:
        raise ValueError(
            "Cannot bitcast data-type of size " + str(src_bits) + "to "
            "data-type of size " + str(dst_bits)
        )
    return tensor(builder.create_bitcast(input.handle, dst_ty.to_ir(builder)), dst_ty)


def _cast(input: tensor, dst_ty: dtype, builder: ir.builder) -> tensor:
    src_ty = input.type
    if isinstance(src_ty, block_type) and not isinstance(dst_ty, block_type):
        dst_ty = block_type(dst_ty, src_ty.get_block_shapes())
    if src_ty == dst_ty:
        return input
    src_sca_ty = src_ty.scalar
    dst_sca_ty = dst_ty.scalar
    # fp8 <=> bf16/fp16
    if (src_sca_ty.is_bf16() or src_sca_ty.is_fp16()) and dst_sca_ty.is_fp8():
        return tensor(
            builder.create_fp_trunc(input.handle, dst_ty.to_ir(builder)), dst_ty
        )
    if src_sca_ty.is_fp8() and (dst_sca_ty.is_bf16() or dst_sca_ty.is_fp16()):
        return tensor(
            builder.create_fp_ext(input.handle, dst_ty.to_ir(builder)), dst_ty
        )
    # bf16 <=> (not fp32)
    if (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()) or (
        dst_sca_ty.is_bf16() and not src_sca_ty.is_fp32()
    ):
        return _cast(_cast(input, float32, builder), dst_sca_ty, builder)

    # FP Truncation
    truncate_fp = (
        src_sca_ty.is_floating()
        and dst_sca_ty.is_floating()
        and src_sca_ty.fp_mantissa_width > dst_sca_ty.fp_mantissa_width
    )
    if truncate_fp:
        return tensor(
            builder.create_fp_trunc(input.handle, dst_ty.to_ir(builder)), dst_ty
        )

    # FP Extension
    ext_fp = (
        src_sca_ty.is_floating()
        and dst_sca_ty.is_floating()
        and src_sca_ty.fp_mantissa_width < dst_sca_ty.fp_mantissa_width
    )
    if ext_fp:
        return tensor(
            builder.create_fp_ext(input.handle, dst_ty.to_ir(builder)), dst_ty
        )

    # Int cast
    if (
        src_sca_ty.is_int()
        and dst_sca_ty.is_int()
        and (
            src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth
            or src_sca_ty.int_signedness != dst_sca_ty.int_signedness
        )
    ):
        sign_extend = src_sca_ty.is_int_signed() and not src_sca_ty.is_bool()
        return tensor(
            builder.create_int_cast(input.handle, dst_ty.to_ir(builder), sign_extend),
            dst_ty,
        )

    # Float to Int
    if src_sca_ty.is_floating() and dst_sca_ty.is_int():
        # TODO: is this correct?
        if dst_sca_ty.is_bool():
            return _not_equal(input, _to_tensor(0, builder), builder)
        else:
            return tensor(
                builder.create_fp_to_si(input.handle, dst_ty.to_ir(builder)), dst_ty
            )

    # int => float
    if src_sca_ty.is_int() and dst_sca_ty.is_floating():
        if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
            return tensor(
                builder.create_ui_to_fp(input.handle, dst_ty.to_ir(builder)), dst_ty
            )
        else:
            return tensor(
                builder.create_si_to_fp(input.handle, dst_ty.to_ir(builder)), dst_ty
            )

    # ptr => int
    if src_sca_ty.is_ptr() and dst_sca_ty.is_int():
        bitwidth = dst_sca_ty.int_bitwidth
        if bitwidth == 64:
            return tensor(
                builder.create_ptr_to_int(input.handle, dst_ty.to_ir(builder)), dst_ty
            )
        if bitwidth == 1:
            return _not_equal(
                _cast(input, int64, builder),
                tensor(builder.get_int64(0), int64),
                builder,
            )

    if not src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
        return tensor(
            builder.create_int_to_ptr(input.handle, dst_ty.to_ir(builder)), dst_ty
        )
    # Ptr . Ptr
    if src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
        return tensor(
            builder.create_bitcast(input.handle, dst_ty.to_ir(builder)), dst_ty
        )
    # * . Bool
    if dst_sca_ty.is_bool():
        if src_sca_ty.is_ptr():
            input = _cast(input, int64, builder)
        other = builder.get_int64(0)
        if src_ty.is_bool():
            assert isinstance(src_ty, block_type)
            other = builder.create_splat(other, src_ty.get_block_shapes())
        return tensor(builder.create_icmpNE(input.handle, other), dst_ty)
    assert False, f"cannot cast {input} to {dst_ty}"


# ===----------------------------------------------------------------------===//
#                               Memory Operators
# ===----------------------------------------------------------------------===//


def _parse_eviction_policy(eviction_policy):
    eviction = ir.EVICTION_POLICY.NORMAL  # default
    if eviction_policy:
        if eviction_policy == "evict_last":
            eviction = ir.EVICTION_POLICY.EVICT_LAST
        elif eviction_policy == "evict_first":
            eviction = ir.EVICTION_POLICY.EVICT_FIRST
        else:
            raise ValueError(f"Eviction policy {eviction_policy} not supported")
    return eviction


def _load(
    ptr: tensor,
    mask: Optional[tensor],
    other: Optional[tensor],
    cache_modifier: str,
    eviction_policy: str,
    is_volatile: bool,
    builder: ir.builder,
) -> tensor:
    if not ptr.type.scalar.is_ptr():
        raise ValueError(
            "Pointer argument of load instruction is " + ptr.type.__repr__()
        )
    if ptr.type.is_block():
        if mask:
            mask = _broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
        if other:
            other = _broadcast_impl_shape(other, ptr.type.get_block_shapes(), builder)

    if other:
        other = _cast(other, ptr.type.scalar.element_ty, builder)
    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty
    # treat bool* as int8*
    if elt_ty == int1:
        elt_ty = int8
        ptr_ty = pointer_type(elt_ty, ptr_ty.address_space)
        ptr = _cast(ptr, ptr_ty, builder)

    # cache modifier
    cache = ir.CACHE_MODIFIER.NONE  # default
    if cache_modifier:
        if cache_modifier == ".ca":
            cache = ir.CACHE_MODIFIER.CA
        elif cache_modifier == ".cg":
            cache = ir.CACHE_MODIFIER.CG
        else:
            raise ValueError(f"Cache modifier {cache_modifier} not supported")

    # eviction policy
    eviction = _parse_eviction_policy(eviction_policy)

    if ptr.type.is_block():
        shape = ptr.type.get_block_shapes()
        dst_ty = block_type(elt_ty, shape)
    else:
        dst_ty = elt_ty

    if not mask and not other:
        return tensor(
            builder.create_load(ptr.handle, cache, eviction, is_volatile), dst_ty
        )
    if not mask:
        raise ValueError("`other` cannot be provided without `mask`")

    if not other:
        other_ir = ir.undef.get(elt_ty.to_ir(builder))
        if ptr.type.is_block():
            other_ir = builder.create_splat(other_ir, ptr.type.get_block_shapes())
        other = tensor(other_ir, dst_ty)

    return tensor(
        builder.create_masked_load(
            ptr.handle, mask.handle, other.handle, cache, eviction, is_volatile
        ),
        dst_ty,
    )


def _store(
    ptr: tensor,
    val: tensor,
    mask: Optional[tensor],
    eviction_policy: str,
    builder: ir.builder,
) -> tensor:
    if not ptr.type.scalar.is_ptr():
        raise ValueError(
            "Pointer argument of store instruction is " + ptr.type.__repr__()
        )
    if ptr.type.is_block():
        val = _broadcast_impl_shape(val, ptr.type.get_block_shapes(), builder)
    if mask:
        mask = _broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty
    # treat bool* as int8*
    if elt_ty == int1:
        # convert to bool first and then store as int8
        val = _cast(val, int1, builder)
        elt_ty = int8
        ptr_ty = pointer_type(elt_ty, ptr_ty.address_space)
        ptr = _cast(ptr, ptr_ty, builder)
    # eviction policy
    eviction = _parse_eviction_policy(eviction_policy)
    # cast to target data-type
    val = _cast(val, elt_ty, builder)
    if not mask:
        return tensor(builder.create_store(ptr.handle, val.handle, eviction), void)
    if not mask.type.scalar.is_bool():
        raise ValueError("Mask must have boolean scalar type")
    return tensor(
        builder.create_masked_store(ptr.handle, val.handle, mask.handle, eviction), void
    )


#########
# atomic
#########


def _atomic_cas(ptr: tensor, cmp: tensor, val: tensor, builder: ir.builder) -> tensor:
    element_ty = ptr.type.scalar.element_ty
    if element_ty.primitive_bitwidth not in [16, 32, 64]:
        raise ValueError("atomic_cas only supports elements with width {16, 32, 64}")
    return tensor(
        builder.create_atomic_cas(ptr.handle, cmp.handle, val.handle), val.type
    )


def _atom_red_typechecking_impl(
    ptr: tensor, val: tensor, mask: tensor, op: str, builder: ir.builder
) -> Tuple[tensor, tensor, tensor]:
    if not ptr.type.scalar.is_ptr():
        raise ValueError(
            "Pointer argument of store instruction is " + ptr.type.__repr__()
        )

    element_ty = ptr.type.scalar.element_ty
    if element_ty is float16 and op != "add":
        raise ValueError("atomic_" + op + " does not support fp16")
    if element_ty in [int1, int8, int16, bfloat16]:
        raise ValueError("atomic_" + op + " does not support " + element_ty)
    if ptr.type.is_block():
        if mask:
            mask = _broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
        if val:
            val = _broadcast_impl_shape(val, ptr.type.get_block_shapes(), builder)
    val = _cast(val, ptr.type.scalar.element_ty, builder)
    if not mask:
        mask_ir = builder.get_int1(True)
        mask_ty = int1
        if ptr.type.is_block():
            mask_ir = builder.create_splat(mask_ir, ptr.type.get_block_shapes())
            mask_ty = block_type(int1, ptr.type.get_block_shapes())
        mask = tensor(mask_ir, mask_ty)
    return ptr, val, mask


def _atomic_max(ptr: tensor, val: tensor, mask: tensor, builder: ir.builder) -> tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "max", builder)
    sca_ty = val.type.scalar
    # direct call to atomic_max for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tensor(
                builder.create_atomic_rmw(
                    ir.ATOMIC_OP.MAX, ptr.handle, val.handle, mask.handle
                ),
                val.type,
            )
        else:
            return tensor(
                builder.create_atomic_rmw(
                    ir.ATOMIC_OP.UMAX, ptr.handle, val.handle, mask.handle
                ),
                val.type,
            )
    # for float
    # return atomic_smax(i_ptr, i_val) if val >= 0
    # return atomic_umin(i_ptr, i_val) if val < 0
    i_val = _bitcast(val, int32, builder)
    i_ptr = _bitcast(ptr, pointer_type(int32, 1), builder)
    pos = _greater_equal(
        val, tensor(ir.constant_float.get(sca_ty.to_ir(builder), 0), sca_ty), builder
    )
    neg = _less_than(
        val, tensor(ir.constant_float.get(sca_ty.to_ir(builder), 0), sca_ty), builder
    )
    pos_ret = tensor(
        builder.create_atomic_rmw(
            ir.ATOMIC_OP.MAX,
            i_ptr.handle,
            i_val.handle,
            _and_(mask, pos, builder).handle,
        ),
        i_val.type,
    )
    neg_ret = tensor(
        builder.create_atomic_rmw(
            ir.ATOMIC_OP.UMIN,
            i_ptr.handle,
            i_val.handle,
            _and_(mask, neg, builder).handle,
        ),
        i_val.type,
    )
    return _where(pos, pos_ret, neg_ret, builder)


def _atomic_min(ptr: tensor, val: tensor, mask: tensor, builder: ir.builder) -> tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "min", builder)
    sca_ty = val.type.scalar
    # direct call to atomic_min for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tensor(
                builder.create_atomic_rmw(
                    ir.ATOMIC_OP.MIN, ptr.handle, val.handle, mask.handle
                ),
                val.type,
            )
        else:
            return tensor(
                builder.create_atomic_rmw(
                    ir.ATOMIC_OP.UMIN, ptr.handle, val.handle, mask.handle
                ),
                val.type,
            )
    # for float
    # return atomic_smin(i_ptr, i_val) if val >= 0
    # return atomic_umax(i_ptr, i_val) if val < 0
    i_val = _bitcast(val, int32, builder)
    i_ptr = _bitcast(ptr, pointer_type(int32, 1), builder)
    pos = _greater_equal(
        val, tensor(ir.constant_float.get(sca_ty.to_ir(builder), 0), sca_ty), builder
    )
    neg = _less_than(
        val, tensor(ir.constant_float.get(sca_ty.to_ir(builder), 0), sca_ty), builder
    )
    pos_ret = tensor(
        builder.create_atomic_rmw(
            ir.ATOMIC_OP.MIN,
            i_ptr.handle,
            i_val.handle,
            _and_(mask, pos, builder).handle,
        ),
        i_val.type,
    )
    neg_ret = tensor(
        builder.create_atomic_rmw(
            ir.ATOMIC_OP.UMAX,
            i_ptr.handle,
            i_val.handle,
            _and_(mask, neg, builder).handle,
        ),
        i_val.type,
    )
    return _where(pos, pos_ret, neg_ret, builder)


def _atomic_add(ptr: tensor, val: tensor, mask: tensor, builder: ir.builder) -> tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "add", builder)
    sca_ty = val.type.scalar
    op = ir.ATOMIC_OP.FADD if sca_ty.is_floating() else ir.ATOMIC_OP.ADD
    return tensor(
        builder.create_atomic_rmw(op, ptr.handle, val.handle, mask.handle), val.type
    )


def _atomic_and(ptr: tensor, val: tensor, mask: tensor, builder: ir.builder) -> tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "and", builder)
    return tensor(
        builder.create_atomic_rmw(
            ir.ATOMIC_OP.AND, ptr.handle, val.handle, mask.handle
        ),
        val.type,
    )


def _atomic_or(ptr: tensor, val: tensor, mask: tensor, builder: ir.builder) -> tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "or", builder)
    return tensor(
        builder.create_atomic_rmw(ir.ATOMIC_OP.OR, ptr.handle, val.handle, mask.handle),
        val.type,
    )


def _atomic_xor(ptr: tensor, val: tensor, mask: tensor, builder: ir.builder) -> tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "xor", builder)
    return tensor(
        builder.create_atomic_rmw(
            ir.ATOMIC_OP.XOR, ptr.handle, val.handle, mask.handle
        ),
        val.type,
    )


def _atomic_xchg(ptr: tensor, val: tensor, mask: tensor, builder: ir.builder) -> tensor:
    ptr, val, mask = _atom_red_typechecking_impl(ptr, val, mask, "xchg", builder)
    return tensor(
        builder.create_atomic_rmw(
            ir.ATOMIC_OP.XCHG, ptr.handle, val.handle, mask.handle
        ),
        val.type,
    )


# ===----------------------------------------------------------------------===//
#                               Linear Algebra
# ===----------------------------------------------------------------------===//


def _dot(
    a: tensor,
    b: tensor,
    trans_a: bool,
    trans_b: bool,
    allow_tf32: bool,
    builder: ir.builder,
) -> tensor:
    in_a = 1 if not trans_a else 0
    in_b = 1 if trans_b else 0
    assert a.type.is_block() and b.type.is_block()
    assert len(a.shape) == 2 and len(b.shape) == 2
    assert a.shape[in_a] == b.shape[in_b]
    assert (
        a.shape[0] >= 16 and a.shape[1] >= 16 and b.shape[1] >= 16
    ), "small blocks not supported!"
    if a.type.scalar.is_int():
        _0 = builder.get_int32(0)
        ret_scalar_ty = int32
    else:
        _0 = builder.get_float32(0)
        ret_scalar_ty = float32
    M = a.type.shape[in_a ^ 1]
    N = b.type.shape[in_b ^ 1]
    _0 = builder.create_splat(_0, [M, N])
    ret_ty = block_type(ret_scalar_ty, [M, N])
    ret = builder.create_dot(a.handle, b.handle, _0, trans_a, trans_b, allow_tf32)
    return tensor(ret, ret_ty)


# ===----------------------------------------------------------------------===//
#                               Indexing
# ===----------------------------------------------------------------------===//


def _where(condition: tensor, x: tensor, y: tensor, builder: ir.builder) -> tensor:
    condition = _cast(condition, int1, builder)
    if condition.type.is_block():
        condition, x = _broadcast_impl_value(condition, x, builder)
        x, y = _broadcast_impl_value(x, y, builder)
        condition, x = _broadcast_impl_value(condition, x, builder)

    x, y = _binary_op_type_checking_impl(
        x,
        y,
        builder=builder,
        allow_lhs_ptr=True,
        allow_rhs_ptr=True,
    )
    if not condition.type.is_block():
        condition, _ = _broadcast_impl_value(condition, x, builder)
    ret_ty = x.type
    return tensor(builder.create_select(condition.handle, x.handle, y.handle), ret_ty)


# ===----------------------------------------------------------------------===//
#                               Reductions
# ===----------------------------------------------------------------------===


def _reduce_impl(
    input: tensor,
    axis: int,
    builder: ir.builder,
    name: str,
    FLOAT_OP: ir.REDUCE_OP,
    INT_OP: ir.REDUCE_OP,
) -> tensor:
    scalar_ty = input.type.scalar
    # input is extended to 32-bits if necessary
    # this increases numerical accuracy and can be done pretty much for free
    # on GPUs
    if scalar_ty.is_int() and scalar_ty.int_bitwidth <= 32:
        input = _cast(input, int32, builder)

    # hardware doesn't support FMAX, FMIN, CMP for bfloat16
    if scalar_ty is bfloat16:
        input = _cast(input, float32, builder)

    # choose the right unsigned operation
    if scalar_ty.is_int_unsigned():
        int_op_to_unit = {
            ir.REDUCE_OP.MIN: ir.REDUCE_OP.UMIN,
            ir.REDUCE_OP.MAX: ir.REDUCE_OP.UMAX,
            ir.REDUCE_OP.ARGMIN: ir.REDUCE_OP.ARGUMIN,
            ir.REDUCE_OP.ARGMAX: ir.REDUCE_OP.ARGUMAX,
        }
        if INT_OP in int_op_to_unit:
            INT_OP = int_op_to_unit[INT_OP]

    # get result type
    shape = input.type.shape
    ret_shape = []
    for i, s in enumerate(shape):
        if i != axis:
            ret_shape.append(s)
    if len(ret_shape) == 0:
        res_ty = scalar_ty
    else:
        res_ty = block_type(scalar_ty, ret_shape)

    if scalar_ty.is_floating():
        return tensor(builder.create_reduce(input.handle, FLOAT_OP, axis), res_ty)
    elif scalar_ty.is_int():
        return tensor(builder.create_reduce(input.handle, INT_OP, axis), res_ty)
    assert False


def _min(input: tensor, axis: int, builder: ir.builder) -> tensor:
    return _reduce_impl(
        input, axis, builder, "min", ir.REDUCE_OP.FMIN, ir.REDUCE_OP.MIN
    )


def _argmin(input: tensor, axis: int, builder: ir.builder) -> tensor:
    return _reduce_impl(
        input, axis, builder, "argmin", ir.REDUCE_OP.ARGFMIN, ir.REDUCE_OP.ARGMIN
    )


def _max(input: tensor, axis: int, builder: ir.builder) -> tensor:
    return _reduce_impl(
        input, axis, builder, "max", ir.REDUCE_OP.FMAX, ir.REDUCE_OP.MAX
    )


def _argmax(input: tensor, axis: int, builder: ir.builder) -> tensor:
    return _reduce_impl(
        input, axis, builder, "argmax", ir.REDUCE_OP.ARGFMAX, ir.REDUCE_OP.ARGMAX
    )


def _sum(input: tensor, axis: int, builder: ir.builder) -> tensor:
    return _reduce_impl(
        input, axis, builder, "sum", ir.REDUCE_OP.FADD, ir.REDUCE_OP.ADD
    )


def _xor_sum(input: tensor, axis: int, builder: ir.builder) -> tensor:
    scalar_ty = input.type.scalar
    if not scalar_ty.is_int():
        raise ValueError("xor_sum only supported for integers")
    return _reduce_impl(input, axis, builder, "sum", ir.REDUCE_OP.XOR, ir.REDUCE_OP.XOR)


# -----------------------
# Utilities
# -----------------------


def _clock(builder: ir.builder) -> tensor:
    return tensor(builder.create_clock(), int64)


def _globaltimer(builder: ir.builder) -> tensor:
    return tensor(builder.create_globaltimer, int64)


# ===----------------------------------------------------------------------===
#                               Math
# ===----------------------------------------------------------------------===


def _umulhi(x: tensor, y: tensor, builder: ir.builder) -> tensor:
    x, y = _binary_op_type_checking_impl(x, y, builder=builder)
    return tensor(builder.create_umulhi(x.handle, y.handle), x.type)


def _exp(x: tensor, builder: ir.builder) -> tensor:
    return tensor(builder.create_exp(x.handle), x.type)


def _log(x: tensor, builder: ir.builder) -> tensor:
    return tensor(builder.create_log(x.handle), x.type)


def _cos(x: tensor, builder: ir.builder) -> tensor:
    return tensor(builder.create_cos(x.handle), x.type)


def _sin(x: tensor, builder: ir.builder) -> tensor:
    return tensor(builder.create_sin(x.handle), x.type)


def _sqrt(x: tensor, builder: ir.builder) -> tensor:
    return tensor(builder.create_sqrt(x.handle), x.type)


##


def _multiple_of(x: tensor, values: List[int]) -> tensor:
    if len(x.shape) != len(values):
        raise ValueError(
            "Shape of input to multiple_of does not match the length of values"
        )
    x.handle.multiple_of(values)
    return x


def _max_contiguous(x: tensor, values: List[int]) -> tensor:
    if len(x.shape) != len(values):
        raise ValueError(
            "Shape of input to max_contiguous does not match the length of values"
        )
    x.handle.max_contiguous(values)
    return x


def _debug_barrier(builder: ir.builder) -> tensor:
    return tensor(builder.create_barrier(""), void)


# -----------------------
# SPMD Programming Model
# -----------------------
def _constexpr_to_value(v):
    if isinstance(v, constexpr):
        return v.value
    return v


@builtin
def program_id(axis, _builder: ir.builder = None):
    """
    Returns the id of the current program instance along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Has to be either 0, 1 or 2.
    :type axis: int
    """
    # if axis == -1:
    #     pid0 = program_id(0, _builder)
    #     pid1 = program_id(1, _builder)
    #     pid2 = program_id(2, _builder)
    #     npg0 = num_programs(0, _builder)
    #     npg1 = num_programs(0, _builder)
    #     return pid0 + pid1*npg0 + pid2*npg0*npg1
    axis = _constexpr_to_value(axis)
    return _program_id(axis, _builder)


@builtin
def num_programs(axis, _builder: ir.builder = None):
    """
    Returns the number of program instances launched along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Has to be either 0, 1 or 2.
    :type axis: int
    """
    axis = _constexpr_to_value(axis)
    return _num_programs(axis, _builder)


# -----------------------
# Block Initialization
# -----------------------


@builtin
def arange(start, end, _builder: ir.builder = None):
    """
    Returns contiguous values within the open interval [:code:`start`, :code:`end`).

    :param start: Start of the interval. Must be a power of two.
    :type start: int
    :param stop: End of the interval. Must be a power of two >= start.
    :type stop: int
    """
    start = _constexpr_to_value(start)
    end = _constexpr_to_value(end)
    return _arange(start, end, _builder)


@builtin
def zeros(shape, dtype, _builder: ir.builder = None):
    """
    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., :code:`float16`
    :type dtype: DType
    """
    for i, d in enumerate(shape):
        if not isinstance(d, constexpr):
            raise TypeError(f"Shape element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(
                f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]"
            )
    shape = [x.value for x in shape]
    dtype = _constexpr_to_value(dtype)
    return _zeros(shape, dtype, _builder)


# -----------------------
# dequantize
# -----------------------


@builtin
def dequantize(input, scale, shift, nbit, dst_ty=float16, _builder: ir.builder = None):
    """
    Tries to dequantize the input to given dtype
    """
    nbit = _constexpr_to_value(nbit)
    return _dequantize(input, scale, shift, nbit, dst_ty, _builder)


# -----------------------
# Shape Manipulation
# -----------------------


@builtin
def broadcast(input, other, _builder: ir.builder = None):
    """
    Tries to broadcast the two given blocks to a common compatible shape.

    :param input: The first input tensor.
    :type input: Block
    :param other: The second input tensor.
    :type other: Block
    """
    return _broadcast_impl_value(input, other, _builder)


@builtin
def broadcast_to(input, shape, _builder: ir.builder = None):
    """
    Tries to broadcast the given tensor to a new :code:`shape`.

    :param input: The input tensor.
    :type input: Block
    :param shape: The desired shape.
    :type shape: Tuple[int]
    """
    return _broadcast_impl_shape(input, shape, _builder)


@builtin
def cat(input, other, _builder: ir.builder = None):
    """
    Concatenate the given blocks

    :param input: The first input tensor.
    :type input:
    :param other: The second input tensor.
    :type other:
    """
    return _cat(input, other, _builder)


@builtin
def reshape(input, shape, _builder: ir.builder = None):
    """
    Tries to reshape the given tensor to a new shape.

    :param input: The input tensor.
    :type input:
    :param shape: The desired shape.
    :type shape: Tuple[int]

    """
    shape = [x.value for x in shape]
    return _reshape(input, shape, _builder)


# -----------------------
# Linear Algebra
# -----------------------


@builtin
def dot(
    input,
    other,
    trans_a=False,
    trans_b=False,
    allow_tf32=True,
    _builder: ir.builder = None,
):
    """
    Returns the matrix product of two blocks.

    The two blocks must be two dimensionals and have compatible inner dimensions.

    :param input: The first tensor to be multiplied.
    :type input: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param other: The second tensor to be multiplied.
    :type other: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    """
    allow_tf32 = _constexpr_to_value(allow_tf32)
    return _dot(input, other, trans_a, trans_b, allow_tf32, _builder)


# -----------------------
# Non-Atomic Memory Operations
# -----------------------


@builtin
def load(
    pointer,
    mask=None,
    other=None,
    cache_modifier="",
    eviction_policy="",
    volatile=False,
    _builder: ir.builder = None,
) -> tensor:
    """
    Return a tensor of data whose values are, elementwise, loaded from memory at location defined by :code:`pointer`.

    :param *:
    :code:`mask` and :code:`other` are implicitly broadcast to :code:`pointer.shape`.

    :code:`other` is implicitly typecast to :code:`pointer.dtype.element_ty`.

    :param pointer: Pointers to the data to be loaded.
    :type pointer: Block of dtype=triton.PointerDType
    :param mask: if mask[idx] is false, do not load the data at address :code:`pointer[idx]`.
    :type mask: Block of triton.int1, optional
    :param other: if mask[idx] is false, return other[idx]
    :type other: Block, optional
    :param cache_modifier: changes cache option in nvidia ptx
    'type cache_modifier: str, optional
    """
    # mask, other can be constexpr
    if mask is not None:
        mask = _to_tensor(mask, _builder)
    if other is not None:
        other = _to_tensor(other, _builder)
    cache_modifier = _constexpr_to_value(cache_modifier)
    eviction_policy = _constexpr_to_value(eviction_policy)
    volatile = _constexpr_to_value(volatile)
    return _load(
        pointer, mask, other, cache_modifier, eviction_policy, volatile, _builder
    )


@builtin
def store(
    pointer,
    value,
    mask=None,
    eviction_policy="",
    _builder: ir.builder = None,
) -> tensor:
    """
    Stores :code:`value` tensor of elements in memory, element-wise, at the memory locations specified by :code:`pointer`.

    :param *:
    :code:`value` is implicitly broadcast to :code:`pointer.shape` and typecast to :code:`pointer.dtype.element_ty`.

    :param pointer: The memory locations where the elements of :code:`value` are stored.
    :type pointer: Block of dtype=triton.PointerDType
    :param value: The tensor of elements to be stored.
    :type value: Block
    :param mask: If mask[idx] is false, do not store :code:`value[idx]` at :code:`pointer[idx]`.
    :type mask: Block of triton.int1, optional
    """
    # value can be constexpr
    value = _to_tensor(value, _builder)
    if mask is not None:
        mask = _to_tensor(mask, _builder)
    return _store(pointer, value, mask, eviction_policy, _builder)


# -----------------------
# Atomic Memory Operations
# -----------------------



def dispatch(
    func,
    *,
    lib_name: str,
    lib_path: str,
    args: list,
    arg_type_symbol_dict: dict,
    ret_shape: Optional[Sequence[int]] = None,
    _builder: ir.builder = None,
) -> tensor:
    """
    Dispatch a function to a library

    :param *:
    :param func: the function to dispatch
    :param lib_name: the name of the library
    :param lib_path: the path of the library
    :param args: the arguments of the function
    :param arg_type_symbol_dict: the type of the arguments
    :param ret_shape: the shape of the return value
    :param _builder: the builder

    :return: the return value of the function
    """
    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(
            f"length of input args does not match."
            f"Expect {len(args)}, got {num_args}"
        )

    arg_types: List[Union[type, dtype]] = []
    arg_list: List[Any] = []
    for arg in args:
        if isinstance(arg, tensor):
            arg_types.append(arg.dtype)
            arg_list.append(arg.handle)
        else:
            arg_types.append(type(arg))
            arg_list.append(arg)
    arg_types_t = tuple(arg_types)

    if arg_types not in arg_type_symbol_dict:
        raise ValueError(
            f"input arg type does not match."
            f"Expect one of {arg_type_symbol_dict.keys()}, got {arg_types_t}"
        )
    else:
        symbol = arg_type_symbol_dict[arg_types_t][0]
        ret_type = arg_type_symbol_dict[arg_types_t][1]
        ret_type = (
            block_type(ret_type, ret_shape) if ret_shape is not None else ret_type
        )
        return tensor(
            func(lib_name, lib_path, symbol, arg_list, ret_type.to_ir(_builder)),
            ret_type,
        )


def elementwise(
    *,
    lib_name: str,
    lib_path: str,
    args: list,
    arg_type_symbol_dict: dict,
    _builder: ir.builder = None,
) -> tensor:
    """
    Dispatch an elementwise function to a library

    :param *:
    :param lib_name: the name of the library
    :param lib_path: the path of the library
    :param args: the arguments of the function
    :param arg_type_symbol_dict: the type of the arguments
    :param _builder: the builder

    :return: the return value of the function
    """
    dispatch_args = args.copy()
    all_scalar = True
    ret_shape = None
    for dispatch_arg in dispatch_args:
        if dispatch_arg.type.is_block():
            all_scalar = False
    if not all_scalar:
        if len(args) == 1:
            dispatch_args[0] = _to_tensor(dispatch_args[0], _builder)
            ret_shape = dispatch_args[0].shape
        elif len(args) == 2:
            dispatch_args[0] = _to_tensor(dispatch_args[0], _builder)
            dispatch_args[1] = _to_tensor(dispatch_args[1], _builder)
            dispatch_args[0], dispatch_args[1] = _binary_op_type_checking_impl(
                dispatch_args[0], dispatch_args[1], builder=_builder
            )
            ret_shape = dispatch_args[0].shape
        else:
            for i in range(len(dispatch_args)):
                dispatch_args[i] = _to_tensor(dispatch_args[i], _builder)
            broadcast_arg = dispatch_args[0]
            # Get the broadcast shape over all the arguments
            for i in range(len(dispatch_args)):
                _, broadcast_arg = _binary_op_type_checking_impl(
                    dispatch_args[i], broadcast_arg, builder=_builder
                )
            # Change the shape of each argument based on the broadcast shape
            for i in range(len(dispatch_args)):
                dispatch_args[i], _ = _binary_op_type_checking_impl(
                    dispatch_args[i], broadcast_arg, builder=_builder
                )
            ret_shape = broadcast_arg.shape
    func = getattr(_builder, "create_extern_elementwise")
    return dispatch(
        func,
        lib_name=lib_name,
        lib_path=lib_path,
        args=dispatch_args,
        arg_type_symbol_dict=arg_type_symbol_dict,
        ret_shape=ret_shape,
        _builder=_builder,
    )


class ExternalFunction:
    """
    A wrapper for external functions
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError(
                "Did you forget to add @triton.jit ? (`_builder` argument must be provided outside of JIT functions.)"
            )
        return self.fn(*args, **kwargs)


def extern(fn):
    """
    A decorator for external functions
    """
    return ExternalFunction(fn)

