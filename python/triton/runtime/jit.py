from __future__ import annotations, division
import ast
import hashlib
import inspect
import itertools
import os
import re
import textwrap
from collections import defaultdict
from functools import cached_property
from typing import Callable, Generic, Iterable, Optional, TypeVar, Union, overload
from ..runtime.driver import driver

TRITON_MODULE = __name__[:-len(".runtime.jit")]

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
        self.hasher = hashlib.sha256(src.encode("utf-8"))
        self.globals = globals

    @property
    def ret(self):
        return self.hasher.hexdigest()

    def visit_Name(self, node):
        if node.id in self.local_names:
            # The global name is hidden by the local name.
            return None
        return self.globals.get(node.id, None)

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        while isinstance(lhs, ast.Attribute):
            lhs = self.visit(lhs.value)
        if lhs is None or (getattr(lhs, "__name__", "") == TRITON_MODULE):
            return None
        return getattr(lhs, node.attr)

    def visit_Call(self, node):

        def is_triton_builtin(func):
            if inspect.isbuiltin(node.func):
                return True
            module = getattr(func, "__module__", "")
            return module.startswith(TRITON_MODULE)

        func = self.visit(node.func)
        assert func is None or is_triton_builtin(func) or isinstance(
            func, JITFunction
        ), f'Function "{func.__name__}" is being called from a Triton function but is not a Triton function itself. Decorate it with @triton.jit to fix this'

        # Traverse arguments as well as node.func so we can find JITFunctions
        # passed to tl.reduce or tl.associative_scan as the combine_fn
        for obj in itertools.chain(
            (func, ),
                map(self.visit, node.args),
            (self.visit(kw.value) for kw in node.keywords),
        ):
            if not isinstance(obj, JITFunction):
                continue
            if is_triton_builtin(obj):
                continue

            func_cache_key = obj.cache_key
            noinline = str(getattr(obj, "noinline", False))

            key = func_cache_key + noinline
            self.hasher.update(key.encode("utf-8"))

    def visit_FunctionDef(self, node):
        # Save the local name which may hide the global name.
        self.local_names = [arg.arg for arg in node.args.args]
        self.generic_visit(node)

    def visit_Assign(self, node):
        _names = []
        for target in node.targets:
            _names += [self.visit(target)]
        if len(_names) == 1:
            self.local_names += _names
        else:
            raise TypeError("Simultaneous multiple assignment is not supported.")
        self.generic_visit(node)


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
    """Represents a parameter (name plus metadata) to a @jit'ed function."""

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
    def annotation_type(self):
        annotation = self.annotation
        for ty1, ty2 in [("uint", 'u'), ("int", 'i')]:
            width = annotation[annotation.find(ty1) + len(ty1):]
            if width and ty1 in annotation:
                return f"{ty2}{width}"
        if annotation == "bool":
            return "u1"
        return ""

    @cached_property
    def is_constexpr(self):
        return "constexpr" in self.annotation

    @cached_property
    def is_const(self):
        return "const" in self.annotation and not self.is_constexpr

    @property
    def default(self):
        return self._param.default

    @property
    def has_default(self):
        return self._param.default != inspect.Parameter.empty


def compute_spec_key(v):

    if hasattr(v, "data_ptr") and (v.data_ptr() % 16 == 0):
        return "D"
    elif isinstance(v, int):
        # bool is a subclass of int, so we don't check explicitly above.
        if (v % 16 == 0):
            return "D"
        elif v == 1:
            return "1"
    return "N"


dtype2str = {}


def mangle_type(arg, is_const=False):

    if arg is None:
        return "none"
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
    else:
        # dtypes are hashable so we can memoize this mapping:
        dsk = (arg.dtype, is_const)
        res = dtype2str.get(dsk, None)
        if res is None:
            res = ("*k" if dsk[1] else "*") + type_canonicalisation_dict[str(dsk[0]).split('.')[-1]]
            dtype2str[dsk] = res
        return res


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


def serialize_specialization_data(name, signature, constants, attrs, options, key):
    constants = {key: str(value) if value.__class__.__name__ == "dtype" else value for key, value in constants.items()}
    import json
    obj = {
        'name': name, 'signature': signature, 'constants': constants, 'attrs': attrs.to_dict(), 'options':
        options.__dict__, 'key': key
    }
    serialized_obj = json.dumps(obj)
    return serialized_obj


def create_function_from_signature(sig, kparams):
    """
    Equivalent to sig.bind followed by apply_defaults. This generates a
    native Python function (using exec) which can be memoized on a per-kernel
    basis to avoid having to run these expensive functions -- which constitute
    much of the kernel launch overhead -- every time we run the kernel.
    """

    assert len(sig.parameters) == len(kparams)

    # Create the function argument list and the dict entries for the return statement
    func_args = []
    dict_entries = []
    constexpr_vals = []
    non_constexpr_vals = []
    signature_types = []
    specialisations = []

    for ((name, sp), kp) in zip(sig.parameters.items(), kparams):
        if sp.default is inspect.Parameter.empty:
            func_args.append(name)
            dict_entries.append(f"'{name}': {name}")
        else:
            func_args.append(f"{name}=default_{name}")
            dict_entries.append(f"'{name}': {name}")
        if kp.is_constexpr:
            constexpr_vals.append(name)
        else:
            non_constexpr_vals.append(name)
            if not kp.do_not_specialize:
                specialisations.append('compute_spec_key(%s)' % name)
            if kp.annotation_type:
                signature_types.append('"%s"' % kp.annotation_type)
            else:
                signature_types.append('mangle_type(%s, %s)' % (name, 'True' if kp.is_const else 'False'))

    cache_key = ''.join([x + ', ' for x in signature_types + specialisations])
    constexpr_vals = ''.join([x + ', ' for x in constexpr_vals])
    non_constexpr_vals = ''.join([x + ', ' for x in non_constexpr_vals])

    func_args.append('**excess_kwargs')

    # Join all arguments into a function definition string
    args_str = ', '.join(func_args)
    dict_str = ', '.join(dict_entries)
    func_body = "def dynamic_func(%s):\n    return {%s}, (%s), (%s), (%s), excess_kwargs" % (
        args_str, dict_str, cache_key, constexpr_vals, non_constexpr_vals)

    # Prepare defaults to be inserted into function namespace
    func_namespace = {
        f"default_{name}": param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }

    func_namespace['mangle_type'] = mangle_type
    func_namespace['compute_spec_key'] = compute_spec_key

    # Execute the function string in func_namespace to create the function
    exec(func_body, func_namespace)

    # Extract the newly created function from the namespace
    return func_namespace['dynamic_func']


type_canonicalisation_dict = {
    "bool": "i1",
    "float8e4nv": "fp8e4nv",
    "float8e5": "fp8e5",
    "float8e4b15": "fp8e4b15",
    "float8_e4m3fn": "fp8e4nv",
    "float8e4b8": "fp8e4b8",
    "float8_e4m3fnuz": "fp8e4b8",
    "float8_e5m2": "fp8e5",
    "float8e5b16": "fp8e5b16",
    "float8_e5m2fnuz": "fp8e5b16",
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

for v in list(type_canonicalisation_dict.values()):
    type_canonicalisation_dict[v] = v


class JITFunction(KernelInterface[T]):
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
    def _spec_of(arg):
        if hasattr(arg, "data_ptr"):
            return arg.data_ptr() % JITFunction.divisibility == 0
        elif isinstance(arg, int):
            return (arg % 16 == 0, arg == 1)
        return (arg is None, )

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

        divisible_by_16 = {
            param.num
            for param, arg in zip(self.params, args)
            if is_divisible_by_16(arg) and not param.do_not_specialize
        }
        equal_to_1 = {
            param.num
            for param, arg in zip(self.params, args)
            if isinstance(arg, int) and not isinstance(arg, bool) and arg == 1 and not param.do_not_specialize
        }
        # folded equal_to_1 and None
        # TODO: method to collect all folded args
        return AttrsDescriptor(tuple(divisible_by_16), tuple(equal_to_1))
        # return _triton.code_gen.instance_descriptor(divisible_by_16,
        # equal_to_1)

    @staticmethod
    def _type_of(key, is_const=False):
        # `None` is nullptr.  Implicitly convert to *i8.
        if key is None:
            return "*i8"
        elif isinstance(key, str):
            return key

        dtype_str = str(key).split(".")[-1]
        dtype_str = type_canonicalisation_dict[dtype_str]
        const_str = "*k" if is_const else "*"
        return const_str + dtype_str

    def _make_constants(self, constexpr_key):
        constants = dict(zip(self.constexprs, constexpr_key))
        return constants

    def _call_hook(
        self,
        key,
        signature,
        device,
        constants,
        options,
        configs,
    ):
        if JITFunction.cache_hook is None:
            return False

        name = self.fn.__name__
        module = self.fn.__module__
        arg_reprs = ", ".join([f"{param.name}: {ty}" for param, ty in zip(self.params, key[1])])
        repr = f"{name}[num_warps={options.num_warps}, num_ctas={options.num_ctas}, num_stages={options.num_stages}, enable_fp_fusion={options.enable_fp_fusion}]({arg_reprs})"

        class JitFunctionInfo:

            def __init__(self, module, name, jit_function):
                self.module = module
                self.name = name
                self.jit_function = jit_function
                pass

        specialization_data = serialize_specialization_data(name, signature, constants, configs[0], options, key)

        kwargs = {
            'signature': signature,
            'device': device,
            'constants': constants,
            'num_warps': options.num_warps,
            'num_ctas': options.num_ctas,
            'num_stages': options.num_stages,
            'enable_fp_fusion': options.enable_fp_fusion,
            'extern_libs': options.extern_libs,
            'configs': configs,
            'specialization_data': specialization_data,
        }

        return JITFunction.cache_hook(
            key=key,
            repr=repr,
            fn=JitFunctionInfo(module, name, self),
            compile={"key": key, **kwargs},
            is_manual_warmup=False,
            already_compiled=False,
        )

    def add_pre_run_hook(self, hook):
        '''
        Add a hook that will be executed prior to the execution of run
        function with args and kwargs passed into the kernel
        '''
        assert callable(hook)
        self.pre_run_hooks.append(hook)

    def create_binder(self):
        """
        Precompute as much as possible.
        """
        from ..compiler import CompiledKernel, compile, ASTSource, make_backend
        self.CompiledKernel = CompiledKernel
        self.compile = compile
        self.ASTSource = ASTSource
        self.make_backend = make_backend
        self.binder = create_function_from_signature(self.signature, self.params)
        self.constexpr_indices = [i for (i, p) in enumerate(self.params) if p.is_constexpr]
        self.non_constexpr_indices = [i for (i, p) in enumerate(self.params) if not p.is_constexpr]
        self.specialised_indices = [
            i for (i, p) in enumerate(self.params) if (not p.do_not_specialize) and (not p.is_constexpr)
        ]

    def run(self, *args, grid, warmup, **kwargs):
        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        kwargs["debug"] = self.debug

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        if self.binder is None:
            self.create_binder()

        bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)

        # compute cache key
        key = ''.join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
        kernel = self.cache[device].get(key, None)

        if kernel is None:
            # Kernel is not cached; we have to compile.
            target = driver.active.get_current_target()
            backend = self.make_backend(target)
            options = backend.parse_options(kwargs)

            # deprecated arguments
            assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
            assert "device" not in kwargs, "device option is deprecated; current device will be used"
            assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"
            for k in excess_kwargs:
                if k not in options.__dict__:
                    raise KeyError("Keyword argument %s was specified but unrecognised" % k)

            bound_vals = tuple(bound_args.values())

            # `None` is nullptr. Implicitly convert to *i8. This needs to be
            # done here rather than when we build the signature as otherwise
            # the kernel cache key could not distinguish between byte pointers
            # and None arguments, resulting in a downstream mismatch:
            sigkeys = [self.params[i].name for i in self.non_constexpr_indices]
            sigvals = sig_and_spec[:len(sigkeys)]
            signature = {k: ('*i8' if (v == 'none') else v) for (k, v) in zip(sigkeys, sigvals)}

            configs = (self._get_config(*bound_vals), )
            constants = {
                p.name: v
                for (v, p) in zip(bound_vals, self.params)
                if p.is_constexpr or p.num in configs[0].equal_to_1 or v is None
            }
            for i, arg in constants.items():
                if callable(arg):
                    raise TypeError(f"Callable constexpr at index {i} is not supported")

            if self._call_hook(key, signature, device, constants, options, configs):
                return None
            # compile the kernel
            src = self.ASTSource(self, signature, constants, configs[0])
            kernel = self.compile(
                src,
                target=target,
                options=options.__dict__,
            )
            self.cache[device][key] = kernel

        if not warmup:
            # canonicalize grid
            assert grid is not None
            if callable(grid):
                # Arguments are passed as a dict to `grid`, by contract.
                # TODO(jlebar): In the new launch API, pass the compiler flags as a
                # second parameter to `grid`.
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1

            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *non_constexpr_vals)
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       self.CompiledKernel.launch_enter_hook, self.CompiledKernel.launch_exit_hook, *non_constexpr_vals)
        return kernel

    def __init__(self, fn, version=None, do_not_specialize=None, debug=None, noinline=None, repr=None,
                 launch_metadata=None):
        do_not_specialize = do_not_specialize if do_not_specialize else []

        self.fn = fn
        self.module = fn.__module__
        self.version = version
        self.signature = inspect.signature(fn)
        self.do_not_specialize = do_not_specialize
        self.starting_line_number = inspect.getsourcelines(fn)[1]
        self.repr = lambda _: fn.__name__ if repr is None else repr(_)
        self.launch_metadata = launch_metadata

        self.binder = None

        self.params = []
        for i, param in enumerate(self.signature.parameters.values()):
            dns = do_not_specialize and (i in do_not_specialize or param.name in do_not_specialize)
            self.params.append(KernelParam(i, param, dns))

        # function source code (without decorators)
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[re.search(r"^def\s+\w+\s*\(", self.src, re.MULTILINE).start():]
        # cache of just-in-time compiled kernels
        self.cache = defaultdict(dict)
        self.hash = None
        # JITFunction can be instantiated as kernel
        # when called with a grid using __getitem__
        self.kernel = None
        self.debug = True if os.environ.get("TRITON_DEBUG", "0") == "1" else debug
        self.noinline = noinline

        # TODO(jlebar): Remove uses of these fields outside this file, then
        # remove the fields here.
        self.arg_names = [p.name for p in self.params]
        self.constexprs = [p.num for p in self.params if p.is_constexpr]

        # Hooks that will be called prior to executing "run"
        self.pre_run_hooks = []

        # reuse docs of wrapped function
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

    def preload(self, specialization_data):
        from ..compiler import AttrsDescriptor, compile, ASTSource
        import json
        import triton.language as tl
        device = driver.active.get_current_device()
        deserialized_obj = json.loads(specialization_data)
        if deserialized_obj['name'] != self.fn.__name__:
            raise RuntimeError(
                f"Specialization data is for {deserialized_obj['name']} but trying to preload for {self.fn.__name__}")
        constants = {
            key: tl.dtype(value) if tl.dtype.is_dtype(value) else value
            for key, value in deserialized_obj['constants'].items()
        }
        signature = dict(deserialized_obj['signature'].items())
        src = ASTSource(self, signature, constants, AttrsDescriptor.from_dict(deserialized_obj['attrs']))
        options = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in deserialized_obj['options'].items()
        }
        key = deserialized_obj['key']
        kernel = compile(src, None, options)
        self.cache[device][key] = kernel
        return kernel

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
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[T], JITFunction[T]]:
    ...


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
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
            from .interpreter import InterpretedFunction
            return InterpretedFunction(fn)
        else:
            return JITFunction(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
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
        self.data = base.data
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

    def cpu(self):
        return TensorWrapper(self.base.cpu(), self.dtype)

    def copy_(self, other):
        self.base.copy_(other.base)

    def to(self, device):
        return TensorWrapper(self.base.to(device), self.dtype)


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
