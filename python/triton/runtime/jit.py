from __future__ import annotations, division
import ast
import copy
import hashlib
import inspect
import itertools
import threading
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Generic, Iterable, Optional, TypeVar, Union, overload, Dict, Any, Tuple

from triton.tools.tensor_descriptor import TensorDescriptor
from types import ModuleType
from .. import knobs
from .driver import driver
from . import _async_compile
from .._utils import find_paths_if, get_iterable_path, type_canonicalisation_dict, canonicalize_dtype
from .cache import get_cache_key
from triton._C.libtriton import get_cache_invalidating_env_vars

TRITON_MODULE = "triton.language"
GLUON_MODULE = "triton.experimental.gluon.language"

T = TypeVar("T")

# -----------------------------------------------------------------------------
# Dependencies Finder
# -----------------------------------------------------------------------------


class DependenciesFinder(ast.NodeVisitor):
    """
    This AST visitor is used to find dependencies of a JITFunction. This can
    be used to invalidate a JITFunction's hash when its source code -- or
    that of its dependencies -- changes.

    This visitor also keeps track of the global variables touched by the
    JITFunction.  When we launch the kernel, we check that these have the same
    values as they did when we ran this visitor.  If not, we raise an error (or
    otherwise we could recompile).
    """

    def __init__(self, name, globals, nonlocals, src) -> None:
        super().__init__()
        self.name = name
        self.hasher = hashlib.sha256(src.encode("utf-8"))

        # This function's __globals__ dict.
        self.globals = globals
        self.nonlocals = nonlocals

        # Python builtins that can be accessed from Triton kernels.
        self.supported_python_builtins = {
            'float',
            'getattr',
            'int',
            'isinstance',
            'len',
            'list',
            'max',
            'min',
            'print',
            'range',
        }
        self.supported_modules = {
            GLUON_MODULE,
            TRITON_MODULE,
            "copy",
            "math",
        }

        # used_global_vals tells us which global variables are used by this
        # function and all those it transitively calls, plus the values of those
        # variables when each function was initially run.  (That is, if A calls
        # C, and B calls C, then the values for C in used_global_vals will be
        # from the first time C was run, either by A or B.)
        #
        # Each function may have a different __globals__ dict, so the global
        # variable `foo` may actually have a different value in the different
        # functions.  Thus this map is actually
        #  (var_name, id(__globals__)) -> (var_value, __globals__).
        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}

        self.visiting_arg_default_value = False

    @property
    def ret(self):
        return self.hasher.hexdigest()

    def _is_triton_builtin(self, node, func):
        if inspect.isbuiltin(node.func):
            return True
        module = getattr(func, "__module__", "")
        return module.startswith(TRITON_MODULE)

    def _update_hash(self, func):
        assert isinstance(func, JITCallable)
        # Merge our used_global_vals with those of the called function,
        # after checking that all overlapping values are consistent.
        for k in self.used_global_vals.keys() & func.used_global_vals.keys():
            var_name, _ = k
            v1, _ = self.used_global_vals[k]
            v2, _ = func.used_global_vals[k]
            if v1 != v2:
                raise RuntimeError(
                    f"Global variable {var_name} has value {v1} when compiling {self.name}, but inner kernel {func.__name__} has conflicting value {v2} from when it was first compiled.  This is not allowed."
                )
        self.used_global_vals.update(func.used_global_vals)
        # update hash
        func_key = func.cache_key
        func_key += str(getattr(func, "noinline", False))
        self.hasher.update(func_key.encode("utf-8"))

    def record_reference(self, val, var_dict=None, name=None):
        from ..language.core import constexpr
        # Only keep track of "interesting" global variables, that non-evil users
        # might change.  Don't consider functions, modules, builtins, etc.  This
        # helps keep the list of vars we have to check small.
        if val is None or type(val) is ModuleType:
            return

        if getattr(val, "__triton_builtin__", False):
            return

        # Stubs that aren't real functions
        if getattr(val, "__module__", "") == "triton.language.extra.libdevice":
            return

        if isinstance(val, JITCallable):
            self._update_hash(val)
            return

        if callable(val) and not isinstance(val, type) and not isinstance(val, constexpr):
            raise RuntimeError(f"Unsupported function referenced: {val}")

        # Python default arguments are resolved only once, when the
        # function is defined.  So if you do `foo(a=A)` and the value of
        # A changes, foo will still use the old value of A.
        # It would be pretty evil if someone did `import x` and then
        # `x = blah`.
        if self.visiting_arg_default_value:
            return

        if var_dict is not None:
            self.used_global_vals[(name, id(var_dict))] = (copy.deepcopy(val), var_dict)
        return

    def visit_Name(self, node):
        if type(node.ctx) is ast.Store:
            return node.id

        if node.id in self.local_names:
            # The global name is hidden by the local name.
            return None

        def name_lookup(name):
            val = self.globals.get(name, None)
            if val is not None:
                return val, self.globals
            val = self.nonlocals.get(name, None)
            if val is not None:
                return val, self.nonlocals
            return None, None

        val, var_dict = name_lookup(node.id)
        if node.id in self.supported_python_builtins:
            return val

        self.record_reference(val, var_dict, node.id)
        return val

    def visit_Tuple(self, node):
        # We need to explicitly return the tuple values so that visit_Assign can
        # access them in the case of `a, b = ...`.
        return [self.visit(elt) for elt in node.elts]

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        while isinstance(lhs, ast.Attribute):
            lhs = self.visit(lhs.value)
        lhs_name = getattr(lhs, "__name__", "")
        if lhs is None or lhs_name in self.supported_modules:
            return None
        ret = getattr(lhs, node.attr)
        self.record_reference(ret)
        return ret

    def visit_FunctionDef(self, node):
        # Save the local name, which may hide the global name.
        self.local_names = {arg.arg for arg in node.args.args}
        self.generic_visit(node)

    def visit_arguments(self, node):
        # The purpose of this function is to visit everything in `arguments`
        # just like `generic_visit`, except when we're visiting default values
        # (i.e. the `foo` part of `def fn(x = foo)`), we set
        # self.visiting_arg_default_value = True.  This allows visit_Name to be
        # aware that we're inside function default values, which have special
        # semantics.

        # According to the AST docs, the arguments node has the following structure.
        #
        # arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
        #              expr* kw_defaults, arg? kwarg, expr* defaults)
        def visit_defaults(defaults):
            try:
                assert not self.visiting_arg_default_value
                self.visiting_arg_default_value = True
                for expr in defaults:
                    if expr is not None:
                        self.visit(expr)
            finally:
                self.visiting_arg_default_value = False

        for arg in itertools.chain(node.posonlyargs, node.args, [node.vararg] if node.vararg else [], node.kwonlyargs):
            self.visit(arg)

        visit_defaults(node.kw_defaults)

        if node.kwarg is not None:
            self.visit(node.kwarg)

        visit_defaults(node.defaults)

    def visitAssnTarget(self, node):
        # Target is either a single string, or a list of strings (if the assn
        # target is a tuple).
        target = self.visit(node)
        if isinstance(target, list):
            self.local_names |= set(target)
        else:
            self.local_names.add(target)

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            # TODO(jlebar): I don't actually know how to hit this.  You don't
            # get it from `a, b = ...` -- in that case, node.targets is a single
            # Tuple, and in fact we *do* need to handle that case if we want
            # existing code to work.
            raise TypeError("Simultaneous multiple assignment is not supported.")

        self.visitAssnTarget(node.targets[0])

        # This will re-visit the target, but that's OK.
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self.visitAssnTarget(node.target)

        # This will re-visit the target, but that's OK.
        self.generic_visit(node)

    def visit_For(self, node):
        self.visitAssnTarget(node.target)

        # This will re-visit the target, but that's fine.
        self.generic_visit(node)


# -----------------------------------------------------------------------------
# JITFunction
# -----------------------------------------------------------------------------


def _normalize_ty(ty) -> str:
    import triton.language.core as core
    if isinstance(ty, str):
        ty = ty.strip()
        if ty.startswith("const "):
            ty = ty.removeprefix("const")
            ty = _normalize_ty(ty)
            assert ty.startswith("*")
            return "*k" + ty[1:]
        if ty.endswith("*"):
            return "*" + _normalize_ty(ty[:-1])
        if ty.startswith("*"):
            return "*" + _normalize_ty(ty[1:])
        if ty.startswith("tl."):
            return _normalize_ty(ty.removeprefix("tl."))
    elif isinstance(ty, core.pointer_type):
        return f"*{_normalize_ty(ty.element_ty)}"
    elif isinstance(ty, core.dtype):
        ty = ty.name
    elif isinstance(ty, type):
        ty = ty.__name__
    else:
        ty = str(ty)
    return type_canonicalisation_dict.get(ty.replace("_t", ""), ty)


class KernelParam:
    """Represents a parameter (name plus metadata) to a @jit'ed function."""

    def __init__(self, num: int, param: inspect.Parameter, do_not_specialize: bool,
                 do_not_specialize_on_alignment: bool):
        self.num = num
        self._param = param
        self.do_not_specialize = do_not_specialize
        self.do_not_specialize_on_alignment = do_not_specialize_on_alignment

    @cached_property
    def name(self):
        return self._param.name

    @cached_property
    def annotation(self) -> str:
        if not self._param.annotation or self._param.annotation == inspect.Parameter.empty:
            return ""
        return _normalize_ty(self._param.annotation)

    @cached_property
    def annotation_type(self) -> str:
        a = self.annotation
        if a.startswith("*k"):
            a = a[2:]
        elif a.startswith("*"):
            a = a[1:]
        if a in set(type_canonicalisation_dict.values()):
            return self.annotation
        return ""

    @cached_property
    def is_constexpr(self):
        return "constexpr" in self.annotation

    @cached_property
    def is_const(self):
        if self.is_constexpr:
            return False
        return "const" in self.annotation or self.annotation.startswith("*k")

    @property
    def default(self):
        return self._param.default

    @property
    def has_default(self):
        return self._param.default != inspect.Parameter.empty


dtype2str = {}
specialize_impl_cache = []


def create_specialize_impl(specialize_extra):

    from ..language import constexpr
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonTensorDescriptor

    def specialize_impl(arg, is_const=False, specialize_value=True, align=True):
        if arg is None:
            return ("constexpr", None)
        elif isinstance(arg, bool):
            return ("u1", None)
        elif isinstance(arg, int):
            key = specialize_extra(arg, "int", align=align) if specialize_value else None
            if arg == 1 and specialize_value:
                return ("constexpr", 1)
            elif -(2**31) <= arg and arg <= 2**31 - 1:
                return ("i32", key)
            elif 2**63 <= arg and arg <= 2**64 - 1:
                return ("u64", key)
            else:
                return ("i64", key)
        elif isinstance(arg, float):
            return ("fp32", None)
        elif hasattr(arg, "data_ptr"):
            # dtypes are hashable so we can memoize this mapping:
            dsk = (arg.dtype, is_const)
            res = dtype2str.get(dsk, None)
            if res is None:
                res = ("*k" if dsk[1] else "*") + canonicalize_dtype(dsk[0])
                dtype2str[dsk] = res
            key = specialize_extra(arg, "tensor", align=align) if specialize_value else None
            return (res, key)
        elif isinstance(arg, JITCallable):
            return ("constexpr", arg.cache_key)
        elif isinstance(arg, constexpr):
            return ("constexpr", arg)
        elif isinstance(arg, tuple):
            spec = [specialize_impl(x) for x in arg]
            make_tuple = lambda vals: type(arg)(*vals) if hasattr(arg, "_fields") else tuple(vals)
            tys = make_tuple([x[0] for x in spec])
            keys = make_tuple([x[1] for x in spec])
            return (tys, keys)
        elif isinstance(arg, TensorDescriptor):
            assert hasattr(arg.base, "data_ptr")
            inner = canonicalize_dtype(arg.base.dtype)
            return (f"tensordesc<{inner}{list(arg.block_shape)}>", None)
        elif isinstance(arg, GluonTensorDescriptor):
            assert hasattr(arg.base, "data_ptr")
            inner = canonicalize_dtype(arg.base.dtype)
            return (f"tensordesc<{inner}{list(arg.block_shape)},{arg.layout!r}>", None)
        else:
            raise TypeError("Unsupported type: %s" % type(arg))

    return specialize_impl


def mangle_type(arg, specialize=False):
    if len(specialize_impl_cache) == 0:
        specialize_impl_cache.append(create_specialize_impl(lambda _, **kwargs: None))
    specialize_impl = specialize_impl_cache[0]
    return specialize_impl(arg, specialize_value=specialize)[0]


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
        'name': name, 'signature': signature, 'constant_keys': [list(x) for x in constants.keys()], 'constant_vals':
        list(constants.values()), 'attrs_keys': [list(x) for x in attrs.keys()], 'attrs_vals': list(attrs.values()),
        'options': options.__dict__, 'key': key
    }
    serialized_obj = json.dumps(obj)
    return serialized_obj


def create_function_from_signature(sig, kparams, backend):
    """
    Equivalent to sig.bind followed by apply_defaults. This generates a
    native Python function (using exec) which can be memoized on a per-kernel
    basis to avoid having to run these expensive functions -- which constitute
    much of the kernel launch overhead -- every time we run the kernel.
    """
    assert len(sig.parameters) == len(kparams)
    # Create the function argument list and the dict entries for the return statement
    specialization = []
    # signature
    for name, kp in zip(sig.parameters.keys(), kparams):
        if kp.is_constexpr:
            specialization.append(f'("constexpr", {name})')
        else:
            is_const = 'True' if kp.is_const else 'False'
            specialize = 'False' if kp.do_not_specialize else 'True'
            align = 'False' if kp.do_not_specialize_on_alignment else 'True'
            ret = f"specialize_impl({name}, {is_const}, {specialize}, {align})"
            if kp.annotation_type:
                if isinstance(kp.annotation_type, str):
                    if kp.annotation_type == "u1" or kp.annotation_type[:2] in ["fp", "bf"]:
                        # we do not specialize non-constexpr floats and bools:
                        specialize = False
                if specialize:
                    specialization.append(f'("{kp.annotation_type}",) + {ret}[1:]')
                else:
                    # skip runtime specialization:
                    specialization.append(f'("{kp.annotation_type}", None)')
            else:
                specialization.append(f"{ret}")

    # compute argument string for a given parameter
    arg = lambda x: x[0] if x[1].default is inspect.Parameter.empty else f"{x[0]}=default_{x[0]}"
    # Join all arguments into a function definition string
    func_body = f"""
def dynamic_func({", ".join(list(map(arg, sig.parameters.items())) + ["**options"])}):
    params = {{{', '.join([f"'{name}': {name}" for name in sig.parameters.keys()])}}}
    specialization = [{','.join(specialization)}]
    return params, specialization, options
"""
    # Prepare defaults to be inserted into function namespace
    func_namespace = {
        f"default_{name}": param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }

    func_namespace["JITCallable"] = JITCallable
    func_namespace["specialize_impl"] = create_specialize_impl(backend.get_arg_specialization)

    # Execute the function string in func_namespace to create the function
    exec(func_body, func_namespace)

    # Extract the newly created function from the namespace
    return func_namespace['dynamic_func']


def get_full_name(fn):
    return f"{fn.__module__}.{fn.__qualname__}"


class JITCallable:

    def __init__(self, fn):
        self.fn = fn
        self.signature = inspect.signature(fn)
        try:
            self.raw_src, self.starting_line_number = inspect.getsourcelines(fn)
        except OSError as e:
            raise ValueError("@jit functions should be defined in a Python file") from e
        self._fn_name = get_full_name(fn)
        self._hash_lock = threading.RLock()

        # function source code (without decorators)
        src = textwrap.dedent("".join(self.raw_src))
        src = src[re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE).start():]
        self._src = src
        self.hash = None

        # Map of global variables used by the function and any functions it
        # transitively calls, plus their values.  The values are collected when
        # the function is first compiled.  Then every time we run the function,
        # we check that the values of the globals match what's expected,
        # otherwise we raise an error.
        #
        # Different functions can have different __globals__ maps, so the map
        # key is actually (var name, id(__globals__)), and the map value is
        # (value, __globals__).
        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}

        # reuse docs of wrapped function
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__qualname__ = fn.__qualname__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__

    def get_capture_scope(self):
        return self.__globals__ | inspect.getclosurevars(self.fn).nonlocals

    @property
    def cache_key(self):
        # TODO : hash should be attribute of `self`
        with self._hash_lock:
            if self.hash is not None:
                return self.hash
            # Set a placeholder hash to break recursion in case the function
            # transitively calls itself. The full hash is set after.
            self.hash = f"recursion:{self._fn_name}"
            nonlocals = inspect.getclosurevars(self.fn).nonlocals
            dependencies_finder = DependenciesFinder(name=self._fn_name, globals=self.__globals__, nonlocals=nonlocals,
                                                     src=self.src)
            dependencies_finder.visit(self.parse())
            self.hash = dependencies_finder.ret + str(self.starting_line_number)
            self.used_global_vals = dict(sorted(dependencies_finder.used_global_vals.items()))

            from triton.language.core import constexpr
            self.hash += str([(name, val)
                              for (name, _), (val, _) in self.used_global_vals.items()
                              if isinstance(val, constexpr)])
            self.hash = hashlib.sha256(self.hash.encode("utf-8")).hexdigest()
        return self.hash

    # we do not parse `src` in the constructor because
    # the user might want to monkey-patch self.src dynamically.
    # Our unit tests do this, for example.
    def parse(self):
        tree = ast.parse(self._src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

    @property
    def type(self):
        from triton.language.core import constexpr_type
        return constexpr_type(self)

    def _unsafe_update_src(self, new_src):
        """
        The only method allowed to modify src.
        Bypasses the __setattr__ restriction by calling super().__setattr__ directly.

        Note that it is the callers responsibility to make sure any triton functions that call this function have the `.hash` value reset to None.
        """
        self.hash = None
        self._src = new_src

    def _set_src(self):
        raise AttributeError("Cannot set attribute 'src' directly. "
                             "Use '_unsafe_update_src()' and manually clear `.hash` of all callers"
                             "instead.")

    def _get_src(self):
        return self._src

    src = property(fget=_get_src, fset=_set_src)


@dataclass
class JitFunctionInfo:
    module: ModuleType
    name: str
    jit_function: JITFunction


def compute_cache_key(kernel_key_cache, specialization, options):
    key = (tuple(specialization), str(options))
    cache_key = kernel_key_cache.get(key, None)
    if cache_key is not None:
        return cache_key

    cache_key = str(specialization) + str(options)
    kernel_key_cache[key] = cache_key
    return cache_key


class JITFunction(JITCallable, KernelInterface[T]):

    def is_gluon(self):
        return False

    def _call_hook(
        self,
        hook,
        key,
        signature,
        device,
        constants,
        options,
        configs,
        is_warmup,
    ) -> bool | None:
        if not hook:
            return None

        name = self.fn.__qualname__
        module = self.fn.__module__
        arg_reprs = ", ".join([f"{param.name}: {ty}" for param, ty in zip(self.params, key[1])])
        repr = f"{name}[num_warps={options.num_warps}, num_ctas={options.num_ctas}, num_stages={options.num_stages}, enable_fp_fusion={options.enable_fp_fusion}, launch_cooperative_grid={options.launch_cooperative_grid}]({arg_reprs})"
        full_name = get_full_name(self.fn)

        specialization_data = serialize_specialization_data(full_name, signature, constants, configs[0], options, key)

        kwargs = {
            'signature': signature,
            'device': device,
            'constants': constants,
            'num_warps': options.num_warps,
            'num_ctas': options.num_ctas,
            'num_stages': options.num_stages,
            'enable_fp_fusion': options.enable_fp_fusion,
            'launch_cooperative_grid': options.launch_cooperative_grid,
            'extern_libs': options.extern_libs,
            'configs': configs,
            'specialization_data': specialization_data,
            'is_warmup': is_warmup,
        }

        return hook(
            key=key,
            repr=repr,
            fn=JitFunctionInfo(module, name, self),
            compile={"key": key, **kwargs},
            is_manual_warmup=is_warmup,
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
        target = driver.active.get_current_target()
        backend = make_backend(target)
        self.CompiledKernel = CompiledKernel
        self.compile = compile
        self.ASTSource = ASTSource
        binder = create_function_from_signature(self.signature, self.params, backend)
        return {}, {}, target, backend, binder

    def _pack_args(self, backend, kwargs, bound_args, specialization, options):
        # options
        options = backend.parse_options(kwargs)
        # signature
        sigkeys = [x.name for x in self.params]
        sigvals = [x[0] for x in specialization]
        signature = {k: v for (k, v) in zip(sigkeys, sigvals)}
        # check arguments
        assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
        assert "device" not in kwargs, "device option is deprecated; current device will be used"
        assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"
        for k in kwargs:
            if k not in options.__dict__ and k not in sigkeys:
                raise KeyError("Keyword argument %s was specified but unrecognised" % k)
        # constexprs
        constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
        constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}
        # attributes
        attrvals = [x[1] for x in specialization]
        attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
        attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}

        return options, signature, constexprs, attrs

    def run(self, *args, grid, warmup, **kwargs):
        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug

        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
        # specialization is list[tuple[str, Any]], where first element of tuple is
        # the type and the second parameter is the 'specialization' value.
        bound_args, specialization, options = binder(*args, **kwargs)

        key = compute_cache_key(kernel_key_cache, specialization, options)
        kernel = kernel_cache.get(key, None)

        # Kernel is not cached; we have to compile.
        if kernel is None:
            options, signature, constexprs, attrs = self._pack_args(backend, kwargs, bound_args, specialization,
                                                                    options)

            kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)
            if kernel is None:
                return None

        # Check that used global values have not changed.
        not_present = object()
        for (name, _), (val, globals_dict) in self.used_global_vals.items():
            if (newVal := globals_dict.get(name, not_present)) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel, from {val} to {newVal}")

        if not warmup:
            # canonicalize grid
            assert grid is not None
            if callable(grid):
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1
            if hasattr(kernel, "result"):
                kernel = kernel.result()
            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
        return kernel

    def repr(self, _):
        return self._fn_name if self._repr is None else self._repr(_)

    def __init__(self, fn, version=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None,
                 noinline=None, repr=None, launch_metadata=None):
        do_not_specialize = do_not_specialize if do_not_specialize else []
        do_not_specialize_on_alignment = do_not_specialize_on_alignment if do_not_specialize_on_alignment else []

        super().__init__(fn)
        self.module = fn.__module__
        self.version = version
        self.do_not_specialize = do_not_specialize
        self.do_not_specialize_on_alignment = do_not_specialize_on_alignment
        self._repr = repr
        self.launch_metadata = launch_metadata

        self.params = []
        for i, param in enumerate(self.signature.parameters.values()):
            dns = i in do_not_specialize or param.name in do_not_specialize
            dns_oa = i in do_not_specialize_on_alignment or param.name in do_not_specialize_on_alignment
            self.params.append(KernelParam(i, param, dns, dns_oa))

        # cache of just-in-time compiled kernels
        self.device_caches = defaultdict(self.create_binder)

        # JITFunction can be instantiated as kernel
        # when called with a grid using __getitem__
        self.kernel = None
        self.debug = debug
        self.noinline = noinline

        # TODO(jlebar): Remove uses of these fields outside this file, then
        # remove the fields here.
        self.arg_names = [p.name for p in self.params]
        self.constexprs = [p.num for p in self.params if p.is_constexpr]

        # Hooks that will be called prior to executing "run"
        self.pre_run_hooks = []

    def warmup(self, *args, grid, **kwargs):
        return self.run(grid=grid, warmup=True, *map(MockTensor.wrap_dtype, args), **kwargs)

    def preload(self, specialization_data):
        import json
        import triton.language as tl
        device = driver.active.get_current_device()
        deserialized_obj = json.loads(specialization_data)
        if deserialized_obj['name'] != self._fn_name:
            raise RuntimeError(
                f"Specialization data is for {deserialized_obj['name']} but trying to preload for {self._fn_name}")
        constant_keys = map(tuple, deserialized_obj['constant_keys'])
        constant_vals = deserialized_obj['constant_vals']
        constexprs = {
            key: tl.dtype(value) if tl.dtype.is_dtype(value) else value
            for key, value in zip(constant_keys, constant_vals)
        }
        attrs_keys = map(tuple, deserialized_obj['attrs_keys'])
        attrs_vals = deserialized_obj['attrs_vals']
        attrs = dict(zip(attrs_keys, attrs_vals))
        signature = dict(deserialized_obj['signature'].items())
        options = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in deserialized_obj['options'].items()
        }
        key = deserialized_obj['key']
        _, _, _, backend, _ = self.device_caches[device]
        options = backend.parse_options(options)
        return self._do_compile(
            key,
            signature,
            device,
            constexprs,
            options,
            attrs,
            warmup=True,
        )

    def _do_compile(self, key, signature, device, constexprs, options, attrs, warmup):
        kernel_cache, _, target, backend, _ = self.device_caches[device]

        if self._call_hook(knobs.runtime.jit_cache_hook, key, signature, device, constexprs, options, [attrs], warmup):
            return None
        src = self.ASTSource(self, signature, constexprs, attrs)

        async_mode = _async_compile.active_mode.get()
        if async_mode is not None:

            env_vars = get_cache_invalidating_env_vars()
            cache_key = get_cache_key(src, backend, options, env_vars)

            def async_compile():
                return self.compile(src, target=target, options=options.__dict__, _env_vars=env_vars)

            def finalize_compile(kernel):
                kernel_cache[key] = kernel
                self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, device, constexprs, options,
                                [attrs], warmup)

            kernel = async_mode.submit(cache_key, async_compile, finalize_compile)
        else:
            kernel = self.compile(src, target=target, options=options.__dict__)
            kernel_cache[key] = kernel
            self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, device, constexprs, options, [attrs],
                            warmup)
        return kernel

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Cannot call @triton.jit'd outside of the scope of a kernel")

    def __repr__(self):
        return f"JITFunction({self.module}:{self.fn.__qualname__})"


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
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
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
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
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
        if knobs.runtime.interpret:
            from .interpreter import InterpretedFunction
            return InterpretedFunction(fn, version=version, do_not_specialize=do_not_specialize,
                                       do_not_specialize_on_alignment=do_not_specialize_on_alignment, debug=debug,
                                       noinline=noinline, repr=repr, launch_metadata=launch_metadata)
        else:
            return JITFunction(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
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

    def __init__(self, dtype, shape=None):
        if shape is None:
            shape = [1]
        self.dtype = dtype
        self.shape = shape

    def stride(self):
        strides = [1]
        for size in self.shape[1:]:
            strides.append(strides[-1] * size)
        return tuple(reversed(strides))

    @staticmethod
    def data_ptr():
        return 0  # optimistically assumes multiple of 16

    @staticmethod
    def ptr_range():
        return 0  # optimistically assumes 32 bit pointer range


class TensorWrapper:

    def __init__(self, base, dtype):
        self.dtype = dtype
        self.base = base
        self.data = base.data
        self.device = base.device
        self.shape = self.base.shape

    def data_ptr(self):
        return self.base.data_ptr()

    def stride(self, *args):
        return self.base.stride(*args)

    def __str__(self) -> str:
        return f"TensorWrapper[{self.dtype}]({self.base})"

    def element_size(self):
        return self.base.element_size()

    def cpu(self):
        return TensorWrapper(self.base.cpu(), self.dtype)

    def copy_(self, other):
        self.base.copy_(other.base)

    def clone(self):
        return TensorWrapper(self.base.clone(), self.dtype)

    def to(self, device):
        return TensorWrapper(self.base.to(device), self.dtype)

    def new_empty(self, sizes):
        return TensorWrapper(self.base.new_empty(sizes), self.dtype)


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


def get_jit_fn_file_line(fn):
    base_fn = fn
    while not isinstance(base_fn, JITCallable):
        base_fn = base_fn.fn
    file_name = base_fn.fn.__code__.co_filename
    begin_line = base_fn.starting_line_number
    # Match the following pattern:
    # @triton.autotune(...) <- foo.__code__.co_firstlineno
    # @triton.heuristics(...)
    # @triton.jit
    # def foo(...): <- this line is the first line
    for idx, line in enumerate(base_fn.raw_src):
        if line.strip().startswith("def "):
            begin_line += idx
            break
    return file_name, begin_line


class BoundConstexprFunction(JITCallable):

    def __init__(self, instance, fn):
        self.__self__ = instance
        self.__func__ = fn

    def __call__(self, *args, **kwargs):
        return self.__func__(self.__self__, *args, **kwargs)


class ConstexprFunction(JITCallable):

    def __init__(self, fn):
        super().__init__(fn)

    def __get__(self, obj, objclass):
        # Create a bound function to support constexpr_function methods
        if obj is not None:
            return BoundConstexprFunction(obj, self)
        return self

    def __call__(self, *args, _semantic=None, **kwargs):
        from triton.language.core import _unwrap_if_constexpr, constexpr
        # de-constexpr arguments and discard the _semantic keyword argument:
        args = [_unwrap_if_constexpr(x) for x in args]
        kwargs = {k: _unwrap_if_constexpr(v) for (k, v) in kwargs.items()}

        # call the raw Python function f:
        res = self.fn(*args, **kwargs)

        if _semantic is None:
            # Not called by triton code generator, e.g. in host code, another constexpr function, or even an aggreate's __init__ function
            return res

        # convert result back to a Triton constexpr:
        if knobs.runtime.interpret:
            return res  # No constexpr in interpreter
        return constexpr(res)


def constexpr_function(fn):
    """
    Wraps an arbitrary Python function so that it can be called at
    compile-time on constexpr arguments in a Triton function and
    returns a constexpr result.
    """
    return ConstexprFunction(fn)
