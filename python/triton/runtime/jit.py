from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import os
import subprocess
import tempfile
import textwrap

import triton
import triton._C.libtriton.triton as _triton
from ..tools.disasm import extract

# -----------------------------------------------------------------------------
# Binary
# -----------------------------------------------------------------------------


class Binary:
    def __init__(self, backend, name, asm, shared_mem, num_warps):
        self.backend = backend
        self.name = name
        self.asm = asm
        self.shared_mem = shared_mem
        self.num_warps = num_warps


class LoadedBinary:
    def __init__(self, device: int, bin: Binary):
        module, kernel = _triton.code_gen.load_binary(bin.backend,
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
# Kernel
# -----------------------------------------------------------------------------


class Kernel:

    def __call__(self, *args, grid, num_warps=4, num_stages=3, **kwargs):
        raise RuntimeError("Not implemented. Public repo implementation will be rewritten to reduce latency.")


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

    def __init__(self, fn, version=None, inline=True, do_not_specialize=None):
        # information of wrapped function
        self.fn = fn
        self.module = fn.__module__
        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]
        self.arg_defaults = [v.default for v in signature.parameters.values()]

        self.version = version
        self.inline = inline
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find("def"):]
        self.do_not_specialize = [] if do_not_specialize is None else do_not_specialize
        self.do_not_specialize = [self.arg_names.index(arg) if isinstance(arg, str) else arg for arg in self.do_not_specialize]
        # cache for callable driver objects (e.g. CUkernel)
        self.bin_cache = dict()
        self.hash = None
        # JITFunction can be instantiated as kernel
        # when called with a grid using __getitem__
        self.kernel_decorators = []
        self.kernel = None
        # annotations
        self.annotations = {self.arg_names.index(name): ty for name, ty in fn.__annotations__.items()}
        self.__annotations__ = fn.__annotations__
        # constexprs
        self.constexprs = [self.arg_names.index(ann) for ann in self.__annotations__.keys()]
        # forward docs
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
    # Some unit tests do this, for example.
    def parse(self):
        tree = ast.parse(self.src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Cannot call @triton.jit'd outside of the scope of a kernel")

    # - when `.src` attribute is set, cache path needs
    #   to be reinitialized
    # - when kernel decorators change, cached kernel
    #   needs to be cleared
    def __setattr__(self, name, value):
        if name == 'kernel_decorators':
            self.kernel = None
        super(JITFunction, self).__setattr__(name, value)
        if name == 'src':
            self.hash = None
            JITFunction.cache_key.fget.cache_clear()

    def _init_kernel(self):
        if self.kernel is None:
            self.kernel = Kernel(self)
            for decorator in reversed(self.kernel_decorators):
                self.kernel = decorator(self.kernel)
        return self.kernel

    def __getitem__(self, grid):
        """
        A JIT function is launched with: fn[grid](*args, **kwargs).
        Hence JITFunction.__getitem__ returns a callable proxy that
        memorizes the grid.
        """
        class Launcher:
            def __init__(self, kernel, grid):
                self.kernel = kernel
                self.grid = grid

            def __call__(self, *wargs, **kwargs):
                return self.kernel(*wargs, **kwargs, grid=self.grid)

        return Launcher(self._init_kernel(), grid)

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
