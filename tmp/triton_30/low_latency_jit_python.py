import inspect
import logging
import os
import textwrap
from collections import defaultdict

from triton.compiler import ASTSource, CompiledKernel, compile, make_backend
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction as _JITFunction
from triton.runtime.jit import KernelArg, KernelParam

logger = logging.getLogger(__name__)


class LowLatencyJITFunctionPython(_JITFunction):
    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(*args, grid=grid, **kwargs)

    def run(
        self,
        *args,
        grid=None,
        warmup=False,
        device_type=None,
        device=None,
        stream=None,
        **kwargs,
    ):
        # deprecated arguments
        assert (
            device_type is None
        ), "device_type option is deprecated; current target will be used"
        assert (
            device is None not in kwargs
        ), "device option is deprecated; current device will be used"
        assert (
            stream is None not in kwargs
        ), "stream option is deprecated; current stream will be used"
        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        target = driver.active.get_current_target()
        backend = make_backend(target)
        kwargs["debug"] = self.debug
        options = backend.parse_options(kwargs)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

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
        args = [
            KernelArg(arg_value, param)
            for (_, arg_value), param in zip(bound_args.arguments.items(), self.params)
        ]
        sig_key = tuple(
            arg.mangled_type() for arg in args if not arg.param.is_constexpr
        )
        spec_key = tuple(
            arg.specialization_key() for arg in args if not arg.param.do_not_specialize
        )
        constexpr_key = tuple(arg.value for arg in args if arg.param.is_constexpr)
        key = (sig_key, constexpr_key, spec_key, options)
        key = str(key)
        # Kernel is not cached; we have to compile.
        if key not in self.cache[device]:
            configs = (self._get_config(*[arg.value for arg in args]),)
            constants = {
                arg.param.name: arg.value
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
                arg.param.name: arg.mangled_type()
                for arg in args
                if not arg.param.is_constexpr
            }

            if self._call_hook(key, signature, device, constants, options, configs):
                return None
            # compile the kernel
            src = ASTSource(self, signature, constants, configs[0])
            self.cache[device][key] = compile(
                src,
                target=target,
                options=options.__dict__,
            )

        kernel = self.cache[device][key]

        # Verify key signature from the cache
        signature = {
            arg.param.name: arg.mangled_type()
            for arg in args
            if not arg.param.is_constexpr
        }
        if kernel.src.signature != signature:
            raise RuntimeError(
                f"Signature mismatch for cached kernel {self.fn.__name__}:\n"
                f"  Cached signature: {kernel.src.signature}\n"
                f"  Call signature:   {signature}"
            )

        if not warmup:
            args = [arg.value for arg in args if not arg.param.is_constexpr]
            launch_metadata = kernel.launch_metadata(grid, stream, *args)
            kernel.run(
                grid_0,
                grid_1,
                grid_2,
                stream,
                kernel.function,
                kernel.metadata,
                launch_metadata,
                CompiledKernel.launch_enter_hook,
                CompiledKernel.launch_exit_hook,
                *args,
            )
        return kernel

    def __init__(
        self,
        fn,
        version=None,
        do_not_specialize=None,
        debug=None,
        noinline=None,
        repr=None,
        launch_metadata=None,
    ):
        do_not_specialize = do_not_specialize if do_not_specialize else []

        self.fn = fn
        self.module = fn.__module__
        self.version = version
        self.signature = inspect.signature(fn)
        self.do_not_specialize = do_not_specialize
        self.starting_line_number = inspect.getsourcelines(fn)[1]
        self.repr = lambda _: fn.__name__ if repr is None else repr(_)
        self.launch_metadata = launch_metadata

        self.params = []
        for i, param in enumerate(self.signature.parameters.values()):
            dns = do_not_specialize and (
                i in do_not_specialize or param.name in do_not_specialize
            )
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

    def __repr__(self):
        return f"LowLatencyJITFunctionPython({self.module}:{self.fn.__name__})"
