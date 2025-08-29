from __future__ import annotations

import functools
import importlib
import os
import re
import subprocess
import sysconfig

from dataclasses import dataclass
from contextlib import contextmanager
from typing import cast, Any, Callable, Generator, Generic, Optional, Protocol, Type, TypeVar, TypedDict, TYPE_CHECKING, Union

from triton._C.libtriton import getenv, getenv_bool  # type: ignore

if TYPE_CHECKING:
    from .runtime.cache import CacheManager, RemoteCacheBackend
    from .runtime.jit import JitFunctionInfo, KernelParam
    from .compiler.compiler import ASTSource, LazyDict, IRSource


class Env:
    pass


env = Env()

propagate_env: bool = True


def setenv(key: str, value: Optional[str]) -> None:
    if not propagate_env:
        return

    if value is not None:
        os.environ[key] = value
    elif key in os.environ:
        del os.environ[key]


def toenv(val: Any) -> Union[None, tuple[Optional[str]]]:
    if val is None:
        return (None, )

    t = type(val)
    if t is bool:
        return ("1" if val else "0", )

    if t is str:
        return (val, )

    if t is int:
        return (str(val), )

    return None


# There's an asymmetry here so that e.g. env_nvidia_tool can be specified with a
# a string but return an NvidiaTool.
SetType = TypeVar("SetType")
GetType = TypeVar("GetType")

_NOTHING = object()


class env_base(Generic[SetType, GetType]):

    def __init__(self, key: str) -> None:
        self.key = key

    def __set_name__(self, objclass: Type[object], name: str) -> None:
        self.name = name

    def __get__(self, obj: Optional[object], objclass: Optional[Type[object]]) -> GetType:
        py_val = obj.__dict__.get(self.name, _NOTHING)
        if py_val is _NOTHING:
            return self.get()
        return self.transform(py_val)

    def get(self) -> GetType:
        raise NotImplementedError()

    def __set__(self, obj: object, value: Union[SetType, Env]) -> None:
        if isinstance(value, Env):
            obj.__dict__.pop(self.name, None)
        else:
            obj.__dict__[self.name] = value
            if env_val := toenv(value):
                setenv(self.key, env_val[0])

    def __delete__(self, obj: object) -> None:
        obj.__dict__.pop(self.name, None)

    def transform(self, val: SetType) -> GetType:
        # See comment about GetType/SetType in their definition above. Only needed
        # if GetType != SetType.
        return cast(GetType, val)


class env_str(env_base[str, str]):

    def __init__(self, key: str, default: str):
        super().__init__(key)
        self.default = default

    def get(self) -> str:
        return getenv(self.key, self.default)


class env_str_callable_default(env_base[str, str]):

    def __init__(self, key: str, default_factory: Callable[[], str]):
        super().__init__(key)
        self.default_factory = default_factory

    def get(self) -> str:
        env_val = getenv(self.key)
        if env_val is None:
            return self.default_factory()
        return env_val


class env_bool(env_base[bool, bool]):

    def __init__(self, key: str, default: bool = False) -> None:
        super().__init__(key)
        self.default = default

    def get(self) -> bool:
        return getenv_bool(self.key, self.default)


class env_int(env_base[int, int]):

    def __init__(self, key: str, default: int = 0) -> None:
        super().__init__(key)
        self.default = default

    def get(self) -> int:
        val = getenv(self.key)
        if val is None:
            return self.default
        try:
            return int(val)
        except ValueError as exc:
            raise RuntimeError(f"Unable to use {self.key}={val}: expected int") from exc


ClassType = TypeVar("ClassType")


class env_class(Generic[ClassType], env_base[Optional[Type[ClassType]], Optional[Type[ClassType]]]):

    def __init__(self, key: str, type: str) -> None:
        super().__init__(key)
        # We can't pass the type directly to avoid import cycles
        self.type = type

    def get(self) -> Optional[Type[ClassType]]:
        val = getenv(self.key)
        if val is None:
            return None
        comps = val.split(":", 1)
        if len(comps) != 2:
            raise RuntimeError(f"Unable to read {self.key}: '{val}' isn't of the form MODULE:CLASS")
        cls = getattr(importlib.import_module(comps[0]), comps[1])

        if not any((c.__name__ == self.type for c in cls.mro())):
            raise RuntimeError(f"Unable to use '{val}' from {self.key}: not of type '{self.type}'")

        return cast(Type[ClassType], cls)


@dataclass
class NvidiaTool:
    path: str
    version: str

    @staticmethod
    @functools.lru_cache
    def from_path(path: str) -> Optional[NvidiaTool]:
        try:
            result = subprocess.check_output([path, "--version"], stderr=subprocess.STDOUT)
            version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
            if version is None:
                return None
            return NvidiaTool(path, version.group(1))
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None


class env_nvidia_tool(env_base[str, NvidiaTool]):

    def __init__(self, binary: str) -> None:
        binary += sysconfig.get_config_var("EXE")
        self.binary = binary
        self.default_path = os.path.join(os.path.dirname(__file__), "backends", "nvidia", "bin", binary)
        super().__init__(f"TRITON_{binary.upper()}_PATH")

    def get(self) -> NvidiaTool:
        return self.transform(getenv(self.key))

    def transform(self, path: str) -> NvidiaTool:
        # We still add default as fallback in case the pointed binary isn't
        # accessible.
        if path is not None:
            paths = [path, self.default_path]
        else:
            paths = [self.default_path]

        for path in paths:
            if tool := NvidiaTool.from_path(path):
                return tool

        raise RuntimeError(f"Cannot find {self.binary}")


# Separate classes so that types are correct
class env_opt_str(env_base[Optional[str], Optional[str]]):

    def get(self) -> Optional[str]:
        return getenv(self.key)


class env_opt_bool(env_base):

    def get(self) -> Optional[str]:
        return getenv_bool(self.key, None)


@dataclass(frozen=True)
class CompileTimes:
    """
    Model holding timing information for an invocation of the compiler.

    All times in microseconds.
    """

    # Duration of make_ir
    ir_initialization: int

    # Ordered mapping from lowering stage to duration spent in that stage.
    # Keyed by stage extension, e.g. ttir, ttgir
    lowering_stages: list[tuple[str, int]]

    # Duration of saving artifacts/metadata to cache
    store_results: int

    @property
    def total_lowering(self) -> int:
        return sum((stage[1] for stage in self.lowering_stages))

    @property
    def total(self) -> int:
        return self.ir_initialization + self.total_lowering + self.store_results


class CompilationListener(Protocol):

    def __call__(self, *, src: Union[ASTSource, IRSource], metadata: dict[str, Any], metadata_group: dict[str, str],
                 times: CompileTimes, cache_hit: bool) -> None:
        ...


knobs_type = TypeVar("knobs_type", bound='base_knobs')


class base_knobs:

    @property
    def knob_descriptors(self) -> dict[str, env_base]:
        return {
            k: v
            # data descriptors live on the class object
            for k, v in type(self).__dict__.items()
            if isinstance(v, env_base)
        }

    @property
    def knobs(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.knob_descriptors.keys()}

    def copy(self: knobs_type) -> knobs_type:
        res = type(self)()
        res.__dict__.update(self.__dict__)
        return res

    def reset(self: knobs_type) -> knobs_type:
        for knob in self.knob_descriptors.keys():
            delattr(self, knob)
        return self

    @contextmanager
    def scope(self) -> Generator[None, None, None]:
        try:
            initial_env = {knob.key: getenv(knob.key) for knob in self.knob_descriptors.values()}
            orig = dict(self.__dict__)
            yield
        finally:
            self.__dict__.clear()
            self.__dict__.update(orig)

            for k, v in initial_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]


class BuildImpl(Protocol):

    def __call__(self, name: str, src: str, srcdir: str, library_dirs: list[str], include_dirs: list[str],
                 libraries: list[str], /) -> str:
        ...


class build_knobs(base_knobs):
    """Configuration controlling how the native compiler is invoked"""
    cc: env_opt_str = env_opt_str("CC")

    cudacrt_path: env_opt_str = env_opt_str("TRITON_CUDACRT_PATH")
    cudart_path: env_opt_str = env_opt_str("TRITON_CUDART_PATH")

    impl: Optional[BuildImpl] = None

    @property
    def backend_dirs(self) -> set[str]:
        return {path for path in (self.cudacrt_path, self.cudart_path) if path is not None}


class redis_knobs(base_knobs):
    key_format: env_str = env_str("TRITON_REDIS_KEY_FORMAT", "triton:{key}:{filename}")
    host: env_str = env_str("TRITON_REDIS_HOST", "localhost")
    port: env_int = env_int("TRITON_REDIS_PORT", 6379)


cache: cache_knobs


class cache_knobs(base_knobs):
    home_dir: env_str = env_str("TRITON_HOME", os.path.expanduser("~/"))

    dump_dir = env_str_callable_default("TRITON_DUMP_DIR", lambda: cache.get_triton_dir("dump"))
    override_dir = env_str_callable_default("TRITON_OVERRIDE_DIR", lambda: cache.get_triton_dir("override"))
    dir = env_str_callable_default("TRITON_CACHE_DIR", lambda: cache.get_triton_dir("cache"))

    manager_class: env_class[CacheManager] = env_class("TRITON_CACHE_MANAGER", "CacheManager")
    remote_manager_class: env_class[RemoteCacheBackend] = env_class("TRITON_REMOTE_CACHE_BACKEND", "RemoteCacheBackend")

    def get_triton_dir(self, dirname: str) -> str:
        return os.path.join(self.home_dir, ".triton", dirname)


class compilation_knobs(base_knobs):
    override: env_bool = env_bool("TRITON_KERNEL_OVERRIDE")
    dump_ir: env_bool = env_bool("TRITON_KERNEL_DUMP")
    store_binary_only: env_bool = env_bool("TRITON_STORE_BINARY_ONLY")
    always_compile: env_bool = env_bool("TRITON_ALWAYS_COMPILE")
    # TODO: Use enum to constrain / 'typecheck' the values
    use_ir_loc: env_opt_str = env_opt_str("USE_IR_LOC")
    enable_asan: env_bool = env_bool("TRITON_ENABLE_ASAN")
    disable_line_info: env_bool = env_bool("TRITON_DISABLE_LINE_INFO")
    front_end_debugging: env_bool = env_bool("TRITON_FRONT_END_DEBUGGING")
    allow_non_constexpr_globals: env_bool = env_bool("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS")
    enable_experimental_consan: env_bool = env_bool("TRITON_ENABLE_EXPERIMENTAL_CONSAN")
    listener: Union[CompilationListener, None] = None


class autotuning_knobs(base_knobs):
    cache: env_bool = env_bool("TRITON_CACHE_AUTOTUNING")
    print: env_bool = env_bool("TRITON_PRINT_AUTOTUNING")


class LaunchHook(Protocol):
    """Hook invoked before and after kernel launching
    """

    def __call__(self, metadata: LazyDict) -> None:
        ...


class InitHandleHook(Protocol):
    """Hook invoked around kernel binary/module loading.
    module/function can be None for the *start* hook (before loading).
    """

    def __call__(
        self,
        module: Optional[object],
        function: Optional[Callable],
        name: str,
        metadata_group: dict[str, str],
        hash: str,
    ) -> None:
        ...


F = TypeVar("F", bound=Callable)


class HookChain(Generic[F]):
    """A chain of hooks of the same type F to be called in order.
    """

    def __init__(self, reversed: bool = False):
        self.calls: list[F] = []
        self.reversed = reversed

    def add(self, func: F) -> None:
        if func not in self.calls:
            self.calls.append(func)

    def remove(self, func: F) -> None:
        if func in self.calls:
            self.calls.remove(func)

    def __call__(self, *args, **kwargs):
        for call in self.calls if not self.reversed else reversed(self.calls):
            call(*args, **kwargs)


# This is of the form [attr_name, attr_val]
# TODO: Use tuple instead of list for better typing.
KernelAttr = list[Union[str, int]]


class JITHookCompileInfo(TypedDict):
    key: str
    signature: dict[KernelParam, str]
    device: int
    constants: None
    num_warps: int
    num_ctas: int
    num_stages: int
    enable_fp_fusion: bool
    launch_cooperative_grid: bool
    extern_libs: tuple[tuple[str, str], ...]
    configs: list[dict[tuple[int, ...], list[KernelAttr]]]
    specialization_data: str
    is_warmup: bool


class JITHook(Protocol):

    def __call__(self, *, key: str, repr: str, fn: JitFunctionInfo, compile: JITHookCompileInfo, is_manual_warmup: bool,
                 already_compiled: bool) -> Optional[bool]:
        ...


class runtime_knobs(base_knobs):
    interpret: env_bool = env_bool("TRITON_INTERPRET")
    # debug is on critical path for kernel launches
    # avoid repeated reads from env-var by calling get directly
    debug: bool = env_bool("TRITON_DEBUG").get()
    override_arch: env_opt_str = env_opt_str("TRITON_OVERRIDE_ARCH")

    launch_enter_hook: HookChain[LaunchHook] = HookChain()
    launch_exit_hook: HookChain[LaunchHook] = HookChain(reversed=True)
    kernel_load_start_hook: HookChain[InitHandleHook] = HookChain()
    kernel_load_end_hook: HookChain[InitHandleHook] = HookChain(reversed=True)

    # Hook for inspecting compiled functions and modules
    jit_cache_hook: Optional[JITHook] = None
    # Hook to signal that a kernel is done compiling and inspect compiled function.
    # jit_cache_hook will always be called before compilation and jit_post_compile_hook after.
    jit_post_compile_hook: Optional[JITHook] = None


class language_knobs(base_knobs):
    fp32_default: env_opt_str = env_opt_str("TRITON_F32_DEFAULT")
    default_fp_fusion: env_bool = env_bool("TRITON_DEFAULT_FP_FUSION", True)


class nvidia_knobs(base_knobs):
    cuobjdump: env_nvidia_tool = env_nvidia_tool("cuobjdump")
    nvdisasm: env_nvidia_tool = env_nvidia_tool("nvdisasm")
    ptxas: env_nvidia_tool = env_nvidia_tool("ptxas")

    dump_nvptx: env_bool = env_bool("NVPTX_ENABLE_DUMP")
    disable_ptxas_opt: env_bool = env_bool("DISABLE_PTXAS_OPT")
    mock_ptx_version: env_opt_str = env_opt_str("TRITON_MOCK_PTX_VERSION")
    dump_ptxas_log: env_bool = env_bool("TRITON_DUMP_PTXAS_LOG")

    libdevice_path: env_opt_str = env_opt_str("TRITON_LIBDEVICE_PATH")
    libcuda_path: env_opt_str = env_opt_str("TRITON_LIBCUDA_PATH")


class amd_knobs(base_knobs):
    use_buffer_ops: env_bool = env_bool("AMDGCN_USE_BUFFER_OPS")
    # Note: This requires use_buffer_ops be true to have any effect
    use_buffer_atomics: env_bool = env_bool("AMDGCN_USE_BUFFER_ATOMICS", True)
    dump_amdgcn: env_bool = env_bool("AMDGCN_ENABLE_DUMP")
    libhip_path: env_opt_str = env_opt_str("TRITON_LIBHIP_PATH")

    # We use strs so that we can have a default value based on other runtime info
    use_block_pingpong: env_opt_bool = env_opt_bool("TRITON_HIP_USE_BLOCK_PINGPONG")
    use_in_thread_transpose: env_opt_bool = env_opt_bool("TRITON_HIP_USE_IN_THREAD_TRANSPOSE")

    global_prefetch: env_int = env_int("TRITON_HIP_GLOBAL_PREFETCH")
    local_prefetch: env_int = env_int("TRITON_HIP_LOCAL_PREFETCH")
    use_async_copy: env_bool = env_bool("TRITON_HIP_USE_ASYNC_COPY")
    scalarize_packed_fops: env_bool = env_bool("AMDGCN_SCALARIZE_PACKED_FOPS")


class proton_knobs(base_knobs):
    cupti_dir: env_opt_str = env_opt_str("TRITON_CUPTI_LIB_PATH")


build = build_knobs()
redis = redis_knobs()
cache = cache_knobs()
compilation = compilation_knobs()
autotuning = autotuning_knobs()
runtime = runtime_knobs()
language = language_knobs()
nvidia = nvidia_knobs()
amd = amd_knobs()
proton = proton_knobs()


def refresh_knobs():
    runtime.debug = env_bool("TRITON_DEBUG").get()
