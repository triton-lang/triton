from __future__ import annotations

import importlib
import os
import re
import subprocess
import sysconfig

from dataclasses import field, dataclass
from typing import overload, Any, Callable, Protocol, Type, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from .cache import CacheManager, FileCacheManager, RemoteCacheBackend
    from .jit import JitFunctionInfo, KernelParam
    from ..compiler.compiler import LazyDict


@overload
def _get_env(env_var: str) -> str | None:
    ...

@overload
def _get_env(env_var: str, default: None) -> str | None:
    ...

@overload
def _get_env(env_var: str, default: str) -> str:
    ...

def _get_env(env_var: str, default: str | None = None) -> str | None:
    res = os.getenv(env_var, default)
    return res if not res else res.strip()

def _get_bool(env_var: str) -> bool:
    return _get_env(env_var, "0").lower() in ("1", "true", "yes", "on", "y")


def _get_set(*env_vars: str) -> set[str]:
    return { val for key in env_vars if (val := _get_env(key)) }


def _load_class_from_env(env_var: str, type: Type[Any]) -> Type[Any] | None:
    cls_module_str = _get_env(env_var)
    if not cls_module_str:
        return None

    comps = cls_module_str.split(":", 1)
    if len(comps) != 2:
        raise RuntimeError(f"Unable to read {env_var}: '{cls_module_str}' isn't of the form MODULE:CLASS")

    module = importlib.import_module(comps[0])
    cls = getattr(module, comps[1])

    if not issubclass(cls, type):
        raise RuntimeError(f"Unable to use '{cls_module_str}' from {env_var}: not a subclass of {type.__name__}")

    return cls


def _get_triton_dir(dirname: str) -> str:
    return os.path.join(
        _get_env("TRITON_HOME") or os.path.expanduser("~/"),
        ".triton",
        dirname,
    )


@dataclass
class _NvidiaTool:
    path: str
    version: str


def _get_nvidia_tool(binary: str) -> _NvidiaTool:
    binary += sysconfig.get_config_var("EXE")
    # triton module root
    base_dir = os.path.dirname(os.path.dirname(__file__))
    paths = [
        _get_env(f"TRITON_{binary.upper()}_PATH"),
        os.path.join(base_dir, "third_party", "cuda", "bin", binary),
    ]
    for path in paths:
        if not path or not os.access(path, os.X_OK):
            continue
        try:
            result = subprocess.check_output([path, "--version"], stderr=subprocess.STDOUT)
            if result is None:
                continue
            version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
            if version is None:
                continue
            return _NvidiaTool(path, version.group(1))
        except subprocess.CalledProcessError:
            pass

    raise RuntimeError(f"Cannot find {binary}")


@dataclass
class _BuildConfig:
    """Configuration controlling how the native compiler is invoked"""
    cc: str | None = _get_env("CC")
    backend_dirs: set[str] = _get_set("TRITON_CUDACRT_PATH", "TRITON_CUDART_PATH")


@dataclass
class _RedisConfig:
    key_format: str = _get_env("TRITON_REDIS_KEY_FORMAT", "triton:{key}:{filename}")
    host: str = _get_env("TRITON_REDIS_HOST", "localhost")
    port: int = int(_get_env("TRITON_REDIS_PORT", "6379"))


@dataclass
class _CacheConfig:
    dump_dir: str = _get_env("TRITON_DUMP_DIR") or _get_triton_dir("dump")
    override_dir: str = _get_env("TRITON_OVERRIDE_DIR") or _get_triton_dir("override")
    dir: str = _get_env("TRITON_CACHE_DIR") or _get_triton_dir("cache")

    manager_class: Type[CacheManager] = FileCacheManager
    remote_manager_class: Type[RemoteCacheBackend] | None = None
    redis: _RedisConfig = _RedisConfig()

    def __post_init__(self) -> None:
        if manager_class := _load_class_from_env("TRITON_CACHE_MANAGER", CacheManager):
            self.manager_class = manager_class

        if remote_manager_class := _load_class_from_env("TRITON_REMOTE_CACHE_BACKEND", RemoteCacheBackend):
            self.remote_manager_class = remote_manager_class


@dataclass
class _CompilationConfig:
    override: bool = _get_bool("TRITON_KERNEL_OVERRIDE")
    dump_ir: bool = _get_bool("TRITON_KERNEL_DUMP")
    store_binary_only: bool = _get_bool("TRITON_STORE_BINARY_ONLY")
    always_compile: bool = _get_bool("TRITON_ALWAYS_COMPILE")
    use_ir_loc: bool = _get_bool("USE_IR_LOC")
    enable_asan: bool = _get_bool("TRITON_ENABLE_ASAN")
    frontend_debugging: bool = _get_bool("TRITON_FRONT_END_DEBUGGING")

    allow_non_constexpr_globals: bool = _get_bool("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS")


@dataclass
class _AutotuningConfig:
    cache: bool = _get_bool("TRITON_CACHE_AUTOTUNING")
    print: bool = _get_bool("TRITON_PRINT_AUTOTUNING")


LaunchHook = Callable[[LazyDict], None]

# This is of the form [attr_name, attr_val]
# TODO: Use tuple instead of list for better typing.
KernelAttr = list[str | int]

class JitHookCompileInfo(TypedDict):
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

class JitHook(Protocol):
    def __call__(
        self,
        *,
        key: str,
        repr: str,
        fn: JitFunctionInfo,
        compile: JitHookCompileInfo,
        is_manual_warmup: bool,
        already_compiled: bool
    ) -> bool:
        ...


@dataclass
class _RuntimeConfig:
    interpret: bool = _get_bool("TRITON_INTERPRET")
    debug: bool = _get_bool("TRITON_DEBUG")

    launch_enter_hook: LaunchHook | None = None
    launch_exit_hook: LaunchHook | None = None

    jit_cache_hook: JitHook | None = None
    jit_post_compile_hook: JitHook | None = None


@dataclass
class _LanguageConfig:
    fp32_default: str | None = _get_env("TRITON_F32_DEFAULT")


@dataclass
class _NvidiaConfig:
    cuobjdump: _NvidiaTool = _get_nvidia_tool("cuobjdump")
    nvdisasm: _NvidiaTool = _get_nvidia_tool("nvdisasm")
    ptxas: _NvidiaTool = _get_nvidia_tool("ptxas")


@dataclass
class _AMDConfig:
    use_buffer_ops: bool = _get_bool("AMDGCN_USE_BUFFER_OPS")


@dataclass
class _TritonConfig:
    build = _BuildConfig()
    cache = _CacheConfig()
    compilation = _CompilationConfig()
    autotuning = _AutotuningConfig()
    runtime = _RuntimeConfig()

    # TODO: ack third_party to find environ, getenv, hook
    nvidia = _NvidiaConfig()
    amd = _AMDConfig()


config = _TritonConfig()
