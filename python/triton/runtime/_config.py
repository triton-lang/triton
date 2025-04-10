from __future__ import annotations

import importlib
import os
import re
import subprocess
import sysconfig

from dataclasses import dataclass
from typing import overload, Any, Callable, Protocol, Type, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from .cache import CacheManager, FileCacheManager, RemoteCacheBackend
    from .jit import JitFunctionInfo, KernelParam
    from ..compiler.compiler import LazyDict


@overload
def _get_str(key: str) -> str | None:
    ...


@overload
def _get_str(key: str, default: None) -> str | None:
    ...


@overload
def _get_str(key: str, default: str) -> str:
    ...


def _get_str(key: str, default: str | None = None) -> str | None:
    res = os.getenv(key, default)
    return res if not res else res.strip()


def _get_bool(key: str, default: bool = False) -> bool:
    return _get_str(key, "1" if default else "0").lower() in ("1", "true", "yes", "on", "y")


def _get_int(key: str, default: int = 0) -> int:
    val = _get_str(key, str(default))
    try:
        return int(val)
    except ValueError as exc:
        raise RuntimeError(f"Unable to use {key}={val}: expected int") from exc


class _propogated_str:
    """
    Helper descriptor class which updates an environment variable on __set__

    This is necessary so that you can update configuration below and it
    correctly propogates to the C++ layer.

    E.g.

    class Foo:
        bar = _propogated_str("BAR")

    f = Foo()
    f.bar = "baz"
    assert os.environ["BAR"] == "baz"  # True
    """

    def __init__(self, key: str, default: str | None = None) -> None:
        self.key = key
        self.value: str | None = _get_str(key, default)

    def __get__(self, obj: object | None, objtype: Type[object] | None = None) -> str | None:
        return self.value

    def __set__(self, obj: object, value: str | None) -> None:
        self.value = value
        if value is None:
            os.unsetenv(self.key)
        else:
            os.putenv(self.key, value)


class _propogated_bool:
    """Similar to _propogated_str but handles bools automatically"""

    def __init__(self, key: str, default: bool = False) -> None:
        self.key = key
        self.value: bool = _get_bool(key, default)

    def __get__(self, obj: object | None, objtype: Type[object] | None = None) -> bool:
        return self.value

    def __set__(self, obj: object, value: bool) -> None:
        self.value = value
        os.putenv(self.key, "1" if value else "0")


def _get_str_set(*env_vars: str) -> set[str]:
    return {val for key in env_vars if (val := _get_str(key))}


def _load_class_from_env(key: str, type: Type[Any]) -> Type[Any] | None:
    cls_module_str = _get_str(key)
    if not cls_module_str:
        return None

    comps = cls_module_str.split(":", 1)
    if len(comps) != 2:
        raise RuntimeError(f"Unable to read {key}: '{cls_module_str}' isn't of the form MODULE:CLASS")

    module = importlib.import_module(comps[0])
    cls = getattr(module, comps[1])

    if not issubclass(cls, type):
        raise RuntimeError(f"Unable to use '{cls_module_str}' from {key}: not a subclass of {type.__name__}")

    return cls


def _get_triton_dir(dirname: str) -> str:
    return os.path.join(
        _get_str("TRITON_HOME") or os.path.expanduser("~/"),
        ".triton",
        dirname,
    )


@dataclass
class _NvidiaTool:
    path: str
    version: str


def _get_nvidia_tool(binary: str) -> _NvidiaTool:
    binary += sysconfig.get_config_var("EXE")
    # nvidia backend root
    paths = [
        _get_str(f"TRITON_{binary.upper()}_PATH"),
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "third_party",
            "nvidia",
            "backend",
            "bin",
            binary,
        )
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
    cc: str | None = _get_str("CC")
    backend_dirs: set[str] = _get_str_set("TRITON_CUDACRT_PATH", "TRITON_CUDART_PATH")


@dataclass
class _RedisConfig:
    key_format: str = _get_str("TRITON_REDIS_KEY_FORMAT", "triton:{key}:{filename}")
    host: str = _get_str("TRITON_REDIS_HOST", "localhost")
    port: int = _get_int("TRITON_REDIS_PORT", 6379)


@dataclass
class _CacheConfig:
    dump_dir: str = _get_str("TRITON_DUMP_DIR") or _get_triton_dir("dump")
    override_dir: str = _get_str("TRITON_OVERRIDE_DIR") or _get_triton_dir("override")
    dir: str = _get_str("TRITON_CACHE_DIR") or _get_triton_dir("cache")

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
    use_ir_loc: _propogated_bool = _propogated_bool("USE_IR_LOC")
    enable_asan: _propogated_bool = _propogated_bool("TRITON_ENABLE_ASAN")
    disable_line_info: _propogated_bool = _propogated_bool("TRITON_DISABLE_LINE_INFO")
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

    def __call__(self, *, key: str, repr: str, fn: JitFunctionInfo, compile: JitHookCompileInfo, is_manual_warmup: bool,
                 already_compiled: bool) -> bool | None:
        ...


@dataclass
class _RuntimeConfig:
    interpret: bool = _get_bool("TRITON_INTERPRET")
    debug: bool = _get_bool("TRITON_DEBUG")
    override_arch: _propogated_str = _propogated_str("TRITON_OVERRIDE_ARCH")

    launch_enter_hook: LaunchHook | None = None
    launch_exit_hook: LaunchHook | None = None

    # Hook for inspecting compiled functions and modules
    jit_cache_hook: JitHook | None = None
    # Hook to signal that a kernel is done compiling and inspect compiled function.
    # jit_cache_hook will always be called before compilation and jit_post_compile_hook after.
    jit_post_compile_hook: JitHook | None = None


@dataclass
class _LanguageConfig:
    fp32_default: _propogated_str = _propogated_str("TRITON_F32_DEFAULT")
    default_fp_fusion: _propogated_bool = _propogated_bool("TRITON_DEFAULT_FP_FUSION", True)


@dataclass
class _NvidiaConfig:
    cuobjdump: _NvidiaTool = _get_nvidia_tool("cuobjdump")
    nvdisasm: _NvidiaTool = _get_nvidia_tool("nvdisasm")
    ptxas: _NvidiaTool = _get_nvidia_tool("ptxas")

    dump_nvptx: bool = _get_bool("NVPTX_ENABLE_DUMP")
    disable_ptxas_opt: bool = _get_bool("DISABLE_PTXAS_OPT")
    mock_ptx_version: str | None = _get_str("TRITON_MOCK_PTX_VERSION")

    libdevice_path: str | None = _get_str("TRITON_LIBDEVICE_PATH")
    libcuda_path: str | None = _get_str("TRITON_LIBCUDA_PATH")


@dataclass
class _AMDConfig:
    use_buffer_ops: _propogated_bool = _propogated_bool("AMDGCN_USE_BUFFER_OPS")
    dump_amdgcn: bool = _get_bool("AMDGCN_ENABLE_DUMP")
    libhip_path: str | None = _get_str("TRITON_LIBHIP_PATH")
    lld_path: str | None = _get_str("TRITON_HIP_LLD_PATH")

    # We use strs so that we can have a default value based on other runtime info
    use_block_pingpong: str | None = _get_str("TRITON_HIP_USE_BLOCK_PINGPONG")
    use_in_thread_transpose: str | None = _get_str("TRITON_HIP_USE_IN_THREAD_TRANSPOSE")

    global_prefetch: int = _get_int("TRITON_HIP_GLOBAL_PREFETCH")
    local_prefetch: int = _get_int("TRITON_HIP_GLOBAL_PREFETCH")
    use_async_copy: bool = _get_bool("TRITON_HIP_GLOBAL_PREFETCH")


@dataclass
class _ProtonConfig:
    cupti_path: str | None = _get_str("TRITON_CUPTI_LIB_PATH")


@dataclass
class _TritonConfig:
    build = _BuildConfig()
    cache = _CacheConfig()
    compilation = _CompilationConfig()
    autotuning = _AutotuningConfig()
    runtime = _RuntimeConfig()

    # third_party configs
    nvidia = _NvidiaConfig()
    amd = _AMDConfig()
    proton = _ProtonConfig()


config = _TritonConfig()
