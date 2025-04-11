from __future__ import annotations

import importlib
import os
import re
import subprocess
import sysconfig

from dataclasses import dataclass
from typing import cast, Callable, Generic, Protocol, Type, TypeVar, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime.cache import CacheManager, RemoteCacheBackend
    from .runtime.jit import JitFunctionInfo, KernelParam
    from .compiler.compiler import LazyDict


class Unset:
    pass


_UNSET = Unset()


def getenv(key: str) -> str | None:
    res = os.getenv(key)
    return res.strip() if res is not None else res


def get_str_set(*env_vars: str) -> set[str]:
    return {val for key in env_vars if (val := getenv(key))}


class env_str:

    def __init__(self, key: str) -> None:
        self.key = key
        self.value: str | Unset | None = _UNSET

    def __get__(self, obj: object | None, objtype: Type[object] | None = None) -> str | None:
        if isinstance(self.value, Unset):
            return getenv(self.key)
        else:
            return self.value

    def __set__(self, obj: object, value: str | None) -> None:
        self.value = value
        if value is None:
            os.unsetenv(self.key)
        else:
            os.putenv(self.key, value)


class env_str_set:

    def __init__(self, *keys: str) -> None:
        self.keys = tuple(keys)
        self.value: set[str] | Unset = _UNSET

    def __get__(self, obj: object | None, objtype: Type[object] | None = None) -> set[str]:
        if isinstance(self.value, Unset):
            return {val for key in self.keys if (val := getenv(key))}
        else:
            return self.value

    def __set__(self, obj: object, value: set[str]) -> None:
        self.value = value


# Separate class so that types are correct (__get__ is not None)
class env_strd(env_str):

    def __init__(self, key: str, default: str | Callable[[], str]) -> None:
        super().__init__(key)
        self.default: Callable[[], str] = (lambda: default) if isinstance(default, str) else default

    def __get__(self, obj: object | None, objtype: Type[object] | None = None) -> str:
        res = super().__get__(obj, objtype)
        return self.default() if res is None else res


class env_bool:

    def __init__(self, key: str, default: bool = False) -> None:
        # Composition because function signatures change below
        self._internal = env_strd(key, "1" if default else "0")

    def __get__(self, obj: object | None, objtype: Type[object] | None = None) -> bool:
        return self._internal.__get__(obj, objtype).lower() in ("1", "true", "yes", "on", "y")

    def __set__(self, obj: object, value: bool) -> None:
        self._internal.__set__(obj, "1" if value else "0")


class env_int:

    def __init__(self, key: str, default: int = 0) -> None:
        # Composition because function signatures change below
        self._internal = env_strd(key, str(default))

    def __get__(self, obj: object | None, objtype: Type[object] | None = None) -> int:
        val = self._internal.__get__(obj, objtype)
        try:
            return int(val)
        except ValueError as exc:
            raise RuntimeError(f"Unable to use {self._internal.key}={val}: expected int") from exc

    def __set__(self, obj: object, value: int) -> None:
        self._internal.__set__(obj, str(value))


T = TypeVar("T")


class env_class(Generic[T]):

    def __init__(self, key: str, type: str) -> None:
        self.key = key
        # We can't pass the type directly to avoid import cycles
        self.type = type
        self.value: Type[T] | Unset | None = _UNSET

    def __get__(self, obj: object | None, objtype: Type[object] | None = None) -> Type[T] | None:
        if isinstance(self.value, Unset):
            cls_module_str = getenv(self.key)
            if cls_module_str is None:
                return None

            comps = cls_module_str.split(":", 1)
            if len(comps) != 2:
                raise RuntimeError(f"Unable to read {self.key}: '{cls_module_str}' isn't of the form MODULE:CLASS")
            cls = getattr(importlib.import_module(comps[0]), comps[1])

            if not any((c.__name__ == self.type for c in cls.mro())):
                raise RuntimeError(f"Unable to use '{cls_module_str}' from {self.key}: not of type '{self.type}'")

            return cast(Type[T], cls)
        else:
            return self.value

    def __set__(self, obj: object, value: Type[T] | None) -> None:
        self.value = value


def get_triton_dir(dirname: str) -> str:
    return os.path.join(
        getenv("TRITON_HOME") or os.path.expanduser("~/"),
        ".triton",
        dirname,
    )


@dataclass
class NvidiaTool:
    path: str
    version: str


def get_nvidia_tool(binary: str) -> NvidiaTool:
    binary += sysconfig.get_config_var("EXE")
    # nvidia backend root
    paths = [
        getenv(f"TRITON_{binary.upper()}_PATH"),
        os.path.join(
            os.path.dirname(__file__),
            "backends",
            "nvidia",
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
            return NvidiaTool(path, version.group(1))
        except subprocess.CalledProcessError:
            pass

    raise RuntimeError(f"Cannot find {binary}")


class build:
    """Configuration controlling how the native compiler is invoked"""
    cc: env_str = env_str("CC")
    backend_dirs: env_str_set = env_str_set("TRITON_CUDACRT_PATH", "TRITON_CUDART_PATH")


class redis:
    key_format: env_strd = env_strd("TRITON_REDIS_KEY_FORMAT", "triton:{key}:{filename}")
    host: env_strd = env_strd("TRITON_REDIS_HOST", "localhost")
    port: env_int = env_int("TRITON_REDIS_PORT", 6379)


class cache:
    dump_dir: env_strd = env_strd("TRITON_DUMP_DIR", lambda: get_triton_dir("dump"))
    override_dir: env_strd = env_strd("TRITON_OVERRIDE_DIR", lambda: get_triton_dir("override"))
    dir: env_strd = env_strd("TRITON_CACHE_DIR", lambda: get_triton_dir("cache"))

    manager_class: env_class[CacheManager] = env_class("TRITON_CACHE_MANAGER", "CacheManager")
    remote_manager_class: env_class[RemoteCacheBackend] = env_class("TRITON_REMOTE_CACHE_BACKEND", "RemoteCacheBackend")


class compilation:
    override: env_bool = env_bool("TRITON_KERNEL_OVERRIDE")
    dump_ir: env_bool = env_bool("TRITON_KERNEL_DUMP")
    store_binary_only: env_bool = env_bool("TRITON_STORE_BINARY_ONLY")
    always_compile: env_bool = env_bool("TRITON_ALWAYS_COMPILE")
    use_ir_loc: env_bool = env_bool("USE_IR_LOC")
    enable_asan: env_bool = env_bool("TRITON_ENABLE_ASAN")
    disable_line_info: env_bool = env_bool("TRITON_DISABLE_LINE_INFO")
    frontend_debugging: env_bool = env_bool("TRITON_FRONT_END_DEBUGGING")
    allow_non_constexpr_globals: env_bool = env_bool("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS")


class autotuning:
    cache: env_bool = env_bool("TRITON_CACHE_AUTOTUNING")
    print: env_bool = env_bool("TRITON_PRINT_AUTOTUNING")


class LaunchHook(Protocol):

    def __call__(self, metadata: LazyDict) -> None:
        ...


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


class runtime:
    interpret: env_bool = env_bool("TRITON_INTERPRET")
    debug: env_bool = env_bool("TRITON_DEBUG")
    override_arch: env_str = env_str("TRITON_OVERRIDE_ARCH")

    launch_enter_hook: LaunchHook | None = None
    launch_exit_hook: LaunchHook | None = None

    # Hook for inspecting compiled functions and modules
    jit_cache_hook: JitHook | None = None
    # Hook to signal that a kernel is done compiling and inspect compiled function.
    # jit_cache_hook will always be called before compilation and jit_post_compile_hook after.
    jit_post_compile_hook: JitHook | None = None


class language:
    fp32_default: env_str = env_str("TRITON_F32_DEFAULT")
    default_fp_fusion: env_bool = env_bool("TRITON_DEFAULT_FP_FUSION", True)


class nvidia:
    cuobjdump: NvidiaTool = get_nvidia_tool("cuobjdump")  #field(default_factory=lambda: _get_nvidia_tool("cuobjdump"))
    nvdisasm: NvidiaTool = get_nvidia_tool("nvdisasm")  #field(default_factory=lambda: _get_nvidia_tool("nvdisasm"))
    ptxas: NvidiaTool = get_nvidia_tool("ptxas")  #field(default_factory=lambda: _get_nvidia_tool("ptxas"))

    dump_nvptx: env_bool = env_bool("NVPTX_ENABLE_DUMP")
    disable_ptxas_opt: env_bool = env_bool("DISABLE_PTXAS_OPT")
    mock_ptx_version: env_str = env_str("TRITON_MOCK_PTX_VERSION")

    libdevice_path: env_str = env_str("TRITON_LIBDEVICE_PATH")
    libcuda_path: env_str = env_str("TRITON_LIBCUDA_PATH")


class amd:
    use_buffer_ops: env_bool = env_bool("AMDGCN_USE_BUFFER_OPS")
    dump_amdgcn: env_bool = env_bool("AMDGCN_ENABLE_DUMP")
    libhip_path: env_str = env_str("TRITON_LIBHIP_PATH")
    lld_path: env_str = env_str("TRITON_HIP_LLD_PATH")

    # We use strs so that we can have a default value based on other runtime info
    use_block_pingpong: env_str = env_str("TRITON_HIP_USE_BLOCK_PINGPONG")
    use_in_thread_transpose: env_str = env_str("TRITON_HIP_USE_IN_THREAD_TRANSPOSE")

    global_prefetch: env_int = env_int("TRITON_HIP_GLOBAL_PREFETCH")
    local_prefetch: env_int = env_int("TRITON_HIP_GLOBAL_PREFETCH")
    use_async_copy: env_bool = env_bool("TRITON_HIP_GLOBAL_PREFETCH")


class proton:
    cupti_path: env_str = env_str("TRITON_CUPTI_LIB_PATH")
