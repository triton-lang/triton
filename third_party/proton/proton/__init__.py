# ruff: noqa

# Select a coherent ROCm runtime before libproton.so is loaded. Explicit
# overrides win; otherwise, when TheRock is installed, all ROCm libraries used
# by Triton and Proton come from that installation.
_therock_library_handles = []


def _ensure_rocm_runtime():
    import ctypes
    import os
    import triton
    from ._libpath import find_rocprofiler_sdk_library, find_therock_rocm_libraries

    for key, value in (
        ("TRITON_ROCPROFILER_SDK_INCLUDE_PATH", triton.knobs.proton.rocprofiler_sdk_include_path),
        ("TRITON_ROCPROFILER_SDK_LIB_PATH", triton.knobs.proton.rocprofiler_sdk_lib_path),
    ):
        if value is not None:
            triton.knobs.setenv(key, value)

    libraries = find_therock_rocm_libraries()
    if libraries is None:
        if triton.knobs.proton.rocprofiler_sdk_lib_path is None:
            library = find_rocprofiler_sdk_library()
            if library is not None:
                triton.knobs.proton.rocprofiler_sdk_lib_path = library
        return

    paths = {
        "TRITON_LIBHIP_PATH": (triton.knobs.amd.libhip_path, libraries["amdhip64"]),
        "TRITON_HSA_RUNTIME_PATH": (triton.knobs.proton.hsa_runtime_path, libraries["hsa-runtime64"]),
        "TRITON_ROCPROFILER_SDK_LIB_PATH": (triton.knobs.proton.rocprofiler_sdk_lib_path, libraries["rocprofiler-sdk"]),
        "TRITON_ROCTRACER_LIB_PATH": (triton.knobs.proton.roctracer_lib_path, libraries["roctracer64"]),
        "TRITON_ROCTX_LIB_PATH": (triton.knobs.proton.roctx_lib_path, libraries["roctx64"]),
    }
    explicit_overrides = {key for key in paths if os.environ.get(key) is not None}
    for key, (_, default) in paths.items():
        if key not in explicit_overrides:
            triton.knobs.setenv(key, default)

    # These libraries are also referenced transitively or by unqualified
    # soname. Load the exact TheRock files globally so the dynamic loader cannot
    # satisfy those references from a system ROCm installation.
    preload_libraries = (
        ("hsa-runtime64", "TRITON_HSA_RUNTIME_PATH"),
        ("amdhip64", "TRITON_LIBHIP_PATH"),
        ("roctx64", "TRITON_ROCTX_LIB_PATH"),
        ("rocprofiler-sdk-roctx", "TRITON_ROCPROFILER_SDK_LIB_PATH"),
    )
    for name, override in preload_libraries:
        if override in explicit_overrides:
            continue
        _therock_library_handles.append(ctypes.CDLL(libraries[name], mode=ctypes.RTLD_GLOBAL | os.RTLD_NOW))


_ensure_rocm_runtime()

from .scope import scope, cpu_timed_scope, enter_scope, exit_scope
from .state import state, enter_state, exit_state, metadata_state
from .profile import (
    start,
    activate,
    deactivate,
    finalize,
    profile,
    DEFAULT_PROFILE_NAME,
)
from . import context, specs, mode, data
