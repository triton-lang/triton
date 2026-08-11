import ctypes
import functools
import os
from pathlib import Path


@functools.lru_cache
def find_therock_rocm_libraries():
    """Find a coherent set of ROCm runtime libraries from TheRock wheels."""
    try:
        import rocm_sdk

        libraries = {
            name: str(rocm_sdk.find_libraries(name)[0])
            for name in (
                "amdhip64",
                "rocprofiler-sdk",
                "rocprofiler-sdk-roctx",
                "roctracer64",
                "roctx64",
            )
        }
        hsa = Path(libraries["amdhip64"]).parent / "libhsa-runtime64.so.1"
        if not hsa.is_file():
            return None
        libraries["hsa-runtime64"] = str(hsa)
        return libraries
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        return None


def find_legacy_rocprofiler_sdk_lib_dir():
    try:
        import _rocm_sdk_core

        lib_dir = Path(_rocm_sdk_core.__file__).parent / "lib"
        if lib_dir.is_dir():
            return str(lib_dir)
    except ImportError:
        pass

    return None


def configure_runtime():
    """Select and preload one coherent ROCm runtime before loading Proton."""
    import triton

    libraries = find_therock_rocm_libraries()
    if libraries is None:
        if triton.knobs.proton.rocprofiler_sdk_lib_path is None:
            lib_dir = find_legacy_rocprofiler_sdk_lib_dir()
            if lib_dir is not None:
                triton.knobs.proton.rocprofiler_sdk_lib_path = lib_dir
        return None

    library_settings = {
        "TRITON_PROTON_HIP_LIB_PATH": ("TRITON_PROTON_HIP_LIBRARY", "amdhip64"),
        "TRITON_HSA_RUNTIME_PATH": ("TRITON_HSA_RUNTIME_LIBRARY", "hsa-runtime64"),
        "TRITON_ROCPROFILER_SDK_LIB_PATH": ("TRITON_ROCPROFILER_SDK_LIBRARY", "rocprofiler-sdk"),
        "TRITON_ROCTRACER_LIB_PATH": ("TRITON_ROCTRACER_LIBRARY", "roctracer64"),
    }
    override_keys = (
        *library_settings,
        *(library_key for library_key, _ in library_settings.values()),
        "TRITON_ROCTX_LIB_PATH",
        "TRITON_ROCTX_LIBRARY",
    )
    explicit_overrides = {key for key in override_keys if key in os.environ}
    for path_key, (library_key, name) in library_settings.items():
        if path_key not in explicit_overrides and library_key not in explicit_overrides:
            path = Path(libraries[name])
            triton.knobs.setenv(path_key, str(path.parent))
            triton.knobs.setenv(library_key, path.name)
    if not {"TRITON_ROCTX_LIB_PATH", "TRITON_ROCTX_LIBRARY"} & explicit_overrides:
        triton.knobs.setenv("TRITON_ROCTX_LIB_PATH", libraries["roctx64"])
        triton.knobs.setenv("TRITON_ROCTX_LIBRARY", libraries["roctx64"])

    # HSA is not registered with rocm_sdk yet, so load and retain it directly.
    hsa_library = os.environ.get("TRITON_HSA_RUNTIME_LIBRARY", "libhsa-runtime64.so.1")
    hsa_path = Path(hsa_library)
    if not hsa_path.is_absolute():
        hsa_dir = os.environ.get("TRITON_HSA_RUNTIME_PATH")
        if hsa_dir:
            hsa_path = Path(hsa_dir) / hsa_path
    hsa_handle = ctypes.CDLL(str(hsa_path), mode=ctypes.RTLD_GLOBAL | os.RTLD_NOW)

    import rocm_sdk

    preload_libraries = (
        ("amdhip64", ("TRITON_PROTON_HIP_LIB_PATH", "TRITON_PROTON_HIP_LIBRARY")),
        ("roctx64", ("TRITON_ROCTX_LIB_PATH", "TRITON_ROCTX_LIBRARY")),
        ("rocprofiler-sdk", ("TRITON_ROCPROFILER_SDK_LIB_PATH", "TRITON_ROCPROFILER_SDK_LIBRARY")),
        ("rocprofiler-sdk-roctx", ("TRITON_ROCPROFILER_SDK_LIB_PATH", "TRITON_ROCPROFILER_SDK_LIBRARY")),
        ("roctracer64", ("TRITON_ROCTRACER_LIB_PATH", "TRITON_ROCTRACER_LIBRARY")),
    )
    preload_names = [
        name for name, overrides in preload_libraries
        if not any(override in explicit_overrides for override in overrides)
    ]
    if preload_names:
        rocm_sdk.preload_libraries(*preload_names)
    return hsa_handle
