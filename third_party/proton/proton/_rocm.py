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


def _set_therock_runtime_environment(libraries):
    """Point Triton and Proton at a coherent set of TheRock libraries."""
    import triton

    explicit_overrides = set(os.environ)
    if "TRITON_LIBHIP_PATH" not in explicit_overrides:
        triton.knobs.setenv("TRITON_LIBHIP_PATH", libraries["amdhip64"])

    library_settings = {
        "TRITON_HSA_RUNTIME_PATH": ("TRITON_HSA_RUNTIME_LIBRARY", "hsa-runtime64"),
        "TRITON_ROCPROFILER_SDK_LIB_PATH": ("TRITON_ROCPROFILER_SDK_LIBRARY", "rocprofiler-sdk"),
        "TRITON_ROCTRACER_LIB_PATH": ("TRITON_ROCTRACER_LIBRARY", "roctracer64"),
    }
    for path_key, (library_key, name) in library_settings.items():
        if {path_key, library_key} & explicit_overrides:
            continue
        path = Path(libraries[name])
        triton.knobs.setenv(path_key, str(path.parent))
        triton.knobs.setenv(library_key, path.name)

    if not {"TRITON_ROCTX_LIB_PATH", "TRITON_ROCTX_LIBRARY"} & explicit_overrides:
        triton.knobs.setenv("TRITON_ROCTX_LIBRARY", libraries["roctx64"])

    return explicit_overrides


def _preload_therock_runtime(explicit_overrides):
    """Preload TheRock libraries that were not explicitly overridden."""
    import rocm_sdk

    preload_libraries = (
        ("amdhip64", ("TRITON_LIBHIP_PATH", )),
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


def _load_hsa_runtime():
    """Load and retain HSA, which is not yet registered with rocm_sdk."""
    hsa_path = Path(os.environ.get("TRITON_HSA_RUNTIME_LIBRARY", "libhsa-runtime64.so.1"))
    if not hsa_path.is_absolute() and (hsa_dir := os.environ.get("TRITON_HSA_RUNTIME_PATH")):
        hsa_path = Path(hsa_dir) / hsa_path
    return ctypes.CDLL(str(hsa_path), mode=ctypes.RTLD_GLOBAL | os.RTLD_NOW)


def configure_runtime():
    """Select and preload one coherent ROCm runtime before loading Proton."""
    libraries = find_therock_rocm_libraries()
    if libraries is None:
        return None
    explicit_overrides = _set_therock_runtime_environment(libraries)
    hsa_handle = _load_hsa_runtime()
    _preload_therock_runtime(explicit_overrides)
    return hsa_handle
