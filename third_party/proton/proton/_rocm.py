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

    path_libraries = {
        "TRITON_PROTON_HIP_LIB_PATH": "amdhip64",
        "TRITON_HSA_RUNTIME_PATH": "hsa-runtime64",
        "TRITON_ROCPROFILER_SDK_LIB_PATH": "rocprofiler-sdk",
        "TRITON_ROCTRACER_LIB_PATH": "roctracer64",
        "TRITON_ROCTX_LIB_PATH": "roctx64",
    }
    explicit_overrides = {key for key in path_libraries if key in os.environ}
    for key, name in path_libraries.items():
        if key not in explicit_overrides:
            triton.knobs.setenv(key, str(Path(libraries[name]).parent))

    # HSA is not registered with rocm_sdk yet, so load and retain it directly.
    hsa_path = Path(os.environ["TRITON_HSA_RUNTIME_PATH"]) / "libhsa-runtime64.so.1"
    hsa_handle = ctypes.CDLL(str(hsa_path), mode=ctypes.RTLD_GLOBAL | os.RTLD_NOW)

    import rocm_sdk

    preload_libraries = (
        ("amdhip64", "TRITON_PROTON_HIP_LIB_PATH"),
        ("roctx64", "TRITON_ROCTX_LIB_PATH"),
        ("rocprofiler-sdk", "TRITON_ROCPROFILER_SDK_LIB_PATH"),
        ("rocprofiler-sdk-roctx", "TRITON_ROCPROFILER_SDK_LIB_PATH"),
        ("roctracer64", "TRITON_ROCTRACER_LIB_PATH"),
    )
    preload_names = [name for name, override in preload_libraries if override not in explicit_overrides]
    if preload_names:
        rocm_sdk.preload_libraries(*preload_names)
    return hsa_handle
