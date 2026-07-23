import os
from pathlib import Path


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
        pass

    return None


def find_rocprofiler_sdk_library():
    """Find rocprofiler-sdk in TheRock wheels, with legacy wheel fallback."""
    libraries = find_therock_rocm_libraries()
    if libraries is not None:
        return libraries["rocprofiler-sdk"]

    try:
        import _rocm_sdk_core

        lib_dir = os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "lib")
        if os.path.isdir(lib_dir):
            return lib_dir
    except ImportError:
        pass

    return None
