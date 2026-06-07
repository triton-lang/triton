# ruff: noqa


# When running in a TheRock virtual environment, ROCm libraries live under
# _rocm_sdk_core/lib/ which isn't on LD_LIBRARY_PATH. Point the C++ backend
# at the correct directory so dlopen() can find librocprofiler-sdk.so et al.
def _ensure_rocm_lib_env():
    import os
    if os.environ.get("TRITON_ROCPROFILER_SDK_LIB_PATH"):
        return
    try:
        import _rocm_sdk_core
        lib_dir = os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "lib")
        if os.path.isdir(lib_dir):
            os.environ["TRITON_ROCPROFILER_SDK_LIB_PATH"] = lib_dir
    except ImportError:
        pass


_ensure_rocm_lib_env()

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
