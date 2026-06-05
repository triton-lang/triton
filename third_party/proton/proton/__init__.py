# ruff: noqa


# Point the C++ backend at custom rocprofiler-sdk include/lib paths before
# libproton.so is loaded. Explicit Proton knobs win; TheRock environments can
# still discover the SDK library through _rocm_sdk_core.
def _ensure_rocprofiler_sdk_env():
    import os
    import triton

    for key, value in (
        ("TRITON_ROCPROFILER_SDK_INCLUDE_PATH", triton.knobs.proton.rocprofiler_sdk_include_path),
        ("TRITON_ROCPROFILER_SDK_LIB_PATH", triton.knobs.proton.rocprofiler_sdk_lib_path),
    ):
        if not os.environ.get(key, None) and value is not None:
            triton.knobs.setenv(key, value)

    if not os.environ.get("TRITON_ROCPROFILER_SDK_LIB_PATH", None):
        try:
            import _rocm_sdk_core
            lib_dir = os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "lib")
            if os.path.isdir(lib_dir):
                triton.knobs.proton.rocprofiler_sdk_lib_path = lib_dir
        except ImportError:
            pass


_ensure_rocprofiler_sdk_env()

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
