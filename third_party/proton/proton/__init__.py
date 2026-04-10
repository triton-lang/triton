# ruff: noqa
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


def _eager_rocprofiler_init():
    try:
        from triton.backends import backends
        if "amd" not in backends:
            return
        from triton._C.libproton import proton as _libproton
        _libproton.ensure_rocprofiler_configured()
    except Exception:
        pass


_eager_rocprofiler_init()
del _eager_rocprofiler_init
