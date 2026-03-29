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

import sys

print("[PROTON_DEBUG __init__] eager_init=YES", file=sys.stderr, flush=True)


def _eager_rocprofiler_init():
    try:
        from triton.backends import backends
        if "amd" not in backends:
            return
        from triton._C.libproton import proton as _libproton
        _libproton.ensure_rocprofiler_configured()
        print("[PROTON_DEBUG __init__] eager_init CALLED force_configure", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[PROTON_DEBUG __init__] eager_init FAILED: {e}", file=sys.stderr, flush=True)


_eager_rocprofiler_init()
del _eager_rocprofiler_init
