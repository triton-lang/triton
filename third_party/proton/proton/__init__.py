# ruff: noqa

# Select a coherent ROCm runtime before libproton.so is loaded. TheRock keeps
# its registered preload handles; HSA is retained here until it is registered.
from ._rocm import configure_runtime

_hsa_runtime_handle = configure_runtime()

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
