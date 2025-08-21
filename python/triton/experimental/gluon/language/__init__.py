from . import _core as _gluon_core  # runtime module reference
from ._core import *  # NOQA: F403
from ._core import __all__ as __core_all
from ._layouts import *  # NOQA: F403
from ._layouts import __all__ as __layouts_all
from ._math import *  # NOQA: F403
from ._math import __all__ as __math_all
from ._standard import *  # NOQA: F403
from ._standard import __all__ as __standard_all
from typing import Any

from . import nvidia
from . import amd
from . import extra

__all__ = [
    *__core_all,
    *__layouts_all,
    *__math_all,
    *__standard_all,
    "amd",
    "nvidia",
    "extra",
]


def __getattr__(name: str) -> Any:
    """Allow attribute-style access to dynamically forwarded symbols.

    This suppresses mypy attr-defined errors for names injected in `_core` at
    runtime (e.g. associative_scan). If the name truly doesn't exist,
    AttributeError propagates as usual.
    """
    try:
        return getattr(_gluon_core, name)
    except AttributeError:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
