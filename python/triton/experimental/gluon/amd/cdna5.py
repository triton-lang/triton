# CDNA 5 is the architecture name for the gfx1250 target; re-export its APIs.
from .gfx1250 import *  # NOQA: F403
from .gfx1250 import __all__ as __gfx1250_all

__all__ = [*__gfx1250_all]
