# TODO: torch needs to be imported first
# or pybind11 shows `munmap_chunk(): invalid pointer`
import torch
# submodules
from .code_gen import cdiv, jit, autotune, heuristics, Config, Autotuner, reinterpret

from . import language
from . import code_gen
from . import testing
from . import ops
# version
__version__ = '1.0.0'