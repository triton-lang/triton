# TODO: torch needs to be imported first
# or pybind11 shows `munmap_chunk(): invalid pointer`
import torch
# submodules
from .code_gen import jit, autotune, heuristics, Config, Autotuner
from .core import *
from triton._C.libtriton.triton.frontend import *

from . import testing
from . import ops
# version
__version__ = '1.0.0'