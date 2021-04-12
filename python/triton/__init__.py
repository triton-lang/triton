# TODO: torch needs to be imported first
# or pybind11 shows `munmap_chunk(): invalid pointer`
import torch
# submodules
from . import testing
from .code_gen import jit, Config
from .core import *
from triton._C.libtriton.triton.frontend import *

# version
__version__ = '1.0.0'