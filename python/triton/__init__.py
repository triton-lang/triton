# TODO: torch needs to be imported first
# or pybind11 shows `munmap_chunk(): invalid pointer`
import torch
# submodules
from . import testing
from .kernel import *
from . import ops

# version
__version__ = '1.0.0'