# TODO: torch needs to be imported first
# or pybind11 shows `munmap_chunk(): invalid pointer`
import torch
# submodules
from .kernel import *
from . import ops
# C bindings
import triton._C.libtriton.torch_utils as _torch_utils

# version
__version__ = '1.0.0'