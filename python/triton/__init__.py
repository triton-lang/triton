# TODO: torch needs to be imported first
# or pybind11 shows `munmap_chunk(): invalid pointer`
import torch

# libtriton resources
import atexit
import triton._C.libtriton as libtriton
@atexit.register
def cleanup():
  libtriton.cleanup()

from .kernel import *
from . import ops