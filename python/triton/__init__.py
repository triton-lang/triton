from .kernel import *
from .function import *
from .utils import *
import triton.ops


# clean-up libtriton resources
import atexit
import triton._C.libtriton as libtriton
@atexit.register
def cleanup():
  libtriton.cleanup()