import functools
import triton
import os

from triton._C.libproton import proton as libproton
from triton.language import core as tl
from triton.language.core import builtin
from .. import language


@builtin
def record(isStart: bool, regionId: int, _builder=None):
    return language.proton_record(isStart, regionId, _builder)
