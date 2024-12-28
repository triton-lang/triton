from __future__ import annotations  # remove after python 3.11
import warnings

from typing import List, Optional, Sequence, Tuple, TypeVar
import numbers

from triton._C.libtriton import ir
#from triton._C.libproton import ttproton
#import triton._C.libproton as proton
from triton.language import core as tl

def proton_record(isStart: bool, regionId: int, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_proton_record(isStart, regionId), tl.void)

#def record(prefix: str, args: List[tl.tensor], hex: bool, builder: ir.builder) -> tl.tensor:
#    # It makes sense visually for prefix to end in ": "; make it so.  Also,
#    # non-empty prefixes should start with " ".
#    if not prefix.endswith(" ") and args:
#        prefix += " "
#    if not prefix.endswith(": ") and args:
#        prefix = prefix[:-1] + ": "
#    if len(prefix) > 2 and not prefix.startswith(" "):
#        prefix = " " + prefix
#    new_args = [arg.handle for arg in args]
#    is_signed = [arg.dtype in (tl.int1, tl.int8, tl.int16, tl.int32, tl.int64) for arg in args]
#    return tl.tensor(builder.create_record(prefix, hex, new_args, is_signed), tl.void)
