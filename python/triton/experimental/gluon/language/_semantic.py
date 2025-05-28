from triton.language import semantic as tl_semantic
from . import _core as ttgl
from triton._C.libtriton import ir


def arange(start, end, layout, builder: ir.builder):
    shape = [end - start]
    ret_ty = ttgl.distributed_type(ttgl.int32, shape, layout)
    return tl_semantic.arange(start, end, builder, ret_ty=ret_ty)
