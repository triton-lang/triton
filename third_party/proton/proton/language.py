from triton._C.libtriton import ir
from triton.language import core as tl


def proton_record(isStart: bool, regionId: int, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_proton_record(isStart, regionId), tl.void)
