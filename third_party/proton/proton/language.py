from triton._C.libtriton import ir
from triton.language import core as tl
from triton.language.core import builtin
import warnings

def proton_record(isStart: bool, regionId: int, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_proton_record(isStart, regionId), tl.void)

@builtin
def record(isStart: bool, regionId: int, _builder=None):
    warnings.warn(
        "\nWarning the proton language module within Proton contains under development features that are not intended to be used outside of the core development team"
    )
    return proton_record(isStart, regionId, _builder)


