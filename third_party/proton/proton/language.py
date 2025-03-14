from triton.language import core as tl
from triton.language.core import builtin
from triton._C.libtriton import ir
import warnings

from .flags import get_instrumentation_on


def record(is_start: tl.constexpr, scope_name: tl.constexpr, builder=ir.builder):
    if not get_instrumentation_on():
        return
    warnings.warn(
        "\nWarning the proton language module within Proton contains under development features that are not intended to be used outside of the core development team"
    )
    is_start = tl._unwrap_if_constexpr(is_start)
    scope_name = tl._unwrap_if_constexpr(scope_name)
    return tl.tensor(builder.create_proton_record(is_start, scope_name), tl.void)


@builtin
def enter_scope(name: tl.constexpr, _builder=None):
    record(is_start=True, scope_name=name, builder=_builder)


@builtin
def exit_scope(name: tl.constexpr, _builder=None):
    record(is_start=False, scope_name=name, builder=_builder)
