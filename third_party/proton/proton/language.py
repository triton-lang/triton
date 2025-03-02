from triton.language import core as tl
from triton.language.core import builtin
from triton._C.libtriton import ir
import warnings


def record(is_start: tl.constexpr, scope_name: tl.constexpr, builder=ir.builder):
    warnings.warn(
        "\nWarning the proton language module within Proton contains under development features that are not intended to be used outside of the core development team"
    )
    is_start = tl._unwrap_if_constexpr(is_start)
    scope_name = tl._unwrap_if_constexpr(scope_name)
    return tl.tensor(_builder.create_proton_record(is_start, scope_name), tl.void)


@builtin
def enter_scope(name: tl.constexpr, _builder=None):
    record(is_start=True, scope_name=name, _builder=_builder)


@builtin
def exit_scope(name: tl.constexpr, _builder=None):
    record(is_start=False, scope_name=name, _builder=_builder)
