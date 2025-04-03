from triton.language import core as tl
from triton.language.core import builtin
from triton._C.libtriton import proton as triton_proton

from .flags import get_instrumentation_on


def record(is_start: tl.constexpr, scope_name: tl.constexpr, triton_builder=None):
    if not get_instrumentation_on():
        return
    assert triton_builder, "triton_builder must be provided"
    is_start = tl._unwrap_if_constexpr(is_start)
    scope_name = tl._unwrap_if_constexpr(scope_name)
    op_builder = triton_builder.get_op_builder()
    return tl.tensor(triton_proton.create_proton_record(op_builder, is_start, scope_name), tl.void)


@builtin
def enter_scope(name: tl.constexpr, _builder=None):
    record(is_start=True, scope_name=name, triton_builder=_builder)


@builtin
def exit_scope(name: tl.constexpr, _builder=None):
    record(is_start=False, scope_name=name, triton_builder=_builder)
