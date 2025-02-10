import triton
from triton.language import core as tl
from triton.language.core import builtin
import warnings


@builtin
def record(is_start: tl.constexpr, scope_id: tl.tensor, _builder=None):
    warnings.warn(
        "\nWarning the proton language module within Proton contains under development features that are not intended to be used outside of the core development team"
    )
    is_start = tl._unwrap_if_constexpr(is_start)
    return tl.tensor(_builder.create_proton_record(is_start, scope_id.handle), tl.void)


@builtin
def init_scope(name: tl.constexpr, _builder=None):
    warnings.warn(
        "\nWarning the proton language module within Proton contains under development features that are not intended to be used outside of the core development team"
    )
    name = tl._unwrap_if_constexpr(name)
    return tl.tensor(_builder.create_proton_init_scope(name), tl.int32)


@triton.jit
def enter_scope(name: tl.constexpr):
    scope_id = init_scope(name)
    record(is_start=True, scope_id=scope_id)
    return scope_id


@triton.jit
def exit_scope(scope_id: tl.tensor):
    record(is_start=False, scope_id=scope_id)
