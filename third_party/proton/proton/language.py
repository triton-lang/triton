from triton.language import core as tl
from triton.language.core import builtin
from triton._C.libtriton import proton as triton_proton

from .flags import get_instrumentation_on


def record(is_start: tl.constexpr, scope_name: tl.constexpr, semantic):
    if not get_instrumentation_on():
        return
    is_start = tl._unwrap_if_constexpr(is_start)
    scope_name = tl._unwrap_if_constexpr(scope_name)
    op_builder = semantic.builder.get_op_builder()
    return tl.tensor(triton_proton.create_proton_record(op_builder, is_start, scope_name), tl.void)


@builtin
def enter_scope(name: tl.constexpr, _semantic=None):
    record(is_start=True, scope_name=name, semantic=_semantic)


@builtin
def exit_scope(name: tl.constexpr, _semantic=None):
    record(is_start=False, scope_name=name, semantic=_semantic)


class scope:

    def __init__(self, name: str, _semantic=None):
        self.name = name
        self.semantic = _semantic

    def __enter__(self):
        enter_scope(self.name, _semantic=self.semantic)

    def __exit__(self, exc_type, exc_value, traceback):
        exit_scope(self.name, _semantic=self.semantic)
