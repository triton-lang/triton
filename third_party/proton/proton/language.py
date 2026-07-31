from triton.language import core as tl
from triton.language.core import builtin
from triton._C.libtriton import proton as triton_proton
from triton.language.semantic import TritonSemantic
from triton.experimental.gluon.language._semantic import GluonSemantic

from .flags import flags

_ALL_SEMANTICS = {
    "triton": TritonSemantic,
    "gluon": GluonSemantic,
}
"""
By default **only Gluon** semantic is enabled.
Instrumenting kernels written in Triton DSL is disable because Triton's higher-level IR undergoes
aggressive compiler rewrites (loop pipelining, instruction re-ordering, IR duplication, etc.).
These transformations can invalidate naïve instrumentation and lead to misleading results.
"""
_SEMANTICS = {_ALL_SEMANTICS["gluon"]}


class _EventType(tl.base_type):
    """Frontend type for an asynchronous event.

    Its IR representation is i32, but retaining a distinct frontend type keeps
    events opaque when they are passed across loops or regions instead of
    reconstructing them as generic tensors.
    """

    def _unflatten_ir(self, handles, cursor):
        return Event(handles[cursor]), cursor + 1

    def _flatten_ir_types(self, builder, out):
        tl.int32._flatten_ir_types(builder, out)

    def __eq__(self, other):
        return isinstance(other, _EventType)

    def __hash__(self):
        return hash(_EventType)

    def mangle(self):
        return "PEvent"

    def __str__(self):
        return "proton.event"


_EVENT_TYPE = _EventType()


class Event(tl.base_value):
    """Opaque event returned by :func:`allocate_event`.

    The event identifies a static asynchronous transaction and may only be
    consumed by :func:`start_event` and :func:`end_event`.
    """

    def __init__(self, handle):
        self.handle = handle
        self.type = _EVENT_TYPE

    def _flatten_ir(self, handles):
        handles.append(self.handle)

    def _set_name(self, builder, name):
        self.handle.set_loc(builder.create_name_loc(name, self.handle.get_loc()))


def _check_supported_semantic(semantic):
    if not isinstance(semantic, tuple(_SEMANTICS)):
        raise TypeError(f"Unsupported semantic type: {type(semantic)}. "
                        f"Supported semantics are: {_SEMANTICS}")


def enable_semantic(semantic_name: str):
    _SEMANTICS.add(_ALL_SEMANTICS[semantic_name])


def disable_semantic(semantic_name: str):
    _SEMANTICS.remove(_ALL_SEMANTICS[semantic_name])


def record(is_start: tl.constexpr, scope_name: tl.constexpr, semantic):
    if not flags.instrumentation_on:
        return
    _check_supported_semantic(semantic)
    is_start = tl._unwrap_if_constexpr(is_start)
    scope_name = tl._unwrap_if_constexpr(scope_name)
    triton_proton.create_proton_record(semantic.builder, is_start, scope_name)


@builtin
def enter_scope(name: tl.constexpr, _semantic=None):
    record(is_start=True, scope_name=name, semantic=_semantic)


@builtin
def exit_scope(name: tl.constexpr, _semantic=None):
    record(is_start=False, scope_name=name, semantic=_semantic)


@builtin
def allocate_event(name: tl.constexpr, _semantic=None):
    if not flags.instrumentation_on:
        return 0
    _check_supported_semantic(_semantic)
    name = tl._unwrap_if_constexpr(name)
    handle = triton_proton.create_proton_allocate_event(_semantic.builder, name)
    return Event(handle)


@builtin
def start_event(event, _semantic=None):
    if not flags.instrumentation_on:
        return
    _check_supported_semantic(_semantic)
    if not isinstance(event, Event):
        raise TypeError("expected an event returned by allocate_event")
    triton_proton.create_proton_start_event(_semantic.builder, event.handle)


@builtin
def end_event(event, _semantic=None):
    if not flags.instrumentation_on:
        return
    _check_supported_semantic(_semantic)
    if not isinstance(event, Event):
        raise TypeError("expected an event returned by allocate_event")
    triton_proton.create_proton_end_event(_semantic.builder, event.handle)


class scope:

    def __init__(self, name: str, _semantic=None):
        self.name = name
        self.semantic = _semantic

    def __enter__(self):
        enter_scope(self.name, _semantic=self.semantic)

    def __exit__(self, exc_type, exc_value, traceback):
        exit_scope(self.name, _semantic=self.semantic)
