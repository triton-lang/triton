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


class _AsyncScopeTokenType(tl.base_type):
    """Frontend type for an async scope token.

    Its IR representation is i32, but retaining a distinct frontend type keeps
    tokens opaque when they are passed across loops or regions instead of
    reconstructing them as generic tensors.
    """

    def _unflatten_ir(self, handles, cursor):
        return AsyncScopeToken(handles[cursor]), cursor + 1

    def _flatten_ir_types(self, builder, out):
        tl.int32._flatten_ir_types(builder, out)

    def __eq__(self, other):
        return isinstance(other, _AsyncScopeTokenType)

    def __hash__(self):
        return hash(_AsyncScopeTokenType)

    def mangle(self):
        return "PAsyncScopeToken"

    def __str__(self):
        return "proton.async_scope_token"


_ASYNC_SCOPE_TOKEN_TYPE = _AsyncScopeTokenType()


class AsyncScopeToken(tl.base_value):
    """Opaque token returned by :func:`allocate_async_token`.

    The token identifies a static asynchronous transaction and may only be
    consumed by :func:`enter_async_scope` and :func:`exit_async_scope`.
    """

    def __init__(self, handle):
        self.handle = handle
        self.type = _ASYNC_SCOPE_TOKEN_TYPE

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
def allocate_async_token(name: tl.constexpr, _semantic=None):
    if not flags.instrumentation_on:
        return 0
    _check_supported_semantic(_semantic)
    name = tl._unwrap_if_constexpr(name)
    handle = triton_proton.create_proton_allocate_async_token(_semantic.builder, name)
    return AsyncScopeToken(handle)


@builtin
def enter_async_scope(token, _semantic=None):
    if not flags.instrumentation_on:
        return
    _check_supported_semantic(_semantic)
    if not isinstance(token, AsyncScopeToken):
        raise TypeError("expected a token returned by allocate_async_token")
    triton_proton.create_proton_async_record(_semantic.builder, True, token.handle)


@builtin
def exit_async_scope(token, _semantic=None):
    if not flags.instrumentation_on:
        return
    _check_supported_semantic(_semantic)
    if not isinstance(token, AsyncScopeToken):
        raise TypeError("expected a token returned by allocate_async_token")
    triton_proton.create_proton_async_record(_semantic.builder, False, token.handle)


class scope:

    def __init__(self, name: str, _semantic=None):
        self.name = name
        self.semantic = _semantic

    def __enter__(self):
        enter_scope(self.name, _semantic=self.semantic)

    def __exit__(self, exc_type, exc_value, traceback):
        exit_scope(self.name, _semantic=self.semantic)
