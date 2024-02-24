import pytest

import triton
import triton.language as tl
from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure
import traceback


def test_err_in_binary_operator():

    @triton.jit
    def kernel():
        0 + "a"

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constants={}))

    try:
        assert "at 2:4:" in str(e.value), "error should point to the 0"
    except AssertionError as assertion_err:
        raise assertion_err from e.value


def test_err_static_assert():

    @triton.jit
    def kernel():
        tl.static_assert(isinstance(0, tl.tensor))

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constants={}))

    try:
        assert isinstance(e.value, CompileTimeAssertionFailure)
        assert e.value.__cause__ is None
        assert "at 2:4:" in str(e.value), "error should point to the static_assert call"
        assert "<source unavailable>" not in str(e.value)
    except AssertionError as assertion_err:
        raise assertion_err from e.value


def test_err_in_unary_op():
    # Currently Triton can't evaluate `not` of a tuple at compile time.  That's
    # ok, but the error message needs to point to the correct spot.
    @triton.jit
    def kernel():
        not (0, 0)

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constants={}))

    try:
        assert e.value.__cause__ is None
        assert "at 2:4:" in str(e.value), "error should point to the `not`"
        assert "<source unavailable>" not in str(e.value)
    except AssertionError as assertion_err:
        raise assertion_err from e.value


def test_err_in_binary_op():

    @triton.jit
    def kernel():
        1.0 << 1

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constants={}))

    try:
        assert "at 2:4:" in str(e.value), "error should point to the 1.0"
        assert "<source unavailable>" not in str(e.value)
    except AssertionError as assertion_err:
        raise assertion_err from e.value


# This has to be defined as a top-level function; jit'ed functions can't call
# nested functions.
@triton.jit
def nested_call():
    xyz  # noqa


def test_err_in_nested_call():

    @triton.jit
    def kernel():
        # this is a comment to push nested_call() onto the next line
        nested_call()

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constants={}))

    try:
        inner = e.value.__cause__
        outer = e.value
        assert "at 2:4:" in str(inner), "error should point to xyz"
        assert "<source unavailable>" not in str(inner)

        assert "at 3:4" in str(outer), "error should point to the nested_call"
        assert "<source unavailable>" not in str(outer)
    except AssertionError as assertion_err:
        raise assertion_err from e.value


def test_err_in_builtin():

    # The root error here comes from core.py.  Make sure the stacktrace reflects
    # this.
    @triton.jit
    def kernel():
        tl.expand_dims(None, -1)

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constants={}))

    try:
        inner = e.value.__cause__
        outer = e.value
        assert "/core.py" in '\n'.join(traceback.format_tb(inner.__traceback__)), "error should point inside core.py"

        assert "at 2:4:" in str(outer), "error should point to expand_dims call"
        assert "<source unavailable>" not in str(outer)
    except AssertionError as assertion_err:
        raise assertion_err from e.value


@triton.jit
def two_returns():
    return tl.arange(0, 4)
    return tl.arange(0, 8)


def test_two_returns_no_err():
    # This program is valid; `a` has shape (10,).
    @triton.jit
    def kernel():
        a = two_returns()
        a + tl.arange(0, 4)  # only works if we took the first return

    triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constants={}))


@triton.jit
def returns_branched_on_constexpr(N: tl.constexpr):
    if N == 0:
        return tl.arange(0, 4)
    # Ideally this would work even without the `else`, but we're not that smart
    # yet.
    else:
        return tl.arange(0, 8)


def test_returns_branched_on_constexpr():

    @triton.jit
    def kernel1(N: tl.constexpr):
        a = returns_branched_on_constexpr(N)
        a + tl.arange(0, 4)

    triton.compile(triton.compiler.ASTSource(fn=kernel1, signature={}, constants={"N": 0}))

    @triton.jit
    def kernel2(N: tl.constexpr):
        a = returns_branched_on_constexpr(N)
        a + tl.arange(0, 8)

    triton.compile(triton.compiler.ASTSource(fn=kernel2, signature={}, constants={"N": 1}))


@triton.jit
def returns_branched_on_non_constexpr(N: int):
    if N == 0:
        return tl.arange(0, 4)
    else:
        return tl.arange(0, 8)


def test_returns_branched_on_non_constexpr():

    @triton.jit
    def kernel(N: int):
        returns_branched_on_non_constexpr(N)

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={'N': 'i32'}, constants={}))

    try:
        assert "at 2:4:" in str(e.value), "error should point to the function call"
        assert "at 5:8:" in str(e.value.__cause__), "error should point to the second `return`"
    except AssertionError as assertion_err:
        raise assertion_err from e.value
