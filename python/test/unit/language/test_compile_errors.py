import pytest

import triton
import triton.language as tl
from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure


def test_err_in_binary_operator():

    @triton.jit
    def kernel():
        0 + "a"

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constants={}))

    try:
        assert "at 2:8:" in str(e.value), "error should point to the opening \" of the string"
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
        assert "at 2:11:" in str(e.value), "error should point to the 1"
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
