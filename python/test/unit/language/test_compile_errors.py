import contextlib
import pytest
import os

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure
import traceback
from triton._internal_testing import is_interpreter, is_cuda, is_hip, is_hip_mi300


def format_exception(type, value, tb):
    list_msg = traceback.format_exception(type, value, tb, chain=False)
    return "\n".join(list_msg)


def test_err_undefined_variable():

    @triton.jit
    def kernel():
        a += 1  # noqa

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))

    try:
        err_msg = format_exception(e.type, value=e.value, tb=e.tb)
        assert "is not defined" in err_msg, "error should mention the undefined variable"
        assert "code_generator.py" not in err_msg
    except AssertionError as assertion_err:
        raise assertion_err from e.value


def test_err_in_binary_operator():

    @triton.jit
    def kernel():
        0 + "a"

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))

    try:
        err_msg = format_exception(e.type, value=e.value, tb=e.tb)
        assert "at 2:4:" in err_msg, "error should point to the 0"
        assert "code_generator.py" not in err_msg
    except AssertionError as assertion_err:
        raise assertion_err from e.value


def test_err_static_assert():

    @triton.jit
    def kernel():
        tl.static_assert(isinstance(0, tl.tensor))

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))

    try:
        assert isinstance(e.value, CompileTimeAssertionFailure)
        assert e.value.__cause__ is None
        err_msg = format_exception(e.type, value=e.value, tb=e.tb)
        print(err_msg)
        assert "at 2:4:" in err_msg, "error should point to the static_assert call"
        assert "<source unavailable>" not in err_msg
        assert "code_generator.py" not in err_msg
    except AssertionError as assertion_err:
        raise assertion_err from e.value


def test_err_in_unary_op():
    # Currently Triton can't evaluate `not` of a tuple at compile time.  That's
    # ok, but the error message needs to point to the correct spot.
    @triton.jit
    def kernel():
        not (0, 0)

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))

    try:
        assert e.value.__cause__ is None
        err_msg = format_exception(e.type, value=e.value, tb=e.tb)
        assert "at 2:4:" in err_msg, "error should point to the `not`"
        assert "<source unavailable>" not in err_msg
        assert "code_generator.py" not in err_msg
    except AssertionError as assertion_err:
        raise assertion_err from e.value


def test_err_in_binary_op():

    @triton.jit
    def kernel():
        1.0 << 1

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))

    try:
        err_msg = format_exception(e.type, value=e.value, tb=e.tb)
        assert "at 2:4:" in err_msg, "error should point to the 1.0"
        assert "<source unavailable>" not in err_msg
        assert "code_generator.py" not in err_msg
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
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))

    try:
        inner_exc = e.value.__cause__
        inner = format_exception(inner_exc.__class__, inner_exc, inner_exc.__traceback__)
        assert "at 2:4:" in inner, "error should point to xyz"
        assert "<source unavailable>" not in inner
        assert "code_generator.py" not in inner

        outer = format_exception(e.type, value=e.value, tb=e.tb)
        assert "at 3:4" in outer, "error should point to the nested_call"
        assert "<source unavailable>" not in outer
        assert "code_generator.py" not in outer
    except AssertionError as assertion_err:
        raise assertion_err from e.value


def test_err_in_builtin():

    # The root error here comes from core.py.  Make sure the stacktrace reflects
    # this.
    @triton.jit
    def kernel():
        tl.expand_dims(None, -1)

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))

    try:
        inner_exc = e.value.__cause__
        inner = format_exception(inner_exc.__class__, inner_exc, inner_exc.__traceback__)
        assert f"{os.sep}core.py" in inner, "error should point inside core.py"
        assert "code_generator.py" not in inner

        outer = format_exception(e.type, value=e.value, tb=e.tb)
        assert "at 2:4:" in outer, "error should point to expand_dims call"
        assert "<source unavailable>" not in outer
        assert "code_generator.py" not in outer
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

    triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))


def test_not_const_annotate_no_err():

    @triton.jit
    def kernel(N: int = 1):
        pass

    triton.compile(triton.compiler.ASTSource(fn=kernel, signature={'N': 'i32'}, constexprs={}))


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

    triton.compile(triton.compiler.ASTSource(fn=kernel1, signature={"N": "constexpr"}, constexprs={"N": 0}))

    @triton.jit
    def kernel2(N: tl.constexpr):
        a = returns_branched_on_constexpr(N)
        a + tl.arange(0, 8)

    triton.compile(triton.compiler.ASTSource(fn=kernel2, signature={"N": "constexpr"}, constexprs={"N": 1}))


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
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={'N': 'i32'}, constexprs={}))

    try:
        assert "at 2:4:" in str(e.value), "error should point to the function call"
        assert "at 5:8:" in str(e.value.__cause__), "error should point to the second `return`"
    except AssertionError as assertion_err:
        raise assertion_err from e.value


def test_power_of_two_shapes():

    @triton.jit
    def kernel():
        tl.arange(2, 7)

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))
    assert str(e.value.__cause__) == "arange's range must be a power of 2"


def test_power_of_two_shapes_2():

    @triton.jit
    def kernel():
        tl.full((33, ), 0, dtype=tl.int64)

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))
    assert str(e.value.__cause__) == "Shape element 0 must be a power of 2"


def test_captured_var_access():

    CAPTURED = 42

    @triton.jit
    def kernel():
        a = CAPTURED  # noqa

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))
    assert "CAPTURED is not defined" in str(e.value)


GLOBAL = 42


def test_global_var_access():

    @triton.jit
    def kernel():
        a = GLOBAL  # noqa

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))
    assert "global variable" in str(e.value)


CONSTEXPR_ANNOTATED_GLOBAL: tl.constexpr = 42


def test_constexpr_annotated_global_var_access():

    @triton.jit
    def kernel():
        a = CONSTEXPR_ANNOTATED_GLOBAL  # noqa

    # No error.
    triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))


CONSTEXPR_GLOBAL = tl.constexpr(42)


def test_constexpr_global_var_access():

    @triton.jit
    def kernel():
        a = CONSTEXPR_GLOBAL  # noqa

    # No error.
    triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))


TYPE_ALIAS = tl.pointer_type(tl.int32)


def test_global_type_alias_access():

    @triton.jit
    def kernel():
        a = TYPE_ALIAS  # noqa

    # No error.
    triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))


def test_global_access_in_fn_default_arg():

    @triton.jit
    def kernel(a=GLOBAL):
        pass

    # No error.
    triton.compile(triton.compiler.ASTSource(fn=kernel, signature={'a': "i32"}, constexprs={}))


def test_defaults_assign_no_err():

    @triton.jit
    def kernel(a=1, B: tl.constexpr = ""):
        pass

    triton.compile(triton.compiler.ASTSource(fn=kernel, signature={'a': 'i32', 'B': 'constexpr'}, constexprs={'B': ""}))


def test_where_warning(fresh_triton_cache):

    @triton.jit
    def kernel():
        a = tl.full((64, ), 0, tl.uint32)
        b = tl.full((64, ), 1, tl.float32)
        c = tl.full((64, ), 2, tl.float32)
        tl.where(a, b, c)

    with pytest.warns(UserWarning):
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))


@pytest.mark.parametrize("dtype", [tl.float8e5, tl.float8e5b16, tl.float8e4nv, tl.float8e4b8, tl.float8e4b15])
def test_fp8_support(dtype):
    warning_dtypes = []
    supported_dtypes = [tl.float8e5]
    if is_cuda():
        cc = torch.cuda.get_device_capability(0)
        supported_dtypes.append(tl.float8e4b15)
        if cc >= (9, 0):
            warning_dtypes.append(tl.float8e4b15)
        if cc >= (8, 9):
            supported_dtypes.append(tl.float8e4nv)
    elif is_hip():
        if is_hip_mi300():
            supported_dtypes += [tl.float8e4nv, tl.float8e4b8, tl.float8e5b16]
    elif is_interpreter():
        supported_dtypes = [tl.float8e5, tl.float8e5b16, tl.float8e4nv, tl.float8e4b8, tl.float8e4b15]

    @triton.jit
    def dtype_kernel(dtype: tl.constexpr):
        _ = tl.full((256, ), 0.0, dtype)

    if dtype in warning_dtypes:
        ctx = pytest.warns(UserWarning, match=r"fp8e4b15 is deprecated in this architecture")
    elif dtype in supported_dtypes:
        ctx = contextlib.nullcontext()
    else:
        ctx = pytest.raises(CompilationError, match="")

    with ctx as e:
        triton.compile(
            triton.compiler.ASTSource(fn=dtype_kernel, signature={"dtype": "constexpr"}, constexprs={"dtype": dtype}))

    if dtype not in supported_dtypes:
        try:
            assert ("not supported in this architecture" in str(e.value.__cause__))
        except AssertionError as assertion_err:
            raise assertion_err from e.value


@pytest.mark.parametrize("dtype", [tl.float8e5, tl.int8, tl.float16])
def test_min_dot_size(dtype):
    error_msg = "Input shapes should have "
    if is_cuda():
        if dtype.primitive_bitwidth == 8:
            error_msg += "M >= 16, N >= 16 and K >= 32"
        else:
            error_msg = "M >= 16, N >= 16 and K >= 16"
    elif is_hip():
        # hip supports arbitrary sizes
        error_msg = None
    else:
        pytest.skip("Test only supported on CUDA and HIP")

    @triton.jit
    def dot_kernel(dtype: tl.constexpr):
        SIZE: tl.constexpr = 8
        a = tl.full((SIZE, SIZE), 0.0, dtype)
        b = tl.full((SIZE, SIZE), 0.0, dtype)
        tl.dot(a, b)

    if error_msg is None:
        triton.compile(
            triton.compiler.ASTSource(fn=dot_kernel, signature={"dtype": "constexpr"}, constexprs={"dtype": dtype}))
    else:
        with pytest.raises(CompilationError) as e:
            triton.compile(
                triton.compiler.ASTSource(fn=dot_kernel, signature={"dtype": "constexpr"}, constexprs={"dtype": dtype}))
        try:
            assert (error_msg in str(e.value.__cause__))
        except AssertionError as assertion_err:
            raise assertion_err from e.value


def test_max_num_imprecise_acc_limit():

    @triton.jit
    def dot_kernel():
        SIZE: tl.constexpr = 64
        a = tl.full((SIZE, SIZE), 0.0, tl.float8e5)
        b = tl.full((SIZE, SIZE), 0.0, tl.float8e5)
        tl.dot(a, b, max_num_imprecise_acc=128)

    with pytest.raises(CompilationError) as e:
        triton.compile(triton.compiler.ASTSource(fn=dot_kernel, signature={}, constexprs={}))
    try:
        assert (str(e.value.__cause__) == "max_num_imprecise_acc (128) must be <= K (64)")
    except AssertionError as assertion_err:
        raise assertion_err from e.value
