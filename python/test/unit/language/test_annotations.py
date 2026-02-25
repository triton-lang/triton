from __future__ import annotations
import torch
import triton
import triton.language as tl
import pytest
import numpy as np


def annotated_function(return_type=None, **arg_types):
    """A decorator to add annotations to a function."""

    def decorator(func):
        func.__annotations__ = {**arg_types, 'return': return_type}
        return func

    return decorator


# Test integer annotations
@pytest.mark.parametrize(("signed", "width"), [
    (signed, width) for signed in [False, True]\
                    for width in [8, 16, 32, 64]
] + [(False, 1)]
                         )
def test_int_annotation(signed, width, device):

    @triton.jit
    @annotated_function(X=torch.tensor, v=f"tl.{'' if signed else 'u'}int{width}")
    def _kernel(X, v):
        tl.store(X + v, v)

    h = _kernel[(1, )](torch.empty(1, device=device), 3)
    pfx = 'si' if signed else 'ui'
    if not signed and width < 64:
        assert "arith.extui %v" in h.asm["ttir"]
    assert f'%v: i{width}' in h.asm["ttir"]
    assert f'arith.{pfx}tofp' in h.asm["ttir"]


# Test that unknown annotations do not emit an error
def test_unknown_annotation(device):

    @triton.jit
    def _kernel(X: torch.Tensor, N: int, BLOCK_SIZE: tl.constexpr):
        pass

    x = torch.empty(1, device=device)
    _kernel[(1, )](x, x.shape[0], 32)
    try:
        _kernel[(1, )](x.shape[0], x.shape[0], 32)
    except AttributeError:
        pass


# Test float annotations are properly respected
@pytest.mark.parametrize(
    ("dtype", "test_val"),
    [(dtype, test_val)
     for dtype in [tl.float16, tl.bfloat16, tl.float32, tl.float64]
     for test_val in [0.0, 42.0, float("inf"), float("nan")]],
)
def test_float_annotation(device, dtype, test_val):

    @triton.jit
    @annotated_function(val=dtype)
    def _kernel(ptr, val):
        tl.static_assert(val.dtype == dtype)
        tl.store(ptr, val)

    ptr = torch.empty(1, device=device, dtype=torch.float32)
    h = _kernel[(1, )](ptr, test_val)
    np.testing.assert_allclose(ptr.cpu().numpy(), [test_val], atol=1e-6)

    # Check that the type is properly emitted in the IR
    if dtype == tl.float16:
        assert "%val: f16" in h.asm["ttir"]
        assert "arith.extf %val : f16 to f32" in h.asm["ttir"]
    elif dtype == tl.bfloat16:
        assert "%val: bf16" in h.asm["ttir"]
        assert "arith.extf %val : bf16 to f32" in h.asm["ttir"]
    elif dtype == tl.float32:
        assert "%val: f32" in h.asm["ttir"]
    elif dtype == tl.float64:
        assert "%val: f64" in h.asm["ttir"]
        assert "arith.truncf %val : f64 to f32" in h.asm["ttir"]


# Test that contiguity/divisibility hints survive JIT function returns
def test_contiguity_hint_across_jit_return(device):
    """tt.contiguity set inside a callee must propagate to the caller's TTIR.

    Regression test for https://github.com/triton-lang/triton/issues/8592
    """

    @triton.jit
    def _inner(ptr, offset):
        ptrs = ptr + offset
        ptrs = tl.max_contiguous(tl.multiple_of(ptrs, 16), 16)
        return ptrs

    @triton.jit
    def _kernel(ptr, out_ptr):
        offsets = tl.arange(0, 128)
        ptrs = _inner(ptr, offsets)
        v = tl.load(ptrs).to(tl.float32)
        tl.store(out_ptr + offsets, v)

    data = torch.randn(128, device=device, dtype=torch.float32)
    out = torch.empty(128, device=device, dtype=torch.float32)
    h = _kernel[(1,)](data, out)

    assert "tt.contiguity" in h.asm["ttir"], \
        "tt.contiguity lost when hint is applied inside a JIT callee and returned"
    assert "tt.divisibility" in h.asm["ttir"]
    torch.testing.assert_close(out, data)
