"""Compile-only tests for RDNA1/RDNA2 (gfx10). No gfx10 hardware required.

gfx10 has no WMMA/MFMA, so tt.dot lowers to FMA. gfx1010 additionally lacks
the v_dot2_f32_f16 / v_dot4_i32_i8 instructions (gfx1011/1012/103x have them).
"""

import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out + offs, tl.load(x + offs, mask=mask) + tl.load(y + offs, mask=mask), mask=mask)


@triton.jit
def dot_kernel(a, b, c, K: tl.constexpr, OUT: tl.constexpr):
    r = tl.arange(0, 16)
    k = tl.arange(0, K)
    A = tl.load(a + r[:, None] * K + k[None, :])
    B = tl.load(b + k[:, None] * 16 + r[None, :])
    tl.store(c + r[:, None] * 16 + r[None, :], tl.dot(A, B, out_dtype=OUT))


def _compile(fn, signature, constexprs, arch):
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=GPUTarget("hip", arch, 32)).asm["amdgcn"]


@pytest.mark.parametrize("arch", ["gfx1010", "gfx1011", "gfx1012", "gfx1030"])
def test_elementwise_compiles(arch):
    sig = {"x": "*fp32", "y": "*fp32", "out": "*fp32", "n": "i32", "BLOCK": "constexpr"}
    asm = _compile(add_kernel, sig, {"BLOCK": 1024}, arch)
    assert "global_load" in asm or "buffer_load" in asm


@pytest.mark.parametrize("arch", ["gfx1010", "gfx1011", "gfx1012", "gfx1030"])
@pytest.mark.parametrize("a_ty,out_ty,out_str", [("*fp16", tl.float32, "*fp32"), ("*i8", tl.int32, "*i32")])
def test_dot_compiles(arch, a_ty, out_ty, out_str):
    sig = {"a": a_ty, "b": a_ty, "c": out_str, "K": "constexpr", "OUT": "constexpr"}
    asm = _compile(dot_kernel, sig, {"K": 16, "OUT": out_ty}, arch)
    assert "wmma" not in asm
    if arch == "gfx1010":
        assert "v_dot2" not in asm and "v_dot4" not in asm
