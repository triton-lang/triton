import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler.errors import CompilationError


def test_e8m0_arithmetic_rejected(fresh_triton_cache):

    @triton.jit
    def kernel(scale, out, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(scale + offs)
        y = x + x
        tl.store(out + offs, y)

    with pytest.raises(CompilationError) as e:
        triton.compile(
            triton.compiler.ASTSource(
                fn=kernel,
                signature={"scale": "*fp8e8m0fnu", "out": "*fp8e8m0fnu", "BLOCK": "constexpr"},
                constexprs={"BLOCK": 16},
            ),
            target=GPUTarget("hip", "gfx950", 64),
        )

    assert "unexpected type fp8e8m0fnu and fp8e8m0fnu" in str(e.value.__cause__ or e.value)


def test_e8m0_cast_rejected(fresh_triton_cache):

    @triton.jit
    def kernel(scale, out, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(scale + offs)
        y = x.to(tl.float32)
        tl.store(out + offs, y)

    with pytest.raises(CompilationError) as e:
        triton.compile(
            triton.compiler.ASTSource(
                fn=kernel,
                signature={"scale": "*fp8e8m0fnu", "out": "*fp32", "BLOCK": "constexpr"},
                constexprs={"BLOCK": 16},
            ),
            target=GPUTarget("hip", "gfx950", 64),
        )

    assert "cannot cast fp8e8m0fnu" in str(e.value.__cause__ or e.value)
