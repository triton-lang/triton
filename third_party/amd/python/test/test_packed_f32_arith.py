"""Test that gfx1250 produces packed f32 arith ops (v_pk_fma_f32 etc).

Compile-only test — no gfx1250 hardware required.
Compiles attn_fwd.ttir for gfx1250 and checks that:
  - LLVM IR contains <2 x float> fmul/fsub/fadd (from packed conversion + VectorCombine)
  - ASM contains v_pk_fma_f32 (from ISel contraction of packed fmul+fsub)
  - Packed FMA lowering clearly dominates scalar FMA lowering in the resulting ASM
"""

import re
from functools import lru_cache
from pathlib import Path

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon._runtime import GluonASTSource
from triton.experimental.gluon import language as gl

GFX1250_TARGET = GPUTarget("hip", "gfx1250", 32)
TTIR_PATH = str(Path(__file__).parent / "attn_fwd.ttir")


@lru_cache(maxsize=None)
def compile_for_target(target):
    return triton.compile(TTIR_PATH, target=target, options={"enable_fp_fusion": True})


def get_func_body_llir(llir):
    func_body = re.findall(
        r"define amdgpu_kernel void .*? \{(.* ret void.*?)}",
        llir,
        flags=re.DOTALL,
    )
    assert len(func_body) >= 1, "couldn't find kernel body in LLVM IR"
    return func_body[0]


def get_func_body_asm(amdgcn, kernel_name="attn_fwd"):
    pattern = rf"^{kernel_name}:(.*); -- End function"
    body = re.findall(pattern, amdgcn, flags=re.DOTALL | re.MULTILINE)
    assert len(body) >= 1, f"couldn't find {kernel_name} body in asm"
    return body[0]


def count_asm_instructions(func_body):
    return {
        "v_pk_fma_f32": func_body.count("v_pk_fma_f32"),
        "v_pk_mul_f32": func_body.count("v_pk_mul_f32"),
        "v_pk_add_f32": func_body.count("v_pk_add_f32"),
        "v_fma_f32": func_body.count("v_fma_f32"),
        "s_fmac_f32": func_body.count("s_fmac_f32"),
    }


@pytest.fixture(scope="module")
def gfx1250_kernel():
    return compile_for_target(GFX1250_TARGET)


def test_gfx1250_packed_f32_in_llir(gfx1250_kernel):
    """GFX1250 should produce packed <2 x float> fmul/fsub in LLVM IR."""
    llir = gfx1250_kernel.asm["llir"]
    func_body = get_func_body_llir(llir)

    packed_fop = re.compile(r"f(mul|sub|add) <2 x float>")
    assert packed_fop.search(func_body), ("Expected packed <2 x float> fmul/fsub/fadd in LLVM IR for gfx1250")

    # Both fmul and fsub must exist for ISel FMA contraction
    assert re.search(r"fmul.*<2 x float>", func_body), ("Expected packed fmul <2 x float> in LLVM IR")
    assert re.search(r"fsub.*<2 x float>", func_body), ("Expected packed fsub <2 x float> in LLVM IR")


def test_gfx1250_v_pk_fma_f32_in_asm(gfx1250_kernel):
    """GFX1250 ASM should contain v_pk_fma_f32 from ISel contraction."""
    amdgcn = gfx1250_kernel.asm["amdgcn"]
    func_body = get_func_body_asm(amdgcn)
    counts = count_asm_instructions(func_body)

    assert counts["v_pk_fma_f32"] > 100, (f"Expected a substantial number of v_pk_fma_f32 instructions, got "
                                          f"{counts['v_pk_fma_f32']}")
    assert counts["v_fma_f32"] < 20, (f"Expected scalar v_fma_f32 instructions to stay low, got "
                                      f"{counts['v_fma_f32']}")


@gluon.jit
def gluon_fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [4], [0])
    offs = gl.arange(0, BLOCK, layout)
    x = gl.load(x_ptr + offs)
    y = gl.load(y_ptr + offs)
    z = gl.load(z_ptr + offs)
    gl.store(out_ptr + offs, gl.fma(x, y, z))


@pytest.mark.parametrize("dtype, packed_inst", [("fp32", "v_pk_fma_f32"), ("bf16", "v_pk_fma_bf16")])
def test_gfx1250_gl_fma_packed(dtype, packed_inst):
    """gl.fma on gfx1250 should lower to packed FMA instructions."""
    ptr_ty = f"*{dtype}"
    src = GluonASTSource(
        gluon_fma_kernel,
        {"x_ptr": ptr_ty, "y_ptr": ptr_ty, "z_ptr": ptr_ty, "out_ptr": ptr_ty, "BLOCK": "constexpr"},
        {"BLOCK": 1024},
    )
    kernel = triton.compile(src, target=GFX1250_TARGET)
    func_body = get_func_body_asm(kernel.asm["amdgcn"], "gluon_fma_kernel")

    # 8 elements per thread -> 4 packed FMAs.
    assert func_body.count(packed_inst) == 4, func_body
    assert "v_fma_f32" not in func_body
    assert "v_dual_fma" not in func_body
