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

GFX1250_TARGET = GPUTarget("hip", "gfx1250", 32)
TTIR_PATH = str(Path(__file__).parent / "attn_fwd.ttir")


@lru_cache(maxsize=None)
def compile_for_target(target):
    return triton.compile(TTIR_PATH, target=target)


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
