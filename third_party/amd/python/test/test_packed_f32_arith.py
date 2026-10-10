"""Test that gfx1250 produces packed arith ops (v_pk_fma_f32 etc).

Compile-only test — no gfx1250 hardware required.
Compiles small Gluon kernels for gfx1250 and checks that:
  - LLVM IR contains <2 x float> fmul/fsub/fadd (from packed conversion)
  - ASM contains v_pk_fma_f32 (from ISel contraction of packed fmul+fsub)
  - gl.fma lowers to packed FMA instructions
"""

import re

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

GFX1250_TARGET = GPUTarget("hip", "gfx1250", 32)
BLOCK = 1024
# Each thread owns 8 elements, so a fully packed elementwise op lowers to 4
# packed instructions per thread.
NUM_PACKED = 4


@gluon.jit
def gluon_binary_kernel(x_ptr, y_ptr, out_ptr, OP: gl.constexpr, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [4], [0])
    offs = gl.arange(0, BLOCK, layout)
    x = gl.load(x_ptr + offs)
    y = gl.load(y_ptr + offs)
    if OP == "add":
        out = x + y
    elif OP == "sub":
        out = x - y
    else:
        out = x * y
    gl.store(out_ptr + offs, out)


@gluon.jit
def gluon_mul_sub_kernel(x_ptr, y_ptr, z_ptr, out_ptr, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [4], [0])
    offs = gl.arange(0, BLOCK, layout)
    x = gl.load(x_ptr + offs)
    y = gl.load(y_ptr + offs)
    z = gl.load(z_ptr + offs)
    gl.store(out_ptr + offs, x * y - z)


@gluon.jit
def gluon_fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [4], [0])
    offs = gl.arange(0, BLOCK, layout)
    x = gl.load(x_ptr + offs)
    y = gl.load(y_ptr + offs)
    z = gl.load(z_ptr + offs)
    gl.store(out_ptr + offs, gl.fma(x, y, z))


def compile_gluon(kernel, signature, constexprs, options=None, aligned=True):
    # With aligned pointers the loads are vectorized, as in real kernels.
    # Otherwise LLVM scalarizes packed fadd/fsub/fmul whose operands come from
    # scalar loads.
    attrs = {}
    if aligned:
        attrs = {(i, ): [["tt.divisibility", 16]] for i, ty in enumerate(signature.values()) if ty.startswith("*")}
    src = gluon.GluonASTSource(kernel, signature, constexprs, attrs)
    return triton.compile(src, target=GFX1250_TARGET, options=options)


def get_func_body_llir(llir):
    func_body = re.findall(
        r"define amdgpu_kernel void .*? \{(.* ret void.*?)}",
        llir,
        flags=re.DOTALL,
    )
    assert len(func_body) >= 1, "couldn't find kernel body in LLVM IR"
    return func_body[0]


def get_func_body_asm(amdgcn, kernel_name):
    pattern = rf"^{kernel_name}:(.*); -- End function"
    body = re.findall(pattern, amdgcn, flags=re.DOTALL | re.MULTILINE)
    assert len(body) >= 1, f"couldn't find {kernel_name} body in asm"
    return body[0]


# There is no v_pk_sub_f32; packed fsub selects v_pk_add_f32 with a neg modifier.
@pytest.mark.parametrize("op, packed_inst", [("add", "v_pk_add_f32"), ("sub", "v_pk_add_f32"), ("mul", "v_pk_mul_f32")])
def test_gfx1250_packed_f32_arith(op, packed_inst):
    """f32 fadd/fsub/fmul on gfx1250 should lower to packed <2 x float> ops."""
    kernel = compile_gluon(
        gluon_binary_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "OP": "constexpr", "BLOCK": "constexpr"},
        {"OP": op, "BLOCK": BLOCK},
    )
    llir = get_func_body_llir(kernel.asm["llir"])
    assert len(re.findall(rf"\bf{op} (?:\w+ )*<2 x float>", llir)) == NUM_PACKED, llir
    assert not re.search(rf"\bf{op} (?:\w+ )*float ", llir), llir

    asm = get_func_body_asm(kernel.asm["amdgcn"], "gluon_binary_kernel")
    assert asm.count(packed_inst) == NUM_PACKED, asm


def test_gfx1250_packed_mul_sub_contracts_to_v_pk_fma_f32():
    """Packed fmul+fsub should be contracted into v_pk_fma_f32 when fp fusion is enabled."""
    kernel = compile_gluon(
        gluon_mul_sub_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "z_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": BLOCK},
        options={"enable_fp_fusion": True},
    )
    llir = get_func_body_llir(kernel.asm["llir"])
    assert re.search(r"\bfmul (?:\w+ )*<2 x float>", llir), llir
    assert re.search(r"\bfsub (?:\w+ )*<2 x float>", llir), llir

    asm = get_func_body_asm(kernel.asm["amdgcn"], "gluon_mul_sub_kernel")
    assert asm.count("v_pk_fma_f32") == NUM_PACKED, asm
    assert "v_fma_f32" not in asm
    assert "v_dual_fma" not in asm


@pytest.mark.parametrize("dtype, packed_inst", [("fp32", "v_pk_fma_f32"), ("bf16", "v_pk_fma_bf16")])
def test_gfx1250_gl_fma_packed(dtype, packed_inst):
    """gl.fma on gfx1250 should lower to packed FMA instructions."""
    ptr_ty = f"*{dtype}"
    kernel = compile_gluon(
        gluon_fma_kernel,
        {"x_ptr": ptr_ty, "y_ptr": ptr_ty, "z_ptr": ptr_ty, "out_ptr": ptr_ty, "BLOCK": "constexpr"},
        {"BLOCK": BLOCK},
        # With vectorized loads LLVM's SLP vectorizer packs scalar FMAs on its
        # own. Use scalar loads so the packing must come from the lowering.
        aligned=False,
    )
    asm = get_func_body_asm(kernel.asm["amdgcn"], "gluon_fma_kernel")
    assert asm.count(packed_inst) == NUM_PACKED, asm
    assert "v_fma_f32" not in asm
    assert "v_dual_fma" not in asm
