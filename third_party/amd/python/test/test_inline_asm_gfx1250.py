"""Compile-only GFX1250 checks for inline assembly."""

import re

import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl


@gluon.jit
def inline_asm_perm_pk16_b4_u4_kernel(idx_ptr, out_ptr, lut_ptr):
    """
    Use 128 logical u4 elements per lane, packed as 64 u8 elements, to index
    into a 16-entry LUT of logical b4 values, packed as two u32 values.
    """
    layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0]],
        warp_bases=[],
        block_bases=[],
        shape=[256, 8],
    )
    rows = ttgl.arange(0, 256, ttgl.SliceLayout(1, layout))[:, None]
    cols = ttgl.arange(0, 8, ttgl.SliceLayout(0, layout))[None, :]
    offsets = rows * 8 + cols
    idx = ttgl.load(idx_ptr + offsets)
    lut0 = ttgl.load(lut_ptr + 0)
    lut1 = ttgl.load(lut_ptr + 1)
    # lut0/lut1 are scalar values, so they broadcast to the idx shape. With
    # pack=8 each scalar contributes eight identical asm operands; use the
    # first copy of each LUT register. This does not affect the final assembly.
    out = ttgl.inline_asm_elementwise(
        "v_perm_pk16_b4_u4 $0, $1, $9, $17",
        ("=v,"
         "s,s,s,s,s,s,s,s,"
         "s,s,s,s,s,s,s,s,"
         "0"),  # The `0` forces the output to be held in the registers used for the indices.
        [lut0, lut1, idx], dtype=idx.dtype, is_pure=True, pack=8,
        operand_vec_sizes=[1, 1, 2],  # Use 2 32-bit registers for idx per asm invocation.
        result_vec_sizes=[2],  # Produce 2 32-bit registers of output per asm invocation.
    )
    ttgl.store(out_ptr + offsets, out)


@gluon.jit
def inline_asm_perm_pk16_b8_u4_kernel(idx_ptr, out_ptr, lut_ptr):
    """
    Use 8 uint8 elements as 16 packed u4 indices. Each asm call produces 16 b8
    values, stored as 8 uint16 elements containing two b8 values each.
    """
    layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0]],
        warp_bases=[],
        block_bases=[],
        shape=[256, 8],
    )
    rows = ttgl.arange(0, 256, ttgl.SliceLayout(1, layout))[:, None]
    cols = ttgl.arange(0, 8, ttgl.SliceLayout(0, layout))[None, :]
    offsets = rows * 8 + cols
    idx = ttgl.load(idx_ptr + offsets)
    lut0 = ttgl.load(lut_ptr + 0)
    lut1 = ttgl.load(lut_ptr + 1)
    out = ttgl.inline_asm_elementwise(
        "v_perm_pk16_b8_u4 $0, $1, $9, $17",
        ("=v,"
         "s,s,s,s,s,s,s,s,"
         "s,s,s,s,s,s,s,s,"
         "v"),
        [lut0, lut1, idx],
        dtype=ttgl.uint16,
        is_pure=True,
        pack=8,
        operand_vec_sizes=[1, 1, 2],
        result_vec_sizes=[4],
    )
    ttgl.store(out_ptr + offsets, out)


@gluon.jit
def inline_asm_perm_pk16_b8_u4_vgpr_lut_kernel(idx_ptr, out_ptr, lut0_ptr, lut1_ptr):
    """
    Use per-element dynamic LUTs from VGPRs instead of uniform scalar LUTs.
    """
    layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0]],
        warp_bases=[],
        block_bases=[],
        shape=[256, 8],
    )
    rows = ttgl.arange(0, 256, ttgl.SliceLayout(1, layout))[:, None]
    cols = ttgl.arange(0, 8, ttgl.SliceLayout(0, layout))[None, :]
    offsets = rows * 8 + cols
    idx = ttgl.load(idx_ptr + offsets)
    lut0 = ttgl.load(lut0_ptr + offsets)
    lut1 = ttgl.load(lut1_ptr + offsets)
    out = ttgl.inline_asm_elementwise(
        "v_perm_pk16_b8_u4 $0, $1, $2, $3",
        "=v,v,v,v",
        [lut0, lut1, idx],
        dtype=ttgl.uint16,
        is_pure=True,
        pack=8,
        operand_vec_sizes=[2, 2, 2],
        result_vec_sizes=[4],
    )
    ttgl.store(out_ptr + offsets, out)


@gluon.jit
def inline_asm_perm_pk16_b8_u4_static_lut_kernel(idx_ptr, out_ptr):
    """
    Materialize a static LUT in registers, then use it for packed u4 indices.
    """
    layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0]],
        warp_bases=[],
        block_bases=[],
        shape=[256, 8],
    )
    rows = ttgl.arange(0, 256, ttgl.SliceLayout(1, layout))[:, None]
    cols = ttgl.arange(0, 8, ttgl.SliceLayout(0, layout))[None, :]
    offsets = rows * 8 + cols
    idx = ttgl.load(idx_ptr + offsets)
    # Produce statically known LUTs without global loads.
    lut0, lut1 = ttgl.inline_asm_elementwise(
        """
        s_mov_b64 $0, 0x8675309
        s_mov_b64 $1, 0x24601
        """,
        "=s,=s",
        args=[],
        dtype=[ttgl.uint64, ttgl.uint64],
        is_pure=True,
        pack=1,
    )
    out = ttgl.inline_asm_elementwise(
        "v_perm_pk16_b8_u4 $0, $1, $9, $17",
        ("=v,"
         "s,s,s,s,s,s,s,s,"
         "s,s,s,s,s,s,s,s,"
         "v"),
        [lut0, lut1, idx],
        dtype=ttgl.uint16,
        is_pure=True,
        pack=8,
        operand_vec_sizes=[1, 1, 2],
        result_vec_sizes=[4],
    )
    ttgl.store(out_ptr + offsets, out)


def compile_kernel(kernel, signature):
    return triton.compile(
        src=gluon._runtime.GluonASTSource(kernel, signature, {}),
        target=GPUTarget("hip", "gfx1250", 32),
        options={"num_warps": 1},
    )


def compile_b4_kernel():
    return compile_kernel(inline_asm_perm_pk16_b4_u4_kernel, {
        "idx_ptr": "*u8",
        "out_ptr": "*u8",
        "lut_ptr": "*u32",
    })


def compile_b8_kernel():
    return compile_kernel(inline_asm_perm_pk16_b8_u4_kernel, {
        "idx_ptr": "*u8",
        "out_ptr": "*u16",
        "lut_ptr": "*u64",
    })


def compile_b8_vgpr_lut_kernel():
    return compile_kernel(inline_asm_perm_pk16_b8_u4_vgpr_lut_kernel, {
        "idx_ptr": "*u8",
        "out_ptr": "*u16",
        "lut0_ptr": "*u8",
        "lut1_ptr": "*u8",
    })


def compile_b8_static_lut_kernel():
    return compile_kernel(inline_asm_perm_pk16_b8_u4_static_lut_kernel, {
        "idx_ptr": "*u8",
        "out_ptr": "*u16",
    })


def get_perm_pk16_instructions(amdgcn, mnemonic):
    pattern = re.compile(rf"{mnemonic}\s+"
                         r"(v\[[^\]]+\]|v\d+),\s*"
                         r"([sv]\[[^\]]+\]|[sv]\d+),\s*"
                         r"([sv]\[[^\]]+\]|[sv]\d+),\s*"
                         r"(v\[[^\]]+\]|v\d+)")
    return pattern.findall(amdgcn)


def test_inline_asm_perm_pk16_b4_u4_in_asm():
    """Vector operands/results should lower to the packed B4 permute instruction."""
    amdgcn = compile_b4_kernel().asm["amdgcn"]
    instructions = get_perm_pk16_instructions(amdgcn, "v_perm_pk16_b4_u4")

    assert len(instructions) >= 4
    assert all(dst == idx for dst, _, _, idx in instructions)
    assert len({(lut0, lut1) for _, lut0, lut1, _ in instructions}) == 1
    assert all(lut0.startswith("s") and lut1.startswith("s") for _, lut0, lut1, _ in instructions)


def test_inline_asm_perm_pk16_b8_u4_in_asm():
    """Vector results should support the GFX1250 B8 permute form."""
    amdgcn = compile_b8_kernel().asm["amdgcn"]
    instructions = get_perm_pk16_instructions(amdgcn, "v_perm_pk16_b8_u4")

    assert len(instructions) >= 4
    assert len({(lut0, lut1) for _, lut0, lut1, _ in instructions}) == 1
    assert all(lut0.startswith("s") and lut1.startswith("s") for _, lut0, lut1, _ in instructions)


def test_inline_asm_perm_pk16_b8_u4_vgpr_lut_in_asm():
    """Dynamic LUT tensors should be usable as adjacent VGPR operands."""
    amdgcn = compile_b8_vgpr_lut_kernel().asm["amdgcn"]
    instructions = get_perm_pk16_instructions(amdgcn, "v_perm_pk16_b8_u4")

    assert len(instructions) >= 4
    assert all(lut0.startswith("v") and lut1.startswith("v") for _, lut0, lut1, _ in instructions)


def test_inline_asm_perm_pk16_b8_u4_static_lut_in_asm():
    """Static LUT values should be materialized with asm, not loaded."""
    amdgcn = compile_b8_static_lut_kernel().asm["amdgcn"]
    instructions = get_perm_pk16_instructions(amdgcn, "v_perm_pk16_b8_u4")

    assert "s_mov_b64" in amdgcn
    assert "0x8675309" in amdgcn
    assert "0x24601" in amdgcn
    assert len(instructions) >= 4
    assert len({(lut0, lut1) for _, lut0, lut1, _ in instructions}) == 1
    assert all(lut0.startswith("s") and lut1.startswith("s") for _, lut0, lut1, _ in instructions)
