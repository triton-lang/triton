# type: ignore

import math

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.amd.gfx1250 import wmma as amd_wmma
from triton.experimental.gluon.language.amd.gfx1250 import tdm as amd_tdm
from triton.experimental.gluon.language.amd.cdna3 import mfma as amd_mfma
from triton.language.target_info import current_target

from triton.tools.triton_to_gluon_translator.common_helpers import *  # noqa: F401,F403
from triton.tools.triton_to_gluon_translator.common_helpers import (
    default_blocked_layout,
    get_num_threads_per_warp,
    tl_dot_decomposed_block_scales_impl,
)

# ---- architecture detection ----


@gluon.constexpr_function
def _is_gfx1250(target=None):
    return target is not None and target.arch == "gfx1250"


@gluon.constexpr_function
def _is_cdna(target=None):
    return target is not None and target.arch in ("gfx942", "gfx950")


@gluon.constexpr_function
def _cdna_version(target=None):
    """Returns 3 for gfx942, 4 for gfx950."""
    return 4 if target is not None and target.arch == "gfx950" else 3


# ---- AMD WMMA layout helpers (gfx1250) ----


@gluon.constexpr_function
def compute_warp_bases(num_warps):
    """Distribute warps across M/N: first bit to N, rest to M."""
    n_bits = int(math.log2(num_warps))
    if n_bits == 0:
        return []
    warp_bases = [[0, 1]]
    for i in range(n_bits - 1):
        warp_bases.append([1 << i, 0])
    return warp_bases


@gluon.constexpr_function
def get_wmma_layout(shape, num_warps):
    warp_bases = compute_warp_bases(num_warps)
    return ttgl.amd.AMDWMMALayout(3, True, warp_bases, [], [16, 16, 32])


@gluon.constexpr_function
def get_wmma_k_width(a_ty, b_ty):
    min_bitwidth = min(a_ty.element_ty.primitive_bitwidth, b_ty.element_ty.primitive_bitwidth)
    return max(128 // min_bitwidth, 1)


# ---- AMD MFMA layout helpers (cdna3/cdna4) ----


@gluon.constexpr_function
def get_mfma_instr_k(element_bitwidth, target=None):
    """K dimension of the MFMA instruction for [32, 32, K]."""
    k_bits = 128 if _cdna_version(target) == 3 else 256
    return k_bits // element_bitwidth


@gluon.constexpr_function
def get_mfma_layout(num_warps, element_bitwidth, target=None):
    instr_k = get_mfma_instr_k(element_bitwidth, target)
    return ttgl.amd.AMDMFMALayout(
        version=_cdna_version(target),
        instr_shape=[32, 32, instr_k],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )


@gluon.constexpr_function
def get_mfma_k_width(a_ty, b_ty, target=None):
    min_bitwidth = min(a_ty.element_ty.primitive_bitwidth, b_ty.element_ty.primitive_bitwidth)
    instr_k = get_mfma_instr_k(min_bitwidth, target)
    return instr_k // 2


# ---- AMD dot paths ----


@gluon.jit
def tl_dot_wmma(a, b, acc, out_dtype):
    """gfx1250 WMMA path."""
    M: ttgl.constexpr = a.type.shape[0]
    N: ttgl.constexpr = b.type.shape[1]
    num_warps: ttgl.constexpr = ttgl.num_warps()

    wmma_layout: ttgl.constexpr = get_wmma_layout([M, N], num_warps)
    k_width: ttgl.constexpr = get_wmma_k_width(a.type, b.type)
    a_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=k_width)
    b_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=k_width)

    a = ttgl.convert_layout(a, a_layout)
    b = ttgl.convert_layout(b, b_layout)

    if acc is not None:
        accumulator = ttgl.convert_layout(acc, wmma_layout)
    else:
        accumulator = ttgl.zeros([M, N], out_dtype, layout=wmma_layout)

    result = amd_wmma(a, b, accumulator)

    if acc is not None:
        ret_layout: ttgl.constexpr = acc.type.layout
    else:
        ret_layout: ttgl.constexpr = default_blocked_layout(result.type.shape, num_warps)
    return ttgl.convert_layout(result, ret_layout)


@gluon.jit
def tl_dot_mfma(a, b, acc, out_dtype):
    """CDNA3/CDNA4 MFMA path."""
    M: ttgl.constexpr = a.type.shape[0]
    N: ttgl.constexpr = b.type.shape[1]
    num_warps: ttgl.constexpr = ttgl.num_warps()
    min_bitwidth: ttgl.constexpr = min(a.type.element_ty.primitive_bitwidth, b.type.element_ty.primitive_bitwidth)
    target: ttgl.constexpr = current_target()

    mfma_layout: ttgl.constexpr = get_mfma_layout(num_warps, min_bitwidth, target)
    k_width: ttgl.constexpr = get_mfma_k_width(a.type, b.type, target)
    a_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=k_width)
    b_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=k_width)

    a = ttgl.convert_layout(a, a_layout)
    b = ttgl.convert_layout(b, b_layout)

    if acc is not None:
        accumulator = ttgl.convert_layout(acc, mfma_layout)
    else:
        accumulator = ttgl.zeros([M, N], out_dtype, layout=mfma_layout)

    result = amd_mfma(a, b, accumulator)

    if acc is not None:
        ret_layout: ttgl.constexpr = acc.type.layout
    else:
        ret_layout: ttgl.constexpr = default_blocked_layout(result.type.shape, num_warps)
    return ttgl.convert_layout(result, ret_layout)


# ---- AMD dot dispatch ----


@gluon.jit
def tl_dot(
    a,
    b,
    acc=None,
    input_precision=None,
    allow_tf32=None,
    max_num_imprecise_acc=None,
    out_dtype=ttgl.float32,
):
    target: ttgl.constexpr = current_target()
    if _is_gfx1250(target):
        return tl_dot_wmma(a, b, acc, out_dtype)
    elif _is_cdna(target):
        return tl_dot_mfma(a, b, acc, out_dtype)


@gluon.jit
def tl_dot_scaled(
    lhs,
    lhs_scale,
    lhs_format,
    rhs,
    rhs_scale,
    rhs_format,
    acc=None,
    fast_math=False,
    lhs_k_pack=True,
    rhs_k_pack=True,
    out_dtype=ttgl.float32,
):
    return tl_dot_decomposed_block_scales_impl(
        tl_dot_scaled,
        tl_dot,
        lhs,
        lhs_scale,
        lhs_format,
        rhs,
        rhs_scale,
        rhs_format,
        acc,
        fast_math,
        lhs_k_pack,
        rhs_k_pack,
        out_dtype,
    )


# ---- AMD TDM tensor descriptors (gfx1250 only) ----


@gluon.constexpr_function
def get_default_tdm_layout(*block_shape):
    block_shape = list(block_shape)
    return ttgl.PaddedSharedLayout.with_identity_for(
        [[block_shape[-1], 4]],
        block_shape,
        list(range(len(block_shape) - 1, -1, -1)),
    )


@tl.core._aggregate
class AMDTensorDescriptorArgs:
    """Wraps a real TDM descriptor alongside the original base pointer.

    The base_ptr is needed by gather/scatter to recreate the descriptor with a different
    block_shape -- Triton uses block_shape=[1, N] but TDM hardware requires [num_indices, N].
    Shape, strides, and block_shape are read from desc (type metadata gives plain Python ints
    for block_shape, tuples for shape/strides)."""
    desc: amd_tdm.tensor_descriptor
    base_ptr: tl.core.tensor


@gluon.jit
def tl_make_tensor_descriptor(base, shape, strides, block_shape, padding_option: ttgl.constexpr = "zero"):
    ttgl.static_assert(_is_gfx1250(current_target()), "tl_make_tensor_descriptor requires gfx1250 target")
    layout: ttgl.constexpr = get_default_tdm_layout(*block_shape)
    desc = amd_tdm.make_tensor_descriptor(base, shape, strides, block_shape, layout)
    return AMDTensorDescriptorArgs(desc, base)


# ---- AMD obj dispatch ----


@gluon.jit
def tl_obj_load_amd(desc, offsets):
    smem = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout)
    amd_tdm.async_load(desc, offsets, smem)
    amd_tdm.async_wait(0)
    ret_layout: ttgl.constexpr = default_blocked_layout(desc.block_shape, ttgl.num_warps())
    return smem.load(ret_layout)


@gluon.jit
def tl_obj_store_amd(desc, offsets, value):
    smem = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout, value)
    amd_tdm.async_store(desc, offsets, smem)
    amd_tdm.async_wait(0)


@gluon.jit
def tl_obj_store(obj, offsets, value):
    if isinstance(obj, AMDTensorDescriptorArgs):
        tl_obj_store_amd(obj.desc, offsets, value)
    elif isinstance(obj, amd_tdm.tensor_descriptor):
        tl_obj_store_amd(obj, offsets, value)
    else:
        return obj.store(offsets, value)


@gluon.jit
def tl_obj_load(obj, offsets):
    if isinstance(obj, AMDTensorDescriptorArgs):
        return tl_obj_load_amd(obj.desc, offsets)
    elif isinstance(obj, amd_tdm.tensor_descriptor):
        return tl_obj_load_amd(obj, offsets)
    else:
        return obj.load(offsets)


@gluon.jit
def tl_obj_gather_amd(desc_args, x_offsets, y_offset):
    NUM_IDX: ttgl.constexpr = x_offsets.shape[0]
    BLOCK_N: ttgl.constexpr = desc_args.desc.block_shape[1]
    smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    gather_block_shape: ttgl.constexpr = [NUM_IDX, BLOCK_N]
    gather_desc = amd_tdm.make_tensor_descriptor(desc_args.base_ptr, desc_args.desc.shape, desc_args.desc.strides,
                                                 gather_block_shape, smem_layout)
    num_warps: ttgl.constexpr = ttgl.num_warps()
    gather_shape: ttgl.constexpr = gather_desc.block_shape
    idx_base: ttgl.constexpr = ttgl.BlockedLayout([gather_shape[0], 1],
                                                  [1, get_num_threads_per_warp(current_target())], [1, num_warps],
                                                  [1, 0])
    idx_layout: ttgl.constexpr = ttgl.SliceLayout(1, idx_base)
    x_offsets = ttgl.convert_layout(x_offsets, idx_layout)
    alloc = ttgl.allocate_shared_memory(desc_args.desc.dtype, list(gather_shape), smem_layout)
    y_off = ttgl.to_tensor(y_offset)
    amd_tdm.async_gather(gather_desc, x_offsets, y_off, alloc)
    amd_tdm.async_wait(0)
    ret_layout: ttgl.constexpr = default_blocked_layout(list(gather_shape), num_warps, current_target())
    out = alloc.load(ret_layout)
    return out


@gluon.jit
def tl_obj_scatter_amd(desc_args, value, x_offsets, y_offset):
    NUM_IDX: ttgl.constexpr = x_offsets.shape[0]
    BLOCK_N: ttgl.constexpr = desc_args.desc.block_shape[1]
    smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    scatter_block_shape: ttgl.constexpr = [NUM_IDX, BLOCK_N]
    scatter_desc = amd_tdm.make_tensor_descriptor(desc_args.base_ptr, desc_args.desc.shape, desc_args.desc.strides,
                                                  scatter_block_shape, smem_layout)
    num_warps: ttgl.constexpr = ttgl.num_warps()
    scatter_shape: ttgl.constexpr = scatter_desc.block_shape
    idx_base: ttgl.constexpr = ttgl.BlockedLayout([scatter_shape[0], 1],
                                                  [1, get_num_threads_per_warp(current_target())], [1, num_warps],
                                                  [1, 0])
    idx_layout: ttgl.constexpr = ttgl.SliceLayout(1, idx_base)
    x_offsets = ttgl.convert_layout(x_offsets, idx_layout)
    alloc = ttgl.allocate_shared_memory(desc_args.desc.dtype, list(scatter_shape), smem_layout, value)
    y_off = ttgl.to_tensor(y_offset)
    amd_tdm.async_scatter(scatter_desc, x_offsets, y_off, alloc)
    amd_tdm.async_wait(0)


@gluon.jit
def tl_obj_gather(obj, x_offsets, y_offset):
    if isinstance(obj, AMDTensorDescriptorArgs):
        return tl_obj_gather_amd(obj, x_offsets, y_offset)
    else:
        return obj.gather(x_offsets, y_offset)


@gluon.jit
def tl_obj_scatter(obj, value, x_offsets, y_offset):
    if isinstance(obj, AMDTensorDescriptorArgs):
        tl_obj_scatter_amd(obj, value, x_offsets, y_offset)
    else:
        obj.scatter(value, x_offsets, y_offset)


# ---- AMD host-side descriptor ----


def convert_host_descriptor(desc):
    from triton.tools.tensor_descriptor import TensorDescriptor

    assert isinstance(desc, TensorDescriptor)
    block_shape = desc.block_shape
    tensor = desc.base

    layout = get_default_tdm_layout(*block_shape)
    return gluon.amd.gfx1250.TensorDescriptor(tensor, list(desc.shape), list(desc.strides), block_shape, layout)
