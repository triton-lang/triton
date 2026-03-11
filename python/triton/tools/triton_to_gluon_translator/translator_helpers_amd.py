# type: ignore
#
# AMD gfx1250 translator helpers. Mirrors translator_helpers.py (NVIDIA) but
# targets WMMA and TDM instead of tcgen05/TMA.

import math
from typing import Any

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.amd.gfx1250 import wmma, wmma_scaled
from triton.experimental.gluon.language.amd.gfx1250 import tdm as amd_tdm
from triton.experimental.gluon.language.amd.gfx1250 import mbarrier as amd_mbarrier
from triton.language.core import _unwrap_if_constexpr

# Target-agnostic helpers reused from the existing module.
from triton.tools.triton_to_gluon_translator.translator_helpers import (
    default_blocked_layout,
    get_num_threads_per_warp,
    get_num_threads_per_program,
    tl_arange,
    tl_full,
    tl_trans,
    tl_cat,
    cat,
    cat_with_permute,
    _wrap_axis,
    reset_to_default_layout,
    set_split_src_layout,
    get_split_src_layout,
    convert_to_expand_dims_layout,
    tl_atomic_add,
    current_target,
    get_int_type,
    tl_dot_decomposed_scale_to_16,
    tl_dot_get_expand_dims_layout,
    tl_dot_get_permute_order,
    tl_dot_get_reshape_shape,
    tl_dot_decomposed_broadcast_scale,
    tl_dot_decomposed_get_transposed_order,
    tl_dot_decomposed_extend_and_broadcast_scale,
    tl_dot_decomposed_mask_nan,
    tl_dot_decomposed_scale_arg,
)

# ---- WMMA layout helpers ----


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
    a_bitwidth = a_ty.element_ty.primitive_bitwidth
    b_bitwidth = b_ty.element_ty.primitive_bitwidth
    min_bitwidth = min(a_bitwidth, b_bitwidth)
    return max(128 // min_bitwidth, 1)


@gluon.constexpr_function
def get_shared_memory_mma_layout(shape, element_bitwidth):
    """PaddedSharedLayout with bank-conflict-avoiding padding."""
    pad_elems = max(128 // element_bitwidth, 8)
    return ttgl.PaddedSharedLayout.with_identity_for(
        [[shape[-1], pad_elems]],
        list(shape),
        [1, 0],
    )


# ---- dot (WMMA) ----


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

    result = wmma(a, b, accumulator)

    if acc is not None:
        ret_layout: ttgl.constexpr = acc.type.layout
    else:
        ret_layout: ttgl.constexpr = default_blocked_layout(result.type.shape, num_warps)
    result = ttgl.convert_layout(result, ret_layout)
    return result


# ---- dot_scaled (decomposed; uses tl_dot above) ----
# Redefined here so the recursive calls resolve to the AMD tl_dot.


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
    return tl_dot_decomposed_block_scales(
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


@gluon.jit
def tl_dot_decomposed_block_scales(
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
    # When only rhs has scales, transpose and recurse with scales on lhs.
    if lhs_scale is None and rhs_scale is not None:
        lhs_trans = tl_trans(lhs)
        rhs_trans = tl_trans(rhs)
        if acc is not None:
            orig_layout: ttgl.constexpr = acc.type.layout
            acc = tl_trans(acc)
        result = tl_dot_scaled(
            rhs_trans,
            rhs_scale,
            rhs_format,
            lhs_trans,
            lhs_scale,
            lhs_format,
            acc,
            fast_math,
            lhs_k_pack,
            rhs_k_pack,
            out_dtype,
        )
        result = tl_trans(result)
        if acc is not None:
            result = ttgl.convert_layout(result, orig_layout)
        return result
    else:
        ttgl.static_assert(not (not lhs_k_pack or not rhs_k_pack), "TODO: support m/n packed formats")
        compute_type: ttgl.constexpr = (ttgl.float16 if
                                        (lhs_format == "fp16" or rhs_format == "fp16") else ttgl.bfloat16)

        scale_a = tl_dot_decomposed_scale_arg(lhs, lhs_scale, lhs_format, 0, compute_type, fast_math)
        scale_b = tl_dot_decomposed_scale_arg(rhs, rhs_scale, rhs_format, 1, compute_type, fast_math)

        return tl_dot(scale_a, scale_b, acc, out_dtype=out_dtype)


# ---- TDM tensor descriptors ----


@gluon.constexpr_function
def get_default_tdm_layout(block_shape, element_bitwidth):
    pad_elems = max(128 // element_bitwidth, 8)
    return ttgl.PaddedSharedLayout.with_identity_for(
        [[block_shape[-1], pad_elems]],
        list(block_shape),
        [1, 0],
    )


@ttgl._core.builtin
def tl_make_tensor_descriptor(base, shape, strides, block_shape, padding_option="zero", _semantic=None):
    element_bitwidth = base.dtype.element_ty.primitive_bitwidth
    layout = get_default_tdm_layout(block_shape, element_bitwidth)
    return amd_tdm.make_tensor_descriptor(base, shape, strides, block_shape, layout, _semantic=_semantic)


@gluon.jit
def tl_load_tensor_descriptor(desc, offsets):
    smem = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout)
    amd_tdm.async_load(desc, offsets, smem)
    amd_tdm.async_wait(0)
    ret_layout: ttgl.constexpr = default_blocked_layout(desc.block_shape, ttgl.num_warps())
    out = smem.load(ret_layout)
    return out


@gluon.jit
def tl_store_tensor_descriptor(desc, offsets, value):
    smem = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout, value)
    amd_tdm.async_store(desc, offsets, smem)
    amd_tdm.async_wait(0)


# ---- obj dispatch (routes desc.load/store/gather/scatter to TDM) ----


@gluon.jit
def tl_obj_store(obj, offsets, value):
    if isinstance(obj, amd_tdm.tensor_descriptor):
        return tl_store_tensor_descriptor(obj, offsets, value)
    else:
        return obj.store(offsets, value)


@gluon.jit
def tl_obj_load(obj, offsets):
    if isinstance(obj, amd_tdm.tensor_descriptor):
        return tl_load_tensor_descriptor(obj, offsets)
    else:
        return obj.load(offsets)


@gluon.jit
def tl_obj_gather(obj, x_offsets, y_offset):
    if isinstance(obj, amd_tdm.tensor_descriptor):
        desc = obj
        desc_shape: ttgl.constexpr = [x_offsets.shape[0], desc.block_shape[1]]
        alloc = ttgl.allocate_shared_memory(desc.dtype, desc_shape, desc.layout)
        amd_tdm.async_gather(desc, x_offsets, y_offset, alloc)
        amd_tdm.async_wait(0)
        ret_layout: ttgl.constexpr = default_blocked_layout(desc_shape, ttgl.num_warps())
        out = alloc.load(ret_layout)
        return out
    else:
        return obj.gather(x_offsets, y_offset)


@gluon.jit
def tl_obj_scatter(obj, value, x_offsets, y_offset):
    if isinstance(obj, amd_tdm.tensor_descriptor):
        desc = obj
        desc_shape: ttgl.constexpr = [x_offsets.shape[0], desc.block_shape[1]]
        alloc = ttgl.allocate_shared_memory(desc.dtype, desc_shape, desc.layout, value)
        amd_tdm.async_scatter(desc, x_offsets, y_offset, alloc)
        amd_tdm.async_wait(0)
    else:
        obj.scatter(value, x_offsets, y_offset)


# ---- host-side descriptor conversion ----


def convert_host_descriptor(desc):
    """Convert a Triton TensorDescriptor to a gfx1250 TDM descriptor."""

    def torch_dtype_to_triton(dtype):
        import torch

        if dtype == torch.float8_e5m2:
            return ttgl.float8e5
        if dtype == torch.float8_e4m3fn:
            return ttgl.float8e4nv
        return getattr(ttgl, str(dtype).split(".")[1])

    from triton.tools.tensor_descriptor import TensorDescriptor

    assert isinstance(desc, TensorDescriptor)
    block_shape = desc.block_shape
    dtype = desc.base.dtype
    element_bitwidth = torch_dtype_to_triton(dtype).primitive_bitwidth
    layout = get_default_tdm_layout(block_shape, element_bitwidth)
    return gluon.amd.gfx1250.TensorDescriptor(
        desc.base, list(desc.shape), list(desc.strides), block_shape, layout
    )
