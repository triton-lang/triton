# type: ignore
#
# AMD translator helpers. Supports gfx1250 (WMMA), cdna3/gfx942 (MFMA v3),
# and cdna4/gfx950 (MFMA v4). Dispatches at compile time via current_target(),
# same pattern as the NVIDIA helpers dispatch between Blackwell and Hopper.

import math
from typing import Any

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.amd.gfx1250 import wmma, wmma_scaled
from triton.experimental.gluon.language.amd.gfx1250 import tdm as amd_tdm
from triton.experimental.gluon.language.amd.gfx1250 import mbarrier as amd_mbarrier
from triton.experimental.gluon.language.amd.cdna3 import mfma
from triton.language.core import _unwrap_if_constexpr

# Warp-size-independent helpers reused from the existing module.
from triton.tools.triton_to_gluon_translator.translator_helpers import (
    tl_trans,
    cat,
    cat_with_permute,
    _wrap_axis,
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

# ---- warp size and blocked layout (target-aware) ----


@gluon.constexpr_function
def get_num_threads_per_warp() -> ttgl.constexpr:
    target = current_target()
    if target is not None and target.backend == "hip":
        gfx_major = int(target.arch[3:-2])
        return ttgl.constexpr(32 if gfx_major >= 10 else 64)
    return ttgl.constexpr(32)


@ttgl._core.builtin
def get_num_threads_per_program(_semantic=None, _generator=None):
    return ttgl.num_warps(_semantic=_semantic, _generator=_generator) * get_num_threads_per_warp(_semantic=_semantic)


@gluon.constexpr_function
def default_blocked_layout(shape: ttgl.constexpr, num_warps: ttgl.constexpr) -> ttgl.constexpr:
    rank = len(shape)
    size_per_thread = [1] * rank
    threads_per_warp = [1] * rank
    threads_per_warp[rank - 1] = get_num_threads_per_warp()
    warps_per_cta = [1] * rank
    warps_per_cta[0] = num_warps
    order = list(range(rank - 1, -1, -1))
    return ttgl.BlockedLayout(
        size_per_thread=size_per_thread,
        threads_per_warp=threads_per_warp,
        warps_per_cta=warps_per_cta,
        order=order,
    )


# ---- layout-dependent helpers (use local default_blocked_layout) ----


@gluon.jit
def tl_arange(start: ttgl.constexpr, stop: ttgl.constexpr = None):
    layout: ttgl.constexpr = default_blocked_layout([stop - start], ttgl.num_warps())
    return ttgl.arange(start, stop, layout=layout)


@gluon.jit
def tl_full(shape, value, dtype=None):
    layout: ttgl.constexpr = default_blocked_layout(shape, ttgl.num_warps())
    return ttgl.full(shape, value, dtype, layout=layout)


@gluon.jit
def tl_cat(lhs, rhs, can_reorder=False):
    if can_reorder:
        return cat(
            lhs,
            rhs,
            can_reorder,
            layout=default_blocked_layout([lhs.shape[0] + rhs.shape[0]], ttgl.num_warps()),
        )
    else:
        return cat_with_permute(lhs, rhs)


@gluon.jit
def reset_to_default_layout(value):
    ty: ttgl.constexpr = value.type
    if isinstance(ty, ttgl.tuple_type):
        out = ()
        for i in ttgl.static_range(len(value)):
            r = ttgl.convert_layout(value[i], layout=default_blocked_layout(value[i].type.shape, ttgl.num_warps()))
            out = out + (r, )
        return out
    elif isinstance(value, ttgl.tensor) and isinstance(value.type, ttgl.distributed_type):
        layout: ttgl.constexpr = default_blocked_layout(ty.shape, ttgl.num_warps())
        return ttgl.convert_layout(value, layout=layout)
    else:
        return value


@gluon.constexpr_function
def get_split_src_layout(shape: ttgl.constexpr, num_warps: ttgl.constexpr) -> ttgl.constexpr:
    rank = len(shape)
    size_per_thread = [1 if i != rank - 1 else 2 for i in range(rank)]
    threads_per_warp = [1 for _ in range(rank)]
    remaining_threads = get_num_threads_per_warp()
    for dim in range(rank - 2, -1, -1):
        threads_per_warp[dim] = min(shape[dim], remaining_threads)
        remaining_threads = remaining_threads // threads_per_warp[dim]
    warps_per_cta = [1 for _ in range(rank)]
    warps_per_cta[0] = num_warps
    order = list(range(rank - 1, -1, -1))
    return ttgl.BlockedLayout(
        size_per_thread=size_per_thread,
        threads_per_warp=threads_per_warp,
        warps_per_cta=warps_per_cta,
        order=order,
    )


@gluon.jit
def set_split_src_layout(value):
    layout: ttgl.constexpr = get_split_src_layout(value.type.shape, ttgl.num_warps())
    return ttgl.convert_layout(value, layout=layout)


@ttgl._core.builtin
def convert_to_expand_dims_layout(value, expand_dims: list[int], _semantic=None, _generator=None) -> Any:
    parent_shape = _unwrap_if_constexpr(value.type.shape)
    if isinstance(parent_shape, ttgl.tuple):
        parent_shape = parent_shape.values
    assert isinstance(parent_shape, list)
    for dim in expand_dims:
        parent_shape.insert(dim, 1)
    num_warps = ttgl.num_warps(_semantic=_semantic, _generator=_generator)
    layout = default_blocked_layout(parent_shape, num_warps)
    for dim in reversed(expand_dims):
        layout = ttgl.SliceLayout(dim=dim, parent=layout)
    return ttgl.convert_layout(value, layout, _semantic=_semantic)


# ---- architecture detection ----


@gluon.constexpr_function
def _is_gfx1250():
    target = current_target()
    return target is not None and target.arch == "gfx1250"


@gluon.constexpr_function
def _is_cdna():
    target = current_target()
    if target is None:
        return False
    return target.arch in ("gfx942", "gfx950")


@gluon.constexpr_function
def _cdna_version():
    """Returns 3 for gfx942, 4 for gfx950."""
    target = current_target()
    if target is not None and target.arch == "gfx950":
        return 4
    return 3


# ---- WMMA layout helpers (gfx1250) ----


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


# ---- MFMA layout helpers (cdna3/cdna4) ----


@gluon.constexpr_function
def get_mfma_instr_k(element_bitwidth):
    """K dimension of the MFMA instruction for [32, 32, K]."""
    k_bits = 128 if _cdna_version() == 3 else 256
    return k_bits // element_bitwidth


@gluon.constexpr_function
def get_mfma_layout(num_warps, element_bitwidth):
    instr_k = get_mfma_instr_k(element_bitwidth)
    return ttgl.amd.AMDMFMALayout(
        version=_cdna_version(),
        instr_shape=[32, 32, instr_k],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )


@gluon.constexpr_function
def get_mfma_k_width(a_ty, b_ty):
    min_bitwidth = min(a_ty.element_ty.primitive_bitwidth, b_ty.element_ty.primitive_bitwidth)
    instr_k = get_mfma_instr_k(min_bitwidth)
    return instr_k // 2


# ---- shared memory layout ----


@gluon.constexpr_function
def get_shared_memory_mma_layout(shape, element_bitwidth):
    """PaddedSharedLayout with bank-conflict-avoiding padding."""
    pad_elems = max(128 // element_bitwidth, 8)
    return ttgl.PaddedSharedLayout.with_identity_for(
        [[shape[-1], pad_elems]],
        list(shape),
        [1, 0],
    )


# ---- dot (dispatches WMMA vs MFMA) ----


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

    result = wmma(a, b, accumulator)

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

    mfma_layout: ttgl.constexpr = get_mfma_layout(num_warps, min_bitwidth)
    k_width: ttgl.constexpr = get_mfma_k_width(a.type, b.type)
    a_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=k_width)
    b_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=k_width)

    a = ttgl.convert_layout(a, a_layout)
    b = ttgl.convert_layout(b, b_layout)

    if acc is not None:
        accumulator = ttgl.convert_layout(acc, mfma_layout)
    else:
        accumulator = ttgl.zeros([M, N], out_dtype, layout=mfma_layout)

    result = mfma(a, b, accumulator)

    if acc is not None:
        ret_layout: ttgl.constexpr = acc.type.layout
    else:
        ret_layout: ttgl.constexpr = default_blocked_layout(result.type.shape, num_warps)
    return ttgl.convert_layout(result, ret_layout)


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
    if _is_gfx1250():
        return tl_dot_wmma(a, b, acc, out_dtype)
    else:
        return tl_dot_mfma(a, b, acc, out_dtype)


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


# ---- TDM tensor descriptors (gfx1250 only) ----


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
