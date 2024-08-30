from triton.language import core
from triton.language import semantic
from triton._C.libtriton import ir

from typing import Sequence

__all__ = [
    "experimental_device_tensormap_create1d",
    "experimental_device_tensormap_create2d",
    "experimental_tensormap_fenceproxy_acquire",
]


def _set_tma_dtype(desc_ptr: core.tensor, element_ty: core.dtype, _builder: ir.builder):
    if element_ty.primitive_bitwidth == 8:
        semantic.tensormap_replace_elemtype(desc_ptr, 0, _builder)
    elif element_ty.primitive_bitwidth == 16:
        semantic.tensormap_replace_elemtype(desc_ptr, 1, _builder)
    elif element_ty.primitive_bitwidth == 32:
        semantic.tensormap_replace_elemtype(desc_ptr, 2, _builder)
    else:
        raise ValueError("element_ty must be a primitive of size 1, 2, or 4 bytes but got")


@core.builtin
def experimental_device_tensormap_create1d(
    desc_out: core.tensor,
    template_desc: core.tensor,
    global_address: core.tensor,
    load_size: core.tensor,
    global_size: core.tensor,
    element_ty: core.dtype,
    _builder: ir.builder,
):
    load_size = core._constexpr_to_value(load_size)
    global_size = core._to_tensor(global_size, _builder)
    element_ty = core._constexpr_to_value(element_ty)

    local_desc = semantic.tensormap_allocate(template_desc, _builder)
    semantic.tensormap_replace_global_address(local_desc, global_address, _builder)
    _set_tma_dtype(local_desc, element_ty, _builder)
    semantic.tensormap_replace_rank(local_desc, 0, _builder)
    semantic.tensormap_replace_box_dim(local_desc, 0, load_size, _builder)
    semantic.tensormap_replace_global_dim(local_desc, 0, global_size, _builder)
    semantic.tensormap_replace_element_stride(local_desc, 0, 1, _builder)
    semantic.tensormap_replace_interleave_layout(local_desc, 0, _builder)
    semantic.tensormap_replace_swizzle_mode(local_desc, 0, _builder)
    semantic.tensormap_replace_fill_mode(local_desc, 0, _builder)

    semantic.tensormap_cp_fenceproxy(desc_out, local_desc, builder=_builder)
    semantic.tensormap_deallocate(local_desc, _builder)


@core.builtin
def experimental_device_tensormap_create2d(
    desc_out: core.tensor,
    template_desc: core.tensor,
    global_address: core.tensor,
    load_size: Sequence[core.constexpr],
    global_size: Sequence[core.tensor],
    element_ty: core.dtype,
    _builder: ir.builder,
):
    assert len(load_size) == 2
    assert len(global_size) == 2
    rank = 2
    load_size = [core._constexpr_to_value(x) for x in load_size]
    global_size = [core._to_tensor(x, _builder) for x in global_size]

    local_desc = semantic.tensormap_allocate(template_desc, _builder)
    semantic.tensormap_replace_global_address(local_desc, global_address, _builder)
    _set_tma_dtype(local_desc, element_ty, _builder)

    semantic.tensormap_replace_rank(local_desc, rank - 1, _builder)

    element_size = element_ty.primitive_bitwidth // 8
    element_size_t = core.full([], element_size, core.int64, _builder=_builder)
    global_stride = semantic.mul(element_size_t, global_size[-1], _builder)
    # Undocumented, but global_stride seems to be divided by 16
    global_stride = semantic.ashr(global_stride, core._to_tensor(4, _builder), _builder)
    semantic.tensormap_replace_global_stride(local_desc, 0, global_stride, _builder)

    contig_dim_size_in_bytes = element_size * load_size[-1]
    if contig_dim_size_in_bytes > 128:
        load_size[-1] = 128 // element_size

    for irank in range(rank):
        # 0th element in tensormap corresponds to this last dimension of the tensor
        idim = rank - (irank + 1)
        semantic.tensormap_replace_box_dim(local_desc, irank, load_size[idim], _builder)
        semantic.tensormap_replace_global_dim(local_desc, irank, global_size[idim], _builder)
        semantic.tensormap_replace_element_stride(local_desc, irank, 1, _builder)

    swizzle = _determine_swizzle_mode_2d(contig_dim_size_in_bytes, load_size)
    semantic.tensormap_replace_interleave_layout(local_desc, 0, _builder)
    semantic.tensormap_replace_swizzle_mode(local_desc, swizzle, _builder)
    semantic.tensormap_replace_fill_mode(local_desc, 0, _builder)

    semantic.tensormap_cp_fenceproxy(desc_out, local_desc, builder=_builder)
    semantic.tensormap_deallocate(local_desc, _builder)


def _determine_swizzle_mode_2d(contig_dim_size_in_bytes, load_size):
    if contig_dim_size_in_bytes >= 128:
        return 3
    elif contig_dim_size_in_bytes >= 64:
        return 2
    elif contig_dim_size_in_bytes >= 32:
        return 1
    else:
        raise ValueError("block size too small")


@core.builtin
def experimental_tensormap_fenceproxy_acquire(desc_ptr: core.tensor, _builder: ir.builder):
    semantic.tensormap_fenceproxy_acquire(desc_ptr, _builder)
