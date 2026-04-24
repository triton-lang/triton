# type: ignore

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma

from triton.tools.triton_to_gluon_translator.common_helpers import *  # noqa: F401,F403
from triton.tools.triton_to_gluon_translator.common_helpers import (
    default_blocked_layout, )

# ---- NVIDIA MMA sync (Ampere) ----


@gluon.constexpr_function
def tl_dot_mma_sync_layout(shape, num_warps):
    rank = len(shape)
    assert rank in [
        2,
        3,
    ], "MMA sync only supports 2D shapes or 3D shapes with a batch outer dimension"
    if rank == 2:
        return ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, 8])
    return ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[num_warps, 1, 1], instr_shape=[1, 16, 8])


@gluon.constexpr_function
def tl_dot_mma_sync_k_width(a_ty, b_ty):
    a_bitwidth = a_ty.element_ty.primitive_bitwidth
    b_bitwidth = b_ty.element_ty.primitive_bitwidth
    min_bitwidth = min(a_bitwidth, b_bitwidth)
    return max(32 // min_bitwidth, 1)


@gluon.jit
def tl_dot_mma_sync(a, b, acc_init=None, input_precision=None, out_dtype=ttgl.float32):
    mma_layout: ttgl.constexpr = tl_dot_mma_sync_layout(a.type.shape, ttgl.num_warps())
    k_width: ttgl.constexpr = tl_dot_mma_sync_k_width(a.type, b.type)
    a_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=mma_layout, operand_index=0, k_width=k_width)
    b_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=mma_layout, operand_index=1, k_width=k_width)
    a = ttgl.convert_layout(a, a_layout)
    b = ttgl.convert_layout(b, b_layout)
    if acc_init is not None:
        acc = ttgl.convert_layout(acc_init, mma_layout)
    else:
        acc = ttgl.full([a.shape[0], a.shape[1], b.shape[2]], 0.0, out_dtype, layout=mma_layout)
    result = mma_v2(a, b, acc, input_precision)
    if acc_init is not None:
        layout: ttgl.constexpr = acc_init.type.layout
    else:
        layout: ttgl.constexpr = default_blocked_layout(result.type.shape, ttgl.num_warps())
    result = ttgl.convert_layout(result, layout)
    return result


@gluon.constexpr_function
def get_shared_memory_mma_layout(type, operand_index, allow_transpose, is_fp4_padded=False, force_transpose=False):
    if not allow_transpose:
        if operand_index == 1:
            transposed = True
        else:
            transposed = False
        if force_transpose:
            transposed = not transposed
    else:
        transposed = operand_index == 1

    shape = type.shape
    swizzle_byte_width = 0
    ele_bit_width = type.element_ty.primitive_bitwidth
    packing_factor = 2 if is_fp4_padded else 1

    contig_dim_size_in_byte = ((shape[0] if transposed else shape[1]) * packing_factor * ele_bit_width // 8)
    if contig_dim_size_in_byte >= 128 and contig_dim_size_in_byte % 128 == 0:
        swizzle_byte_width = 128
    elif contig_dim_size_in_byte >= 64 and contig_dim_size_in_byte % 64 == 0:
        swizzle_byte_width = 64
    elif contig_dim_size_in_byte >= 32 and contig_dim_size_in_byte % 32 == 0:
        swizzle_byte_width = 32
    else:
        swizzle_byte_width = 0

    flatten_outer_dim = 1
    for dim in shape:
        flatten_outer_dim *= dim
    if len(shape) < 2 or flatten_outer_dim < 8:
        swizzle_byte_width = 0
    return ttgl.NVMMASharedLayout(
        swizzle_byte_width=swizzle_byte_width,
        transposed=transposed,
        element_bitwidth=ele_bit_width,
        rank=len(shape),
        fp4_padded=is_fp4_padded,
    )


@gluon.jit
def get_shared_memory_mma_operand(value, operand_index, allow_transpose, is_fp4_padded=False, force_transpose=False):
    layout: ttgl.constexpr = get_shared_memory_mma_layout(value.type, operand_index, allow_transpose, is_fp4_padded,
                                                          force_transpose)
    return ttgl.allocate_shared_memory(value.dtype, value.shape, layout, value)


# ---- NVIDIA TMA tensor descriptors ----


@gluon.jit
def tl_make_tensor_descriptor(base, shape, strides, block_shape, padding_option: ttgl.constexpr = "zero"):
    layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for(block_shape, base.dtype.element_ty)
    return tma.make_tensor_descriptor(base, shape, strides, block_shape, layout, padding_option)


@gluon.jit
def tl_store_tensor_descriptor(desc, offsets, value):
    alloc = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout, value)
    fence_async_shared()
    tma.async_copy_shared_to_global(desc, offsets, alloc)
    tma.store_wait(0)
    alloc._keep_alive()


@gluon.jit
def tl_load_tensor_descriptor(desc, offsets):
    smem = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_load(desc, offsets, bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    ret_layout: ttgl.constexpr = default_blocked_layout(desc.block_shape, ttgl.num_warps())
    out = smem.load(ret_layout)
    return out


# ---- NVIDIA obj dispatch ----


@gluon.jit
def tl_obj_store(obj, offsets, value):
    if isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor):
        return tl_store_tensor_descriptor(obj, offsets, value)
    else:
        return obj.store(offsets, value)


@gluon.jit
def tl_obj_load(obj, offsets):
    if isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor):
        return tl_load_tensor_descriptor(obj, offsets)
    else:
        return obj.load(offsets)


# ---- NVIDIA host-side descriptor ----


def convert_host_descriptor(desc):

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
    tensor = desc.base

    layout = ttgl.NVMMASharedLayout.get_default_for(block_shape, torch_dtype_to_triton(dtype))
    return gluon.nvidia.hopper.TensorDescriptor(tensor, desc.shape, desc.strides, block_shape, layout)
