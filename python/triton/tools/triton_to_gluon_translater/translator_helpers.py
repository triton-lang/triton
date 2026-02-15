from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tcgen05_mma,
    tcgen05_mma_scaled,
    tcgen05_commit,
)
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.hopper import tma, fence_async_shared
from triton.experimental.gluon.language.nvidia.blackwell import tma as tma_blackwell


@gluon.constexpr_function
def tl_dot_mma_sync_layout(shape, num_warps):
    rank = len(shape)
    assert rank in [2, 3], "MMA sync only supports 2D shapes or 3D shapes with a batch outer dimension"
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
        result = ttgl.convert_layout(result, acc_init.type.layout)
    return result


@gluon.constexpr_function
def tl_dot_mmav5_supported(a_ty, b_ty, num_warps, input_precision, allow_tf32, max_num_imprecise_acc):
    assert max_num_imprecise_acc is None, "max_num_imprecise_acc only applies to Hopper warp_group_dot"
    assert input_precision is None or allow_tf32 is None, "Only one of input_precision and allow_tf32 can be specified"
    if input_precision is None and (allow_tf32 or allow_tf32 is None):
        input_precision = "tf32"

    M = a_ty.shape[0]
    N = b_ty.shape[1]
    K = a_ty.shape[1]
    min_K = 256 // a_ty.element_ty.primitive_bitwidth
    if a_ty.element_ty.is_int() or b_ty.element_ty.is_int():
        return False
    if min(a_ty.element_ty.primitive_bitwidth, b_ty.element_ty.primitive_bitwidth) >= 32 and input_precision != "tf32":
        return False
    return num_warps in [4, 8] and len(a_ty.shape) == 2 and len(b_ty.shape) == 2 and K >= min_K and M >= 64 and N >= 16


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

    contig_dim_size_in_byte = (shape[0] if transposed else shape[1]) * packing_factor * ele_bit_width // 8
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
    return ttgl.NVMMASharedLayout(swizzle_byte_width=swizzle_byte_width, transposed=transposed,
                                  element_bitwidth=ele_bit_width, rank=len(shape), fp4_padded=is_fp4_padded)


@gluon.jit
def get_shared_memory_mma_operand(value, operand_index, allow_transpose, is_fp4_padded=False, force_transpose=False):
    layout: ttgl.constexpr = get_shared_memory_mma_layout(value.type, operand_index, allow_transpose, is_fp4_padded,
                                                          force_transpose)
    return ttgl.allocate_shared_memory(value.dtype, value.shape, layout, value)


@gluon.jit
def tl_dot_blackwell(a, b, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None,
                     out_dtype=ttgl.float32):
    M: ttgl.constexpr = a.type.shape[0]
    N: ttgl.constexpr = b.type.shape[1]

    allow_transpose = not a.type.element_ty.is_fp32()
    a_smem = get_shared_memory_mma_operand(a, 0, allow_transpose)
    b_smem = get_shared_memory_mma_operand(b, 1, allow_transpose)

    # MMA instruction shape
    m: ttgl.constexpr = 128 if M >= 128 else 64
    n: ttgl.constexpr = 256 if N >= 256 else N

    acc_dtype: ttgl.constexpr = acc.dtype if acc is not None else out_dtype
    col_stride: ttgl.constexpr = 32 // acc_dtype.primitive_bitwidth
    acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout([m, n], col_stride=col_stride)

    tmem_reg_layout: ttgl.constexpr = get_tmem_reg_layout(acc_dtype, (M, N), acc_tmem_layout, ttgl.num_warps())
    if acc is not None:
        acc_temp = ttgl.convert_layout(acc, tmem_reg_layout)
    else:
        acc_temp = ttgl.zeros([M, N], out_dtype, layout=tmem_reg_layout)
    acc_tmem = allocate_tensor_memory(acc_temp.dtype, [M, N], acc_tmem_layout, acc_temp)
    fence_async_shared()
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # Load back from TMEM using a register layout and convert to acc layout
    out = acc_tmem.load(tmem_reg_layout)
    ret_layout: ttgl.constexpr = default_blocked_layout([M, N], ttgl.num_warps())
    out = ttgl.convert_layout(out, ret_layout)
    return out


@gluon.jit
def tl_dot(a, b, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=ttgl.float32):
    num_warps: ttgl.constexpr = ttgl.num_warps()
    if tl_dot_mmav5_supported(a.type, b.type, num_warps, input_precision, allow_tf32, max_num_imprecise_acc):
        return tl_dot_blackwell(a, b, acc, input_precision, allow_tf32, max_num_imprecise_acc, out_dtype)
    else:
        return tl_dot_mma_sync(a, b, acc, input_precision, out_dtype)


@gluon.constexpr_function
def tl_dot_scaled_mmav5_supported(a_ty, b_ty, num_warps):
    M = a_ty.shape[0]
    N = b_ty.shape[1]
    K = a_ty.shape[1]
    min_K = 256 // a_ty.element_ty.primitive_bitwidth
    return num_warps in [4, 8] and len(a_ty.shape) == 2 and len(b_ty.shape) == 2 and K >= min_K and M >= 128 and N >= 16


@gluon.constexpr_function
def get_swizzle_byte_width(bitwidth):
    swizzle = min(bitwidth, 128)
    swizzle = 0 if swizzle < 32 else swizzle
    return swizzle


@gluon.constexpr_function
def get_int_type(bitwidth):
    if bitwidth == 64:
        return ttgl.int64
    elif bitwidth == 32:
        return ttgl.int32
    elif bitwidth == 16:
        return ttgl.int16
    elif bitwidth == 8:
        return ttgl.int8
    else:
        assert False, f"Unsupported bitwidth: {bitwidth}"


@gluon.jit
def tl_dot_decomposed_scale_to_16(scale, compute_type):
    large_fp_type: ttgl.constexpr = ttgl.float32 if compute_type == ttgl.float16 else compute_type
    int_width: ttgl.constexpr = large_fp_type.primitive_bitwidth
    int_type: ttgl.constexpr = get_int_type(int_width)

    zexted = ttgl.cast(scale, int_type)
    shift_value: ttgl.constexpr = large_fp_type.fp_mantissa_width
    shl_res = zexted << shift_value
    scale_fp = ttgl.cast(shl_res, large_fp_type, bitcast=True)
    if large_fp_type != compute_type:
        scale_fp = ttgl.cast(scale_fp, compute_type)
    return scale_fp


@gluon.constexpr_function
def tl_dot_get_expand_dims_layout(scale_ty, num_warps, rank):
    shape = scale_ty.shape.values + [1]
    blocked = default_blocked_layout(shape, num_warps)
    slice = ttgl.SliceLayout(rank, blocked)
    return slice


@gluon.constexpr_function
def tl_dot_get_permute_order(rank, dim):
    order = list(range(rank))
    order.insert(dim + 1, rank)
    return order


@gluon.constexpr_function
def tl_dot_get_reshape_shape(scale_ty, dim):
    shape = list(scale_ty.shape.values)
    shape.pop()
    shape[dim] *= 32
    return shape


@gluon.jit
def tl_dot_decomposed_broadcast_scale(scale, dim):
    scale_ty: ttgl.constexpr = scale.type
    rank: ttgl.constexpr = len(scale_ty.shape)

    num_warps: ttgl.constexpr = ttgl.num_warps()
    slice_enc: ttgl.constexpr = tl_dot_get_expand_dims_layout(scale_ty, num_warps, rank)
    scale = ttgl.convert_layout(scale, slice_enc)
    expand_scale = scale.expand_dims(rank)
    broadcast_scale = expand_scale.broadcast_to(scale.type.shape + (32, ))
    permute_order: ttgl.constexpr = tl_dot_get_permute_order(rank, dim)
    transposed_scale = broadcast_scale.permute(permute_order.value)
    reshape_shape: ttgl.constexpr = tl_dot_get_reshape_shape(broadcast_scale.type, dim)
    return transposed_scale.reshape(reshape_shape)


@gluon.constexpr_function
def tl_dot_decomposed_get_transposed_order(rank):
    assert rank >= 2
    order = list(range(rank - 2))
    order += [rank - 1, rank - 2]
    return order


@gluon.jit
def tl_dot_decomposed_extend_and_broadcast_scale(v, scale, compute_type, operand_index):
    rank: ttgl.constexpr = len(v.type.shape)
    k_dim: ttgl.constexpr = rank - 1 if operand_index == 0 else rank - 2

    if operand_index == 1:
        order: ttgl.constexpr = tl_dot_decomposed_get_transposed_order(rank)
        scale = ttgl.permute(scale, order.value)

    scale16 = tl_dot_decomposed_scale_to_16(scale, compute_type)
    reshape_scale = tl_dot_decomposed_broadcast_scale(scale16, k_dim)
    return ttgl.convert_layout(reshape_scale, v.type.layout), scale


@gluon.jit
def tl_dot_decomposed_mask_nan(mxfp, scale, fast_math):
    ttgl.static_assert(fast_math, "TODO: support non-fast-math")
    return mxfp


@gluon.jit
def tl_dot_decomposed_scale_arg(v, scale, arg_format, operand_index, compute_type, fast_math):
    is_fp4: ttgl.constexpr = arg_format == "e2m1"
    rank: ttgl.constexpr = len(v.type.shape)
    k_dim: ttgl.constexpr = rank - 1 if operand_index == 0 else rank - 2

    if is_fp4:
        v = ttgl.fp4_to_fp(v, compute_type, k_dim)
    else:
        v = ttgl.cast(v, compute_type)
    if scale is None:
        return v
    else:
        reshape_scale, scale = tl_dot_decomposed_extend_and_broadcast_scale(v, scale, compute_type, operand_index)
        mxfp = ttgl.mul(v, reshape_scale)
        return tl_dot_decomposed_mask_nan(mxfp, scale, fast_math)


@gluon.jit
def tl_dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, fast_math=False, lhs_k_pack=True,
                  rhs_k_pack=True, out_dtype=ttgl.float32):
    if tl_dot_scaled_mmav5_supported(lhs.type, rhs.type,
                                     ttgl.num_warps() and lhs_scale is not None and rhs_scale is not None):
        return tl_dot_scaled_blackwell(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, fast_math,
                                       lhs_k_pack, rhs_k_pack, out_dtype)
    else:
        return tl_dot_decomposed_block_scales(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, fast_math,
                                              lhs_k_pack, rhs_k_pack, out_dtype)


@gluon.jit
def tl_dot_decomposed_block_scales(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, fast_math=False,
                                   lhs_k_pack=True, rhs_k_pack=True, out_dtype=ttgl.float32):
    if lhs_scale is None and rhs_scale is not None:
        lhs_trans = tl_trans(lhs)
        rhs_trans = tl_trans(rhs)
        if acc is not None:
            orig_layout: ttgl.constexpr = acc.type.layout
            acc = tl_trans(acc)
        result = tl_dot_scaled(rhs_trans, rhs_scale, rhs_format, lhs_trans, lhs_scale, lhs_format, acc, fast_math,
                               lhs_k_pack, rhs_k_pack, out_dtype)
        result = tl_trans(result)
        if acc is not None:
            result = ttgl.convert_layout(result, orig_layout)
        return result
    else:
        ttgl.static_assert(not (not lhs_k_pack or not rhs_k_pack), "TODO: support m/n packed formats")
        compute_type: ttgl.constexpr = ttgl.float16 if (lhs_format == "fp16" or rhs_format == "fp16") else ttgl.bfloat16

        scale_a = tl_dot_decomposed_scale_arg(lhs, lhs_scale, lhs_format, 0, compute_type, fast_math)
        scale_b = tl_dot_decomposed_scale_arg(rhs, rhs_scale, rhs_format, 1, compute_type, fast_math)

        return tl_dot(scale_a, scale_b, acc, out_dtype=out_dtype)


@gluon.jit
def tl_dot_scaled_blackwell(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, fast_math=False,
                            lhs_k_pack=True, rhs_k_pack=True, out_dtype=ttgl.float32):
    is_a_fp4: ttgl.constexpr = lhs_format == "e2m1"
    is_b_fp4: ttgl.constexpr = rhs_format == "e2m1"

    mixed_prec: ttgl.constexpr = lhs_format != rhs_format
    is_a_mixed_prec_fp4: ttgl.constexpr = mixed_prec and is_a_fp4
    is_b_mixed_prec_fp4: ttgl.constexpr = mixed_prec and not is_a_fp4 and is_b_fp4

    is_mmav5_fp4_padded_a: ttgl.constexpr = is_a_mixed_prec_fp4 or not lhs_k_pack
    is_mmav5_fp4_padded_b: ttgl.constexpr = is_b_mixed_prec_fp4 or not rhs_k_pack

    a_smem = get_shared_memory_mma_operand(lhs, 0, allow_transpose=not is_a_fp4, is_fp4_padded=is_mmav5_fp4_padded_a,
                                           force_transpose=not lhs_k_pack)
    b_smem = get_shared_memory_mma_operand(rhs, 1, allow_transpose=not is_b_fp4, is_fp4_padded=is_mmav5_fp4_padded_b,
                                           force_transpose=not rhs_k_pack)

    M: ttgl.constexpr = lhs.type.shape[0]
    N: ttgl.constexpr = rhs.type.shape[1]

    m: ttgl.constexpr = 128
    n: ttgl.constexpr = 256 if N >= 256 else N

    acc_dtype: ttgl.constexpr = acc.dtype if acc is not None else out_dtype
    col_stride: ttgl.constexpr = 32 // acc_dtype.primitive_bitwidth
    acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout([m, n], col_stride=col_stride)
    tmem_reg_layout: ttgl.constexpr = get_tmem_reg_layout(acc_dtype, (M, N), acc_tmem_layout, ttgl.num_warps())
    if acc is not None:
        acc_temp = ttgl.convert_layout(acc, tmem_reg_layout)
    else:
        acc_temp = ttgl.zeros([M, N], out_dtype, layout=tmem_reg_layout)
    acc_tmem = allocate_tensor_memory(acc_temp.dtype, [M, N], acc_tmem_layout, acc_temp)
    fence_async_shared()

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    scale_layout_reg_lhs: ttgl.constexpr = get_tmem_reg_layout(lhs_scale.dtype, lhs_scale.type.shape, scale_layout,
                                                               ttgl.num_warps())
    scale_layout_reg_rhs: ttgl.constexpr = get_tmem_reg_layout(rhs_scale.dtype, rhs_scale.type.shape, scale_layout,
                                                               ttgl.num_warps())
    lhs_scale = ttgl.convert_layout(lhs_scale, scale_layout_reg_lhs)
    rhs_scale = ttgl.convert_layout(rhs_scale, scale_layout_reg_rhs)
    a_scale_tmem = allocate_tensor_memory(lhs_scale.dtype, lhs_scale.shape, scale_layout, lhs_scale)
    b_scale_tmem = allocate_tensor_memory(rhs_scale.dtype, rhs_scale.shape, scale_layout, rhs_scale)

    tcgen05_mma_scaled(a_smem, b_smem, acc_tmem, a_scale_tmem, b_scale_tmem, lhs_format, rhs_format, use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    # Load back from TMEM using a register layout and convert to acc layout
    out = acc_tmem.load(tmem_reg_layout)
    ret_layout: ttgl.constexpr = default_blocked_layout([M, N], ttgl.num_warps())
    out = ttgl.convert_layout(out, ret_layout)
    return out


@gluon.constexpr_function
def get_num_threads_per_warp() -> ttgl.constexpr:
    return ttgl.constexpr(32)


@ttgl._core.builtin
def get_num_threads_per_program(_semantic=None, _generator=None):
    return ttgl.num_warps(_semantic=_semantic, _generator=_generator) * get_num_threads_per_warp(_semantic=_semantic)


@gluon.constexpr_function
def default_blocked_layout(shape: ttgl.constexpr, num_warps: ttgl.constexpr) -> ttgl.constexpr:
    rank = len(shape)
    # 1 element per thread for all dimensions
    size_per_thread = [1 for _ in range(rank)]
    # Distribute 32 threads per warp across dimensions (simple heuristic: last-fastest)
    threads_per_warp = [1 for _ in range(rank)]
    # TODO: pick a better layout based on shape. Using this allows to not have to convert layout when broadcasting but may blow up register pressure.
    threads_per_warp[rank - 1] = get_num_threads_per_warp()
    # remaining_threads = get_num_threads_per_warp()
    # for dim in range(rank - 1, -1, -1):
    #     threads_per_warp[dim] = min(remaining_threads, shape[dim])
    #     remaining_threads = remaining_threads // threads_per_warp[dim]
    # Use provided num_warps to distribute warps per CTA (put all on first dim)
    warps_per_cta = [1 for _ in range(rank)]
    warps_per_cta[0] = num_warps
    # Natural order [rank-1, rank-2, ..., 0]
    order = [i for i in range(rank - 1, -1, -1)]
    return ttgl.BlockedLayout(size_per_thread=size_per_thread, threads_per_warp=threads_per_warp,
                              warps_per_cta=warps_per_cta, order=order)


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


@gluon.jit
def tl_obj_gather(obj, x_offsets, y_offset):
    if isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor):
        desc = obj
        desc_shape: ttgl.constexpr = [x_offsets.shape[0], desc.block_shape[1]]
        alloc = ttgl.allocate_shared_memory(desc.dtype, desc_shape, desc.layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)
        x_offsets_layout: ttgl.constexpr = ttgl.SliceLayout(
            0, ttgl.BlockedLayout([1, 4], [get_num_threads_per_warp(), 1], [1, ttgl.num_warps()], [1, 0]))
        x_offsets = ttgl.convert_layout(x_offsets, x_offsets_layout)
        mbarrier.expect(bar, x_offsets.shape[0] * obj.block_type.nbytes)
        tma_blackwell.async_gather(desc, x_offsets, y_offset, bar, alloc)
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)
        # Load from shared memory into a register tensor using a reasonable default layout
        ret_layout: ttgl.constexpr = default_blocked_layout(desc.block_shape, ttgl.num_warps())
        out = alloc.load(ret_layout)
        return out
    else:
        return obj.gather(x_offsets, y_offset)


@gluon.jit
def tl_obj_scatter(obj, value, x_offsets, y_offset):
    if isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor):
        desc = obj
        desc_shape: ttgl.constexpr = [x_offsets.shape[0], desc.block_shape[1]]
        alloc = ttgl.allocate_shared_memory(desc.dtype, desc_shape, desc.layout, value)
        fence_async_shared()
        x_offsets_layout: ttgl.constexpr = ttgl.SliceLayout(
            0, ttgl.BlockedLayout([1, 4], [get_num_threads_per_warp(), 1], [1, ttgl.num_warps()], [1, 0]))
        x_offsets = ttgl.convert_layout(x_offsets, x_offsets_layout)
        tma_blackwell.async_scatter(desc, x_offsets, y_offset, alloc)
        tma.store_wait(0)
    else:
        obj.scatter(value, x_offsets, y_offset)


@ttgl._core.builtin
def tl_make_tensor_descriptor(base, shape, strides, block_shape, padding_option="zero", _semantic=None):
    layout = ttgl.NVMMASharedLayout.get_default_for(block_shape, base.dtype.element_ty)
    return tma.make_tensor_descriptor(base, shape, strides, block_shape, layout, padding_option, _semantic=_semantic)


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
    # Issue async copy from global (descriptor) to shared memory and wait for completion
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_copy_global_to_shared(desc, offsets, bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    # Load from shared memory into a register tensor using a reasonable default layout
    ret_layout: ttgl.constexpr = default_blocked_layout(desc.block_shape, ttgl.num_warps())
    out = smem.load(ret_layout)
    return out


@gluon.jit
def tl_arange(start: ttgl.constexpr, stop: ttgl.constexpr = None):
    layout: ttgl.constexpr = default_blocked_layout([stop - start], ttgl.num_warps())
    return ttgl.arange(start, stop, layout=layout)


@gluon.jit
def tl_full(shape, value, dtype=None):
    layout: ttgl.constexpr = default_blocked_layout(shape, ttgl.num_warps())
    return ttgl.full(shape, value, dtype, layout=layout)


@ttgl._core.builtin
def tl_trans(value, *dims, _semantic=None):
    return value.trans(*dims, _semantic=_semantic)


@ttgl._core.builtin
def cat(input, other, can_reorder=False, layout=None, _semantic=None):
    """
    Concatenate the two tensors.

    Args:
        input (tensor): The first input tensor.
        other (tensor): The second input tensor.
        can_reorder (bool): Compiler hint. If true, the compiler is allowed to reorder elements while concatenating inputs.  Only use if the order does not matter (e.g., result is only used in reduction ops).  Current implementation of `cat` supports only can_reorder=True.
        layout (DistributedLayout): The destination layout of the output tensor.

    Returns:
        tensor: The concatenated tensor.
    """
    can_reorder = ttgl._core._unwrap_if_constexpr(can_reorder)
    layout = ttgl._core._unwrap_if_constexpr(layout)
    return _semantic.cat(input, other, can_reorder, layout)


@gluon.jit
def tl_cat(lhs, rhs, can_reorder=False):
    return cat(lhs, rhs, can_reorder, layout=default_blocked_layout([lhs.shape[0] + rhs.shape[0]], ttgl.num_warps()))


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
    # Distribute 32 threads per warp across dimensions (simple heuristic: last-fastest)
    threads_per_warp = [1 for _ in range(rank)]
    remaining_threads = get_num_threads_per_warp()
    for dim in range(rank - 2, -1, -1):
        threads_per_warp[dim] = min(shape[dim], remaining_threads)
        remaining_threads = remaining_threads // threads_per_warp[dim]
    # Use provided num_warps to distribute warps per CTA (put all on first dim)
    warps_per_cta = [1 for _ in range(rank)]
    warps_per_cta[0] = num_warps
    # Natural order [rank-1, rank-2, ..., 0]
    order = [i for i in range(rank - 1, -1, -1)]
    return ttgl.BlockedLayout(size_per_thread=size_per_thread, threads_per_warp=threads_per_warp,
                              warps_per_cta=warps_per_cta, order=order)


@gluon.jit
def set_split_src_layout(value):
    layout: ttgl.constexpr = get_split_src_layout(value.type.shape, ttgl.num_warps())
    return ttgl.convert_layout(value, layout=layout)


def convert_host_descriptor(desc):

    def torch_dtype_to_triton(dtype):
        import torch
        if dtype == torch.float8_e5m2:
            return ttgl.float8e5
        if dtype == torch.float8_e4m3fn:
            return ttgl.float8e4nv
        return getattr(ttgl, str(dtype).split('.')[1])

    from triton.tools.tensor_descriptor import TensorDescriptor
    assert isinstance(desc, TensorDescriptor)
    block_shape = desc.block_shape
    dtype = desc.base.dtype
    tensor = desc.base
    layout = ttgl.NVMMASharedLayout.get_default_for(block_shape, torch_dtype_to_triton(dtype))
    return gluon.nvidia.hopper.TensorDescriptor(tensor, desc.shape, desc.strides, block_shape, layout)


# hacks to workaround limited dependencies tracking.
# TODO: fix this by pulling imports into the generated file.
def current_target():
    from triton.runtime import driver
    try:
        active_driver = driver.active
    except RuntimeError:
        # If there is no active driver, return None
        return None
    return active_driver.get_current_target()


current_target.__triton_builtin__ = True
