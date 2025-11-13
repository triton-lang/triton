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


@gluon.jit
def tl_dot_mma_sync(a, b, acc_init=None, input_precision=None):
    mma_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(
        version=[2, 0],
        warps_per_cta=[ttgl.num_warps(), 1],
        instr_shape=[16, 8],
    )
    a_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=mma_layout, operand_index=0, k_width=2)
    b_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=mma_layout, operand_index=1, k_width=2)
    a = ttgl.convert_layout(a, a_layout)
    b = ttgl.convert_layout(b, b_layout)
    if acc_init is not None:
        acc = ttgl.convert_layout(acc_init, mma_layout)
    else:
        acc = ttgl.full([a.shape[0], a.shape[1], b.shape[2]], 0.0, ttgl.float32, layout=mma_layout)
    result = mma_v2(a, b, acc, input_precision)
    if acc is not None:
        result = ttgl.convert_layout(result, acc_init.type.layout)
    return result


@gluon.constexpr_function
def get_swizzle_byte_width(bitwidth):
    swizzle = min(bitwidth, 128)
    swizzle = 0 if swizzle < 32 else swizzle
    return swizzle


@gluon.jit
def tl_dot_blackwell(a, b, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None,
                     out_dtype=ttgl.float32):
    M: ttgl.constexpr = a.type.shape[0]
    N: ttgl.constexpr = b.type.shape[1]
    K: ttgl.constexpr = a.type.shape[1]
    ttgl.static_assert(M >= 64 and N >= 16 and K >= 16, "TODO: support smaller shapes using mmav2")
    # Shared memory layouts for inputs (simple default)
    swizzle_byte_with_a: ttgl.constexpr = get_swizzle_byte_width(a.dtype.primitive_bitwidth * K)
    swizzle_byte_with_b: ttgl.constexpr = get_swizzle_byte_width(b.dtype.primitive_bitwidth * N)
    nvmma_layout_a: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=swizzle_byte_with_a, transposed=False,
                                                            element_bitwidth=a.dtype.primitive_bitwidth, rank=2)
    nvmma_layout_b: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=swizzle_byte_with_b, transposed=True,
                                                            element_bitwidth=a.dtype.primitive_bitwidth, rank=2)
    a_smem = ttgl.allocate_shared_memory(a.dtype, [M, K], nvmma_layout_a, a)
    b_smem = ttgl.allocate_shared_memory(b.dtype, [K, N], nvmma_layout_b, b)
    acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout([M, N], col_stride=1)
    acc_dtype: ttgl.constexpr = acc.dtype if acc is not None else out_dtype
    tmem_reg_layout: ttgl.constexpr = get_tmem_reg_layout(acc_dtype, (M, N), acc_tmem_layout, ttgl.num_warps())
    if acc is not None:
        acc_temp = ttgl.convert_layout(acc, tmem_reg_layout)
    else:
        acc_temp = ttgl.zeros([M, N], out_dtype, layout=tmem_reg_layout)
    acc_tmem = allocate_tensor_memory(acc_temp.dtype, [M, N], acc_tmem_layout, acc_temp)
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
    if ttgl.num_warps() < 4:
        return tl_dot_mma_sync(a, b, acc, input_precision)
    else:
        return tl_dot_blackwell(a, b, acc, input_precision, allow_tf32, max_num_imprecise_acc, out_dtype)


@gluon.constexpr_function
def _constexpr_min(a, b):
    return min(a, b)


@gluon.jit
def tl_dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, fast_math=False, lhs_k_pack=True,
                  rhs_k_pack=True, out_dtype=ttgl.float32):
    # TODO: check if MMAv5_scaled cannot be used and fallback to mmav5/mmav3 or mmav2
    M: ttgl.constexpr = lhs_scale.shape[0]
    N: ttgl.constexpr = rhs_scale.shape[0]
    K: ttgl.constexpr = lhs_scale.shape[1] * 32
    ttgl.static_assert(M >= 128 and N >= 16 and K >= 16, "TODO: support smaller shapes using mmav2")
    fp4_padded_a: ttgl.constexpr = lhs_format == "e2m1" and rhs_format != "e2m1"
    fp4_padded_b: ttgl.constexpr = lhs_format != "e2m1" and rhs_format == "e2m1"
    swizzle_byte_with_a: ttgl.constexpr = get_swizzle_byte_width(lhs.dtype.primitive_bitwidth * K)
    swizzle_byte_with_b: ttgl.constexpr = get_swizzle_byte_width(rhs.dtype.primitive_bitwidth * N)
    nvmma_layout_a: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=swizzle_byte_with_a, transposed=False,
                                                            element_bitwidth=8, rank=2, fp4_padded=fp4_padded_a)
    nvmma_layout_b: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=swizzle_byte_with_b, transposed=True,
                                                            element_bitwidth=8, rank=2, fp4_padded=fp4_padded_b)

    a_smem = ttgl.allocate_shared_memory(lhs.dtype, lhs.shape, nvmma_layout_a, lhs)
    b_smem = ttgl.allocate_shared_memory(rhs.dtype, rhs.shape, nvmma_layout_b, rhs)
    acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout([M, N], col_stride=1)
    acc_dtype: ttgl.constexpr = acc.dtype if acc is not None else out_dtype
    tmem_reg_layout: ttgl.constexpr = get_tmem_reg_layout(acc_dtype, (M, N), acc_tmem_layout, ttgl.num_warps())
    if acc is not None:
        acc_temp = ttgl.convert_layout(acc, tmem_reg_layout)
    else:
        acc_temp = ttgl.zeros([M, N], out_dtype, layout=tmem_reg_layout)
    acc_tmem = allocate_tensor_memory(acc_temp.dtype, [M, N], acc_tmem_layout, acc_temp)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    scale_layout_reg_lhs: ttgl.constexpr = get_tmem_reg_layout(lhs_scale.type.element_ty, lhs_scale.type.shape,
                                                               scale_layout, ttgl.num_warps())
    scale_layout_reg_rhs: ttgl.constexpr = get_tmem_reg_layout(rhs_scale.type.element_ty, rhs_scale.type.shape,
                                                               scale_layout, ttgl.num_warps())
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
