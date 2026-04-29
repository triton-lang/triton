# type: ignore

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_mma,
    tcgen05_mma_scaled,
)
from triton.experimental.gluon.language.nvidia.blackwell import tma as tma_blackwell

from triton.tools.triton_to_gluon_translator.common_helpers import *  # noqa: F401,F403
from triton.tools.triton_to_gluon_translator.common_helpers import (
    default_blocked_layout,
    get_num_threads_per_warp,
    tl_dot_decomposed_block_scales_impl,
)
from triton.tools.triton_to_gluon_translator.nvidia_helpers import *  # noqa: F401,F403
from triton.tools.triton_to_gluon_translator.nvidia_helpers import (
    tl_dot_mma_sync,
    get_shared_memory_mma_operand,
)

# ---- NVIDIA Blackwell dot ----


@gluon.constexpr_function
def tl_dot_mmav5_supported(a_ty, b_ty, num_warps, input_precision, allow_tf32, max_num_imprecise_acc):
    assert max_num_imprecise_acc in [0, None], ("max_num_imprecise_acc only applies to Hopper warp_group_dot")
    assert input_precision is None or allow_tf32 is None, (
        "Only one of input_precision and allow_tf32 can be specified")
    if input_precision is None and (allow_tf32 or allow_tf32 is None):
        input_precision = "tf32"

    M = a_ty.shape[0]
    N = b_ty.shape[1]
    K = a_ty.shape[1]
    min_K = 256 // a_ty.element_ty.primitive_bitwidth
    if a_ty.element_ty.is_int() or b_ty.element_ty.is_int():
        return False
    if (min(a_ty.element_ty.primitive_bitwidth, b_ty.element_ty.primitive_bitwidth) >= 32
            and input_precision != "tf32"):
        return False
    return (num_warps in [4, 8] and len(a_ty.shape) == 2 and len(b_ty.shape) == 2 and K >= min_K and M >= 64
            and N >= 16)


@gluon.jit
def tl_dot_blackwell(
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

    allow_transpose = not a.type.element_ty.is_fp32()
    a_smem = get_shared_memory_mma_operand(a, 0, allow_transpose)
    b_smem = get_shared_memory_mma_operand(b, 1, allow_transpose)

    m: ttgl.constexpr = 128 if M >= 128 else 64
    n: ttgl.constexpr = 256 if N >= 256 else N

    acc_dtype: ttgl.constexpr = acc.dtype if acc is not None else out_dtype
    col_stride: ttgl.constexpr = 32 // acc_dtype.primitive_bitwidth
    acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout([m, n], col_stride=col_stride)
    acc_tmem = allocate_tensor_memory(acc_dtype, [M, N], acc_tmem_layout)
    tmem_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    if acc is not None:
        acc_temp = ttgl.convert_layout(acc, tmem_reg_layout)
    else:
        acc_temp = ttgl.zeros([M, N], out_dtype, layout=tmem_reg_layout)
    acc_tmem.store(acc_temp)
    fence_async_shared()
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out = acc_tmem.load()
    ret_layout: ttgl.constexpr = default_blocked_layout([M, N], ttgl.num_warps())
    out = ttgl.convert_layout(out, ret_layout)
    return out


# ---- NVIDIA dot dispatch ----


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
    num_warps: ttgl.constexpr = ttgl.num_warps()
    if tl_dot_mmav5_supported(a.type, b.type, num_warps, input_precision, allow_tf32, max_num_imprecise_acc):
        return tl_dot_blackwell(a, b, acc, input_precision, allow_tf32, max_num_imprecise_acc, out_dtype)
    else:
        return tl_dot_mma_sync(a, b, acc, input_precision, out_dtype)


# ---- NVIDIA dot-scaled ----


@gluon.constexpr_function
def tl_dot_scaled_mmav5_supported(a_ty, b_ty, num_warps):
    M = a_ty.shape[0]
    N = b_ty.shape[1]
    K = a_ty.shape[1]
    min_K = 256 // a_ty.element_ty.primitive_bitwidth
    return (num_warps in [4, 8] and len(a_ty.shape) == 2 and len(b_ty.shape) == 2 and K >= min_K and M >= 128
            and N >= 16)


@gluon.jit
def tl_dot_scaled_blackwell(
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
    is_a_fp4: ttgl.constexpr = lhs_format == "e2m1"
    is_b_fp4: ttgl.constexpr = rhs_format == "e2m1"

    mixed_prec: ttgl.constexpr = lhs_format != rhs_format
    is_a_mixed_prec_fp4: ttgl.constexpr = mixed_prec and is_a_fp4
    is_b_mixed_prec_fp4: ttgl.constexpr = mixed_prec and not is_a_fp4 and is_b_fp4

    is_mmav5_fp4_padded_a: ttgl.constexpr = is_a_mixed_prec_fp4 or not lhs_k_pack
    is_mmav5_fp4_padded_b: ttgl.constexpr = is_b_mixed_prec_fp4 or not rhs_k_pack

    a_smem = get_shared_memory_mma_operand(
        lhs,
        0,
        allow_transpose=not is_a_fp4,
        is_fp4_padded=is_mmav5_fp4_padded_a,
        force_transpose=not lhs_k_pack,
    )
    b_smem = get_shared_memory_mma_operand(
        rhs,
        1,
        allow_transpose=not is_b_fp4,
        is_fp4_padded=is_mmav5_fp4_padded_b,
        force_transpose=not rhs_k_pack,
    )

    M: ttgl.constexpr = lhs.type.shape[0]
    N: ttgl.constexpr = rhs.type.shape[1]

    m: ttgl.constexpr = 128
    n: ttgl.constexpr = 256 if N >= 256 else N

    acc_dtype: ttgl.constexpr = acc.dtype if acc is not None else out_dtype
    col_stride: ttgl.constexpr = 32 // acc_dtype.primitive_bitwidth
    acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout([m, n], col_stride=col_stride)
    acc_tmem = allocate_tensor_memory(acc_dtype, [M, N], acc_tmem_layout)
    tmem_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    if acc is not None:
        acc_temp = ttgl.convert_layout(acc, tmem_reg_layout)
    else:
        acc_temp = ttgl.zeros([M, N], out_dtype, layout=tmem_reg_layout)
    acc_tmem.store(acc_temp)
    fence_async_shared()

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(lhs_scale.dtype, lhs_scale.shape, scale_layout)
    b_scale_tmem = allocate_tensor_memory(rhs_scale.dtype, rhs_scale.shape, scale_layout)
    scale_layout_reg_lhs: ttgl.constexpr = a_scale_tmem.get_reg_layout()
    scale_layout_reg_rhs: ttgl.constexpr = b_scale_tmem.get_reg_layout()
    lhs_scale = ttgl.convert_layout(lhs_scale, scale_layout_reg_lhs)
    rhs_scale = ttgl.convert_layout(rhs_scale, scale_layout_reg_rhs)
    a_scale_tmem.store(lhs_scale)
    b_scale_tmem.store(rhs_scale)

    tcgen05_mma_scaled(a_smem, b_smem, acc_tmem, a_scale_tmem, b_scale_tmem, lhs_format, rhs_format, use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    out = acc_tmem.load()
    ret_layout: ttgl.constexpr = default_blocked_layout([M, N], ttgl.num_warps())
    out = ttgl.convert_layout(out, ret_layout)
    return out


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
    return tl_dot_decomposed_block_scales_impl(
        tl_dot_scaled,
        tl_dot,
        lhs,
        lhs_scale,
        lhs_format,
        rhs,
        rhs_scale,
        rhs_format,
        acc=acc,
        fast_math=fast_math,
        lhs_k_pack=lhs_k_pack,
        rhs_k_pack=rhs_k_pack,
        out_dtype=out_dtype,
    )


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
    if (tl_dot_scaled_mmav5_supported(lhs.type, rhs.type, ttgl.num_warps()) and lhs_scale is not None
            and rhs_scale is not None):
        return tl_dot_scaled_blackwell(
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
    else:
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


# ---- NVIDIA TMA tensor descriptors ----


@gluon.jit
def tl_gather_tensor_descriptor(desc, x_offsets, y_offset):
    desc_shape: ttgl.constexpr = [x_offsets.shape[0], desc.block_shape[1]]
    alloc = ttgl.allocate_shared_memory(desc.dtype, desc_shape, desc.layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    x_offsets_layout: ttgl.constexpr = ttgl.SliceLayout(
        0,
        ttgl.BlockedLayout([1, 4], [get_num_threads_per_warp(), 1], [1, ttgl.num_warps()], [1, 0]),
    )
    x_offsets = ttgl.convert_layout(x_offsets, x_offsets_layout)
    mbarrier.expect(bar, x_offsets.shape[0] * desc.block_type.nbytes)
    tma_blackwell.async_gather(desc, x_offsets, y_offset, bar, alloc)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    ret_layout: ttgl.constexpr = default_blocked_layout(desc.block_shape, ttgl.num_warps())
    out = alloc.load(ret_layout)
    return out


@gluon.jit
def tl_scatter_tensor_descriptor(desc, value, x_offsets, y_offset):
    desc_shape: ttgl.constexpr = [x_offsets.shape[0], desc.block_shape[1]]
    alloc = ttgl.allocate_shared_memory(desc.dtype, desc_shape, desc.layout, value)
    fence_async_shared()
    x_offsets_layout: ttgl.constexpr = ttgl.SliceLayout(
        0,
        ttgl.BlockedLayout([1, 4], [get_num_threads_per_warp(), 1], [1, ttgl.num_warps()], [1, 0]),
    )
    x_offsets = ttgl.convert_layout(x_offsets, x_offsets_layout)
    tma_blackwell.async_scatter(desc, x_offsets, y_offset, alloc)
    tma.store_wait(0)


# ---- NVIDIA obj dispatch ----


@gluon.jit
def tl_obj_gather(obj, x_offsets, y_offset):
    if isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor):
        return tl_gather_tensor_descriptor(obj, x_offsets, y_offset)
    else:
        return obj.gather(x_offsets, y_offset)


@gluon.jit
def tl_obj_scatter(obj, value, x_offsets, y_offset):
    if isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor):
        return tl_scatter_tensor_descriptor(obj, value, x_offsets, y_offset)
    else:
        return obj.scatter(value, x_offsets, y_offset)
