# type: ignore

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

# hack to workaround limited dependencies tracking.
# TODO: fix this by pulling imports into the generated file.
from triton.language.target_info import current_target  # noqa: F401

# ---- layout utilities ----


@gluon.constexpr_function
def get_num_threads_per_warp(target=None) -> ttgl.constexpr:
    if target is None:
        target = current_target()
    if target is not None and target.backend == "hip":
        gfx_major = int(target.arch[3:-2])
        return ttgl.constexpr(32 if gfx_major >= 10 else 64)
    return ttgl.constexpr(32)


@gluon.jit
def get_num_threads_per_program():
    return ttgl.num_warps() * get_num_threads_per_warp(current_target())


@gluon.constexpr_function
def default_blocked_layout(shape: ttgl.constexpr, num_warps: ttgl.constexpr, target=None) -> ttgl.constexpr:
    rank = len(shape)
    size_per_thread = [1] * rank
    threads_per_warp = [1] * rank
    # TODO: pick a better layout based on shape. Using this allows to not have to convert layout when broadcasting but may blow up register pressure.
    threads_per_warp[rank - 1] = get_num_threads_per_warp(target)
    warps_per_cta = [1] * rank
    warps_per_cta[0] = num_warps
    order = list(range(rank - 1, -1, -1))
    return ttgl.BlockedLayout(
        size_per_thread=size_per_thread,
        threads_per_warp=threads_per_warp,
        warps_per_cta=warps_per_cta,
        order=order,
    )


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


# ---- portable ops ----


@gluon.jit
def tl_arange(start: ttgl.constexpr, stop: ttgl.constexpr = None):
    layout: ttgl.constexpr = default_blocked_layout([stop - start], ttgl.num_warps())
    return ttgl.arange(start, stop, layout=layout)


@gluon.jit
def tl_full(shape, value, dtype=None):
    layout: ttgl.constexpr = default_blocked_layout(shape, ttgl.num_warps())
    return ttgl.full(shape, value, dtype, layout=layout)


@gluon.jit
def tl_trans(value, *dims):
    return value.trans(*dims)


@gluon.constexpr_function
def cat_permute_order(rank, dim):
    order = list(range(rank))
    order.insert(dim, rank)
    return order


@gluon.constexpr_function
def cat_result_shape(input_shape, dim):
    result_shape = list(input_shape)
    result_shape[dim] *= 2
    return result_shape


@gluon.jit
def tl_cat(input, other, can_reorder=False, dim=0):
    c = ttgl.join(input, other)
    order: ttgl.constexpr = cat_permute_order(len(input.shape), dim)
    c = ttgl.permute(c, order)
    shape: ttgl.constexpr = cat_result_shape(input.shape, dim)
    c = ttgl.reshape(c, shape)
    return reset_to_default_layout(c)


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
def get_split_src_layout(shape: ttgl.constexpr, num_warps: ttgl.constexpr, target=None) -> ttgl.constexpr:
    rank = len(shape)
    size_per_thread = [1 if i != rank - 1 else 2 for i in range(rank)]
    threads_per_warp = [1 for _ in range(rank)]
    remaining_threads = get_num_threads_per_warp(target)
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


@gluon.constexpr_function
def build_expand_dims_layout(shape, expand_dims, num_warps):
    if isinstance(shape, ttgl.tuple):
        shape = shape.values
    assert isinstance(shape, list), (f"expected shape to be a list, got {shape} which is {type(shape)}")
    parent_shape = list(shape)
    for dim in expand_dims:
        parent_shape.insert(dim, 1)
    layout = default_blocked_layout(parent_shape, num_warps)
    for dim in reversed(expand_dims):
        layout = ttgl.SliceLayout(dim=dim, parent=layout)
    return layout


@gluon.jit
def convert_to_expand_dims_layout(value, expand_dims: list[int]):
    layout: ttgl.constexpr = build_expand_dims_layout(value.shape, expand_dims, ttgl.num_warps())
    return ttgl.convert_layout(value, layout)


# ---- dot-scaled sub-helpers (vendor-neutral) ----


@gluon.jit
def tl_dot_decomposed_scale_to_16(scale, compute_type: ttgl.constexpr):
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
def tl_dot_get_reshape_shape(scale_ty, dim, scale_factor):
    shape = list(scale_ty.shape.values)
    shape.pop()
    shape[dim] *= scale_factor
    return shape


@gluon.jit
def tl_dot_decomposed_broadcast_scale(scale, dim, scale_factor: ttgl.constexpr):
    scale_ty: ttgl.constexpr = scale.type
    rank: ttgl.constexpr = len(scale_ty.shape)

    num_warps: ttgl.constexpr = ttgl.num_warps()
    slice_enc: ttgl.constexpr = tl_dot_get_expand_dims_layout(scale_ty, num_warps, rank)
    scale = ttgl.convert_layout(scale, slice_enc)
    expand_scale = scale.expand_dims(rank)
    broadcast_scale = expand_scale.broadcast_to(scale.type.shape + (scale_factor, ))
    permute_order: ttgl.constexpr = tl_dot_get_permute_order(rank, dim)
    transposed_scale = broadcast_scale.permute(permute_order)
    reshape_shape: ttgl.constexpr = tl_dot_get_reshape_shape(broadcast_scale.type, dim, scale_factor)
    return transposed_scale.reshape(reshape_shape)


@gluon.constexpr_function
def tl_dot_decomposed_get_transposed_order(rank):
    assert rank >= 2
    order = list(range(rank - 2))
    order += [rank - 1, rank - 2]
    return order


@gluon.jit
def tl_dot_decomposed_extend_and_broadcast_scale(v, scale, compute_type: ttgl.constexpr, operand_index: ttgl.constexpr,
                                                 scale_factor: ttgl.constexpr):
    rank: ttgl.constexpr = len(v.type.shape)
    k_dim: ttgl.constexpr = rank - 1 if operand_index == 0 else rank - 2

    if operand_index == 1:
        order: ttgl.constexpr = tl_dot_decomposed_get_transposed_order(rank)
        scale = ttgl.permute(scale, order)

    scale16 = tl_dot_decomposed_scale_to_16(scale, compute_type)
    reshape_scale = tl_dot_decomposed_broadcast_scale(scale16, k_dim, scale_factor)
    return ttgl.convert_layout(reshape_scale, v.type.layout), scale


@gluon.jit
def tl_dot_decomposed_mask_nan(mxfp, scale, fast_math):
    ttgl.static_assert(fast_math, "TODO: support non-fast-math")
    return mxfp


@gluon.jit
def tl_dot_decomposed_scale_arg(v, scale, arg_format: ttgl.constexpr, operand_index: ttgl.constexpr,
                                compute_type: ttgl.constexpr, fast_math: ttgl.constexpr, scale_factor: ttgl.constexpr):
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
        reshape_scale, scale = tl_dot_decomposed_extend_and_broadcast_scale(v, scale, compute_type, operand_index,
                                                                            scale_factor)
        mxfp = ttgl.mul(v, reshape_scale)
        return tl_dot_decomposed_mask_nan(mxfp, scale, fast_math)


@gluon.constexpr_function
def tl_dot_decomposed_deduce_scale_factor(v, scale, arg_format, operand_index, k_pack):
    if scale is None:
        return 0
    if scale.numel == 1:
        return 0

    k_dim = len(v.shape) - 1 if operand_index == 0 else len(v.shape) - 2
    unpack_factor = 2 if arg_format == "e2m1" and k_pack else 1
    k_size = v.shape[k_dim] * unpack_factor
    scale_factor = k_size // scale.shape[-1]
    assert scale_factor in (16, 32), f"scale factor must be 16 or 32. Got {scale_factor}"
    return scale_factor


@gluon.jit
def tl_dot_decomposed_block_scales_impl(
    tl_dot_scaled_fn: ttgl.constexpr,
    tl_dot_fn: ttgl.constexpr,
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
    if lhs_scale is None and rhs_scale is not None:
        lhs_trans = tl_trans(lhs)
        rhs_trans = tl_trans(rhs)
        if acc is not None:
            orig_layout: ttgl.constexpr = acc.type.layout
            acc = tl_trans(acc)
        result = tl_dot_scaled_fn(
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
        lhs_scale_factor: ttgl.constexpr = tl_dot_decomposed_deduce_scale_factor(lhs, lhs_scale, lhs_format, 0,
                                                                                 lhs_k_pack)
        rhs_scale_factor: ttgl.constexpr = tl_dot_decomposed_deduce_scale_factor(rhs, rhs_scale, rhs_format, 1,
                                                                                 rhs_k_pack)
        scale_factor: ttgl.constexpr = lhs_scale_factor or rhs_scale_factor or 32
        ttgl.static_assert(
            lhs_scale_factor == 0 or rhs_scale_factor == 0 or lhs_scale_factor == rhs_scale_factor,
            "Operands must have the same scale factor",
        )

        scale_a = tl_dot_decomposed_scale_arg(lhs, lhs_scale, lhs_format, 0, compute_type, fast_math, scale_factor)
        scale_b = tl_dot_decomposed_scale_arg(rhs, rhs_scale, rhs_format, 1, compute_type, fast_math, scale_factor)

        return tl_dot_fn(scale_a, scale_b, acc, out_dtype=out_dtype)
