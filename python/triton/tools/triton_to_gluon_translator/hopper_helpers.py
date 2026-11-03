# type: ignore

import builtins

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_wait,
    warpgroup_mma_init,
)

from triton.tools.triton_to_gluon_translator.common_helpers import *  # noqa: F401,F403
from triton.tools.triton_to_gluon_translator.common_helpers import (
    tl_dot_decomposed_block_scales_impl,
    default_blocked_layout,
)
from triton.tools.triton_to_gluon_translator.nvidia_helpers import *  # noqa: F401,F403
from triton.tools.triton_to_gluon_translator.nvidia_helpers import (
    tl_dot_mma_sync,
    get_shared_memory_mma_operand,
)

# ---- NVIDIA Hopper dot ----


@gluon.constexpr_function
def _get_default_max_num_imprecise_acc(
    a_ty: ttgl.block_type,
    b_ty: ttgl.block_type,
    max_num_imprecise_acc: int | None,
) -> int:
    # FIXME: Get this from builder options. Default is 2**30 for Hopper.
    max_num_imprecise_acc_default = 2**30
    if max_num_imprecise_acc is not None:
        return max_num_imprecise_acc
    if a_ty.element_ty.is_fp8() and b_ty.element_ty.is_fp8():
        return max_num_imprecise_acc_default
    return 0


@gluon.constexpr_function
def _get_default_input_precision(
    allow_tf32: bool | None,
    input_precision: str | None,
) -> str:
    assert input_precision is None or allow_tf32 is None, "Only one of input_precision and allow_tf32 can be specified"
    if input_precision is not None:
        return input_precision

    allow_tf32 = allow_tf32 or True
    return "tf32" if allow_tf32 else "ieee"


@gluon.constexpr_function
def _operand_supports_mmav3(dtype: ttgl.dtype) -> bool:
    return (dtype.is_fp8() or dtype.is_fp16() or dtype.is_bf16() or dtype.is_fp32() or dtype.is_fp64()
            or dtype.is_int8() or dtype.is_uint8())


@gluon.constexpr_function
def tl_dot_mmav3_supported(
    a_ty: ttgl.block_type,
    b_ty: ttgl.block_type,
    num_warps: int,
    input_precision: str,
    max_num_imprecise_acc: int,
    out_dtype: ttgl.dtype,
) -> bool:

    M = a_ty.shape[0]
    N = b_ty.shape[1]
    K = a_ty.shape[1]

    a_dtype = a_ty.element_ty
    b_dtype = b_ty.element_ty

    # Minimum MMA instruction K shape.
    mma_min_K = 256 // a_dtype.primitive_bitwidth
    if K < mma_min_K:
        return False

    # Only rank 2 supported.
    if len(a_ty.shape) != 2 or len(b_ty.shape) != 2:
        return False

    # Minimum 4 warps.
    if num_warps % 4 != 0:
        return False

    # Minimum MMA instruction shape along M and N.
    if M % 64 != 0 or N % 8 != 0:
        return False

    # Accepted LHS dtypes: float8e5m2, float8e4m3fn, u/int8, float16, bfloat16, float32.
    if a_dtype not in [
            ttgl.float8e5,
            ttgl.float8e4nv,
            ttgl.int8,
            ttgl.uint8,
            ttgl.float16,
            ttgl.bfloat16,
            ttgl.float32,
    ]:
        return False

    # Check float8 -> float32 accumulation with imprecise acc.
    if max_num_imprecise_acc < 32 and a_dtype in [ttgl.float8e5, ttgl.float8e4nv] and out_dtype.is_fp32():
        return False

    # fp32 operands require tf32.
    if a_dtype.is_fp32() and b_dtype.is_fp32():
        return input_precision == "tf32"

    # Check supported operand dtypes.
    return _operand_supports_mmav3(a_dtype) and _operand_supports_mmav3(b_dtype)


@gluon.constexpr_function
def _mmav3_acc_layout(
    num_warps: int,
    c_shape: list[int],
    a_dtype: ttgl.dtype,
    b_dtype: ttgl.dtype,
    out_dtype: ttgl.dtype,
):
    k = 256 // a_dtype.primitive_bitwidth
    assert c_shape[0] % 64 == 0 and c_shape[1] % 8 == 0, "c_shape must be divisible [64, 8]"

    # fmt: off
    if a_dtype.is_floating():
        valid_n = [256, 248, 240, 232, 224, 216, 208, 200, 192, 184, 176,
                   168, 160, 152, 144, 136, 128, 120, 112, 104, 96,  88,  # noqa: E241
                    80,  72,  64,  56,  48,  40,  32,  24,  16,  8]       # noqa: E241, E127
    else:
        assert a_dtype.is_int()
        valid_n = [224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 24, 16, 8]
    # fmt: on

    m = 16
    m_warps = max(c_shape[0] // m, 1)
    n_warps = max(num_warps // m_warps, 1)
    max_n = max(c_shape[1] // n_warps, 8)
    instr_shape = None
    for n in valid_n:
        if c_shape[1] % n == 0 and n <= max_n:
            instr_shape = [m, n, k]
            break

    assert instr_shape is not None, "Failed to find valid instruction shape"

    warps_per_tile = [4, 1]
    shape_per_warp = [16, instr_shape[1]]
    while True:
        if warps_per_tile[0] * warps_per_tile[1] >= num_warps:
            break
        if c_shape[0] > shape_per_warp[0] * warps_per_tile[0]:
            warps_per_tile[0] *= 2
        else:
            warps_per_tile[1] *= 2

    return ttgl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_tile,
        instr_shape=instr_shape,
    )


@gluon.jit
def tl_dot_mmav3(
    a: ttgl.tensor,
    b: ttgl.tensor,
    acc: ttgl.tensor | None,
    input_precision: builtins.str,
    max_num_imprecise_acc: int,
    out_dtype: ttgl.dtype,
):
    el_ty: ttgl.constexpr = a.type.element_ty
    allow_transpose: ttgl.constexpr = el_ty.is_fp16() or el_ty.is_bf16()
    a_smem = get_shared_memory_mma_operand(a, 0, allow_transpose, is_fp4_padded=False, force_transpose=False)
    b_smem = get_shared_memory_mma_operand(b, 1, allow_transpose, is_fp4_padded=False, force_transpose=False)

    c_shape: ttgl.constexpr = a.shape[:-1] + [b.shape[-1]]
    if acc is not None:
        ttgl.static_assert(acc.shape == c_shape, "accumulator shape is incompatible")
        ttgl.static_assert(acc.type.element_ty == out_dtype, "accumulator dtype is incompatible")

    mma_layout: ttgl.constexpr = _mmav3_acc_layout(
        ttgl.num_warps(),
        c_shape,
        a.type.element_ty,
        b.type.element_ty,
        out_dtype,
    )

    if acc is None:
        ret_layout: ttgl.constexpr = default_blocked_layout(c_shape, ttgl.num_warps())
        acc = ttgl.zeros(c_shape, out_dtype, layout=mma_layout)
    else:
        ret_layout: ttgl.constexpr = acc.type.layout
        acc = ttgl.convert_layout(acc, mma_layout)

    fence_async_shared()
    wgmma_acc = warpgroup_mma_init(acc)
    wgmma_acc = warpgroup_mma(a_smem, b_smem, wgmma_acc, is_async=True)
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(wgmma_acc, ))

    return ttgl.convert_layout(acc, ret_layout)


@gluon.jit
def tl_dot(
    a: ttgl.tensor,
    b: ttgl.tensor,
    acc: ttgl.tensor | None = None,
    input_precision: builtins.str | None = None,
    allow_tf32: builtins.bool | None = None,
    max_num_imprecise_acc: int | None = None,
    out_dtype: ttgl.dtype = ttgl.float32,
):
    input_prec: ttgl.constexpr = _get_default_input_precision(
        allow_tf32,
        input_precision,
    )
    max_num_imprecise: ttgl.constexpr = _get_default_max_num_imprecise_acc(
        a.type,
        b.type,
        max_num_imprecise_acc,
    )
    num_warps: ttgl.constexpr = ttgl.num_warps()

    if tl_dot_mmav3_supported(a.type, b.type, num_warps, input_prec, max_num_imprecise, out_dtype):
        return tl_dot_mmav3(a, b, acc, input_prec, max_num_imprecise, out_dtype)
    else:
        return tl_dot_mma_sync(a, b, acc, input_precision, out_dtype)


# ---- NVIDIA Hopper dot-scaled ----


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
        acc=acc,
        fast_math=fast_math,
        lhs_k_pack=lhs_k_pack,
        rhs_k_pack=rhs_k_pack,
        out_dtype=out_dtype,
    )


# ---- NVIDIA obj dispatch ----


@gluon.jit
def tl_obj_gather(obj, x_offsets, y_offset):
    ttgl.static_assert(not isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor),
                       "descriptor gather is not supported on Hopper")
    return obj.gather(x_offsets, y_offset)


@gluon.jit
def tl_obj_scatter(obj, value, x_offsets, y_offset):
    ttgl.static_assert(not isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor),
                       "descriptor scatter is not supported on Hopper")
    return obj.scatter(value, x_offsets, y_offset)
