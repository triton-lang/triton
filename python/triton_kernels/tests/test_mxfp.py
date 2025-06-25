import math

import pytest
import torch

import triton
from triton_kernels.numerics_details.mxfp import (
    DequantScaleRoundingMode,
    SwizzlingType,
    downcast_to_mxfp,
    downcast_to_mxfp_torch,
    get_max_quant_val,
    perm_tensor_from_contig,
    perm_tuple_to_contig,
    swizzle_mx_scale_bw,
    swizzle_mxfp4_scale_hopper,
    swizzle_mxfp4_value_hopper,
    unswizzle_mx_scale_bw_torch,
    unswizzle_mxfp4_scale_hopper_torch,
    unswizzle_mxfp4_value_hopper_torch,
    upcast_from_mxfp,
    upcast_from_mxfp_torch,
)
from triton_kernels.testing import assert_close, assert_equal
from triton_kernels.target_info import is_hip, is_hip_cdna3


def dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    return torch.uint8 if dtype_str == "float4_e2m1" else getattr(torch, dtype_str)


@pytest.mark.parametrize("dst_dtype", ["float16", "bfloat16"])
def test_mxfp4_rounding_cases(dst_dtype):
    dst_dtype = dtype_str_to_torch(dst_dtype)
    x = torch.tensor([6, 0, 0.24, 0.25, 0.75, 0.99, 1.2, 1.3]).cuda().bfloat16().view(1, -1, 1)
    quant, scale, _ = downcast_to_mxfp(x, torch.uint8, axis=1)
    dequant = upcast_from_mxfp(quant, scale, dst_dtype, axis=1)
    assert dequant.flatten().tolist() == [6, 0, 0, 0.5, 1.0, 1.0, 1.0, 1.5], f"{dequant=}"

    quant_torch, scale_torch = downcast_to_mxfp_torch(x, torch.uint8, axis=1)
    assert_equal(quant_torch, quant)
    assert_equal(scale_torch, scale)

    dequant_torch = upcast_from_mxfp_torch(quant_torch, scale_torch, dst_dtype, axis=1)
    assert_equal(dequant_torch, dequant)


@pytest.mark.parametrize("src_dtype", ["float4_e2m1", "float8_e5m2", "float8_e4m3fn"])
@pytest.mark.parametrize("dst_dtype", ["float16", "bfloat16"])
def test_mxfp_quant_dequant(src_dtype, dst_dtype):
    if "float8" in src_dtype and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Float8 not tested on A100")
    limit_range = src_dtype == "float8_e5m2" and dst_dtype == "float16"

    # This test checks that quantization and dequantization kernels produce the exact values for some inputs
    # that can be represented exactly in the quantized format.
    src_dtype = dtype_str_to_torch(src_dtype)
    dst_dtype = dtype_str_to_torch(dst_dtype)
    max_val = get_max_quant_val(src_dtype)
    if limit_range:
        # FP16 can't represent the full range of MXFP8, so we limit the max value here
        max_val = 128

    # These are all the valid mxfp4 positive values.
    pos_vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, max_val], device="cuda", dtype=dst_dtype)
    neg_vals = -pos_vals
    k_dim = torch.cat([pos_vals, neg_vals])
    k_dim = k_dim.reshape([k_dim.shape[0], 1])

    # We pick power of 2 scales since both the scales and their inverse only require exponent bits to be exactly
    # represented. This means we can store the scales exactly in the e8m0 format.
    powers = torch.arange(-8, 8, device="cuda", dtype=dst_dtype)
    scales = 2**powers
    scales = scales.reshape([1, powers.shape[0]])
    weight = k_dim * scales
    weight = weight.repeat((9, 32))  # Repeat the dimensions to test multi block launches.
    weight = weight.reshape([1, weight.shape[0], weight.shape[1]])
    weight = weight.mT.contiguous().mT
    quant, scale, _ = downcast_to_mxfp(weight, src_dtype, axis=1)
    dequant = upcast_from_mxfp(quant, scale, dst_dtype, axis=1)
    assert_equal(weight, dequant)


@pytest.mark.parametrize(
    "shape",
    [
        (3, 4096, 1024),
        (10, 254, 60),
        (1, 320, 160),
        (2, 16, 512),
        (3, 2, 36),
    ],
)
def test_mxfp_swizzle(shape: tuple[int, ...]):
    """
    Test that unswizzle is the inverse of swizzle, after removing padding.
    """
    x = torch.randn(shape, device="cuda")
    assert_equal(x, unswizzle_mx_scale_bw_torch(swizzle_mx_scale_bw(x))[..., :shape[-2], :shape[-1]])


# fmt: off
@pytest.mark.parametrize(
    "shape, axis, swizzle_axis, block_out_dim, block_quant_dim, quant_dtype, rounding_mode",
    [
        ((3, 4096, 1024), 1, -1, 128, 32, "float4_e2m1", DequantScaleRoundingMode.ROUND_UP),
        ((10, 254, 60), 0, 1, 128, 32, "float4_e2m1", DequantScaleRoundingMode.ROUND_DOWN),
        ((1, 320, 160), 2, 1, 128, 32, "float8_e5m2", DequantScaleRoundingMode.ROUND_UP),
        ((2, 16, 512), -1, 0, 128, 32, "float8_e4m3fn", DequantScaleRoundingMode.ROUND_DOWN),
    ],
)
# fmt: on
@pytest.mark.parametrize("user_allocated_output", [False, True])
@pytest.mark.parametrize("dequant_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("swizzle_value, swizzle_scale", [(None, None), (SwizzlingType.HOPPER, None),
                                                          (None, SwizzlingType.HOPPER),
                                                          (SwizzlingType.HOPPER, SwizzlingType.HOPPER),
                                                          (None, SwizzlingType.BLACKWELL)])
def test_mxfp_casting(
    shape: tuple[int, ...],
    axis: int,
    swizzle_axis: int | None,
    swizzle_value: SwizzlingType | None,
    swizzle_scale: SwizzlingType | None,
    block_out_dim,
    block_quant_dim,
    quant_dtype: str,
    dequant_dtype: str,
    rounding_mode: DequantScaleRoundingMode,
    user_allocated_output: bool,
):
    if "float8" in quant_dtype and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Float8 not tested on A100")
    if swizzle_value == SwizzlingType.HOPPER:
        if "float4" not in quant_dtype:
            pytest.skip("NYI. Hopper swizzle just implemented for mxfp4")
        if shape[axis] % 64 != 0 or shape[swizzle_axis] % 128 != 0:
            # Automatic padding not implemented for Hopper swizzle
            pytest.skip("Hopper swizzle not supported for tile not multiple of 64x128")
    if swizzle_scale == SwizzlingType.HOPPER:
        if shape[axis] % 64 != 0 or shape[swizzle_axis] % 128 != 0:
            # Automatic padding not implemented for Hopper swizzle
            pytest.skip("Hopper swizzle not supported for tile not multiple of 64x128")
    if user_allocated_output and any([swizzle_value, swizzle_scale]):
        pytest.skip("User-allocated output not supported together with swizzling")
    if is_hip():
        if swizzle_value is not None or swizzle_scale is not None:
            pytest.skip("Other swizzling patterns are not supported by AMD GPU")
        if quant_dtype == 'float8_e4m3fn' and is_hip_cdna3():
            pytest.skip("float8_e4m3fn cast hasn't been fully tested on AMD CDNA3")
        if quant_dtype == 'float8_e5m2' and is_hip_cdna3():
            pytest.skip("float8_e5m2 cast hasn't been fully tested on AMD CDNA3")

    swizzle_axis = swizzle_axis if (swizzle_value or swizzle_scale) else None
    quant_torch_type = dtype_str_to_torch(quant_dtype)
    dequant_torch_type = dtype_str_to_torch(dequant_dtype)
    # Generate random input tensor that is contiguous once axis is the last dimension
    # and swizzle_axis is the second to last dimension.
    x = torch.randn(shape, device="cuda", dtype=dequant_torch_type)

    # Allocate output tensor if needed
    if user_allocated_output:
        shape_list = list(perm_tuple_to_contig(shape, axis, swizzle_axis))
        scale_shape_list = list(shape_list)
        scale_shape_list[-1] = triton.cdiv(scale_shape_list[-1], 32)
        if "float4" in quant_dtype:
            shape_list[-1] //= 2

        out_tensor = torch.empty(shape_list, device="cuda", dtype=quant_torch_type)
        out_tensor_scale = torch.empty(scale_shape_list, device="cuda", dtype=torch.uint8)

        out_tensor = perm_tensor_from_contig(out_tensor, axis, swizzle_axis)
        out_tensor_scale = perm_tensor_from_contig(out_tensor_scale, axis, swizzle_axis)
    else:
        out_tensor = None
        out_tensor_scale = None

    # Quantize and check equivalence
    quant, scale, _ = downcast_to_mxfp(
        x,
        quant_torch_type,
        axis,
        swizzle_axis,
        swizzle_value=swizzle_value,
        swizzle_scale=swizzle_scale,
        out_quant_tensor=out_tensor,
        out_scale=out_tensor_scale,
        DEQUANT_SCALE_ROUNDING_MODE=rounding_mode,
        BLOCK_OUT_DIM=block_out_dim,
        BLOCK_QUANT_DIM=block_quant_dim,
    )
    if out_tensor is not None:
        assert_equal(out_tensor, quant)
    if out_tensor_scale is not None:
        assert_equal(out_tensor_scale, scale)

    quant_torch, scale_torch = downcast_to_mxfp_torch(
        x,
        quant_torch_type,
        axis,
        swizzle_axis,
        swizzle_value=swizzle_value,
        swizzle_scale=swizzle_scale,
        out_quant_tensor=out_tensor,
        out_scale=out_tensor_scale,
        DEQUANT_SCALE_ROUNDING_MODE=rounding_mode,
    )
    if out_tensor is not None:
        assert_equal(out_tensor, quant_torch)
    if out_tensor_scale is not None:
        assert_equal(out_tensor_scale, scale_torch)

    assert_equal(quant_torch, quant)
    assert_equal(scale_torch, scale)
    assert_equal(1, quant.stride(axis))
    assert_equal(1, quant_torch.stride(axis))

    # Dequantize and check equivalence
    dequant = upcast_from_mxfp(
        quant,
        scale,
        dequant_torch_type,
        axis,
        swizzle_axis,
        swizzle_value=swizzle_value,
        swizzle_scale=swizzle_scale,
        BLOCK_OUT_DIM=block_out_dim,
        BLOCK_QUANT_DIM=block_quant_dim,
    )
    dequant_torch = upcast_from_mxfp_torch(quant_torch, scale_torch, dequant_torch_type, axis, swizzle_axis,
                                           swizzle_value, swizzle_scale)
    assert_equal(dequant, dequant_torch)

    # Dequantized result should be close to the original, though tolerance is large due to the precision loss.
    assert_close(x, dequant, maxtol=0.5, rmstol=0.15)


def test_unswizzle_mxfp4_value_golden_value():
    shape = (16, 32)
    x = torch.arange(math.prod(shape)).view(shape).to(torch.uint8)
    x = x.mT
    res = swizzle_mxfp4_value_hopper(x, op_idx=1, mma_version=3)
    # res = res.mT.view(torch.uint32).mT
    # Thread 0
    assert res[0:16, 0].tolist() == [0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28]
    # Thread 1
    assert res[16:32, 0].tolist() == [1, 1, 5, 5, 9, 9, 13, 13, 17, 17, 21, 21, 25, 25, 29, 29]


@pytest.mark.parametrize("shape", [(16, 32), (16, 64), (32, 32), (32, 64), (64, 128), (128, 128)])
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("op_idx", [0, 1])
@pytest.mark.parametrize("mma_version", [2, 3])
def test_swizzle_mxfp4_value(shape, trans, op_idx, mma_version):
    x = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    if trans:
        x = x.mT
    k_dim = 1 - op_idx
    if x.shape[k_dim] < 32:
        pytest.skip("Not enough elements along K")

    # PyTorch implementation of unswizzle_mxfp4_value
    res = swizzle_mxfp4_value_hopper(x, op_idx, mma_version)
    res = unswizzle_mxfp4_value_hopper_torch(res, op_idx, mma_version)
    assert (res == x).all()


@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("shape", [(256, 64), (256, 128), (256, 256)])
def test_swizzle_mxfp4_scale(shape, num_warps):
    x = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    res = swizzle_mxfp4_scale_hopper(x, num_warps=num_warps)
    res = unswizzle_mxfp4_scale_hopper_torch(res, num_warps=num_warps)
    assert (res == x).all()
