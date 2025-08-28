import pytest
import torch
from triton_kernels.numerics_details.mxfp import (
    DequantScaleRoundingMode,
    downcast_to_mxfp,
    downcast_to_mxfp_torch,
    get_max_quant_val,
    upcast_from_mxfp,
    upcast_from_mxfp_torch,
)
from triton_kernels.testing import assert_close, assert_equal


def dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    return torch.uint8 if dtype_str == "float4_e2m1" else getattr(torch, dtype_str)


@pytest.mark.parametrize("dst_dtype", ["float16", "bfloat16", "float32"])
def test_mxfp4_rounding_cases(dst_dtype):
    dst_dtype = dtype_str_to_torch(dst_dtype)
    x = torch.tensor([6, 0, 0.24, 0.25, 0.75, 0.99, 1.2, 1.3]).cuda().bfloat16().view(1, -1, 1)
    quant, scale = downcast_to_mxfp(x, torch.uint8, axis=1)
    dequant = upcast_from_mxfp(quant, scale, dst_dtype, axis=1)
    assert dequant.flatten().tolist() == [6, 0, 0, 0.5, 1.0, 1.0, 1.0, 1.5], f"{dequant=}"

    quant_torch, scale_torch = downcast_to_mxfp_torch(x, torch.uint8, axis=1)
    assert_equal(quant_torch, quant)
    assert_equal(scale_torch, scale)

    dequant_torch = upcast_from_mxfp_torch(quant_torch, scale_torch, dst_dtype, axis=1)
    assert_equal(dequant_torch, dequant)


@pytest.mark.parametrize("src_dtype", ["float4_e2m1", "float8_e5m2", "float8_e4m3fn"])
@pytest.mark.parametrize("dst_dtype", ["float16", "bfloat16", "float32"])
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
    quant, scale = downcast_to_mxfp(weight, src_dtype, axis=1)
    dequant = upcast_from_mxfp(quant, scale, dst_dtype, axis=1)
    assert_equal(weight, dequant)


# fmt: off
@pytest.mark.parametrize(
    "shape, axis, quant_dtype, rounding_mode",
    [
        ((3, 4096, 1024), 1, "float4_e2m1", DequantScaleRoundingMode.ROUND_UP),
        ((10, 254, 60), 0, "float4_e2m1", DequantScaleRoundingMode.ROUND_DOWN),
        ((1, 320, 160), 2, "float8_e5m2", DequantScaleRoundingMode.ROUND_UP),
        ((2, 16, 512), -1, "float8_e4m3fn", DequantScaleRoundingMode.ROUND_DOWN),
    ],
)
# fmt: on
@pytest.mark.parametrize("dequant_dtype", ["float16", "bfloat16", "float32"])
def test_mxfp_casting(
    shape: tuple[int, ...],
    axis: int,
    quant_dtype: str,
    dequant_dtype: str,
    rounding_mode: DequantScaleRoundingMode,
):
    if "float8" in quant_dtype and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Float8 not tested on A100")
    quant_torch_type = dtype_str_to_torch(quant_dtype)
    dequant_torch_type = dtype_str_to_torch(dequant_dtype)
    # Generate random input tensor that is contiguous once axis is the last dimension
    x = torch.randn(shape, device="cuda", dtype=dequant_torch_type)

    # Quantize and check equivalence
    quant, scale = downcast_to_mxfp(x, quant_torch_type, axis, DEQUANT_SCALE_ROUNDING_MODE=rounding_mode)
    quant_torch, scale_torch = downcast_to_mxfp_torch(x, quant_torch_type, axis,
                                                      DEQUANT_SCALE_ROUNDING_MODE=rounding_mode)

    assert_equal(quant_torch, quant)
    assert_equal(scale_torch, scale)
    assert_equal(1, quant.stride(axis))
    assert_equal(1, quant_torch.stride(axis))

    # Dequantize and check equivalence
    dequant = upcast_from_mxfp(quant, scale, dequant_torch_type, axis)
    dequant_torch = upcast_from_mxfp_torch(quant_torch, scale_torch, dequant_torch_type, axis)
    assert_equal(dequant, dequant_torch)

    # Dequantized result should be close to the original, though tolerance is large due to the precision loss.
    assert_close(x, dequant, maxtol=0.5, rmstol=0.15)
