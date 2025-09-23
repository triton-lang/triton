from functools import partial

import pytest
import torch
import triton
from triton_kernels.numerics_details.mxfp import (
    MXFP_BLOCK_SIZE,
    DequantScaleRoundingMode,
    downcast_to_mxfp,
    downcast_to_mxfp_torch,
    get_max_quant_val,
    upcast_from_mxfp,
    upcast_from_mxfp_torch,
)
from triton_kernels.target_info import is_cuda
from triton_kernels.testing import assert_close, assert_equal


def dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    return torch.uint8 if dtype_str == "float4_e2m1" else getattr(torch, dtype_str)


@pytest.mark.parametrize("dst_dtype", ["float16", "bfloat16", "float32"])
def test_mxfp4_rounding_cases(dst_dtype, device):
    dst_dtype = dtype_str_to_torch(dst_dtype)
    x = torch.tensor([6, 0, 0.24, 0.25, 0.75, 0.99, 1.2, 1.3, 1.25, -1.25]).to(device).bfloat16().view(1, -1, 1)
    quant, scale = downcast_to_mxfp(x, torch.uint8, axis=1)
    dequant = upcast_from_mxfp(quant, scale, dst_dtype, axis=1)
    # Tie-breaking cases (RTNE):
    # - 0.25 is exactly halfway between 0.0 and 0.5. RTNE selects the even quantized value 0.0
    #   (binary LSB of target is 0). Rounding away from zero would pick 0.5; towards zero also picks 0.0.
    # - 0.75 is halfway between 0.5 and 1.0. RTNE selects the even value 1.0 (LSB 0). Away-from-zero would pick 1.0;
    #   towards-zero would pick 0.5.
    # - 1.25 is halfway between 1.0 and 1.5. RTNE selects the even value 1.0. Away-from-zero would pick 1.5;
    #   towards-zero would pick 1.0.
    # - -1.25 is halfway between -1.0 and -1.5. RTNE selects -1.0 (even). Away-from-zero would pick -1.5;
    #   towards-zero would pick -1.0.
    assert dequant.flatten().tolist() == [6, 0, 0, 0.0, 1.0, 1.0, 1.0, 1.5, 1.0, -1.0], f"{dequant=}"

    quant_torch, scale_torch = downcast_to_mxfp_torch(x, torch.uint8, axis=1)
    assert_equal(quant_torch, quant)
    assert_equal(scale_torch, scale)

    dequant_torch = upcast_from_mxfp_torch(quant_torch, scale_torch, dst_dtype, axis=1)
    assert_equal(dequant_torch, dequant)


@pytest.mark.parametrize("src_dtype", ["float4_e2m1", "float8_e5m2", "float8_e4m3fn"])
@pytest.mark.parametrize("dst_dtype", ["float16", "bfloat16", "float32"])
def test_mxfp_quant_dequant(src_dtype, dst_dtype, device):
    if "float8" in src_dtype and (is_cuda() and torch.cuda.get_device_capability()[0] < 9):
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
    pos_vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, max_val], device=device, dtype=dst_dtype)
    neg_vals = -pos_vals
    k_dim = torch.cat([pos_vals, neg_vals])
    k_dim = k_dim.reshape([k_dim.shape[0], 1])

    # We pick power of 2 scales since both the scales and their inverse only require exponent bits to be exactly
    # represented. This means we can store the scales exactly in the e8m0 format.
    powers = torch.arange(-8, 8, device=device, dtype=dst_dtype)
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
        # Zero-sized arrays
        ((0, 4096, 1024), 1, "float4_e2m1", DequantScaleRoundingMode.ROUND_UP),
        ((3, 4096, 0), 1, "float4_e2m1", DequantScaleRoundingMode.ROUND_DOWN),
        ((10, 0, 1024), 2, "float8_e5m2", DequantScaleRoundingMode.ROUND_UP),
        ((0, 0, 1024), 2, "float8_e4m3fn", DequantScaleRoundingMode.ROUND_DOWN),

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
    device,
):
    if "float8" in quant_dtype and (is_cuda() and torch.cuda.get_device_capability()[0] < 9):
        pytest.skip("Float8 not tested on A100")
    quant_torch_type = dtype_str_to_torch(quant_dtype)
    dequant_torch_type = dtype_str_to_torch(dequant_dtype)
    # Generate random input tensor that is contiguous once axis is the last dimension
    x = torch.randn(shape, device=device, dtype=dequant_torch_type)

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


def _benchmark_mxfp_quantization(shape, src_dtype: torch.dtype, target_quant_dtype: torch.dtype, n_iters=1000):
    x = torch.randn(*shape, dtype=src_dtype, device="cuda")
    elapsed = (triton.testing.do_bench(
        partial(downcast_to_mxfp, x, target_quant_dtype, axis=-1),
        rep=n_iters,
        return_mode="min",
    ) / 1e3)

    # Each call reads x (2 Bytes) and writes the output tensor (1B or 0.5B) once.
    # -> 3B * numel
    gbytes = ((3 if target_quant_dtype == torch.float8_e4m3fn else 2.5) * x.numel()) / 1e9

    bw = gbytes / elapsed
    return bw


def _benchmark_mxfp_dequantization(shape, src_quant_dtype: torch.dtype, target_dtype: torch.dtype, n_iters=1000):
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda").to(src_quant_dtype)
    scale_shape = shape[:-1] + (triton.cdiv(shape[-1], MXFP_BLOCK_SIZE), )
    x_scale = torch.randint(0, 256, scale_shape, device="cuda", dtype=torch.uint8)
    elapsed = (triton.testing.do_bench(
        partial(upcast_from_mxfp, x, x_scale, target_dtype, axis=-1),
        rep=n_iters,
        return_mode="min",
    ) / 1e3)

    # Each call reads x (1B or 0.5B) and writes the output tensor (2 Bytes) once.
    # -> 3B * numel
    gbytes = ((3 if src_quant_dtype == torch.float8_e4m3fn else 2.5) * x.numel()) / 1e9

    bw = gbytes / elapsed
    return bw


if __name__ == "__main__":
    tests = [
        ((1024, 8192), torch.float16),
        ((4096, 8192), torch.float16),
        ((1024, 8192), torch.bfloat16),
        ((4096, 8192), torch.bfloat16),
    ]

    table = []
    for shape, dtype in tests:
        mxfp8_q_bw = _benchmark_mxfp_quantization(shape, dtype, torch.float8_e4m3fn)
        mxfp8_dq_bw = _benchmark_mxfp_dequantization(shape, torch.float8_e4m3fn, dtype)
        mxfp4_q_bw = _benchmark_mxfp_quantization(shape, dtype, torch.uint8)
        mxfp4_dq_bw = _benchmark_mxfp_dequantization(shape, torch.uint8, dtype)
        table.append(shape + (dtype, mxfp8_q_bw, mxfp8_dq_bw, mxfp4_q_bw, mxfp4_dq_bw))

    from tabulate import tabulate
    print(
        tabulate(
            table,
            headers=["M", "N", "dtype", "mxfp8_quant_bw", "mxfp8_dequant_bw", "mxfp4_quant_bw", "mxfp4_dequant_bw"]))
