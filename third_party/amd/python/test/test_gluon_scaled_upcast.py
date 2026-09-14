import pytest
import torch

from triton._internal_testing import is_hip_cdna3, is_hip_cdna4
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

IS_CDNA3_OR_CDNA4 = is_hip_cdna3() or is_hip_cdna4()


@gluon.jit
def _compact_scaled_upcast_fp4_kernel(x_ptr, scale_ptr, out_ptr, M: ttgl.constexpr, K_PACKED: ttgl.constexpr,
                                      OUT_K: ttgl.constexpr, SCALE_K: ttgl.constexpr, SPT_PACKED: ttgl.constexpr,
                                      USE_CDNA4: ttgl.constexpr):
    packed_layout: ttgl.constexpr = ttgl.BlockedLayout([1, SPT_PACKED], [8, 8], [1, 1], [1, 0])

    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, packed_layout))
    offs_k_packed = ttgl.arange(0, K_PACKED, layout=ttgl.SliceLayout(0, packed_layout))
    x_offsets = offs_m[:, None] * K_PACKED + offs_k_packed[None, :]
    x = ttgl.load(x_ptr + x_offsets)

    scale_factor: ttgl.constexpr = OUT_K // SCALE_K
    scale_layout: ttgl.constexpr = ttgl.amd.get_scaled_upcast_fp4_scale_layout(x, scale_factor, ttgl.bfloat16, axis=1)
    offs_scale_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_layout))
    offs_scale_k = ttgl.arange(0, SCALE_K, layout=ttgl.SliceLayout(0, scale_layout))
    scale_offsets = offs_scale_m[:, None] * SCALE_K + offs_scale_k[None, :]
    scale = ttgl.load(scale_ptr + scale_offsets)

    if USE_CDNA4:
        out = ttgl.amd.cdna4.scaled_upcast(x, scale, ttgl.bfloat16, axis=1)
    else:
        out = ttgl.amd.cdna3.scaled_upcast(x, scale, ttgl.bfloat16, axis=1)

    out_layout: ttgl.constexpr = out.type.layout
    offs_out_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, out_layout))
    offs_out_k = ttgl.arange(0, OUT_K, layout=ttgl.SliceLayout(0, out_layout))
    out_offsets = offs_out_m[:, None] * OUT_K + offs_out_k[None, :]
    ttgl.store(out_ptr + out_offsets, out)


@pytest.mark.skipif(not IS_CDNA3_OR_CDNA4, reason="Requires CDNA3 or CDNA4")
@pytest.mark.parametrize(
    "k_packed, scale_k, spt_packed", [(128, 8, 16), (256, 16, 32), (256, 8, 32), (512, 16, 64)], ids=[
        "one_scale_register", "two_scale_registers", "one_scale_register_64_elements", "two_scale_registers_64_elements"
    ])
def test_runtime_scaled_upcast_fp4_compact_scale(k_packed, scale_k, spt_packed):
    M = 8
    out_k = 2 * k_packed
    elements_per_scale = out_k // scale_k

    # Packed 0x1 E2M1 values are 0.5. Use a different E8M0 exponent for
    # every compact scale block to exercise the scale-register mapping.
    x = torch.full((M, k_packed), 0x11, dtype=torch.uint8, device="cuda")
    scale_exponents = torch.arange(M * scale_k, device="cuda").reshape(M, scale_k) % 7 - 3
    scale = (scale_exponents + 0x7f).to(torch.uint8)
    out = torch.empty((M, out_k), dtype=torch.bfloat16, device="cuda")

    use_cdna4 = is_hip_cdna4()
    program = _compact_scaled_upcast_fp4_kernel[(1, )](x, scale, out, M, k_packed, out_k, scale_k, spt_packed,
                                                       use_cdna4, num_warps=1)

    expected = torch.ldexp(torch.full_like(scale_exponents, 0.5, dtype=torch.float32),
                           scale_exponents).repeat_interleave(elements_per_scale, dim=1).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if use_cdna4:
        assert "v_cvt_scalef32_pk_bf16_fp4" in program.asm["amdgcn"]
    else:
        assert "v_cvt_scalef32_pk_bf16_fp4" not in program.asm["amdgcn"]


@gluon.jit
def _scaled_upcast_scale_edges_kernel(x_ptr, scale_ptr, out_ptr, FP4: ttgl.constexpr, USE_CDNA4: ttgl.constexpr):
    layout: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [8, 8], [1, 1], [1, 0])
    input_k: ttgl.constexpr = 32 if FP4 else 64
    dtype: ttgl.constexpr = out_ptr.dtype.element_ty
    rows = ttgl.arange(0, 8, layout=ttgl.SliceLayout(1, layout))
    cols = ttgl.arange(0, input_k, layout=ttgl.SliceLayout(0, layout))
    x = ttgl.load(x_ptr + rows[:, None] * input_k + cols[None, :])

    if FP4:
        scale_layout: ttgl.constexpr = ttgl.amd.get_scaled_upcast_fp4_scale_layout(x, 32, out_ptr.dtype.element_ty,
                                                                                   axis=1)
        scale_k: ttgl.constexpr = 2
    else:
        scale_layout: ttgl.constexpr = layout
        scale_k: ttgl.constexpr = 64
    scale_rows = ttgl.arange(0, 8, layout=ttgl.SliceLayout(1, scale_layout))
    scale_cols = ttgl.arange(0, scale_k, layout=ttgl.SliceLayout(0, scale_layout))
    scale = ttgl.load(scale_ptr + scale_rows[:, None] * scale_k + scale_cols[None, :])
    axis: ttgl.constexpr = 1 if FP4 else None
    if USE_CDNA4:
        out = ttgl.amd.cdna4.scaled_upcast(x, scale, dtype, axis)
    else:
        out = ttgl.amd.cdna3.scaled_upcast(x, scale, dtype, axis)

    out_layout: ttgl.constexpr = out.type.layout
    out_rows = ttgl.arange(0, 8, layout=ttgl.SliceLayout(1, out_layout))
    out_cols = ttgl.arange(0, 64, layout=ttgl.SliceLayout(0, out_layout))
    ttgl.store(out_ptr + out_rows[:, None] * 64 + out_cols[None, :], out)


@pytest.mark.skipif(not IS_CDNA3_OR_CDNA4, reason="Requires CDNA3 or CDNA4")
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("nan_scale", [False, True])
def test_scaled_upcast_scale_edges(dtype, out_dtype, nan_scale):
    fp4 = dtype == torch.uint8
    small_value = 0 if nan_scale else (0x11 if fp4 else 0.5)
    values = [small_value] * 4 + [0x66 if fp4 else 4.0] * 4
    x = torch.tensor(values, dtype=dtype)[:, None].expand(8, 32 if fp4 else 64).contiguous().cuda()
    scale = torch.tensor([255] * 8 if nan_scale else [0, 1, 127, 254] * 2, dtype=torch.uint8, device="cuda")
    scale = scale[:, None].expand(8, 2 if fp4 else 64).contiguous()
    out = torch.empty((8, 64), dtype=out_dtype, device="cuda")
    _scaled_upcast_scale_edges_kernel[(1, )](x, scale, out, fp4, is_hip_cdna4(), num_warps=1, allow_flush_denorm=False)

    # Both subnormal and normal BF16 results survive; the smallest FP16
    # results underflow after scaling.
    expected_values = [float("nan")] * 8 if nan_scale else [
        2.0**-128, 2.0**-127, 0.5, 2.0**126, 2.0**-125, 2.0**-124, 4.0,
        float("inf")
    ]
    expected = torch.tensor(expected_values, dtype=torch.float64).to(out_dtype)[:, None].expand(8, 64)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)
