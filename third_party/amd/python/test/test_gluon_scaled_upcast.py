import pytest
import torch

from triton._internal_testing import is_hip_cdna3, is_hip_cdna4
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

IS_CDNA3_OR_CDNA4 = is_hip_cdna3() or is_hip_cdna4()


@gluon.jit
def _compact_scaled_upcast_fp4_kernel(x_ptr, scale_ptr, out_ptr, USE_CDNA4: ttgl.constexpr):
    packed_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 16], [8, 8], [1, 1], [1, 0])
    compact_scale_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [8, 8], [1, 1], [1, 0])

    offs_m = ttgl.arange(0, 8, layout=ttgl.SliceLayout(1, packed_layout))
    offs_k_packed = ttgl.arange(0, 128, layout=ttgl.SliceLayout(0, packed_layout))
    x_offsets = offs_m[:, None] * 128 + offs_k_packed[None, :]
    x = ttgl.load(x_ptr + x_offsets)

    offs_scale_m = ttgl.arange(0, 8, layout=ttgl.SliceLayout(1, compact_scale_layout))
    offs_scale_k = ttgl.arange(0, 8, layout=ttgl.SliceLayout(0, compact_scale_layout))
    scale_offsets = offs_scale_m[:, None] * 8 + offs_scale_k[None, :]
    scale = ttgl.load(scale_ptr + scale_offsets)

    if USE_CDNA4:
        out = ttgl.amd.cdna4.scaled_upcast(x, scale, ttgl.bfloat16, axis=1)
    else:
        out = ttgl.amd.cdna3.scaled_upcast(x, scale, ttgl.bfloat16, axis=1)

    out_layout: ttgl.constexpr = out.type.layout
    offs_out_m = ttgl.arange(0, 8, layout=ttgl.SliceLayout(1, out_layout))
    offs_out_k = ttgl.arange(0, 256, layout=ttgl.SliceLayout(0, out_layout))
    out_offsets = offs_out_m[:, None] * 256 + offs_out_k[None, :]
    ttgl.store(out_ptr + out_offsets, out)


@pytest.mark.skipif(not IS_CDNA3_OR_CDNA4, reason="Requires CDNA3 or CDNA4")
def test_runtime_scaled_upcast_fp4_compact_scale():
    # Packed 0x1 E2M1 values are 0.5. Use a different E8M0 exponent for
    # every compact scale block to exercise the scale-register mapping.
    x = torch.full((8, 128), 0x11, dtype=torch.uint8, device="cuda")
    scale_exponents = torch.arange(64, device="cuda").reshape(8, 8) % 7 - 3
    scale = (scale_exponents + 0x7f).to(torch.uint8)
    out = torch.empty((8, 256), dtype=torch.bfloat16, device="cuda")

    use_cdna4 = is_hip_cdna4()
    program = _compact_scaled_upcast_fp4_kernel[(1, )](x, scale, out, use_cdna4, num_warps=1)

    expected = torch.ldexp(torch.full_like(scale_exponents, 0.5, dtype=torch.float32),
                           scale_exponents).repeat_interleave(32, dim=1).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if use_cdna4:
        assert "v_cvt_scalef32_pk_bf16_fp4" in program.asm["amdgcn"]
    else:
        assert "v_cvt_scalef32_pk_bf16_fp4" not in program.asm["amdgcn"]
