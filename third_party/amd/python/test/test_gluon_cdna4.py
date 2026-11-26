import pytest
import torch

from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl
from triton._internal_testing import fp6_values, is_hip_cdna4, pack_fp6


@gluon.jit
def scaled_downcast_fp4_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    BLOCK_M: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
):
    # BLOCK_K is the packed output width along the scaled axis; the unpacked
    # input is twice as wide.
    out_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [8, 8], [1, 1], [1, 0])
    in_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [8, 8], [1, 1], [1, 0])
    scale_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[8, 0]],
        lane_bases=[
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 0],
            [2, 0],
            [4, 0],
        ],
        warp_bases=[],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_K // 16],
    )
    in_offsets_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, in_layout))
    in_offsets_k = ttgl.arange(0, 2 * BLOCK_K, layout=ttgl.SliceLayout(0, in_layout))
    in_offsets = in_offsets_m[:, None] * (2 * BLOCK_K) + in_offsets_k[None, :]
    input = ttgl.load(input_ptr + in_offsets)

    scale_offsets_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, scale_layout))
    scale_offsets_k = ttgl.arange(0, BLOCK_K // 16, layout=ttgl.SliceLayout(0, scale_layout))
    scale_offsets = (scale_offsets_m[:, None] * (BLOCK_K // 16) + scale_offsets_k[None, :])
    scale = ttgl.load(scale_ptr + scale_offsets)

    output = ttgl.amd.cdna4.scaled_downcast(input, scale, "e2m1", axis=1)

    out_offsets_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, out_layout))
    out_offsets_k = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, out_layout))
    out_offsets = out_offsets_m[:, None] * BLOCK_K + out_offsets_k[None, :]
    ttgl.store(output_ptr + out_offsets, output)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("scale_byte", [0, 1, 127])
@pytest.mark.parametrize(
    "dtype,instruction_suffix",
    [
        (torch.float16, "f16"),
        (torch.bfloat16, "bf16"),
        (torch.float32, "f32"),
    ],
)
def test_runtime_scaled_downcast_fp4(dtype, instruction_suffix, scale_byte):
    block_m, block_k = 16, 32
    # Interleave scaled 1.0 and 2.0 so dividing by the scale packs 0x42.
    input = torch.empty((block_m, 2 * block_k), dtype=dtype)
    input[:, 0::2] = 2.0**(scale_byte - 127)
    input[:, 1::2] = 2.0**(scale_byte - 126)
    input = input.cuda()
    scale = torch.full((block_m, block_k // 16), scale_byte, dtype=torch.uint8).cuda()
    output = torch.empty((block_m, block_k), dtype=torch.uint8, device="cuda")

    program = scaled_downcast_fp4_kernel[(1, )](
        input,
        scale,
        output,
        block_m,
        block_k,
        num_warps=1,
        allow_flush_denorm=False,
    )

    # The smallest scaled inputs already underflow in FP16.
    expected_byte = 0 if dtype == torch.float16 and scale_byte < 127 else 0x42
    expected = torch.full((block_m, block_k), expected_byte, dtype=torch.uint8)
    torch.testing.assert_close(output.cpu(), expected)
    assert (f"v_cvt_scalef32_pk_fp4_{instruction_suffix}" in program.asm["amdgcn"])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("fmt", ["e2m3", "e3m2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_runtime_scaled_downcast_fp6(fmt, dtype):

    @gluon.jit
    def kernel(x_ptr, scale_ptr, out_ptr, FMT: ttgl.constexpr, M: ttgl.constexpr, K: ttgl.constexpr):
        # Each thread converts 32 consecutive values with one instruction.
        layout: ttgl.constexpr = ttgl.BlockedLayout([1, 32], [16, 4], [4, 1], [1, 0])
        scale_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [16, 4], [4, 1], [1, 0])
        offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))[:, None]
        offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, layout))[None, :]
        x = ttgl.load(x_ptr + offs_m * K + offs_k)
        scale_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_layout))[:, None]
        scale_k = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, scale_layout))[None, :]
        scale = ttgl.load(scale_ptr + scale_m * (K // 32) + scale_k)
        ttgl.store(out_ptr + offs_m * K + offs_k, ttgl.amd.cdna4.scaled_downcast(x, scale, FMT, axis=1))

    M, K = 64, 128
    scale = torch.randint(124, 131, (M, K // 32), dtype=torch.uint8, device="cuda")
    scale_f = torch.exp2(scale.double() - 127).repeat_interleave(32, dim=1)
    # Non-negative FP6 values in code order.
    fp6_magnitudes = fp6_values(fmt, torch.arange(32, device="cuda")).double()
    fp6_max = fp6_magnitudes[-1]
    x = (torch.randn((M, K), dtype=torch.float64, device="cuda") * fp6_max * scale_f).to(dtype)
    out = torch.empty((M, K), dtype=torch.uint8, device="cuda")
    compiled = kernel[(1, )](x, scale, out, fmt, M, K, num_warps=4)
    assert f"v_cvt_scalef32_2xpk16_{'fp6' if fmt == 'e2m3' else 'bf6'}_f32" in compiled.asm["amdgcn"]

    # Round x / scale to nearest, ties to even code, saturating to the largest magnitude.
    a = (x.double() / scale_f).abs()
    dist = (a[..., None] - fp6_magnitudes).abs()
    nearest = dist == dist.min(dim=-1, keepdim=True).values
    even = nearest & (torch.arange(32, device="cuda") % 2 == 0)
    code = torch.where(even.any(-1), even.int().argmax(-1), nearest.int().argmax(-1))
    code = torch.where(a >= fp6_max, 31, code) | (torch.signbit(x.float()).int() << 5)
    groups = pack_fp6(code).reshape(M, -1, 12)
    expected = torch.cat([groups, torch.zeros_like(groups[..., :4])], -1).reshape(M, K)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
