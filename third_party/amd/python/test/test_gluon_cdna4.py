import pytest
import torch

from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl
from triton._internal_testing import create_mxfp_operand, is_hip_cdna4


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


@gluon.jit
def scaled_downcast_layout_kernel(x_ptr, scale_ptr, y_ptr, BLOCK_M: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                                  OUT_K: ttgl.constexpr, SCALE_FACTOR: ttgl.constexpr, FORMAT: ttgl.constexpr,
                                  LAYOUT: ttgl.constexpr, SCALE_LAYOUT: ttgl.constexpr):
    offs_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, LAYOUT))
    offs_k = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, LAYOUT))
    offs = offs_m[:, None] * BLOCK_K + offs_k[None, :]
    x = ttgl.load(x_ptr + offs)

    scale_k: ttgl.constexpr = BLOCK_K // SCALE_FACTOR
    offs_scale_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, SCALE_LAYOUT))
    offs_scale_k = ttgl.arange(0, scale_k, layout=ttgl.SliceLayout(0, SCALE_LAYOUT))
    scale = ttgl.load(scale_ptr + offs_scale_m[:, None] * scale_k + offs_scale_k[None, :])

    y = ttgl.amd.cdna4.scaled_downcast(x, scale, FORMAT, axis=1)
    out_layout: ttgl.constexpr = y.type.layout
    offs_out_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, out_layout))
    offs_out_k = ttgl.arange(0, OUT_K, layout=ttgl.SliceLayout(0, out_layout))
    ttgl.store(y_ptr + offs_out_m[:, None] * OUT_K + offs_out_k[None, :], y)


def per_lane_layouts(scale_factor):
    # Each lane holds one whole scale block.
    layout = ttgl.BlockedLayout([1, scale_factor], [8, 8], [4, 1], [1, 0])
    scale_layout = ttgl.BlockedLayout([1, 1], [8, 8], [4, 1], [1, 0])
    return layout, scale_layout


def repeated_register_layouts(scale_factor):
    # Register 1 repeats the element of register 0, so each thread holds the
    # unique columns 0, 1, 8 and 9 of its rows. Columns 8 and 9 are in a
    # different scale block from columns 0 and 1, and the lowering must look up
    # their scale by unique value rather than by register.
    reg_bases = [[0, 1], [0, 0], [0, 8], [8, 0], [16, 0]]
    lane_bases = [[0, 2], [0, 4], [0, 16], [0, 32], [0, 64], [1, 0]]
    warp_bases = [[2, 0], [4, 0]]

    # Scale column c // scale_factor serves input column c.
    def to_scale(bases):
        return [[row, col // scale_factor] for row, col in bases]

    layout = ttgl.DistributedLinearLayout(reg_bases, lane_bases, warp_bases, [], [32, 128])
    scale_layout = ttgl.DistributedLinearLayout(to_scale(reg_bases), to_scale(lane_bases), to_scale(warp_bases), [],
                                                [32, 128 // scale_factor])
    return layout, scale_layout


# Each entry maps a scale factor to the input and scale layouts of a 32x128
# tile with 4 warps.
SCALED_DOWNCAST_LAYOUTS = {
    "per_lane": per_lane_layouts,
    "repeated_register": repeated_register_layouts,
}


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("layouts", SCALED_DOWNCAST_LAYOUTS)
@pytest.mark.parametrize("scale_factor", [2, 4, 8])
@pytest.mark.parametrize("format,out_dtype,out_mnemonic", [
    ("e2m1", torch.uint8, "fp4"),
    ("e4m3", torch.float8_e4m3fn, "fp8"),
    ("e5m2", torch.float8_e5m2, "bf8"),
])
@pytest.mark.parametrize("dtype,in_suffix", [
    (torch.float16, "f16"),
    (torch.bfloat16, "bf16"),
    (torch.float32, "f32"),
])
def test_runtime_scaled_downcast_layout(layouts, scale_factor, format, out_dtype, out_mnemonic, dtype, in_suffix):
    # Blocks of fewer than 8 values fail the CDNA5 pk8 rule, but the CDNA4
    # instructions convert one pair at a time, so these layouts are valid on
    # CDNA4.
    BLOCK_M, BLOCK_K = 32, 128
    layout, scale_layout = SCALED_DOWNCAST_LAYOUTS[layouts](scale_factor)
    torch.manual_seed(0)
    expected, values_ref = create_mxfp_operand(0, BLOCK_M, BLOCK_K, format)
    # Each scale block of a row gets a different scale, so a scale that is applied
    # to the wrong block changes the output.
    num_blocks = BLOCK_K // scale_factor
    scale = (124 + torch.arange(BLOCK_M * num_blocks) % 7).to(torch.uint8).reshape(BLOCK_M, num_blocks)
    scale_ref = torch.exp2((scale.to(torch.int16) - 127).to(torch.float32))
    scaled_blocks = values_ref.reshape(BLOCK_M, num_blocks, scale_factor) * scale_ref[:, :, None]
    x = scaled_blocks.reshape(BLOCK_M, BLOCK_K).to(dtype)

    out_k = BLOCK_K // 2 if format == "e2m1" else BLOCK_K
    y = torch.empty((BLOCK_M, out_k), dtype=out_dtype, device="cuda")
    program = scaled_downcast_layout_kernel[(1, )](x.cuda(), scale.cuda(), y, BLOCK_M, BLOCK_K, out_k, scale_factor,
                                                   format, layout, scale_layout, num_warps=4)

    assert f"v_cvt_scalef32_pk_{out_mnemonic}_{in_suffix}" in program.asm["amdgcn"]
    torch.testing.assert_close(y.view(torch.uint8).cpu(), expected, atol=0, rtol=0)
