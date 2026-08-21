import pytest
import torch

from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl
from triton._internal_testing import is_hip_cdna4


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
@pytest.mark.parametrize(
    "dtype,instruction_suffix",
    [
        (torch.float16, "f16"),
        (torch.bfloat16, "bf16"),
        (torch.float32, "f32"),
    ],
)
def test_runtime_scaled_downcast_fp4(dtype, instruction_suffix):
    block_m, block_k = 16, 32
    # Interleave 1.0 (low nibble) and 2.0 (high nibble) so every packed byte is
    # e2m1(1.0) | (e2m1(2.0) << 4) == 0x42.
    input = torch.empty((block_m, 2 * block_k), dtype=dtype)
    input[:, 0::2] = 1.0
    input[:, 1::2] = 2.0
    input = input.cuda()
    scale = torch.full((block_m, block_k // 16), 127, dtype=torch.uint8).cuda()
    output = torch.empty((block_m, block_k), dtype=torch.uint8, device="cuda")

    program = scaled_downcast_fp4_kernel[(1, )](
        input,
        scale,
        output,
        block_m,
        block_k,
        num_warps=1,
    )

    expected = torch.full((block_m, block_k), 0x42, dtype=torch.uint8)
    torch.testing.assert_close(output.cpu(), expected)
    assert (f"v_cvt_scalef32_pk_fp4_{instruction_suffix}" in program.asm["amdgcn"])
