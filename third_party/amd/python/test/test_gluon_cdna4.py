import pytest
import torch

from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl
from triton._internal_testing import is_hip_cdna4
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


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
def mfma_scaled_kernel(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, out_ptr, A_FMT: ttgl.constexpr, B_FMT: ttgl.constexpr,
                       MN: ttgl.constexpr, K: ttgl.constexpr):
    mfma_layout: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=4, instr_shape=[MN, MN, K], transposed=True,
                                                         warps_per_cta=[1, 1])
    a_layout: ttgl.constexpr = ttgl.DotOperandLayout(0, mfma_layout, 16)
    b_layout: ttgl.constexpr = ttgl.DotOperandLayout(1, mfma_layout, 16)
    KA: ttgl.constexpr = K // 2 if A_FMT == "e2m1" else K
    KB: ttgl.constexpr = K // 2 if B_FMT == "e2m1" else K

    a_m = ttgl.arange(0, MN, layout=ttgl.SliceLayout(1, a_layout))[:, None]
    a_k = ttgl.arange(0, KA, layout=ttgl.SliceLayout(0, a_layout))[None, :]
    b_k = ttgl.arange(0, KB, layout=ttgl.SliceLayout(1, b_layout))[:, None]
    b_n = ttgl.arange(0, MN, layout=ttgl.SliceLayout(0, b_layout))[None, :]
    a = ttgl.load(a_ptr + a_m * KA + a_k)
    b = ttgl.load(b_ptr + b_k * MN + b_n)
    acc = ttgl.zeros((MN, MN), ttgl.float32, layout=mfma_layout)

    if a_scale_ptr is not None:
        a_scale_layout: ttgl.constexpr = ttgl.amd.cdna4.get_mfma_scale_layout(a_layout, [MN, K // 32])
        b_scale_layout: ttgl.constexpr = ttgl.amd.cdna4.get_mfma_scale_layout(b_layout, [MN, K // 32])
        sm = ttgl.arange(0, MN, layout=ttgl.SliceLayout(1, a_scale_layout))[:, None]
        sk = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, a_scale_layout))[None, :]
        a_scale = ttgl.load(a_scale_ptr + sm * (K // 32) + sk)
        sn = ttgl.arange(0, MN, layout=ttgl.SliceLayout(1, b_scale_layout))[:, None]
        sk = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, b_scale_layout))[None, :]
        b_scale = ttgl.load(b_scale_ptr + sn * (K // 32) + sk)
        result = ttgl.amd.cdna4.mfma_scaled(a, a_scale, A_FMT, b, b_scale, B_FMT, acc)
    else:
        result = ttgl.amd.cdna4.mfma_scaled(a, None, A_FMT, b, None, B_FMT, acc)

    out_m = ttgl.arange(0, MN, layout=ttgl.SliceLayout(1, mfma_layout))[:, None]
    out_n = ttgl.arange(0, MN, layout=ttgl.SliceLayout(0, mfma_layout))[None, :]
    ttgl.store(out_ptr + out_m * MN + out_n, result)


def _mx_operand(logical_shape, fmt, pack_dim):
    if fmt == "e4m3":
        raw = torch.randint(20, 40, logical_shape, dtype=torch.uint8)
        return raw, raw.view(torch.float8_e4m3fn).to(torch.float32)
    if fmt == "e5m2":
        raw = torch.randint(20, 40, logical_shape, dtype=torch.uint8)
        return raw, raw.view(torch.float8_e5m2).to(torch.float32)
    mx = MXFP4Tensor(size=logical_shape).random()
    return mx.to_packed_tensor(pack_dim), mx.to(torch.float32)


def _mx_scale(rows, k):
    scale = MXScaleTensor(size=(rows, k // 32)).random(1 / 32, 32)
    return scale.data, scale.to(torch.float32).repeat_interleave(32, dim=1)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("mn, k", [(32, 64), (16, 128)])
@pytest.mark.parametrize("a_fmt, b_fmt", [("e4m3", "e4m3"), ("e2m1", "e2m1"), ("e5m2", "e2m1"), ("e2m1", "e4m3")])
@pytest.mark.parametrize("has_scale", [False, True])
def test_runtime_mfma_scaled(has_scale, a_fmt, b_fmt, mn, k):
    torch.manual_seed(0)
    # Non-uniform payloads: a constant 1.0 tile hides a wrong dot-operand or scale layout.
    a, a_ref = _mx_operand((mn, k), a_fmt, pack_dim=1)
    b, b_ref = _mx_operand((k, mn), b_fmt, pack_dim=0)
    a = a.cuda()
    b = b.cuda()
    if has_scale:
        a_scale, a_scale_ref = _mx_scale(mn, k)
        b_scale, b_scale_ref = _mx_scale(mn, k)
        a_scale = a_scale.cuda()
        b_scale = b_scale.cuda()
        expected = torch.matmul(a_ref * a_scale_ref, b_ref * b_scale_ref.T.contiguous())
    else:
        a_scale = b_scale = None
        expected = torch.matmul(a_ref, b_ref)
    output = torch.empty((mn, mn), dtype=torch.float32, device="cuda")

    program = mfma_scaled_kernel[(1, )](a, b, a_scale, b_scale, output, a_fmt, b_fmt, mn, k, num_warps=1)

    torch.testing.assert_close(output.cpu(), expected)

    # Without scales the non-scaled F8F6F4 MFMA must be used, otherwise the scaled one.
    shape = f"{mn}x{mn}x{k}"
    amdgcn = program.asm["amdgcn"]
    if has_scale:
        assert f"v_mfma_scale_f32_{shape}_f8f6f4" in amdgcn
    else:
        assert f"v_mfma_f32_{shape}_f8f6f4" in amdgcn
        assert "v_mfma_scale" not in amdgcn
