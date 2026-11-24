import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import get_current_target, is_hip_cdna5


def make_inputs(k_dim, a_type, b_type):
    torch_a = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[a_type]
    torch_b = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[b_type]
    torch.manual_seed(42)
    a = torch.randint(-4, 5, (32, k_dim)).float()
    b = torch.randint(-4, 5, (k_dim, 32)).float()
    return a @ b, a.to(torch_a).view(torch.uint8).cuda(), b.to(torch_b).view(torch.uint8).cuda()


@pytest.mark.skipif(not is_hip_cdna5(), reason="Requires CDNA5")
@pytest.mark.parametrize("k_dim", [128, 256])
@pytest.mark.parametrize("b_type", ["e4m3", "e5m2"])
@pytest.mark.parametrize("a_type", ["e4m3", "e5m2"])
def test_wmma_fallback_correctness(a_type, b_type, k_dim):
    tl_a = {"e4m3": tl.float8e4nv, "e5m2": tl.float8e5}[a_type]
    tl_b = {"e4m3": tl.float8e4nv, "e5m2": tl.float8e5}[b_type]

    @triton.jit
    def kernel(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, plain_ptr, scaled_ptr, BLOCK_M: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, A_DTYPE: tl.constexpr, B_DTYPE: tl.constexpr,
               type_a: tl.constexpr, type_b: tl.constexpr):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        offs_s = tl.arange(0, BLOCK_K // 32)
        a = tl.load(a_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :])
        a_scale = tl.load(a_scale_ptr + offs_m[:, None] * (BLOCK_K // 32) + offs_s[None, :])
        b_scale = tl.load(b_scale_ptr + offs_n[:, None] * (BLOCK_K // 32) + offs_s[None, :])
        plain = tl.dot(a.to(A_DTYPE, bitcast=True), b.to(B_DTYPE, bitcast=True))
        scaled = tl.dot_scaled(a, a_scale, type_a, b, b_scale, type_b)
        tl.store(plain_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], plain)
        tl.store(scaled_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], scaled)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = k_dim

    expected, a, b = make_inputs(k_dim, a_type, b_type)
    a_scale = torch.full((BLOCK_M, BLOCK_K // 32), 127, dtype=torch.uint8, device="cuda")
    b_scale = torch.full((BLOCK_N, BLOCK_K // 32), 127, dtype=torch.uint8, device="cuda")
    plain = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.float32, device="cuda")
    scaled = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.float32, device="cuda")

    pgm = kernel[(1, )](a, b, a_scale, b_scale, plain, scaled, BLOCK_M, BLOCK_N, BLOCK_K, tl_a, tl_b, a_type, b_type,
                        num_warps=4)

    assert "v_wmma_scale_f32_16x16x128_f8f6f4" in pgm.asm["amdgcn"]
    torch.testing.assert_close(plain.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(scaled.cpu(), expected, atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna5(), reason="Requires CDNA5")
@pytest.mark.parametrize("scale_value, factor", [(127, 1), (128, 2)])
def test_dot_scaled_unit_scale_value(scale_value, factor):

    @triton.jit
    def kernel(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_K: tl.constexpr):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        offs_s = tl.arange(0, BLOCK_K // 32)
        a = tl.load(a_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :])
        a_scale = tl.load(a_scale_ptr + offs_m[:, None] * (BLOCK_K // 32) + offs_s[None, :])
        b_scale = tl.load(b_scale_ptr + offs_n[:, None] * (BLOCK_K // 32) + offs_s[None, :])
        out = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3")
        tl.store(out_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], out)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 128

    expected, a, b = make_inputs(BLOCK_K, "e4m3", "e4m3")
    a_scale = torch.full((BLOCK_M, BLOCK_K // 32), scale_value, dtype=torch.uint8, device="cuda")
    b_scale = torch.full((BLOCK_N, BLOCK_K // 32), 127, dtype=torch.uint8, device="cuda")
    out = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.float32, device="cuda")

    pgm = kernel[(1, )](a, b, a_scale, b_scale, out, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4)

    assert "v_wmma_scale_f32_16x16x128_f8f6f4" in pgm.asm["amdgcn"]
    torch.testing.assert_close(out.cpu(), expected * factor, atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna5(), reason="Requires CDNA5")
@pytest.mark.parametrize("b_type", ["e4m3", "e5m2"])
@pytest.mark.parametrize("a_type", ["e4m3", "e5m2"])
def test_fp8_dot_x64_matches_x128(a_type, b_type):
    tl_a = {"e4m3": tl.float8e4nv, "e5m2": tl.float8e5}[a_type]
    tl_b = {"e4m3": tl.float8e4nv, "e5m2": tl.float8e5}[b_type]
    a_mnemonic = {"e4m3": "fp8", "e5m2": "bf8"}[a_type]
    b_mnemonic = {"e4m3": "fp8", "e5m2": "bf8"}[b_type]

    @triton.jit
    def kernel(a_ptr, b_ptr, full_ptr, split_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               A_DTYPE: tl.constexpr, B_DTYPE: tl.constexpr):
        HALF_K: tl.constexpr = BLOCK_K // 2
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        offs_h = tl.arange(0, HALF_K)
        a = tl.load(a_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :]).to(A_DTYPE, bitcast=True)
        b = tl.load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :]).to(B_DTYPE, bitcast=True)
        a0 = tl.load(a_ptr + offs_m[:, None] * BLOCK_K + offs_h[None, :]).to(A_DTYPE, bitcast=True)
        a1 = tl.load(a_ptr + offs_m[:, None] * BLOCK_K + HALF_K + offs_h[None, :]).to(A_DTYPE, bitcast=True)
        b0 = tl.load(b_ptr + offs_h[:, None] * BLOCK_N + offs_n[None, :]).to(B_DTYPE, bitcast=True)
        b1 = tl.load(b_ptr + (HALF_K + offs_h[:, None]) * BLOCK_N + offs_n[None, :]).to(B_DTYPE, bitcast=True)
        tl.store(full_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], tl.dot(a, b))
        tl.store(split_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], tl.dot(a1, b1, tl.dot(a0, b0)))

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 128

    expected, a, b = make_inputs(BLOCK_K, a_type, b_type)
    full = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.float32, device="cuda")
    split = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.float32, device="cuda")

    pgm = kernel[(1, )](a, b, full, split, BLOCK_M, BLOCK_N, BLOCK_K, tl_a, tl_b, num_warps=4)

    if get_current_target().arch == "gfx1250-strict":
        assert "v_wmma_scale_f32_16x16x128_f8f6f4" in pgm.asm["amdgcn"]
    else:
        assert f"v_wmma_f32_16x16x128_{b_mnemonic}_{a_mnemonic}" in pgm.asm["amdgcn"]
    assert f"v_wmma_f32_16x16x64_{b_mnemonic}_{a_mnemonic}" in pgm.asm["amdgcn"]
    torch.testing.assert_close(full.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(split.cpu(), expected, atol=0, rtol=0)
