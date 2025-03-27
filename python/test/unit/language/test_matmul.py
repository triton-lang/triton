import math
import pytest
import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
from test_mxfp import MXFP4Tensor, MXScaleTensor
import re
from triton._internal_testing import is_cuda, is_hip, is_hip_cdna3, is_hip_cdna4, is_hip_cdna


def f8_to_f16(x, dtype):

    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, x, mask=mask)

    ret = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )
    dtype = getattr(tl, dtype)
    kernel[grid](ret, triton.reinterpret(x, dtype), ret.numel(), BLOCK_SIZE=1024)
    return ret


@triton.jit
def matmul_kernel(  #
        a_ptr, b_ptr, output_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr, SCALE_A: tl.constexpr = None, PRECISION: tl.constexpr = "ieee"):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        if SCALE_A is not None:
            a = a * SCALE_A
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, acc=accumulator, out_dtype=output_ptr.dtype.element_ty, input_precision=PRECISION)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(output_ptrs, accumulator, mask=mask_c)


def get_src_element_ty_size(dtype_str):
    if dtype_str == "float8e5":
        return 1
    if dtype_str == "float16":
        return 2
    if dtype_str == "float32" or dtype_str == "tensorfloat32":
        return 4
    raise ValueError(f"Unknown dtype {dtype_str}")


@pytest.mark.parametrize("dtype_src_str", ["float32", "tensorfloat32", "float16", "float8e5"])
@pytest.mark.parametrize("dtype_dst_str", ["float32", "float16"])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES", [(128, 128, 16, 4), (64, 128, 32, 4), (32, 32, 32, 4),
                                                                   (256, 128, 32, 4), (64, 512, 32, 2),
                                                                   (512, 64, 32, 2), (64, 16, 16, 4)])
@pytest.mark.parametrize("NUM_CTAS", [1, 2])
@pytest.mark.parametrize("NUM_WARPS", [4, 8])
def test_simple_matmul(dtype_src_str, dtype_dst_str, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, NUM_WARPS, NUM_CTAS,
                       device):
    if NUM_CTAS > 1 and (not is_cuda() or torch.cuda.get_device_capability()[0] < 9):
        pytest.skip("Clusters requires nvidia compute capability >= 9")
    if is_hip() and ((BLOCK_K * BLOCK_M + BLOCK_K * BLOCK_N) * NUM_STAGES * get_src_element_ty_size(dtype_src_str)
                     > 65536):
        pytest.skip("HIP path requires less than 64KB of shared memory")
    if is_hip() and (not is_hip_cdna3()) and dtype_src_str == "tensorfloat32":
        pytest.skip("tensorfloat32 is only supported on HIP CDNA3")
    if dtype_src_str == "float8e5" and BLOCK_K == 16:
        pytest.skip("Skipping cases small K for float8")
    if dtype_src_str == "float8e5" and device == "cuda" and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Float8 requires compute capability >= 9")
    if "float32" in dtype_src_str and dtype_dst_str == "float16":
        pytest.skip("Skipping unsupported case")
    if "float32" == dtype_src_str and NUM_CTAS > 1:
        pytest.skip("FMA matmul not supported for multiple CTAs")
    if (BLOCK_M < 64 or (BLOCK_M == 64 and BLOCK_N == 16)) and NUM_CTAS > 1:
        pytest.skip("multi-CTAs is broken for mmav2")
    M, N, K = 1024, 512, 256
    torch.manual_seed(42)
    precision = "tf32" if dtype_src_str == "tensorfloat32" else "ieee"
    dtype_src_str = "float32" if dtype_src_str == "tensorfloat32" else dtype_src_str
    if dtype_src_str == "float8e5":
        a = torch.randint(20, 40, (M, K), dtype=torch.int8, device=device).view(torch.float8_e5m2)
        b = torch.randint(20, 40, (K, N), dtype=torch.int8, device=device).view(torch.float8_e5m2)
        A = f8_to_f16(a, dtype_src_str)
        B = f8_to_f16(b, dtype_src_str)
    else:
        dtype_src = getattr(torch, dtype_src_str)
        a = torch.randn(M, K, dtype=dtype_src, device=device)
        b = torch.randn(K, N, dtype=dtype_src, device=device)
        A = a
        B = b
    dtype_dst = getattr(torch, dtype_dst_str)
    output = torch.empty((M, N), dtype=dtype_dst, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    k = matmul_kernel[grid](a, b, output, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), output.stride(0),
                            output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES=NUM_STAGES, PRECISION=precision,
                            num_warps=NUM_WARPS, num_ctas=NUM_CTAS)
    ref_out = torch.matmul(A, B).to(torch.float32)
    output = output.to(torch.float32)
    if dtype_src_str == "float32":
        # TF32 has lower precision than torch.float32
        atol = 0.03
        rtol = 0.03
    elif dtype_dst_str == "float16":
        atol = 0.06
        rtol = 0.06
    else:
        atol = 0.01
        rtol = 0.01
    torch.testing.assert_close(ref_out, output, atol=atol, rtol=rtol)
    # Make sure the mma is pipelined by checking if in the TTGIR we are waiting for the
    # barrier coming from the loop args (previous iteration).
    # This applies only if TCv5 MMA is used (M % 64 == 0 and N % 8 == 0) and
    # when MMA arguments loads are pipelined (N > 16)
    if (device == "cuda" and torch.cuda.get_device_capability()[0] == 10 and NUM_STAGES > 1 and BLOCK_M % 64 == 0
            and BLOCK_N % 8 == 0 and BLOCK_N > 16 and not (precision == "ieee" and dtype_src_str == "float32")):
        ttgir = k.asm["ttgir"]
        pattern = (r"ttng.wait_barrier %arg")
        assert re.search(pattern, str(ttgir)), "The TTGIR does not match the expected pattern."


# persistent matmul with fused loops
@triton.jit
def simple_persistent_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,  #
                             stride_bk, stride_bn,  #
                             stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                             BLOCK_SIZE_K: tl.constexpr,  #
                             GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr,
                             DISALLOW_ACC_MULTI_BUFFER: tl.constexpr):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    tile_id_c = start_pid - NUM_SMS  # remat value to use in the epilogue
    ki = -1

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in tl.range(0, k_tiles * tiles_per_SM, disallow_acc_multi_buffer=DISALLOW_ACC_MULTI_BUFFER):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCK_SIZE_M
            start_n = pid_n * BLOCK_SIZE_N
            offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
            offs_am = tl.where(offs_am < M, offs_am, 0)
            offs_bn = tl.where(offs_bn < N, offs_bn, 0)
            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        if ki == k_tiles - 1:
            tile_id_c += NUM_SMS
            group_id = tile_id_c // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id_c % group_size_m)
            pid_n = (tile_id_c % num_pid_in_group) // group_size_m

            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            if (c_ptr.dtype == tl.float8e4nv):
                c = accumulator.to(tl.float8e4nv)
            else:
                c = accumulator.to(tl.float16)
            tl.store(c_ptrs, c, mask=c_mask)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 128, 16), (64, 128, 32), (32, 32, 32), (256, 128, 16),
                                                       (64, 512, 16), (512, 64, 16), (64, 16, 16)])
@pytest.mark.parametrize("NUM_WARPS", [4, 8])
@pytest.mark.parametrize("DISALLOW_ACC_MULTI_BUFFER", [True, False])
def test_simple_persistent_matmul(BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, DISALLOW_ACC_MULTI_BUFFER, device):
    M, N, K = 1024, 512, 256
    NUM_STAGES = 3
    a = torch.randn(M, K, dtype=torch.float16, device=device)
    b = torch.randn(K, N, dtype=torch.float16, device=device)
    output = torch.empty((M, N), dtype=torch.float16, device=device)

    # Fake small number of SMS to test that persistent kernel works reliably
    NUM_SMS = 8

    grid = (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), )
    k = simple_persistent_kernel[grid](
        a, b, output,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        output.stride(0), output.stride(1),  #
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,  #
        GROUP_SIZE_M=8, NUM_SMS=NUM_SMS, DISALLOW_ACC_MULTI_BUFFER=DISALLOW_ACC_MULTI_BUFFER, num_stages=NUM_STAGES,
        num_warps=NUM_WARPS)
    ref_out = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(torch.float16)

    torch.testing.assert_close(ref_out, output, atol=0.01, rtol=0.01)

    # Make sure the mma is pipelined by checking if in the TTGIR we are waiting for the
    # barrier coming from the loop args (previous iteration).
    # This applies only if TCv5 MMA is used (M % 64 == 0 and N % 8 == 0) and
    # when MMA arguments loads are pipelined (N > 16)
    if (device == "cuda" and torch.cuda.get_device_capability()[0] == 10 and BLOCK_M % 64 == 0 and BLOCK_N % 8 == 0
            and BLOCK_N > 16):
        ttgir = k.asm["ttgir"]
        pattern = "ttng.wait_barrier %arg"
        assert ttgir.count(pattern) > 0, "Expect barrier coming from the previous iteration."


@triton.jit
def mxfp_matmul(  #
        a_ptr, b_ptr, output_ptr,  #
        a_scale, b_scale,  #
        M, N, K,  #
        stride_scale: tl.constexpr,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    offs_scale_k = tl.arange(0, BLOCK_K // 32)
    a_scale_ptr = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
    b_scale_ptr = b_scale + offs_bn[:, None] * stride_scale + offs_scale_k[None, :]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        k_remaining = K - k * BLOCK_K
        valid_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=valid_k[None, :], other=0.)
        b = tl.load(b_ptrs, mask=valid_k[:, None], other=0.)
        scale_a = tl.load(a_scale_ptr)
        scale_b = tl.load(b_scale_ptr)
        accumulator = tl.dot_scaled(a, scale_a, "e5m2", b, scale_b, "e5m2", accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        a_scale_ptr += BLOCK_K // 32
        b_scale_ptr += BLOCK_K // 32
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)


def fp8e8m0_to_float32(scale):
    scale = scale.view(torch.uint8)
    scale = scale.to(torch.int32)
    scale = scale << 23
    scale = scale.view(torch.float32)
    return scale


@pytest.mark.parametrize("M, N, K", [(1024, 512, 256), (128, 256, 256), (128, 128, 128), (2, 4, 32), (2, 4, 64),
                                     (256, 16, 32)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 128, 128), (256, 128, 128), (128, 256, 128),
                                                       (128, 256, 256), (128, 128, 64), (128, 64, 128)])
@pytest.mark.parametrize("NUM_STAGES", [1, 3])
@pytest.mark.parametrize("NUM_WARPS", [4, 8])
@pytest.mark.parametrize("nonKDim", ([0, 16, 32] if is_hip_cdna() else [0]))
def test_mxfp(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, nonKDim, NUM_WARPS, device):
    if is_cuda() and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Requires compute capability >= 10")
    elif is_hip():
        if not is_hip_cdna4():
            pytest.skip("Scaled mxfp8 matmul is only natively supported on CDNA4")
        if (M == 2 and N == 4 and K == 32) or (M == 256 and N == 16 and K == 32):
            pytest.skip(f"Input shape {M=}, {N=}, {K=} is not supported yet")
        if (nonKDim == 16 and BLOCK_K < 128) or (nonKDim == 32 and BLOCK_K < 64):
            pytest.skip(f"CDNA4 does not support {BLOCK_K=} for scaled mfma {nonKDim=} variants")

    if BLOCK_N == 256 and BLOCK_K == 256:
        NUM_STAGES = min(NUM_STAGES, 2)
    torch.manual_seed(42)
    dtype_src_str = "float8e5"
    dtype_dst_str = "float32"
    a = torch.randint(20, 40, (M, K), dtype=torch.uint8, device=device).view(torch.float8_e5m2)
    a_f16 = f8_to_f16(a, dtype_src_str)
    b = torch.randint(20, 40, (K, N), dtype=torch.uint8, device=device).view(torch.float8_e5m2)
    b_f16 = f8_to_f16(b, dtype_src_str)
    a_scale = torch.randint(130, (M, K // 32), dtype=torch.uint8, device=device)
    b_scale = torch.randint(130, (N, K // 32), dtype=torch.uint8, device=device)

    dtype_dst = getattr(torch, dtype_dst_str)
    output = torch.empty((M, N), dtype=dtype_dst, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    kernel_kwargs = {}
    if is_hip():
        kernel_kwargs["matrix_instr_nonkdim"] = nonKDim
    out = mxfp_matmul[grid](a, b, output, a_scale, b_scale, M, N, K, a_scale.stride(0), a.stride(0), a.stride(1),
                            b.stride(0), b.stride(1), output.stride(0), output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K,
                            NUM_STAGES=NUM_STAGES, **kernel_kwargs, num_warps=NUM_WARPS)
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)
    a_scale_f32 = a_scale_f32.repeat_interleave(32, dim=1)
    b_scale_f32 = b_scale_f32.repeat_interleave(32, dim=1)

    # b_scales are always col major
    b_scale_f32 = b_scale_f32.T.contiguous()

    a = a_f16 * a_scale_f32
    b = b_f16 * b_scale_f32
    ref_out = torch.matmul(a, b).to(torch.float32)
    output = output.to(torch.float32)
    atol = 0.0001
    rtol = 0.0001
    torch.testing.assert_close(ref_out, output, atol=atol, rtol=rtol)

    # Pipelining of dot_scaled requires tmem_copy to be used, which in turn
    # requires the scales to be in the blocked layout in global memory.
    assert "ttng.wait_barrier" not in out.asm["ttgir"]


def _knob_promote_lhs_to_tmem(monkeypatch):
    # Promoting the LHS to TMEM should be patched because it will otherwise
    # unintentionally be enabled for all consecutive tests if using os.environ
    monkeypatch.setenv("ALLOW_LHS_TMEM_LAYOUT_CONVERSION", "1")


@triton.jit
def block_scale_mxfp_matmul(  #
        a_ptr, b_ptr, output_ptr,  #
        a_scale, b_scale,  #
        M, N, K,  #
        stride_sk, stride_sb, stride_sc, stride_sd: tl.constexpr,  # Need tl.constexpr to pipeline scale load. Why?
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr, USE_2D_SCALE_LOAD: tl.constexpr):
    ## This kernel assumes a_scale and b_scale are coming in with shapes
    ## [BLOCK_M(or N) // 128, BLOCK_K // 128, 32, 4, 4] for optimial performance
    ## on nvidia sm100+ HW
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    offs_sm = (pid_m * (BLOCK_M // 128) + tl.arange(0, BLOCK_M // 128))
    offs_sn = (pid_n * (BLOCK_N // 128) + tl.arange(0, BLOCK_N // 128))

    if USE_2D_SCALE_LOAD:
        offs_inner = tl.arange(0, (BLOCK_K // 128) * 32 * 4 * 4)
        a_scale_ptr = a_scale + offs_sm[:, None] * stride_sk + offs_inner[None, :]
        b_scale_ptr = b_scale + offs_sn[:, None] * stride_sk + offs_inner[None, :]
    else:
        offs_sk = tl.arange(0, (BLOCK_K // 128))
        offs_sc = tl.arange(0, 32)
        offs_sd = tl.arange(0, 4)
        a_scale_ptr = a_scale + (offs_sm[:, None, None, None, None] * stride_sk + offs_sk[None, :, None, None, None] *
                                 stride_sb + offs_sc[None, None, :, None, None] * stride_sc +
                                 offs_sd[None, None, None, :, None] * stride_sd + offs_sd[None, None, None, None, :])
        b_scale_ptr = b_scale + (offs_sn[:, None, None, None, None] * stride_sk + offs_sk[None, :, None, None, None] *
                                 stride_sb + offs_sc[None, None, :, None, None] * stride_sc +
                                 offs_sd[None, None, None, :, None] * stride_sd + offs_sd[None, None, None, None, :])

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        scale_a = tl.load(a_scale_ptr)
        scale_b = tl.load(b_scale_ptr)

        if USE_2D_SCALE_LOAD:
            scale_a = scale_a.reshape(BLOCK_M // 128, BLOCK_K // 128, 32, 4, 4)
            scale_b = scale_b.reshape(BLOCK_N // 128, BLOCK_K // 128, 32, 4, 4)

        # Scales are comming in for optimial peformance, but we reshape here for
        # the canonical inputs to dot_scaled
        # These reshapes and transposes will be optimized away during lowering
        scale_a = scale_a.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // 32)
        scale_b = scale_b.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // 32)
        accumulator = tl.dot_scaled(a, scale_a, "e5m2", b, scale_b, "e5m2", accumulator)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        a_scale_ptr += BLOCK_K // 128 * stride_sb
        b_scale_ptr += BLOCK_K // 128 * stride_sb
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)


@pytest.mark.parametrize("M, N, K", [(1024, 512, 512), (998, 111, 512), (63, 128, 512)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 128, 128), (256, 128, 128), (128, 256, 128),
                                                       (128, 128, 256), (128, 256, 256)])
@pytest.mark.parametrize("NUM_STAGES", [1, 2, 4])
@pytest.mark.parametrize("USE_2D_SCALE_LOAD", [False, True])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 10, reason="Requires compute capability >= 10")
def test_blocked_scale_mxfp(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, USE_2D_SCALE_LOAD, device, monkeypatch):
    if BLOCK_N == 256 and BLOCK_K == 256:
        NUM_STAGES = min(NUM_STAGES, 2)
    elif BLOCK_K == 256:
        NUM_STAGES = min(NUM_STAGES, 3)

    torch.manual_seed(42)
    dtype_src_str = "float8e5"
    dtype_dst_str = "float32"
    a = torch.randint(20, 40, (M, K), dtype=torch.uint8, device=device).view(torch.float8_e5m2)
    A = f8_to_f16(a, dtype_src_str)
    b = torch.randint(20, 40, (K, N), dtype=torch.uint8, device=device).view(torch.float8_e5m2)
    B = f8_to_f16(b, dtype_src_str)
    ceildiv = lambda a, b: math.ceil(a / b)
    a_scale = torch.randint(130, (ceildiv(M, 128), ceildiv(K, 128), 32, 4, 4), dtype=torch.uint8).to(device)
    b_scale = torch.randint(130, (ceildiv(N, 128), ceildiv(K, 128), 32, 4, 4), dtype=torch.uint8).to(device)

    dtype_dst = getattr(torch, dtype_dst_str)
    output = torch.empty((M, N), dtype=dtype_dst, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    out = block_scale_mxfp_matmul[grid](a, b, output, a_scale, b_scale, M, N, K, a_scale.stride(0), a_scale.stride(1),
                                        a_scale.stride(2), a_scale.stride(3), a.stride(0), a.stride(1), b.stride(0),
                                        b.stride(1), output.stride(0), output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K,
                                        NUM_STAGES=NUM_STAGES, USE_2D_SCALE_LOAD=USE_2D_SCALE_LOAD)
    ttgir = out.asm["ttgir"]
    ptx = out.asm["ptx"]

    def flatten_scale(scale):
        num_chunk_m, num_chunk_k, _, _, _ = scale.shape
        return scale.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()

    a_scale_f32 = flatten_scale(fp8e8m0_to_float32(a_scale))[:M]
    b_scale_f32 = flatten_scale(fp8e8m0_to_float32(b_scale))[:N]
    a_scale_f32 = a_scale_f32.repeat_interleave(32, dim=1)
    b_scale_f32 = b_scale_f32.repeat_interleave(32, dim=1)

    # b_scales are always col major
    b_scale_f32 = b_scale_f32.T.contiguous()

    a = A * a_scale_f32
    b = B * b_scale_f32
    ref_out = torch.matmul(a, b).to(torch.float32)
    output = output.to(torch.float32)
    atol = 0.0001
    rtol = 0.0001
    torch.testing.assert_close(ref_out, output, atol=atol, rtol=rtol)

    if USE_2D_SCALE_LOAD:
        # Due to an issue in the coalescing pass, tmem_copy can not be generated for the 5D load.
        # The issue is fixed using the patch from https://github.com/triton-lang/triton/pull/4914
        assert "tcgen05.cp" in ptx
    if NUM_STAGES > 1:
        if BLOCK_M == BLOCK_K and BLOCK_N == BLOCK_K:
            load_pipelined = ttgir.count(f"ttg.local_alloc : () -> !ttg.memdesc<{NUM_STAGES}x{BLOCK_M}x{BLOCK_K}") == 2
        else:
            load_pipelined = (ttgir.count(f"ttg.local_alloc : () -> !ttg.memdesc<{NUM_STAGES}x{BLOCK_M}x{BLOCK_K}")
                              and ttgir.count(f"ttg.local_alloc : () -> !ttg.memdesc<{NUM_STAGES}x{BLOCK_K}x{BLOCK_N}"))

        if load_pipelined and USE_2D_SCALE_LOAD:
            # If load is pipelined and tmem_copy is used,  MMA pipelining should also kick in
            assert "ttng.wait_barrier" in ttgir
        elif not load_pipelined:
            # The behavior of load pipelining seems to depend on the size of input tensors.
            # In this test, it fails to pipeline the RHS tensor when N is not a multiple of 128. Pipelining of the LHS tensor
            # does not seem to be affected by the value of M, though.
            print(f"SWP failed for M = {M}, N = {N}")


@triton.jit
def lhs_in_tmem_kernel(  #
        a_ptr, b_ptr, output_ptr,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, A_TRANS: tl.constexpr, BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    if not A_TRANS:
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    else:
        a_ptrs = a_ptr + (offs_k[:, None] * stride_am + offs_am[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=1):
        k_remaining = K - k * BLOCK_K
        valid_k = offs_k < k_remaining
        m_remaining = M - pid_m * BLOCK_M
        valid_m = offs_am < m_remaining
        a = tl.load(a_ptrs, mask=(valid_k[None, :] & valid_m[:, None]), other=0.0)
        if A_TRANS:
            a = a.T
        n_remaining = N - pid_n * BLOCK_N
        valid_n = offs_bn < n_remaining
        b = tl.load(b_ptrs, mask=(valid_k[:, None] & valid_n[None, :]), other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=mask_c)


@pytest.mark.parametrize("M, N, K", [(128, 64, 64), (1024, 512, 256)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 128, 128), (256, 128, 128), (128, 256, 128),
                                                       (128, 256, 256), (128, 128, 64), (128, 64, 128)])
@pytest.mark.parametrize("a_trans", [False, True])
@pytest.mark.parametrize("dtype_src_str", ["float32", "float16", "float8e5"])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 10, reason="Requires compute capability >= 10")
def test_lhs_in_tmem(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, a_trans, dtype_src_str, device, monkeypatch):
    _knob_promote_lhs_to_tmem(monkeypatch)
    if M != BLOCK_M or N != BLOCK_N or K != BLOCK_K:
        # TODO: Make LHS TMEM promotion work for all problem sizes regardless of block dims
        pytest.xfail(
            "LHS TMEM promotion produces incorrect results when the workload dimensions are not equal to the block dims"
        )
    torch.manual_seed(42)
    if dtype_src_str == "float8e5":
        a = torch.randint(20, 40, (M, K), dtype=torch.int8, device=device).view(torch.float8_e5m2)
        if (a_trans):
            a = a.T
        b = torch.randint(20, 40, (K, N), dtype=torch.int8, device=device).view(torch.float8_e5m2)
        A = f8_to_f16(a, dtype_src_str)
        B = f8_to_f16(b, dtype_src_str)
    else:
        dtype_src = getattr(torch, dtype_src_str)
        a = torch.randn(M, K, dtype=dtype_src, device=device)
        if (a_trans):
            a = a.T
        b = torch.randn(K, N, dtype=dtype_src, device=device)
        A = a
        B = b
    output = torch.empty((M, N), dtype=torch.float16, device=device)
    grid = (1, 1)
    lhs_in_tmem_kernel[grid](a, b, output, a.stride(0), a.stride(1), b.stride(0), b.stride(1), output.stride(0),
                             output.stride(1), M, N, K, A_TRANS=a_trans, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                             BLOCK_K=BLOCK_K)
    ref_out = torch.matmul(A if not a_trans else A.T, B).to(torch.float16)

    atol = 0.03
    rtol = 0.03
    torch.testing.assert_close(ref_out, output, atol=atol, rtol=rtol)


@triton.jit
def lhs_in_tmem_kernel_mxfp(  #
        a_ptr, b_ptr, output_ptr,  #
        a_scale, b_scale,  #
        stride_scale,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_am = tl.arange(0, M)
    offs_bn = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    offs_scale_k = tl.arange(0, K // 32)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    a_scale_ptr = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
    b_scale_ptr = b_scale + offs_bn[:, None] * stride_scale + offs_scale_k[None, :]
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    scale_a = tl.load(a_scale_ptr)
    scale_b = tl.load(b_scale_ptr)
    accumulator = tl.dot_scaled(a, scale_a, "e5m2", b, scale_b, "e5m2")
    offs_cm = tl.arange(0, M)
    offs_cn = tl.arange(0, N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(output_ptrs, accumulator)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 10, reason="Requires compute capability >= 10")
def test_lhs_in_tmem_mxfp(device, monkeypatch):
    _knob_promote_lhs_to_tmem(monkeypatch)
    M, N, K = 128, 64, 32
    torch.manual_seed(42)
    a = torch.randint(20, 40, (M, K), dtype=torch.uint8, device=device)
    b = torch.randint(20, 40, (K, N), dtype=torch.uint8, device=device)
    A = f8_to_f16(a, "float8e5")
    B = f8_to_f16(b, "float8e5")
    a_scale = torch.randint(124, 130, (M, K // 32), dtype=torch.uint8, device=device)
    b_scale = torch.randint(124, 130, (N, K // 32), dtype=torch.uint8, device=device)
    output = torch.empty((M, N), dtype=torch.float16, device=device)
    grid = (1, 1)
    lhs_in_tmem_kernel_mxfp[grid](a, b, output, a_scale, b_scale, a_scale.stride(0), a.stride(0), a.stride(1),
                                  b.stride(0), b.stride(1), output.stride(0), output.stride(1), M, N, K)
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)
    a_scale_f32 = a_scale_f32.repeat_interleave(32, dim=1)
    b_scale_f32 = b_scale_f32.repeat_interleave(32, dim=1)

    # b_scales are always col major
    b_scale_f32 = b_scale_f32.T.contiguous()

    a = A * a_scale_f32
    b = B * b_scale_f32
    ref_out = torch.matmul(a, b).to(torch.float16)
    atol = 0.003
    rtol = 0.003
    torch.testing.assert_close(ref_out, output, atol=atol, rtol=rtol)


@triton.jit
def block_scale_fp4_matmul(  #
        a_ptr, b_ptr, output_ptr,  #
        a_scale, b_scale,  #
        M, N, K,  #
        stride_scale,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr):  #
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    # Two e2m1 values per K
    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_scale_k = tl.arange(0, BLOCK_K // VEC_SIZE)
    if a_scale is not None:
        a_scale_ptr = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
    if b_scale is not None:
        b_scale_ptr = b_scale + offs_bn[:, None] * stride_scale + offs_scale_k[None, :]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        k_remaining = tl.cdiv(K - k * BLOCK_K, 2)
        valid_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=valid_k[None, :], other=0)
        b = tl.load(b_ptrs, mask=valid_k[:, None], other=0)
        if a_scale is not None:
            scale_a = tl.load(a_scale_ptr)
        else:
            scale_a = None
        if b_scale is not None:
            scale_b = tl.load(b_scale_ptr)
        else:
            scale_b = None
        accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", accumulator)
        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        if a_scale is not None:
            a_scale_ptr += BLOCK_K // VEC_SIZE
        if b_scale is not None:
            b_scale_ptr += BLOCK_K // VEC_SIZE
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)


@pytest.mark.parametrize("M, N, K", [(1024, 512, 256), (2, 4, 64)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 128, 128), (256, 128, 128), (128, 256, 128),
                                                       (128, 256, 256), (128, 128, 64), (128, 64, 128)])
@pytest.mark.parametrize("with_a_scale", [True, False])
@pytest.mark.parametrize("with_b_scale", [True, False])
@pytest.mark.parametrize(("scale_type", "VEC_SIZE"), [("float8_e8m0fnu", 32), ("float8_e4m3fn", 16)],
                         ids=["mxfp4", "nvfp4"])
@pytest.mark.parametrize("nonKDim", ([0, 16, 32] if is_hip_cdna() else [0]))
def test_block_scale_fp4(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, VEC_SIZE, with_a_scale, with_b_scale, scale_type, nonKDim,
                         device):
    if is_cuda():
        if torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("Requires compute capability >= 10")
        if not (with_a_scale and with_b_scale):
            pytest.skip("None aScale/bScale is only tested on AMD backend for now")
    elif is_hip():
        if not is_hip_cdna4():
            pytest.skip("Scaled fp4 matmul is only natively supported on CDNA4")
        if scale_type != 'float8_e8m0fnu':
            pytest.skip("CDNA4 only supports E8M0 scale")
        if (nonKDim == 16 and BLOCK_K < 128) or (nonKDim == 32 and BLOCK_K < 64):
            pytest.skip(f"CDNA4 does not support {BLOCK_K=} for scaled mfma {nonKDim=} variants")

    NUM_STAGES = 1
    torch.manual_seed(42)
    a_mxfp4 = MXFP4Tensor(size=(M, K), device=device).random()
    a = a_mxfp4.to_packed_tensor(dim=1)
    # Generate b with k-major layout, pack two e2m1 along k, then logical transpose to K, N
    b_mxfp4 = MXFP4Tensor(size=(N, K), device=device).random()
    b = b_mxfp4.to_packed_tensor(dim=1).T
    # No need to pack along K since we convert each e2m1 to f32 directly for the reference matmul
    b_ref = b_mxfp4.to(torch.float32).T

    a_size = (M, (K + VEC_SIZE - 1) // VEC_SIZE)
    b_size = (N, (K + VEC_SIZE - 1) // VEC_SIZE)
    a_scale = torch.rand(a_size, device=device)
    b_scale = torch.rand(b_size, device=device)
    if scale_type == "float8_e8m0fnu":
        a_scale_ref = MXScaleTensor(a_scale)
        b_scale_ref = MXScaleTensor(b_scale)
        a_scale = a_scale_ref.data
        b_scale = b_scale_ref.data
    elif scale_type == "float8_e4m3fn":
        a_scale = a_scale.to(torch.float8_e4m3fn)
        b_scale = b_scale.to(torch.float8_e4m3fn)
        a_scale_ref = a_scale
        b_scale_ref = b_scale

    a_scale_ref = a_scale_ref.to(torch.float32).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_ref = b_scale_ref.to(torch.float32).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    stride_scale = a_scale.stride(0)
    if not with_a_scale:
        a_scale = None
        a_scale_ref = 1.0
    if not with_b_scale:
        b_scale = None
        b_scale_ref = 1.0
    ref_out = torch.matmul(a_mxfp4.to(torch.float32) * a_scale_ref, b_ref * b_scale_ref)

    output = a.new_empty((M, N), dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    kernel_kwargs = {}
    if is_hip():
        kernel_kwargs["matrix_instr_nonkdim"] = nonKDim
    block_scale_fp4_matmul[grid](a, b, output, a_scale, b_scale, M, N, K, stride_scale, a.stride(0), a.stride(1),
                                 b.stride(0), b.stride(1), output.stride(0), output.stride(1), VEC_SIZE, BLOCK_M,
                                 BLOCK_N, BLOCK_K, NUM_STAGES=NUM_STAGES, **kernel_kwargs)

    torch.testing.assert_close(ref_out, output, atol=1e-2, rtol=1e-2)


@triton.jit
def mxfp8_mxfp4_matmul(  #
        a_ptr, b_ptr, output_ptr,  #
        a_scale, b_scale,  #
        M, N, K,  #
        stride_scale,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        tensor_scale: tl.constexpr,  #
        DTYPE_A: tl.constexpr,  #
        DTYPE_B: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr):  #
    DIV_FACTOR_A: tl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: tl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_ak = tl.arange(0, BLOCK_K // DIV_FACTOR_A)
    offs_bk = tl.arange(0, BLOCK_K // DIV_FACTOR_B)
    offs_scale_k = tl.arange(0, BLOCK_K // 32)

    if a_scale is not None:
        a_scale_ptr = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
    if b_scale is not None:
        b_scale_ptr = b_scale + offs_bn[:, None] * stride_scale + offs_scale_k[None, :]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)

    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if a_scale is not None:
            if tensor_scale:
                scale_a = tl.load(a_scale_ptr)
            else:
                scale_a = tl.full(a_scale_ptr.shape, a_scale.to(tl.int8), dtype=tl.int8)
        else:
            scale_a = None
        if b_scale is not None:
            scale_b = tl.load(b_scale_ptr)
        else:
            scale_b = None
        accumulator = tl.dot_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)
        a_ptrs += (BLOCK_K // DIV_FACTOR_A) * stride_ak
        b_ptrs += (BLOCK_K // DIV_FACTOR_B) * stride_bk
        if a_scale is not None:
            a_scale_ptr += BLOCK_K // 32
        if b_scale is not None:
            b_scale_ptr += BLOCK_K // 32

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)


@pytest.mark.parametrize("M, N, K", [(1024, 512, 512)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 128, 128), (256, 128, 128), (128, 256, 128),
                                                       (128, 256, 256), (128, 128, 64), (128, 64, 128)])
@pytest.mark.parametrize("NUM_STAGES", [1, 3])
@pytest.mark.parametrize("B_TRANS", [True, False])
@pytest.mark.parametrize("CONST_SCALE", [True, False])
@pytest.mark.parametrize("A_DATA_TYPE", ["float8e5", "float8e4nv", "float4"])
@pytest.mark.parametrize("B_DATA_TYPE", ["float8e5", "float8e4nv", "float4"])
@pytest.mark.parametrize("WITH_A_SCALE", [True, False])
@pytest.mark.parametrize("WITH_B_SCALE", [True, False])
@pytest.mark.parametrize("nonKDim", ([0, 16, 32] if is_hip_cdna() else [0]))
def test_mxfp8_mxfp4_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, B_TRANS, CONST_SCALE, A_DATA_TYPE,
                            B_DATA_TYPE, WITH_A_SCALE, WITH_B_SCALE, nonKDim, device):
    if is_cuda():
        if torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("Requires compute capability >= 10")
        if not (WITH_A_SCALE and WITH_B_SCALE):
            pytest.skip("None scale has not been tested on NV backend")
        if not (A_DATA_TYPE == "float8e5" and B_DATA_TYPE == "float4"):
            pytest.skip(f"(A: {A_DATA_TYPE}, B: {B_DATA_TYPE}) has not been tested on NV backend")
    elif is_hip():
        if not is_hip_cdna4():
            pytest.skip("Scaled mxfp4 & mxfp8 matmul is only natively supported on CDNA4")
        if CONST_SCALE:
            pytest.skip("Constant scale is not supported in AMD backend for now")
        if (nonKDim == 16 and BLOCK_K < 128) or (nonKDim == 32 and BLOCK_K < 64):
            pytest.skip(f"CDNA4 does not support {BLOCK_K=} for scaled mfma {nonKDim=} variants")
        if (A_DATA_TYPE == 'float4' and not WITH_A_SCALE) or (B_DATA_TYPE == 'float4' and not WITH_B_SCALE):
            pytest.skip("Float4 without scale is tested in test_block_scale_fp4")

    if BLOCK_N == 256 and BLOCK_K == 256:
        NUM_STAGES = 2

    torch.manual_seed(42)

    def create_operand(dtype: str, size0: int, size1: int, k_dim: int, transpose: bool = True):
        if dtype == "float8e5":
            if transpose:
                v = torch.randint(20, 40, (size0, size1), dtype=torch.uint8).view(torch.float8_e5m2).to(device)
                v_ref = f8_to_f16(v.view(torch.float8_e5m2), dtype).to(torch.float32)
            else:
                v = torch.randint(20, 40, (size1, size0), dtype=torch.uint8).view(torch.float8_e5m2).to(device).T
                v_ref = f8_to_f16(v.view(torch.float8_e5m2).T, dtype).to(torch.float32).T
        elif dtype == "float8e4nv":
            if transpose:
                v = torch.randint(20, 40, (size0, size1), dtype=torch.uint8).view(torch.float8_e4m3fn).to(device)
                v_ref = f8_to_f16(v.view(torch.float8_e4m3fn), dtype).to(torch.float32)
            else:
                v = torch.randint(20, 40, (size1, size0), dtype=torch.uint8).view(torch.float8_e4m3fn).to(device).T
                v_ref = f8_to_f16(v.view(torch.float8_e4m3fn).T, dtype).to(torch.float32).T
        else:
            # float4
            if transpose:
                v_mxfp4 = MXFP4Tensor(size=(size0, size1), device=device).random()
                v = v_mxfp4.to_packed_tensor(dim=k_dim)
                v_ref = v_mxfp4.to(torch.float32)
            else:
                v_mxfp4 = MXFP4Tensor(size=(size1, size0), device=device).random()
                v = v_mxfp4.to_packed_tensor(dim=(k_dim + 1) % 2).T
                v_ref = v_mxfp4.to(torch.float32).T
        return v, v_ref

    dtype_converter = {'float8e5': 'e5m2', 'float8e4nv': 'e4m3', 'float4': 'e2m1'}

    a, a_ref = create_operand(A_DATA_TYPE, M, K, 1)
    b, b_ref = create_operand(B_DATA_TYPE, K, N, 0, B_TRANS)

    a_scale_mxfp4 = MXScaleTensor(size=(M, (K + 32 - 1) // 32), device=device).random(high=32.0)
    b_scale_mxfp4 = MXScaleTensor(size=(N, (K + 32 - 1) // 32), device=device).random(high=32.0)
    a_scale = a_scale_mxfp4.data
    b_scale = b_scale_mxfp4.data

    a_scale_ref = a_scale_mxfp4.to(torch.float32).repeat_interleave(32, dim=1)[:M, :K]
    if CONST_SCALE:
        a_scale_ref = torch.full_like(a_scale_ref, 2.0)
        a_scale = 128  # 2.0 in e8m0
    b_scale_ref = b_scale_mxfp4.to(torch.float32).repeat_interleave(32, dim=1).T.contiguous()[:K, :N]
    stride_scale = b_scale.stride(0)
    if not WITH_A_SCALE:
        a_scale = None
        a_scale_ref = 1.0
    if not WITH_B_SCALE:
        b_scale = None
        b_scale_ref = 1.0

    ref_out = torch.matmul(a_ref * a_scale_ref, b_ref * b_scale_ref)

    output = a.new_empty((M, N), dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    kernel_kwargs = {}
    if is_hip():
        kernel_kwargs["matrix_instr_nonkdim"] = nonKDim
    out = mxfp8_mxfp4_matmul[grid](a, b, output, a_scale, b_scale, M, N, K, stride_scale, a.stride(0), a.stride(1),
                                   b.stride(0), b.stride(1), output.stride(0), output.stride(1), not CONST_SCALE,
                                   dtype_converter[A_DATA_TYPE], dtype_converter[B_DATA_TYPE], BLOCK_M, BLOCK_N,
                                   BLOCK_K, NUM_STAGES=NUM_STAGES, **kernel_kwargs)
    if is_cuda():
        ttgir = out.asm["ttgir"]
        assert "fp4Padded = true" in ttgir

    torch.testing.assert_close(ref_out, output, atol=1e-3, rtol=1e-3)
