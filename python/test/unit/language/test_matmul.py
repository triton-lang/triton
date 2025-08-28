import math
import pytest
import torch
import triton
import triton.language as tl
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
        NUM_STAGES: tl.constexpr, SCALE_A: tl.constexpr = None, PRECISION: tl.constexpr = "ieee",
        A_TRANS: tl.constexpr = False, EPILOGUE_SUBTILE: tl.constexpr = False, dummy: tl.constexpr = 0):
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
        a_ptrs = a_ptr + (offs_k[:, None] * stride_ak + offs_am[None, :] * stride_am)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        if SCALE_A is not None:
            a = a * SCALE_A
        if A_TRANS:
            a = a.T
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, acc=accumulator, out_dtype=output_ptr.dtype.element_ty, input_precision=PRECISION)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    if EPILOGUE_SUBTILE:
        acc = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
        output_ptrs0 = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        output_ptrs1 = output_ptrs0 + stride_cn * (BLOCK_N // 2)
        tl.store(output_ptrs0, acc0)
        tl.store(output_ptrs1, acc1)
    else:
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(output_ptrs, accumulator)


def get_src_element_ty_size(dtype_str):
    if dtype_str == "float8e5":
        return 1
    if dtype_str == "float16":
        return 2
    if dtype_str == "float32" or dtype_str == "tensorfloat32":
        return 4
    if dtype_str == "float64":
        return 8
    raise ValueError(f"Unknown dtype {dtype_str}")


@pytest.mark.parametrize("dtype_src_str", ["float32", "tensorfloat32", "float16", "float8e5", "float64"])
@pytest.mark.parametrize("dtype_dst_str", ["float32", "float16", "float64"])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES", [(128, 128, 16, 4), (64, 128, 32, 4), (32, 32, 32, 4),
                                                                   (256, 128, 32, 4), (64, 512, 32, 2),
                                                                   (512, 64, 32, 2), (64, 16, 64, 4)])
@pytest.mark.parametrize("NUM_CTAS", [1, 2])
@pytest.mark.parametrize("NUM_WARPS", [4, 8])
@pytest.mark.parametrize("EPILOGUE_SUBTILE", [True, False])
@pytest.mark.parametrize("LAYOUT_16x256", [True, False])
def test_simple_matmul(dtype_src_str, dtype_dst_str, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, NUM_WARPS, NUM_CTAS, device,
                       EPILOGUE_SUBTILE, LAYOUT_16x256, monkeypatch):
    if NUM_CTAS > 1 and (not is_cuda() or torch.cuda.get_device_capability()[0] < 9):
        pytest.skip("Clusters requires nvidia compute capability >= 9")
    shared_mem_accum = (BLOCK_K * BLOCK_M + BLOCK_K * BLOCK_N) * NUM_STAGES * get_src_element_ty_size(dtype_src_str)
    shared_mem_avail = triton.runtime.driver.active.utils.get_device_properties(0)["max_shared_mem"]
    if shared_mem_accum > shared_mem_avail:
        pytest.skip("Skipped due to insufficient shared memory on this GPU.")
    if is_hip() and (not is_hip_cdna3()) and dtype_src_str == "tensorfloat32":
        pytest.skip("tensorfloat32 is only supported on HIP CDNA3")
    if dtype_src_str == "float8e5" and BLOCK_K == 16:
        pytest.skip("Skipping cases small K for float8")
    if dtype_src_str == "float8e5" and device == "cuda" and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Float8 requires compute capability >= 9")
    if (dtype_src_str == "float64") != (dtype_dst_str == "float64"):
        pytest.skip("Skipping unsupported case")
    if "float32" in dtype_src_str and dtype_dst_str == "float16":
        pytest.skip("Skipping unsupported case")
    if "float32" == dtype_src_str and NUM_CTAS > 1:
        pytest.skip("FMA matmul not supported for multiple CTAs")
    if (BLOCK_M < 64 or (BLOCK_M == 64 and BLOCK_N == 16)) and NUM_CTAS > 1:
        pytest.skip("multi-CTAs is broken for mmav2")
    if EPILOGUE_SUBTILE and (is_hip() or NUM_CTAS > 1 or BLOCK_N >= 512):
        pytest.skip("creates convert layout too big to fit in smem")
    if LAYOUT_16x256 and (not is_cuda() or torch.cuda.get_device_capability()[0] < 10):
        pytest.skip("skip forcing tmem layout on non blackwell targets.")
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
    # pass a dummy constexpr argument to force recompilation.
    if LAYOUT_16x256:
        monkeypatch.setenv("TRITON_PREFER_TMEM_16x256_LAYOUT", "1")
    dtype_dst = getattr(torch, dtype_dst_str)
    output = torch.empty((M, N), dtype=dtype_dst, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    k = matmul_kernel[grid](a, b, output, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), output.stride(0),
                            output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES=NUM_STAGES, PRECISION=precision,
                            num_warps=NUM_WARPS, num_ctas=NUM_CTAS, EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
                            dummy=LAYOUT_16x256)
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
    # Make sure the mma is pipelined by checking if in the TTGIR we see two mmav5
    # operations. (Pipeliner will add additional mma operation by peeling the prologue.)
    # This applies only if TCv5 MMA is used (M % 64 == 0 and N % 8 == 0) and
    # when MMA arguments loads are pipelined (N > 16)
    if (device == "cuda" and torch.cuda.get_device_capability()[0] == 10 and NUM_STAGES > 1 and BLOCK_M % 64 == 0
            and BLOCK_N % 8 == 0 and BLOCK_N > 16
            and not (precision == "ieee" and (dtype_src_str == "float32" or dtype_src_str == "float64"))):
        ttgir = k.asm["ttgir"]
        count = ttgir.count("ttng.tc_gen5_mma")
        assert count == 2, "The TTGIR does not match the expected pattern."
        ptx = k.asm["ptx"]
        if LAYOUT_16x256:
            assert "16x256b" in ptx, "PTX does not contain 16x256b"
        else:
            if "32x32b" not in ptx and "16x32b" not in ptx:
                print(ptx)
            assert ("32x32b" in ptx) or ("16x32b" in ptx), "PTX does not contain 32x32b or 16x32b"


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

    # Make sure the mma is pipelined by checking if in the TTGIR we have peeled mmav5 ops.
    # This applies only if TCv5 MMA is used (M % 64 == 0 and N % 8 == 0) and
    # when MMA arguments loads are pipelined (N > 16)
    if (device == "cuda" and torch.cuda.get_device_capability()[0] == 10 and BLOCK_M % 64 == 0 and BLOCK_N % 8 == 0
            and BLOCK_N > 16):
        ttgir = k.asm["ttgir"]
        pattern = "ttng.tc_gen5_mma"
        assert ttgir.count(pattern) > 0, "Expect peeled mmav5 operations."


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
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
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


@pytest.mark.parametrize("M, N, K", [(1024, 512, 256), (128, 256, 256), (128, 128, 128), (2, 4, 64)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 128, 128), (256, 128, 128), (128, 256, 128),
                                                       (128, 256, 256), (128, 128, 64), (128, 64, 128)])
@pytest.mark.parametrize("NUM_STAGES", [1, 3])
@pytest.mark.parametrize("NUM_WARPS", [4, 8])
@pytest.mark.parametrize("nonKDim", ([0, 16, 32] if is_hip_cdna() else [0]))
def test_mxfp(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, nonKDim, NUM_WARPS, device):
    if K % BLOCK_K != 0:
        pytest.skip("Kernel requires shapes aligned by K dimension")
    if is_cuda() and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Requires compute capability >= 10")
    elif is_hip():
        if not is_hip_cdna4():
            pytest.skip("Scaled mxfp8 matmul is only natively supported on CDNA4")
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
    mxfp_matmul[grid](a, b, output, a_scale, b_scale, M, N, K, a_scale.stride(0), a.stride(0), a.stride(1), b.stride(0),
                      b.stride(1), output.stride(0), output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES=NUM_STAGES,
                      **kernel_kwargs, num_warps=NUM_WARPS)
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

        # Scales are coming in for optimial performance, but we reshape here for
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


@triton.jit
def _gemm_afp4_wfp4_kernel_preshuffled_scales_cdna4(a_ptr, b_ptr, c_ptr, a_scales_ptr, b_scales_ptr, M, N, K, stride_am,
                                                    stride_ak, stride_bk, stride_bn, stride_ck, stride_cm, stride_cn,
                                                    stride_asm, stride_ask, stride_bsn, stride_bsk,
                                                    # Meta-parameters
                                                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                                    mfma_nonkdim: tl.constexpr, preshuffle: tl.constexpr):
    """Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32

    if preshuffle:
        NON_K_PRESHUFFLE_BLOCK_SIZE: tl.constexpr = 32
    else:
        NON_K_PRESHUFFLE_BLOCK_SIZE: tl.constexpr = 1

    num_k_iter = tl.cdiv(K, BLOCK_K // 2)
    # Create pointers for first block of A and B input matrices
    # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_k_split = offs_k
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Create pointers for the first block of A and B scales
    offs_asn = (pid_n *
                (BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE) + tl.arange(0, (BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE))) % N
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * NON_K_PRESHUFFLE_BLOCK_SIZE)

    # B scales are N x K even though B operand is K x N.
    b_scale_ptrs = (b_scales_ptr + offs_asn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk)
    offs_asm = (pid_m *
                (BLOCK_M // NON_K_PRESHUFFLE_BLOCK_SIZE) + tl.arange(0, (BLOCK_M // NON_K_PRESHUFFLE_BLOCK_SIZE))) % M
    a_scale_ptrs = (a_scales_ptr + offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, num_k_iter):
        if preshuffle:
            # Here we "undo" the shuffle done in global memory (shuffle_scales_cdna4 function).
            if mfma_nonkdim == 32:
                a_scales = tl.load(a_scale_ptrs).reshape(BLOCK_M // NON_K_PRESHUFFLE_BLOCK_SIZE,
                                                         BLOCK_K // SCALE_GROUP_SIZE // 8, 2, 32, 4,
                                                         1).permute(0, 3, 1, 4, 2,
                                                                    5).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
                b_scales = tl.load(b_scale_ptrs).reshape(BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE,
                                                         BLOCK_K // SCALE_GROUP_SIZE // 8, 2, 32, 4,
                                                         1).permute(0, 3, 1, 4, 2,
                                                                    5).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)
            elif mfma_nonkdim == 16:
                a_scales = tl.load(a_scale_ptrs).reshape(BLOCK_M // NON_K_PRESHUFFLE_BLOCK_SIZE,
                                                         BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2,
                                                         1).permute(0, 5, 3, 1, 4, 2,
                                                                    6).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
                b_scales = tl.load(b_scale_ptrs).reshape(BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE,
                                                         BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2,
                                                         1).permute(0, 5, 3, 1, 4, 2,
                                                                    6).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)
        else:
            a_scales = tl.load(a_scale_ptrs)
            b_scales = tl.load(b_scale_ptrs)

        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs, cache_modifier=None)

        accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

        # Advance the ptrs to the next K block.
        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        if preshuffle:
            a_scale_ptrs += BLOCK_K * stride_ask
            b_scale_ptrs += BLOCK_K * stride_bsk
        else:
            a_scale_ptrs += (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask
            b_scale_ptrs += (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk

    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_ptrs = (c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


@pytest.mark.parametrize("M, N, K", [(1024, 1024, 1024)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 128, 256), (64, 64, 512), [32, 32, 64]])
@pytest.mark.parametrize("mfma_nonkdim", [16, 32])
@pytest.mark.parametrize("preshuffle", [True, False])
@pytest.mark.skipif(is_cuda() and torch.cuda.get_device_capability()[0] == 10, reason="Compilation bug for GB200.")
@pytest.mark.skipif(is_hip() and not is_hip_cdna4(), reason="Scaled dot is not emulated on other archs yet.")
def test_preshuffle_scale_mxfp_cdna4(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, mfma_nonkdim, preshuffle, device):
    # This test primarily evaluates correctness for efficient scale packing for MFMA-scaled instructions.
    #
    # Scales are stored as 8-bit tensors, where each element scales 32 values from the A or B operand tensors.
    # Since MFMA instructions are wave-level instructions, that means that each thread provides a fixed set of operand values to MFMA instructions.
    #
    # For example, in an MFMA instruction with shape 16x16x128:
    # - 4 threads contribute elements along the K dimension.
    # - 16 threads contribute elements along the M or N dimension.
    #
    # From the perspective of the scales tensor, even if the K dimension is stored contiguously in LDS,
    # each thread sees its elements along K dim as strided due to interleaving with other threads.
    # This striding limits the ability to load scale values using vectorized memory access.
    #
    # Our goal is to reorganize the scale tensor so that:
    # 1. Each thread stores the 4 scale values it needs for 4 MFMA ops in contiguous memory.
    # 2. Continuous threads access contiguous memory locations improving global memory coalescing when bypassing LDS,
    #    which is especially beneficial for "skinny" matmuls.
    #
    # We consider two MFMA cases: one with non-K dimension 16, and one with 32.
    # In both, the minimum tile size for preshuffling is 32x32x256.
    # For example, for a 32x256 operand tile, the corresponding scale tensor has shape 32x8,
    # where each scale covers 32 elements along the K dimension.
    #
    # Each thread holds one scale per MFMA operation. We pack the 4 scale values (for 4 different MFMA ops)
    # next to each other in memory.
    #
    # Case 1: mfma_scaled_16x16x128
    #
    # Packing order: mfma_op_0, mfma_op_2, mfma_op_1, mfma_op_3
    #
    #            K = 128       K = 128
    #        +------------+ +------------+
    #    M=16|  MFMA op 0 | |  MFMA op 1 |
    #        +------------+ +------------+
    #    M=16|  MFMA op 2 | |  MFMA op 3 |
    #        +------------+ +------------+
    #
    # Case 2: mfma_scaled_32x32x64
    #
    # Packing order: mfma_op_0, mfma_op_1, mfma_op_2, mfma_op_3
    #
    #            K=64     K=64     K=64     K=64
    #        +--------+ +--------+ +--------+ +--------+
    #    M=32| op 0   | | op 1   | | op 2   | | op 3   |
    #        +--------+ +--------+ +--------+ +--------+

    if preshuffle and (BLOCK_M < 32 or BLOCK_N < 32 or BLOCK_K < 256):
        pytest.skip("Minimal tile size for preshuffling is 32x32x256")

    def shuffle_scales_cdna4(scales: torch.Tensor):
        if not preshuffle:
            return scales

        scales_shuffled = scales.clone()

        sm, sn = scales_shuffled.shape
        if mfma_nonkdim == 32:
            scales_shuffled = scales_shuffled.view(sm // 32, 32, sn // 8, 4, 2, 1)
            scales_shuffled = scales_shuffled.permute(0, 2, 4, 1, 3, 5).contiguous()
        elif mfma_nonkdim == 16:
            scales_shuffled = scales_shuffled.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
            scales_shuffled = scales_shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()

        scales_shuffled = scales_shuffled.view(sm // 32, sn * 32)
        return scales_shuffled

    def e8m0_to_f32(x):
        x_f32 = 2**((x - 127).to(torch.float32))
        x_f32[x_f32 == 128] = float("nan")
        return x_f32

    def run_torch(x, w, x_scales, w_scales, dtype):
        # First convert the x and w inputs to f32.
        SCALE_GROUP_SIZE = 32
        x_f32 = x.to(torch.float32)
        w_f32 = w.to(torch.float32)
        # Next convert the e8m0 scales to f32.
        x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
        x_scales_f32 = e8m0_to_f32(x_scales)
        x_f32 = x_f32 * x_scales_f32
        w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
        w_scales_f32 = e8m0_to_f32(w_scales)
        w_f32 = w_f32 * w_scales_f32
        return torch.mm(x_f32, w_f32.T).to(dtype)

    def generate_gemm_afp4wfp4_inputs(M, N, K):
        torch.manual_seed(5)
        SCALE_GROUP_SIZE = 32

        x = MXFP4Tensor(size=(M, K), device="cuda").random()
        w = MXFP4Tensor(size=(N, K), device="cuda").random()

        x_scales = torch.randint(124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device="cuda")
        w_scales = torch.randint(124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda")
        x_scales = x_scales.T
        w_scales = w_scales.T
        x_scales_shuffled = shuffle_scales_cdna4(x_scales)
        w_scales_shuffled = shuffle_scales_cdna4(w_scales)

        return (
            x,
            w,
            x_scales,
            w_scales,
            x_scales_shuffled,
            w_scales_shuffled,
        )

    x_mxfp4, w_mxfp4, x_scales, w_scales, x_scales_triton, w_scales_triton = generate_gemm_afp4wfp4_inputs(M, N, K)

    x = x_mxfp4.to_packed_tensor(dim=1)
    w = w_mxfp4.to_packed_tensor(dim=1)

    torch_out = run_torch(x_mxfp4, w_mxfp4, x_scales, w_scales, torch.float32)
    M, K = x.shape
    N, K = w.shape
    w = w.T
    triton_out = torch.empty((M, N), device=x.device)

    kernel_kwargs = {}
    if is_hip():
        kernel_kwargs["matrix_instr_nonkdim"] = mfma_nonkdim

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    _gemm_afp4_wfp4_kernel_preshuffled_scales_cdna4[grid](x, w, triton_out, x_scales_triton, w_scales_triton, M, N, K,
                                                          x.stride(0), x.stride(1), w.stride(0), w.stride(1), 0,
                                                          triton_out.stride(0), triton_out.stride(1),
                                                          x_scales_triton.stride(0), x_scales_triton.stride(1),
                                                          w_scales_triton.stride(0), w_scales_triton.stride(1), BLOCK_M,
                                                          BLOCK_N, BLOCK_K, mfma_nonkdim, preshuffle, num_warps=8,
                                                          num_stages=1, **kernel_kwargs)
    triton_out = triton_out.to(torch.float32)
    torch.testing.assert_close(torch_out, triton_out)


@pytest.mark.parametrize("M, N, K", [(1024, 512, 512), (998, 111, 512), (63, 128, 512)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 128, 128), (256, 128, 128), (128, 256, 128),
                                                       (128, 128, 256), (128, 256, 256)])
@pytest.mark.parametrize("NUM_STAGES", [1, 2, 4])
@pytest.mark.parametrize("USE_2D_SCALE_LOAD", [False, True])
@pytest.mark.skipif(is_hip() or torch.cuda.get_device_capability()[0] != 10, reason="Requires compute capability == 10")
def test_blocked_scale_mxfp(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, USE_2D_SCALE_LOAD, device):
    if BLOCK_N == 256 and BLOCK_K == 256:
        NUM_STAGES = min(NUM_STAGES, 2)
    elif BLOCK_K == 256:
        NUM_STAGES = min(NUM_STAGES, 3)
    #since the block size are big we use num_warps = 8 to avoid pressure problems.
    num_warps = 8
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
                                        NUM_STAGES=NUM_STAGES, USE_2D_SCALE_LOAD=USE_2D_SCALE_LOAD, num_warps=num_warps)
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


@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 128, 64), (128, 64, 128), (64, 128, 32), (128, 256, 32),
                                                       (256, 64, 32)])
@pytest.mark.parametrize("a_trans", [False, True])
@pytest.mark.parametrize("dtype_src_str", ["float32", "float16", "float8e5"])
@pytest.mark.skipif(is_hip() or torch.cuda.get_device_capability()[0] != 10, reason="Requires compute capability == 10")
def test_lhs_in_tmem(BLOCK_M, BLOCK_N, BLOCK_K, a_trans, dtype_src_str, device, monkeypatch):
    M = 1024
    N = 512
    K = 256
    _knob_promote_lhs_to_tmem(monkeypatch)
    torch.manual_seed(42)
    if dtype_src_str == "float8e5":
        a = torch.randint(20, 40, (M, K), dtype=torch.int8, device=device).view(torch.float8_e5m2)
        b = torch.randint(20, 40, (K, N), dtype=torch.int8, device=device).view(torch.float8_e5m2)
        if a_trans:
            a = a.T.contiguous().T
        A = f8_to_f16(a, dtype_src_str)
        B = f8_to_f16(b, dtype_src_str)
    else:
        dtype_src = getattr(torch, dtype_src_str)
        a = torch.randn(M, K, dtype=dtype_src, device=device)
        b = torch.randn(K, N, dtype=dtype_src, device=device)
        if a_trans:
            a = a.T.contiguous().T
        A = a
        B = b
    output = torch.empty((M, N), dtype=torch.float32, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    k = matmul_kernel[grid](a, b, output, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), output.stride(0),
                            output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES=1, SCALE_A=None, PRECISION="tf32",
                            A_TRANS=a_trans)
    ref_out = torch.matmul(A, B).to(torch.float32)
    atol = 0.03
    rtol = 0.03
    torch.testing.assert_close(ref_out, output, atol=atol, rtol=rtol)
    pattern = r"%\w+\s*=\s*ttng\.tmem_alloc[\s\S]*?tng\.tc_gen5_mma\s+%\w+,"
    ttgir = k.asm["ttgir"]
    assert re.search(pattern, ttgir)


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


@pytest.mark.skipif(is_hip() or torch.cuda.get_device_capability()[0] != 10, reason="Requires compute capability == 10")
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
        NUM_STAGES: tl.constexpr, PACK_ALONG_K: tl.constexpr):  #
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    PACKING_ALONG_M_N: tl.constexpr = 1 if PACK_ALONG_K else 2
    offs_am_packed = (pid_m * (BLOCK_M // PACKING_ALONG_M_N) + tl.arange(0, BLOCK_M // PACKING_ALONG_M_N))
    offs_bn_packed = (pid_n * (BLOCK_N // PACKING_ALONG_M_N) + tl.arange(0, BLOCK_N // PACKING_ALONG_M_N))
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2 if PACK_ALONG_K else BLOCK_K

    # Two e2m1 values per K
    offs_k = tl.arange(0, BLOCK_K_PACKED)
    offs_scale_k = tl.arange(0, BLOCK_K // VEC_SIZE)
    if a_scale is not None:
        a_scale_ptr = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
    if b_scale is not None:
        b_scale_ptr = b_scale + offs_bn[:, None] * stride_scale + offs_scale_k[None, :]
    a_ptrs = a_ptr + (offs_am_packed[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn_packed[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if a_scale is not None:
            scale_a = tl.load(a_scale_ptr)
        else:
            scale_a = None
        if b_scale is not None:
            scale_b = tl.load(b_scale_ptr)
        else:
            scale_b = None
        accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", accumulator, lhs_k_pack=PACK_ALONG_K,
                                    rhs_k_pack=PACK_ALONG_K)
        a_ptrs += (BLOCK_K_PACKED) * stride_ak
        b_ptrs += (BLOCK_K_PACKED) * stride_bk
        if a_scale is not None:
            a_scale_ptr += BLOCK_K // VEC_SIZE
        if b_scale is not None:
            b_scale_ptr += BLOCK_K // VEC_SIZE
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)


@pytest.mark.parametrize("M, N, K", [(1024, 512, 256)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 128, 128), (256, 128, 128), (128, 256, 128),
                                                       (128, 256, 256), (128, 128, 64), (128, 64, 128)])
@pytest.mark.parametrize("with_a_scale", [True, False])
@pytest.mark.parametrize("with_b_scale", [True, False])
@pytest.mark.parametrize("pack_along_k", [True, False])
@pytest.mark.parametrize(("scale_type", "VEC_SIZE"), [("float8_e8m0fnu", 32), ("float8_e4m3fn", 16)],
                         ids=["mxfp4", "nvfp4"])
@pytest.mark.parametrize("nonKDim", ([0, 16, 32] if is_hip_cdna() else [0]))
def test_block_scale_fp4(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, VEC_SIZE, with_a_scale, with_b_scale, pack_along_k,
                         scale_type, nonKDim, device):
    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0
    if is_cuda():
        if scale_type == "float8_e4m3fn" and not pack_along_k:
            pytest.skip("Packing along K is required for float8_e4m3fn")
        if torch.cuda.get_device_capability()[0] != 10:
            pytest.skip("Requires compute capability == 10")
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
    packing_dim = 1 if pack_along_k else 0
    a_mxfp4 = MXFP4Tensor(size=(M, K), device=device).random()
    a = a_mxfp4.to_packed_tensor(dim=packing_dim)
    # Generate b with k-major layout, pack two e2m1 along k or n, then logical transpose to K, N
    b_mxfp4 = MXFP4Tensor(size=(N, K), device=device).random()
    b = b_mxfp4.to_packed_tensor(dim=packing_dim).T
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
    k = block_scale_fp4_matmul[grid](a, b, output, a_scale, b_scale, M, N, K, stride_scale, a.stride(0), a.stride(1),
                                     b.stride(0), b.stride(1), output.stride(0), output.stride(1), VEC_SIZE, BLOCK_M,
                                     BLOCK_N, BLOCK_K, NUM_STAGES=NUM_STAGES, PACK_ALONG_K=pack_along_k,
                                     **kernel_kwargs)
    torch.testing.assert_close(ref_out, output, atol=1e-2, rtol=1e-2)
    if is_cuda():
        ptx = k.asm["ptx"]
        if pack_along_k:
            assert "kind::mxf4" in ptx
        else:
            assert "kind::mxf8f6f4" in ptx


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
        NUM_STAGES: tl.constexpr,  #
        PACK_B_ALONG_K: tl.constexpr = True):  #
    DIV_FACTOR_A: tl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: tl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    DIV_FACTOR_B_K: tl.constexpr = DIV_FACTOR_B if PACK_B_ALONG_K else 1
    DIV_FACTOR_B_N: tl.constexpr = 1 if PACK_B_ALONG_K else DIV_FACTOR_B
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_bn = (pid_n * BLOCK_N // DIV_FACTOR_B_N + tl.arange(0, BLOCK_N // DIV_FACTOR_B_N))
    offs_bn_scale = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_ak = tl.arange(0, BLOCK_K // DIV_FACTOR_A)
    offs_bk = tl.arange(0, BLOCK_K // DIV_FACTOR_B_K)
    offs_scale_k = tl.arange(0, BLOCK_K // 32)

    if a_scale is not None:
        a_scale_ptr = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
    if b_scale is not None:
        b_scale_ptr = b_scale + offs_bn_scale[:, None] * stride_scale + offs_scale_k[None, :]
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
        accumulator = tl.dot_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator, rhs_k_pack=PACK_B_ALONG_K)
        a_ptrs += (BLOCK_K // DIV_FACTOR_A) * stride_ak
        b_ptrs += (BLOCK_K // DIV_FACTOR_B_K) * stride_bk
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
@pytest.mark.parametrize("PACK_B_ALONG_K", [True, False])
@pytest.mark.parametrize("CONST_SCALE", [True, False])
@pytest.mark.parametrize("A_DATA_TYPE", ["float8e5", "float8e4nv", "float4"])
@pytest.mark.parametrize("B_DATA_TYPE", ["float8e5", "float8e4nv", "float4"])
@pytest.mark.parametrize("WITH_A_SCALE", [True, False])
@pytest.mark.parametrize("WITH_B_SCALE", [True, False])
@pytest.mark.parametrize("nonKDim", ([0, 16, 32] if is_hip_cdna() else [0]))
def test_mxfp8_mxfp4_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, B_TRANS, PACK_B_ALONG_K, CONST_SCALE,
                            A_DATA_TYPE, B_DATA_TYPE, WITH_A_SCALE, WITH_B_SCALE, nonKDim, device):
    if is_cuda():
        if torch.cuda.get_device_capability()[0] != 10:
            pytest.skip("Requires compute capability == 10")
        if not (WITH_A_SCALE and WITH_B_SCALE):
            pytest.skip("None scale has not been tested on NV backend")
        if not (A_DATA_TYPE == "float8e5" and B_DATA_TYPE == "float4"):
            pytest.skip(f"(A: {A_DATA_TYPE}, B: {B_DATA_TYPE}) has not been tested on NV backend")
    elif is_hip():
        if not is_hip_cdna4():
            pytest.skip("Scaled mxfp4 & mxfp8 matmul is only natively supported on CDNA4")
        if (nonKDim == 16 and BLOCK_K < 128) or (nonKDim == 32 and BLOCK_K < 64):
            pytest.skip(f"CDNA4 does not support {BLOCK_K=} for scaled mfma {nonKDim=} variants")
        if (A_DATA_TYPE == 'float4' and not WITH_A_SCALE) or (B_DATA_TYPE == 'float4' and not WITH_B_SCALE):
            pytest.skip("Float4 without scale is tested in test_block_scale_fp4")
    if not PACK_B_ALONG_K and B_DATA_TYPE != "float4":
        pytest.skip("Pack along K can only be False for float4")
    if BLOCK_N == 256 and BLOCK_K == 256:
        NUM_STAGES = 2

    torch.manual_seed(42)

    def create_operand(dtype: str, size0: int, size1: int, k_dim: int, transpose: bool = True,
                       pack_along_k: bool = True):
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
            if pack_along_k:
                pack_dim = k_dim
            else:
                pack_dim = (k_dim + 1) % 2
            if transpose:
                v_mxfp4 = MXFP4Tensor(size=(size0, size1), device=device).random()
                v = v_mxfp4.to_packed_tensor(dim=pack_dim)
                v_ref = v_mxfp4.to(torch.float32)
            else:
                v_mxfp4 = MXFP4Tensor(size=(size1, size0), device=device).random()
                v = v_mxfp4.to_packed_tensor(dim=(pack_dim + 1) % 2).T
                v_ref = v_mxfp4.to(torch.float32).T
        return v, v_ref

    dtype_converter = {'float8e5': 'e5m2', 'float8e4nv': 'e4m3', 'float4': 'e2m1'}

    a, a_ref = create_operand(A_DATA_TYPE, M, K, 1)
    b, b_ref = create_operand(B_DATA_TYPE, K, N, 0, B_TRANS, PACK_B_ALONG_K)

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
                                   BLOCK_K, PACK_B_ALONG_K=PACK_B_ALONG_K, NUM_STAGES=NUM_STAGES, **kernel_kwargs)
    if is_cuda():
        ttgir = out.asm["ttgir"]
        assert "fp4Padded = true" in ttgir

    torch.testing.assert_close(ref_out, output, atol=1e-3, rtol=1e-3)
