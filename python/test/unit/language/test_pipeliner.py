# End-to-end tests to check the correctness of the pipeliner

import pytest
import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor

from triton._internal_testing import is_cuda, is_hopper, is_hip_cdna, is_hip_mi200


def check_capabilities():
    if is_cuda():
        cc = torch.cuda.get_device_capability()
        if cc[0] < 8:
            pytest.skip("CUDA 8.0+ required")


@triton.jit
def matmul_kernel(  #
        a_ptr, scale_ptr, b_ptr, output_ptr,  #
        M, N, K_MXFP,  # K_MXFP is the number of mxfp vectors in a row of a. Otherwise it's just K
        stride_am, stride_ak,  #
        stride_sm, stride_sk,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr, a_type: tl.constexpr, b_type: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    IS_SCALED: tl.constexpr = a_type is not None and b_type is not None
    DIV_FACTOR: tl.constexpr = 2 if IS_SCALED and a_type == "e2m1" else 1
    # We pass K_MXFP to make explicit that KB is multiple of 32 and KA is multiple of 16 or 32
    # for the pipeliner divisibility condition
    KA = K_MXFP if not IS_SCALED else K_MXFP * (32 // DIV_FACTOR)
    KB = K_MXFP if not IS_SCALED else K_MXFP * 32
    BLOCK_AK: tl.constexpr = BLOCK_K // DIV_FACTOR
    offs_k = tl.arange(0, BLOCK_K)
    offs_ak = tl.arange(0, BLOCK_AK)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    if IS_SCALED:
        BLOCK_SK: tl.constexpr = BLOCK_K // 32
        offs_sk = tl.arange(0, BLOCK_SK)
        scale_ptrs = scale_ptr + (offs_am[:, None] * stride_sm + offs_sk[None, :] * stride_sk)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(KB, BLOCK_K), num_stages=NUM_STAGES):
        mask_a = (offs_am[:, None] < M) & (offs_ak[None, :] + k * BLOCK_AK < KA)
        mask_b = ((offs_k[:, None] + k * BLOCK_K) < KB) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b = tl.load(b_ptrs, mask=mask_b, other=0)
        if IS_SCALED:
            # Adapted scale indexing and dot_scaled operation
            mask_scale = (offs_am[:, None] < M) & (offs_sk[None, :] + k * BLOCK_SK < K_MXFP)
            a_scale = tl.load(scale_ptrs, mask=mask_scale, other=0)
            accumulator = tl.dot_scaled(a, a_scale, a_type, b, None, b_type, acc=accumulator)
        else:
            accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_AK * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        if IS_SCALED:
            scale_ptrs += BLOCK_SK * stride_sk
    OUT_DTYPE = tl.bfloat16 if IS_SCALED else tl.float16
    accumulator = accumulator.to(OUT_DTYPE)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(output_ptrs, accumulator, mask=mask_c)


@triton.jit
def matmul_kernel_tma(  #
        a_ptr, b_ptr, output_ptr,  #
        M, N, K,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M) % M
    offs_bn = (pid_n * BLOCK_N) % N
    offs_am = tl.multiple_of(offs_am, BLOCK_M)
    offs_bn = tl.multiple_of(offs_bn, BLOCK_N)
    offs_k = 0
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl._experimental_descriptor_load(a_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K], tl.float16)
        b = tl._experimental_descriptor_load(b_ptr, [offs_k, offs_bn], [BLOCK_K, BLOCK_N], tl.float16)
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_K
    accumulator = accumulator.to(tl.float16)
    tl._experimental_descriptor_store(output_ptr, accumulator, [offs_am, offs_bn])


@triton.jit
def vecadd_kernel(a_ptr, b_ptr, output_ptr, n_elements, num_blocks, BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * num_blocks
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    for _ in tl.range(0, num_blocks, num_stages=NUM_STAGES):
        mask = offsets < n_elements
        x = tl.load(a_ptr + offsets, mask=mask)
        y = tl.load(b_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
        offsets += BLOCK_SIZE


@triton.jit
def mxfp_to_bf16_kernel(
    x_ptr,
    scale_ptr,
    mxfp_ptr,
    N,
    e_bits: tl.constexpr,
    m_bits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # x.shape ==     (N, 32) for fp8 or (N, 16) for fp4
    # scale.shape == (N,)
    # out.shape   == (N, 32)
    is_fp8: tl.constexpr = e_bits + m_bits == 7
    # fp8: BLOCK_SIZE -> BLOCK_SIZE // 32, 32
    # fp4: BLOCK_SIZE // 2 -> BLOCK_SIZE // 32 , 16
    PARALLEL_DIM: tl.constexpr = BLOCK_SIZE // 32
    LAST_DIM: tl.constexpr = 32 if is_fp8 else 16
    LOAD_SIZE: tl.constexpr = LAST_DIM * PARALLEL_DIM

    offsets = (tl.program_id(0) * LOAD_SIZE + tl.arange(0, PARALLEL_DIM)[:, None] * LAST_DIM +
               tl.arange(0, LAST_DIM)[None, :])
    x = tl.load(x_ptr + offsets, mask=offsets < N * LAST_DIM)

    offsets = tl.program_id(0) * PARALLEL_DIM + tl.arange(0, PARALLEL_DIM)[:, None]
    scale = tl.load(scale_ptr + offsets, mask=offsets < N)
    tl.static_assert(scale.dtype == tl.uint8)
    tl.static_assert(x.dtype == tl.uint8)

    scale_bf16 = (scale.to(tl.uint16) << 7).to(tl.bfloat16, bitcast=True)
    if is_fp8:
        if e_bits == 5 and m_bits == 2:
            x_f8 = x.to(tl.float8e5, bitcast=True)
            x_bf16 = x_f8.to(tl.bfloat16)
            # Preserve infs and nans. FIXME Fp8E5M2_to_Bf16 doesn't preserve them!
            non_finite_mask: tl.constexpr = ((1 << e_bits) - 1) << m_bits
            non_finite_mask_bf16: tl.constexpr = ((1 << 8) - 1) << 7
            x_bf16 = tl.where(
                x & non_finite_mask == non_finite_mask,
                (x_bf16.to(tl.uint16, bitcast=True) | non_finite_mask_bf16).to(tl.bfloat16, bitcast=True),
                x_bf16,
            )
        else:
            tl.static_assert(e_bits == 4 and m_bits == 3)
            x_f8 = x.to(tl.float8e4nv, bitcast=True)
            x_bf16 = x_f8.to(tl.bfloat16)
    else:
        # e2m1
        em0 = x & 0x70
        em1 = x & 0x7
        x0 = (em0.to(tl.uint16) << 2) | ((x & 0x80).to(tl.uint16) << 8)
        x1 = (em1.to(tl.uint16) << (2 + 4)) | ((x & 0x8).to(tl.uint16) << (8 + 4))
        # Three cases:
        # 1) x is normal and non-zero: Correct bias
        x0 = tl.where((em0 & 0x60) != 0, x0 + ((127 - 1) << 7), x0)
        x1 = tl.where((em1 & 0x6) != 0, x1 + ((127 - 1) << 7), x1)
        # 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in bf16
        x0 = tl.where(em0 == 0x10, 16128 | (x0 & 0x8000), x0)
        x1 = tl.where(em1 == 0x1, 16128 | (x1 & 0x8000), x1)
        # 3) x is zero, do nothing
        x_bf16 = tl.interleave(x0, x1).to(tl.bfloat16, bitcast=True)
    # Multiplication preserves infs and NaNs in x_bf16
    mxfp = x_bf16 * scale_bf16
    # If scale is NaN, we encode it as an bf16 inf, so we need to correct for that
    mxfp = tl.where(scale == 0xFF, float("nan"), mxfp)

    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(mxfp_ptr + offsets, tl.ravel(mxfp), mask=offsets < N * 32)


def dot_scale_ref(x, scale, y, type_x, type_y):
    e_bits, m_bits = {"e2m1": (2, 1), "e4m3": (4, 3), "e5m2": (5, 2)}[type_x]
    type_fp8_y = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[type_y]

    comp_dtype = torch.float32
    out_dtype = torch.bfloat16

    x = x.contiguous()
    x_upcast = x.new_empty(scale.shape[:-1] + (32 * scale.shape[-1], ), dtype=comp_dtype)

    N = x_upcast.numel()
    BLOCK_SIZE = 512
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    mxfp_to_bf16_kernel[grid](x, scale, x_upcast, scale.numel(), e_bits, m_bits, BLOCK_SIZE, num_warps=4)
    y_upcast = y.view(type_fp8_y)

    class AccumulateInFp32:

        def __enter__(self):
            self.prev_value = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        def __exit__(self, exc_type, exc_val, exc_tb):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = self.prev_value

    with AccumulateInFp32():
        return torch.matmul(x_upcast.to(out_dtype), y_upcast.to(out_dtype))


@pytest.mark.parametrize("scale", [True, False])
def test_pipeline_matmul(scale, device):
    check_capabilities()
    if scale and not (is_cuda() or is_hip_cdna()):
        pytest.skip("NYI: scale_dot just implemented in CUDA/HIP")
    M, N, K = 512, 512, 128
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    NUM_STAGES = 4

    if scale:
        # TODO Use e5m2 for Ampere, as it does not support fp_to_fp conversions for fp8e4m3
        BLOCK_K = 64  # 32 NYI
        K = BLOCK_K * NUM_STAGES
        a_type = "e2m1"
        DIV_FACTOR = 2 if a_type == "e2m1" else 1
        a = torch.randint(256, (M, K // DIV_FACTOR), device=device, dtype=torch.uint8)
        # Sample small-ish scales to avoid overflow
        scale_a = torch.randint(74, (M, K // 32), device=device, dtype=torch.uint8)
        # Ampere does not support fp8e4m3
        b_type = "e4m3" if is_hopper() else "e5m2"
        b = torch.randint(256, (K, N), device=device, dtype=torch.uint8)
        # e5m2 has too many non-finite values when sampled uniformly (1 / 32) and
        # Fp8E5M2_to_Bf16 doesn't preserve NaNs (fixme)
        if b_type == "e5m2":
            finite = torch.arange(K * N, device=device, dtype=torch.uint8).reshape(K, N) % 0x7C
            b = torch.where(b & 0x7C == 0x7C, finite | (0x80 & b), b)
        output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    else:
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)
        scale_a = None
        a_type, b_type = None, None
        output = torch.empty((M, N), dtype=torch.float16, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    use_tma = not scale and is_hopper()

    if use_tma:
        a_tma = triton.tools.experimental_descriptor.create_2d_tma_descriptor(a.data_ptr(), M, K, BLOCK_M, BLOCK_K,
                                                                              a.element_size())
        b_tma = triton.tools.experimental_descriptor.create_2d_tma_descriptor(b.data_ptr(), K, N, BLOCK_K, BLOCK_N,
                                                                              b.element_size())
        output_tma = triton.tools.experimental_descriptor.create_2d_tma_descriptor(output.data_ptr(), M, N, BLOCK_M,
                                                                                   BLOCK_N, output.element_size())
        handler = matmul_kernel_tma[grid](a_tma, b_tma, output_tma, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                          NUM_STAGES=NUM_STAGES)
    else:
        # Pass K_MXFP to make explicit that KB is multiple of 32 and KA is multiple of 16 or 32º
        if scale:
            K = scale_a.shape[-1]
        stride_sm, stride_sk = scale_a.stride() if scale else (0, 0)
        handler = matmul_kernel[grid](a, scale_a, b, output, M, N, K, a.stride(0), a.stride(1), stride_sm, stride_sk,
                                      b.stride(0), b.stride(1), output.stride(0), output.stride(1), BLOCK_M, BLOCK_N,
                                      BLOCK_K, NUM_STAGES=NUM_STAGES, a_type=a_type, b_type=b_type)
    if scale:
        ref_out = dot_scale_ref(a, scale_a, b, a_type, b_type)
    else:
        ref_out = torch.matmul(a, b)
    # Bigger tolerance for AMD MI200 devices.
    # MI200 devices use reduced precision fp16 and bf16 and flush input and
    # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    atol = 1e-2 if is_hip_mi200() or scale else None
    rtol = 1e-2 if is_hip_mi200() or scale else None
    torch.testing.assert_close(ref_out, output, atol=atol, rtol=rtol, equal_nan=scale)
    if is_cuda():
        ttgir = handler.asm["ttgir"]
        if use_tma:
            assert ttgir.count("triton_nvidia_gpu.async_tma_copy_global_to_local") != 0, "async tma copy not found"
            assert ttgir.count(f"num = {NUM_STAGES} : i32") == 0, "num_stages not match"
            # a_tma, b_tma, output_tma, barriar
            assert ttgir.count("triton_gpu.local_alloc") == 4, "alloc number not match"
            assert ttgir.count("triton_nvidia_gpu.barrier_expect") != 0, "barrier_expect not found"
            assert ttgir.count("triton_nvidia_gpu.wait_barrier") != 0, "wait_barrier not found"
            assert ttgir.count("triton_nvidia_gpu.warp_group_dot") != 0, "warp_group_dot not found"
        else:
            # 1. check async
            assert ttgir.count("triton_gpu.async_copy_global_to_local") != 0, "async copy not found"
            # 2. check number of stages
            assert ttgir.count(f"num = {NUM_STAGES} : i32") != 0, "num_stages not match"
            # 3. check alloc
            assert ttgir.count("triton_gpu.local_alloc") == 2, "alloc number not match"
            # 4. check dot
            cc = torch.cuda.get_device_capability()
            if cc[0] >= 9:
                ttgir.count("triton_nvidia_gpu.warp_group_dot") != 0, "warp_group_dot not found"
            else:
                ttgir.count("triton_gpu.dot") != 0, "dot not found"


def test_pipeline_vecadd(device):
    check_capabilities()
    SIZE = 4096
    NUM_BLOCKS = 4
    BLOCK_SIZE = 256
    NUM_STAGES = 3
    a = torch.randn(SIZE, dtype=torch.float16, device=device)
    b = torch.randn(SIZE, dtype=torch.float16, device=device)
    output = torch.empty(SIZE, dtype=torch.float16, device=device)
    grid = (triton.cdiv(SIZE, NUM_BLOCKS * BLOCK_SIZE), 1)
    handler = vecadd_kernel[grid](a, b, output, SIZE, NUM_BLOCKS, BLOCK_SIZE, NUM_STAGES)
    ref_out = a + b
    torch.testing.assert_close(ref_out, output)
    if is_cuda():
        ttgir = handler.asm["ttgir"]
        # 1. check async
        assert ttgir.count("triton_gpu.async_copy_global_to_local") != 0, "async copy not found"
        # 2. check number of stages
        assert ttgir.count(f"num = {NUM_STAGES} : i32") != 0, "num_stages not match"
        # 3. check alloc
        assert ttgir.count("triton_gpu.local_alloc") == 2, "alloc number not match"


@pytest.mark.parametrize("ROW_COUNT", [0, 1, 2, 3])
@pytest.mark.parametrize("NUM_STAGES", [1, 2, 3, 4, 5])
def test_pipeline_epilogue(ROW_COUNT, NUM_STAGES, device):

    @triton.jit
    def kernel_up(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                  NUM_STAGES: tl.constexpr):
        row_step = tl.num_programs(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        for row_idx in tl.range(0, n_rows, row_step, num_stages=NUM_STAGES):
            row_start_ptr = input_ptr + row_idx * input_row_stride
            input_ptrs = row_start_ptr + col_offsets
            val = tl.load(input_ptrs, mask=mask, other=-float('inf'))
            val += 1.0
            output_row_start_ptr = output_ptr + row_idx * output_row_stride
            output_ptrs = output_row_start_ptr + col_offsets
            tl.store(output_ptrs, val, mask=mask)

    width = ROW_COUNT
    depth = 78
    x = torch.zeros(width, depth, device=device)
    y0 = torch.rand_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    kernel_up[(1, )](y0, x, x.stride(0), y0.stride(0), n_rows, n_cols, BLOCK_SIZE, NUM_STAGES)
    assert (y0 == torch.ones_like(x)).all()
