import os
from contextlib import contextmanager

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_cuda


@contextmanager
def enable_diagnostics_context(value):
    try:
        os.environ["MLIR_ENABLE_DIAGNOSTICS"] = value
        yield
    finally:
        os.environ["MLIR_ENABLE_DIAGNOSTICS"] = ""


def test_mma_remark(capfd, fresh_triton_cache):
    if is_cuda():
        capability = torch.cuda.get_device_capability()
        if capability[0] != 9:
            pytest.skip("Requires sm = 90 to run")

    @triton.jit
    def matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
    ):
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(0, 0),
            block_shape=(32, 128),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, 0),
            block_shape=(128, 32),
            order=(0, 1),
        )
        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            offsets=(0, 0),
            block_shape=(32, 32),
            order=(1, 0),
        )
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        c = tl.dot(a, b)
        tl.store(c_block_ptr, c)

    signature = {
        "a_ptr": "*fp32",
        "b_ptr": "*fp32",
        "c_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
    }
    with enable_diagnostics_context('remarks'):
        triton.compile(triton.compiler.ASTSource(
            fn=matmul_kernel,
            signature=signature,
            constexprs={},
        ))
    captured = capfd.readouterr()

    assert ("can't use MMA V3 for the dot op" in captured.err), "expect MMA V3 remark"
    assert "note: see current operation:" not in captured.err

    with enable_diagnostics_context('remarks,operations,stacktraces'):
        triton.compile(triton.compiler.ASTSource(
            fn=matmul_kernel,
            signature=signature,
            constexprs={},
        ))
    captured = capfd.readouterr()
    assert "note: diagnostic emitted with trace:" in captured.err
    assert "note: see current operation:" in captured.err


def test_remark_vectorization(capfd, fresh_triton_cache):

    @triton.jit
    def ldst_vec(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, XBLOCK: tl.constexpr):
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        x0 = xindex % 9
        x2 = (xindex // 3456) % 512
        x1 = (xindex // 9) % 384
        x4 = xindex
        tmp0 = tl.load(in_ptr0 + (x2 + (512 * x0)), None, eviction_policy="evict_last")
        tmp1 = tmp0 + 520
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tmp9 = (-4) + tmp3
        tmp12 = tl.full([1], 512, tl.int64)
        tmp14 = tmp9 < tmp12
        tmp16 = tl.load(in_ptr3 + (x1), tmp14, eviction_policy="evict_last", other=0.0)
        tmp18 = tmp16.to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
        tmp21 = tl.where(tmp14, tmp19, tmp20)
        tmp22 = tmp21.to(tl.float32)
        tl.store(out_ptr0 + (x4), tmp22, None)

    XBLOCK = 1024

    astsource_args = {
        "fn": ldst_vec,
        "signature": {
            "in_ptr0": "*i64",
            "in_ptr1": "*i64",
            "in_ptr2": "*fp16",
            "in_ptr3": "*fp32",
            "out_ptr0": "*fp16",
            "XBLOCK": "constexpr",
        },
        "constexprs": {"XBLOCK": XBLOCK},
    }

    with enable_diagnostics_context('remarks'):
        triton.compile(
            triton.compiler.ASTSource(**astsource_args),
            options={"num_warps": 1},
        )

    _, err = capfd.readouterr()
    assert ("remark: Warning: vectorization fails" in err), "expect vectorization failure remark"
    assert "note: see current operation:" not in err

    with enable_diagnostics_context('remarks,operations,stacktraces'):
        triton.compile(
            triton.compiler.ASTSource(**astsource_args),
            options={"num_warps": 1},
        )

    _, err = capfd.readouterr()
    assert "note: see current operation:" in err
    assert "note: diagnostic emitted with trace:" in err


def test_remark_swp_op_before_operands(capfd, fresh_triton_cache):

    @triton.jit
    def kernel_pipe_error(in_ptr, out_ptr):
        SIZE: tl.constexpr = 64
        in_ptrs = in_ptr + tl.arange(0, SIZE)
        val = tl.zeros((SIZE, ), dtype=tl.float32)
        k = 0
        for i in tl.range(0, 64, num_stages=3):
            in_ptrs = in_ptr + tl.arange(0, SIZE) + SIZE * k
            val = tl.load(in_ptrs)
            out_ptrs = out_ptr + (tl.arange(0, SIZE) + i * SIZE)
            tl.store(out_ptrs, val)
            if tl.max(val) > 0:
                k += 1

    i = torch.empty(64 * 64, dtype=torch.float32).cuda()
    o = torch.empty(64 * 64, dtype=torch.float32).cuda()
    kernel_pipe_error[(1, )](i, o)


def test_remark_swp_op_before_operands_persistent_matmul(capfd, fresh_triton_cache):

    # this example is from https://github.com/triton-lang/triton/issues/5172
    @triton.jit
    def matmul_kernel_persistent(a_ptr, b_ptr, c_ptr,  #
                                 M, N, K,  #
                                 stride_am, stride_ak,  #
                                 stride_bk, stride_bn,  #
                                 stride_cm, stride_cn,  #
                                 BLOCK_SIZE_M: tl.constexpr,  #
                                 BLOCK_SIZE_N: tl.constexpr,  #
                                 BLOCK_SIZE_K: tl.constexpr,  #
                                 GROUP_SIZE_M: tl.constexpr,  #
                                 NUM_SMS: tl.constexpr,  #
                                 ):
        start_pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
        num_tiles = num_pid_m * num_pid_n

        tiles_per_SM = num_tiles // NUM_SMS
        if start_pid < num_tiles % NUM_SMS:
            tiles_per_SM += 1

        # tile_id = start_pid - NUM_SMS
        tile_id = start_pid

        ki = -1

        offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

        num_pid_in_group = GROUP_SIZE_M * num_pid_n

        pid_m = 0
        pid_n = 0
        offs_am = tl.arange(0, BLOCK_SIZE_M)
        offs_bn = tl.arange(0, BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for _ in range(0, k_tiles * tiles_per_SM):
            ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
            if ki == 0:
                # tile_id += NUM_SMS
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
                offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                c_ptrs = (c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])
                c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
                if c_ptr.dtype.element_ty == tl.float8e4nv:
                    c = accumulator.to(tl.float8e4nv)
                else:
                    c = accumulator.to(tl.float16)
                tl.store(c_ptrs, c, mask=c_mask)
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                tile_id += NUM_SMS  # this line is newly added
            else:  # deliberately increasing test coverage
                tile_id = min(tile_id, tile_id - NUM_SMS)
                tile_id += NUM_SMS

    with enable_diagnostics_context('warnings'):
        M = 8192
        N = 8192
        K = 512

        dtype = torch.float16

        a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)

        b = b.T.contiguous().T

        # Check constraints.
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        assert a.dtype == b.dtype, "Incompatible dtypes"

        # equals to 132 on H100
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        M, K = a.shape
        K, N = b.shape
        dtype = a.dtype
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=dtype)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (min(NUM_SMS,
                                 triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
        matmul_kernel_persistent[grid](
            a, b, c,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            BLOCK_SIZE_M=128,  #
            BLOCK_SIZE_N=256,  #
            BLOCK_SIZE_K=64,  #
            GROUP_SIZE_M=8,  #
            NUM_SMS=NUM_SMS,  #
            num_stages=3,  #
            num_warps=8,  #
        )

    _, err = capfd.readouterr()
    # Split the output into lines for easier processing
    lines = err.splitlines()
    # Define the expected strings in order
    expected_strings = [
        "warning: The software pipeliner failed due to a dependency conflict", "for _ in range",
        "note: The loop body is divided into 3 stages to optimize GPU I/O and computation resources.",
        "tile_id += NUM_SMS", "tile_id += NUM_SMS", "group_id = tile_id",
        "pid_m = first_pid_m + (tile_id % group_size_m)", "pid_n = (tile_id % num_pid_in_group)"
    ]
    # Initialize an index to track the position in expected_strings
    index = 0
    # Iterate over each line in the output
    for line in lines:
        # Check if the current expected string is in the line
        if expected_strings[index] in line:
            # Move to the next expected string
            index += 1
            # If all expected strings have been found, break out of the loop
            if index == len(expected_strings):
                break
        else:
            print("line: ", line)
            print("expected strings: ", expected_strings[index])
    # Check if all expected strings were found
    if index != len(expected_strings):
        missing_string = expected_strings[index]
        raise AssertionError(f"Missing expected string: '{missing_string}' from {err}")


def test_remark_swp_op_before_operands_fused_matmul(capfd, fresh_triton_cache):

    @triton.jit()
    def fused_matmul_kernel(
        a_ptr,
        b_left_ptr,
        b_right_ptr,
        bias_left_ptr,
        bias_right_ptr,
        c_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        NUM_SMS: tl.constexpr,
    ):
        start_pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
        num_tiles = num_pid_m * num_pid_n

        tiles_per_SM = num_tiles // NUM_SMS
        if start_pid < num_tiles % NUM_SMS:
            tiles_per_SM += 1

        tile_id = start_pid - NUM_SMS
        ki = -1

        offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

        num_pid_in_group = GROUP_SIZE_M * num_pid_n

        pid_m = 0
        pid_n = 0
        offs_am = tl.arange(0, BLOCK_SIZE_M)
        offs_bn = tl.arange(0, BLOCK_SIZE_N)

        acc_left = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_right = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for _ in range(0, k_tiles * tiles_per_SM):
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
                bias_left_ptrs = bias_left_ptr + offs_bn[None, :]
                bias_right_ptrs = bias_right_ptr + offs_bn[None, :]
                bias_left = tl.load(bias_left_ptrs).to(tl.float32)
                bias_right = tl.load(bias_right_ptrs).to(tl.float32)
                acc_left = bias_left.broadcast_to(BLOCK_SIZE_M, BLOCK_SIZE_N)  # num_stage = 0
                acc_right = bias_right.broadcast_to(BLOCK_SIZE_M, BLOCK_SIZE_N)
            # else:
            # we implicit yield acc_left and acc_right from the previous iteration. This causes conflicts in scheduling.

            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_left_ptrs = b_left_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            b_right_ptrs = b_right_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b_left = tl.load(
                b_left_ptrs,
                mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K,
                other=0.0,
            )
            b_right = tl.load(
                b_right_ptrs,
                mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K,
                other=0.0,
            )

            acc_left = tl.dot(a, b_left, acc_left)  # num_stage = 2
            acc_right = tl.dot(a, b_right, acc_right)

            if ki == k_tiles - 1:
                offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
                c = (acc_left * tl.sigmoid(acc_right)).to(c_ptr.dtype.element_ty)
                tl.store(c_ptrs, c, mask=c_mask)

    with enable_diagnostics_context('warnings'):

        # Define constants
        NUM_ROWS = 291840
        D_IN = 384
        D_OUT = 384
        device = torch.device("cuda")
        dtype = torch.bfloat16
        # Define matrix dimensions
        M = NUM_ROWS
        N = D_OUT * 2
        K = D_IN
        # Generate input tensors
        a = torch.randn((M, K), device=device, dtype=dtype, requires_grad=True)
        w = torch.randn((K, N), dtype=dtype, device=device)
        b = torch.randn((N, ), dtype=dtype, device=device)
        # Split the b matrix in half column-wise
        N_half = N // 2
        b_left, b_right = w.split(N_half, dim=-1)
        bias_left, bias_right = b.split(N_half)
        # Allocate output tensor
        c = torch.empty((M, N_half), device=device, dtype=dtype)
        # Get the number of streaming multiprocessors
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        # Define kernel configuration
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        }
        # Define grid size
        grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_half, META["BLOCK_SIZE_N"]),
        ), )
        # Launch the kernel
        fused_matmul_kernel[grid](
            a,
            b_left,
            b_right,
            bias_left,
            bias_right,
            c,
            M,
            N_half,
            K,
            a.stride(0),
            a.stride(1),
            w.stride(0),
            w.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            NUM_SMS=NUM_SMS,
            num_stages=config["num_stages"],
            num_warps=config["num_warps"],
        )

    _, err = capfd.readouterr()
    # Split the output into lines for easier processing
    lines = err.splitlines()
    # Define the expected strings in order
    expected_strings = [
        "warning: The software pipeliner failed due to a dependency conflict", "for _ in range",
        "note: The loop body is divided into", "acc_left = tl.dot(a, b_left, acc_left)",
        "if ki == 0:"  #implicit use from the else branch
    ]
    # Initialize an index to track the position in expected_strings
    index = 0
    # Iterate over each line in the output
    for line in lines:
        # Check if the current expected string is in the line
        if expected_strings[index] in line:
            # Move to the next expected string
            index += 1
            # If all expected strings have been found, break out of the loop
            if index == len(expected_strings):
                break
        else:
            print("line: ", line)
            print("expected strings: ", expected_strings[index])
    # Check if all expected strings were found
    if index != len(expected_strings):
        missing_string = expected_strings[index]
        raise AssertionError(f"Missing expected string: '{missing_string}' from {err}")
