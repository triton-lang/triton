# RUN: python3 %s | FileCheck %s

import triton
import triton.language as tl


def print_test_name_and_run(f):
    print(f"Test: {f.__name__}")
    f()


# CHECK: Test: test_dump_kernel
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
                             USE_BUFFER_OPS: tl.constexpr,  #
                             ):
    if USE_BUFFER_OPS:
        tl.assume(M > 0)
        tl.assume(N > 0)
        tl.assume(K > 0)
        tl.assume(stride_am > 0)
        tl.assume(stride_ak > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_bn > 0)
        tl.assume(stride_cm > 0)
        tl.assume(stride_cn > 0)

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    tl.assume(1 < k_tiles)
    tl.assume(k_tiles < 10)

    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    tl.assume(1 < tiles_per_SM)
    tl.assume(tiles_per_SM < 10)

    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tl.assume(1 < tiles_per_SM)
    tl.assume(tiles_per_SM < 10)

    tile_id = start_pid - NUM_SMS
    ki = -1

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    pid_m = 0
    pid_n = 0
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    k = k_tiles * tiles_per_SM
    tl.assume(k < 10)

    for _ in range(0, k):
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
            offs_am = tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tl.arange(0, BLOCK_SIZE_N)
            offs_am = tl.where(offs_am < M - start_m, offs_am, 0)
            offs_bn = tl.where(offs_bn < N - start_n, offs_bn, 0)
            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # CHECK: amdgpu.buffer_load %arg0
        a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
        # CHECK: amdgpu.buffer_load %arg1
        b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        if ki == k_tiles - 1:
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            if c_ptr.dtype == tl.float8e4nv:
                c = accumulator.to(tl.float8e4nv)
            else:
                c = accumulator.to(tl.float16)
            tl.store(c_ptrs, c, mask=c_mask)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


@print_test_name_and_run
def test_matmul_kernel_persistent():

    kernel = triton.compile(
        triton.compiler.ASTSource(
            fn=matmul_kernel_persistent,
            signature={
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
                "BLOCK_SIZE_M": "constexpr",
                "BLOCK_SIZE_N": "constexpr",
                "BLOCK_SIZE_K": "constexpr",
                "GROUP_SIZE_M": "constexpr",
                "NUM_SMS": "constexpr",
                "USE_BUFFER_OPS": "constexpr",
            },
            constexprs={
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
                "NUM_SMS": 4,
                "USE_BUFFER_OPS": 1,
            },
        ))

    print(kernel.asm["ttgir"])
