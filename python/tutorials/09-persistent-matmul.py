"""
Persistent Matmul
=====================
This script demonstrates persistent kernel implementations of matrix multiplication using Triton.
Various matmul methods are included, such as naive, persistent, and TMA (Tensor Memory Accelerator) based approaches.
The kernels support both FP16 and FP8 data types but the FP8 implementation is only available on CUDA devices with compute capability >= 9.0.

Triton and cuBLAS implementations are benchmarked under different configurations and evaluated using the proton profiler.
Users can pass command-line arguments to specify matrix dimensions and iteration steps flexibly.

.. code-block:: bash

    # FP8
    python 09-persistent-matmul.py --prec fp8 --K_range 128 1024 --K_step 128

    # FP16
    python 09-persistent-matmul.py --prec fp16 --K_range 128 1024 --K_step 128

Note that currently this tutorial will fail on devices with a small shared memory size, such as RTX-4090.
"""

import argparse

import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
import triton.profiler as proton
from contextlib import contextmanager

from typing import Optional

if torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)

if HAS_TMA_DESC:
    print("TMA benchmarks will be running with experimental grid constant TMA descriptor.", )
else:
    print("TMA benchmarks will be running without grid constant TMA descriptor.", )


# TmaAutoTuneHelper used in htyu's PR #5622
class TmaAutoTuneHelper:

    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:

        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = (triton.runtime.driver.active.utils.fill_1d_tma_descriptor)
        self.fill_2d_tma_descriptor_inner = (triton.runtime.driver.active.utils.fill_2d_tma_descriptor)
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8)
        else:
            self.cuda_descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8)

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]


def matmul_get_configs():
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K" : BK, "GROUP_SIZE_M" : 8}, num_stages=s, num_warps=w) \
        for BM in [128] \
        for BN in [128, 256] \
        for BK in [64,128] \
        for s in ([3,4]) \
        for w in [4,8] \
    ]


@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel(a_ptr, b_ptr, c_ptr,  #
                  M, N, K,  #
                  stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_cm, stride_cn,  #
                  BLOCK_SIZE_M: tl.constexpr,  #
                  BLOCK_SIZE_N: tl.constexpr,  #
                  BLOCK_SIZE_K: tl.constexpr,  #
                  GROUP_SIZE_M: tl.constexpr,  #
                  ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    return c


@triton.jit
def _compute_tile_and_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    tile_id += NUM_SMS
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return tile_id, pid_m, pid_n


@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
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

    # NOTE: There is currently a bug in blackwell pipelining that means it can't handle a value being
    # used in both the prologue and epilogue, so we duplicate the counters as a work-around.
    tile_id = start_pid - NUM_SMS
    tile_id_c = start_pid - NUM_SMS
    ki = -1

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id, pid_m, pid_n = _compute_tile_and_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
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
            tile_id_c, pid_m, pid_n = _compute_tile_and_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M,
                                                            NUM_SMS)
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            if (c_ptr.dtype.element_ty == tl.float8e4nv):
                c = accumulator.to(tl.float8e4nv)
            else:
                c = accumulator.to(tl.float16)
            tl.store(c_ptrs, c, mask=c_mask)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_persistent(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_persistent[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
    )
    return c


def matmul_tma_persistent_get_configs():
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K" : BK, "GROUP_SIZE_M" : 8, "EPILOGUE_SUBTILE" : SUBTILE}, num_stages=s, num_warps=w) \
        for BM in [128] \
        for BN in [128, 256] \
        for BK in [64, 128] \
        for s in ([3, 4]) \
        for w in [4, 8] \
        for SUBTILE in [True, False] \
    ]


@triton.autotune(
    configs=matmul_tma_persistent_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(a_desc_ptr, b_desc_ptr, c_desc_ptr,  #
                                 M, N, K,  #
                                 BLOCK_SIZE_M: tl.constexpr,  #
                                 BLOCK_SIZE_N: tl.constexpr,  #
                                 BLOCK_SIZE_K: tl.constexpr,  #
                                 GROUP_SIZE_M: tl.constexpr,  #
                                 FP8_OUTPUT: tl.constexpr,  #
                                 EPILOGUE_SUBTILE: tl.constexpr,  #
                                 NUM_SMS: tl.constexpr):  #
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    tile_id_c = start_pid - NUM_SMS

    ki = -1

    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id, pid_m, pid_n = _compute_tile_and_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype)
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            tile_id_c, pid_m, pid_n = _compute_tile_and_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M,
                                                            NUM_SMS)

            offs_am_c = pid_m * BLOCK_SIZE_M
            offs_bn_c = pid_n * BLOCK_SIZE_N

            # Epilogue subtiling is a technique to break our computation and stores into multiple pieces
            # By subtiling we can reduce shared memory consumption by the epilogue and instead use that
            # memory to increase our stage count.
            # In this case we partition the accumulator into 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(dtype)
                tl._experimental_descriptor_store(c_desc_ptr, c0, [offs_am_c, offs_bn_c])
                c1 = acc1.to(dtype)
                tl._experimental_descriptor_store(c_desc_ptr, c1, [offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            else:
                accumulator = accumulator.to(dtype)
                tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am_c, offs_bn_c])

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_tma_persistent(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    desc_helper = TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("a")
    desc_helper.init_tma_descriptor("b")
    desc_helper.init_tma_descriptor("c")

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(META):
        nonlocal desc_helper
        desc_helper.fill_2d_tma_descriptor(
            "a",
            a.data_ptr(),
            M,
            K,
            META["BLOCK_SIZE_M"],
            META["BLOCK_SIZE_K"],
            a.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "b",
            b.data_ptr(),
            N,
            K,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            b.element_size(),
        )

        store_block_n = META["BLOCK_SIZE_N"]

        if META["EPILOGUE_SUBTILE"]:
            store_block_n = store_block_n // 2

        desc_helper.fill_2d_tma_descriptor(
            "c",
            c.data_ptr(),
            M,
            N,
            META["BLOCK_SIZE_M"],
            store_block_n,
            c.element_size(),
        )

        return (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

    desc_a = desc_helper.get_tma_descriptor_kernel_param("a")
    desc_b = desc_helper.get_tma_descriptor_kernel_param("b")
    desc_c = desc_helper.get_tma_descriptor_kernel_param("c")

    matmul_kernel_tma_persistent[grid](
        desc_a, desc_b, desc_c,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
    )
    return c


@triton.autotune(
    configs=matmul_tma_persistent_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_descriptor_persistent(a_ptr, b_ptr, c_ptr,  #
                                        M, N, K,  #
                                        BLOCK_SIZE_M: tl.constexpr,  #
                                        BLOCK_SIZE_N: tl.constexpr,  #
                                        BLOCK_SIZE_K: tl.constexpr,  #
                                        GROUP_SIZE_M: tl.constexpr,  #
                                        EPILOGUE_SUBTILE: tl.constexpr,  #
                                        NUM_SMS: tl.constexpr):  #
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl._experimental_make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl._experimental_make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl._experimental_make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2],
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    tile_id_c = start_pid - NUM_SMS
    ki = -1

    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id, pid_m, pid_n = _compute_tile_and_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            tile_id_c, pid_m, pid_n = _compute_tile_and_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M,
                                                            NUM_SMS)
            offs_cm = pid_m * BLOCK_SIZE_M
            offs_cn = pid_n * BLOCK_SIZE_N

            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(dtype)
                c_desc.store([offs_cm, offs_cn], c0)
                c1 = acc1.to(dtype)
                c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
            else:
                c = accumulator.to(dtype)
                c_desc.store([offs_cm, offs_cn], c)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_descriptor_persistent(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_descriptor_persistent[grid](
        a, b, c,  #
        M, N, K,  #
        NUM_SMS=NUM_SMS,  #
    )
    return c


def cublas_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(f"cublas [M={M}, N={N}, K={K}]",
                      {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2. * M * N * K}):
        cublas.matmul(a, b, c)
    return c


def torch_matmul(a, b):
    M, K = a.shape
    N, K = b.shape
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(f"torch [M={M}, N={N}, K={K}]",
                      {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2. * M * N * K}):
        c = torch.matmul(a, b.T)
    return c


@contextmanager
def proton_context():
    proton.activate(0)
    try:
        yield
    finally:
        proton.deactivate(0)


def bench_fn(reps, warmup_reps, fn, *args):
    for _ in range(warmup_reps):
        fn(*args)
    with proton_context():
        for _ in range(reps):
            fn(*args)


def bench(K, dtype, reps=1000, warmup_reps=10000):
    M = 8192
    N = 8192
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)

    b = b.T.contiguous()

    if cublas is not None:
        bench_fn(reps, warmup_reps, cublas_matmul, a, b)
    if dtype == torch.float16:
        bench_fn(reps, warmup_reps, torch_matmul, a, b)
    bench_fn(reps, warmup_reps, matmul, a, b.T)
    bench_fn(reps, warmup_reps, matmul_persistent, a, b.T)
    if supports_tma():
        bench_fn(reps, warmup_reps, matmul_tma_persistent, a, b)
        bench_fn(reps, warmup_reps, matmul_descriptor_persistent, a, b)


def validate(M, N, K, dtype):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = b.T.contiguous()

    torch_result = torch_matmul(a, b) if dtype == torch.float16 else None
    cublas_result = cublas_matmul(a, b) if cublas is not None else None
    naive_result = matmul(a, b.T)
    persistent_result = matmul_persistent(a, b.T)
    tma_persistent_result = matmul_tma_persistent(a, b) if supports_tma() else None
    descriptor_persistent_result = matmul_descriptor_persistent(a, b) if supports_tma() else None

    if torch_result is not None:
        naive_vs_torch = "✅" if torch.allclose(naive_result.to(torch.float16), torch_result.to(torch.float16),
                                               atol=1.0) else "❌"
    if cublas_result is not None:
        naive_vs_cublas = "✅" if torch.allclose(naive_result.to(torch.float16), cublas_result.to(torch.float16),
                                                atol=1.0) else "❌"
    naive_vs_persistent = "✅" if torch.allclose(naive_result.to(torch.float16), persistent_result.to(torch.float16),
                                                atol=1.0) else "❌"
    if tma_persistent_result is not None:
        naive_vs_tma_persistent = "✅" if torch.allclose(cublas_result.to(torch.float16),
                                                        tma_persistent_result.to(torch.float16), atol=1.0) else "❌"
    if descriptor_persistent_result is not None:
        naive_vs_descriptor_persistent = "✅" if torch.allclose(cublas_result.to(
            torch.float16), descriptor_persistent_result.to(torch.float16), atol=1.0) else "❌"
    print(f"M={M}, N={N}, K={K} verification naive vs: ", end="")
    if torch_result is not None:
        print(f"torch: {naive_vs_torch} ", end="")
    if cublas_result is not None:
        print(f"cublas: {naive_vs_cublas} ", end="")
    print(f"persistent: {naive_vs_persistent} ", end="")
    if tma_persistent_result is not None:
        print(f"TMA persistent: {naive_vs_tma_persistent} ", end="")
    if descriptor_persistent_result is not None:
        print(f"Tensor descriptor persistent: {naive_vs_descriptor_persistent} ", end="")
    print()


def show_profile(precision, profile_name):
    import triton.profiler.viewer as proton_viewer
    metric_names = ["time/ms"]
    if precision == 'fp8':
        metric_names = ["tflop8/s"] + metric_names
    elif precision == 'fp16':
        metric_names = ["tflop16/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=False, default=512)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--prec", type=str, choices=["fp8", "fp16"], default="fp16")
    args = parser.parse_args()

    if args.prec == 'fp8' and (not hasattr(torch, "float8_e4m3fn") or not is_cuda()):
        print("This example requires CUDA with fp8 support.")
        exit(1)

    dtype = torch.float8_e4m3fn if args.prec == 'fp8' else torch.float16

    if args.K and args.K_range is None:
        args.K_range = [args.K, args.K]
        args.K_step = 1  # doesn't matter as long as it's not 0

    torch.manual_seed(0)

    validate(32, 32, 32, dtype)
    validate(8192, 8192, args.K_range[0], dtype)

    proton.start("matmul", hook="triton")
    for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
        bench(K, dtype)
    proton.finalize()
    show_profile(args.prec, "matmul")
