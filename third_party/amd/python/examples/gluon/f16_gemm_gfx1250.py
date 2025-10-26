# ruff: noqa: E402
import hip

hip.hip.hipInit(0)
# Needed for internal dev flow for now; will remove later

import pytest
import torch

import triton
from triton.experimental import gluon
from triton.language.core import _aggregate as aggregate
import triton.experimental.gluon.language as ttgl


@aggregate
class PersistentTileScheduler:
    pid_start: ttgl.tensor
    pid_end: ttgl.tensor
    num_pid_m: ttgl.tensor

    @gluon.constexpr_function
    def __init__(self, pid_start, pid_end, num_pid_m):
        self.pid_start = pid_start
        self.pid_end = pid_end
        self.num_pid_m = num_pid_m

    @gluon.jit
    def initialize(M, N, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr):
        kernel_id = ttgl.program_id(axis=0)
        num_kernels = ttgl.num_programs(axis=0)
        num_pid_m = ttgl.cdiv(M, BLOCK_M)
        num_pid_n = ttgl.cdiv(N, BLOCK_N)
        num_pid = num_pid_m * num_pid_n
        pid_per_kernel = ttgl.cdiv(num_pid, num_kernels)
        pid_start = kernel_id * pid_per_kernel
        pid_end = min(pid_start + pid_per_kernel, num_pid)
        return PersistentTileScheduler(pid_start, pid_end, num_pid_m)

    @gluon.jit
    def get_num_tiles(self):
        return self.pid_end - self.pid_start

    @gluon.jit
    def get_tile(self, idx):
        # Delinearize the tile ID along M.
        pid = self.pid_start + idx
        pid_m = pid % self.num_pid_m
        pid_n = pid // self.num_pid_m
        return pid_m, pid_n


@gluon.jit
def issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K: ttgl.constexpr,
                NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr):
    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am, producer * BLOCK_K],  #
                                    a_buffer.index(producer % NUM_BUFFERS))
    if not TRANSPOSE_B:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [producer * BLOCK_K, off_bn],  #
                                        b_buffer.index(producer % NUM_BUFFERS))
    else:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, producer * BLOCK_K],  #
                                        b_buffer.index(producer % NUM_BUFFERS))
    producer += 1
    return producer


@gluon.jit
def issue_wmma(consumer, a_buffer, a_layout: ttgl.constexpr, b_buffer, b_layout: ttgl.constexpr, accumulator,
               wait_producers_cnt, NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr):
    ttgl.amd.gfx1250.tdm.async_wait(wait_producers_cnt)

    a = a_buffer.index(consumer % NUM_BUFFERS).load(layout=a_layout)
    if not TRANSPOSE_B:
        b = b_buffer.index(consumer % NUM_BUFFERS).load(layout=b_layout)
    else:
        b = b_buffer.index(consumer % NUM_BUFFERS).permute([1, 0]).load(layout=b_layout)

    accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)
    consumer += 1
    return consumer, accumulator


@gluon.constexpr_function
def create_shared_layouts(BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                          TRANSPOSE_B: ttgl.constexpr):
    SHARED_LAYOUT_A: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K],
                                                                                [1, 0])
    if not TRANSPOSE_B:
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 16]], [BLOCK_K, BLOCK_N],
                                                                                    [1, 0])
    else:
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_N, BLOCK_K],
                                                                                    [1, 0])

    return (SHARED_LAYOUT_A, SHARED_LAYOUT_B)


@gluon.jit
def persistent_gemm_tdm_pipelined_kernel(a_ptr, b_ptr, c_ptr,  #
                                         M, N, K,  #
                                         stride_am, stride_ak,  #
                                         stride_bk, stride_bn,  #
                                         stride_cm, stride_cn,  #
                                         BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,  #
                                         NUM_BUFFERS: ttgl.constexpr,  #
                                         TRANSPOSE_B: ttgl.constexpr,  #
                                         NUM_WARPS: ttgl.constexpr):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, [NUM_WARPS // 2, 2], [16, 16, 32])
    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(  #
        base=a_ptr,  #
        shape=(M, K),  #
        strides=(stride_am, stride_ak),  #
        block_shape=(BLOCK_M, BLOCK_K),  #
        layout=SHARED_LAYOUT_A)

    if not TRANSPOSE_B:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(  #
            base=b_ptr,  #
            shape=(K, N),  #
            strides=(stride_bk, stride_bn),  #
            block_shape=(BLOCK_K, BLOCK_N),  #
            layout=SHARED_LAYOUT_B)
    else:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(  #
            base=b_ptr,  #
            shape=(N, K),  #
            strides=(stride_bn, stride_bk),  #
            block_shape=(BLOCK_N, BLOCK_K),  #
            layout=SHARED_LAYOUT_B)

    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    scheduler = PersistentTileScheduler.initialize(M, N, BLOCK_M, BLOCK_N)

    for tile_idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(tile_idx)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        producer = 0
        consumer = 0
        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

        for _ in ttgl.static_range(NUM_BUFFERS - 1):
            producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                   TRANSPOSE_B)

        for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
            producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                   TRANSPOSE_B)
            consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                               accumulator, (NUM_BUFFERS - 1) * 2, NUM_BUFFERS, TRANSPOSE_B)

        for i in ttgl.static_range(NUM_BUFFERS - 1):
            consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                               accumulator, (NUM_BUFFERS - 2 - i) * 2, NUM_BUFFERS, TRANSPOSE_B)

        offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
        offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)


@gluon.jit
def persistent_gemm_tdm_pipelined_with_prefetch_kernel(a_ptr, b_ptr, c_ptr,  #
                                                       M, N, K,  #
                                                       stride_am, stride_ak,  #
                                                       stride_bk, stride_bn,  #
                                                       stride_cm, stride_cn,  #
                                                       BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                                       BLOCK_K: ttgl.constexpr,  #
                                                       NUM_BUFFERS: ttgl.constexpr,  #
                                                       TRANSPOSE_B: ttgl.constexpr,  #
                                                       NUM_WARPS: ttgl.constexpr):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, [NUM_WARPS // 2, 2], [16, 16, 32])
    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(  #
        base=a_ptr,  #
        shape=(M, K),  #
        strides=(stride_am, stride_ak),  #
        block_shape=(BLOCK_M, BLOCK_K),  #
        layout=SHARED_LAYOUT_A)

    if not TRANSPOSE_B:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(  #
            base=b_ptr,  #
            shape=(K, N),  #
            strides=(stride_bk, stride_bn),  #
            block_shape=(BLOCK_K, BLOCK_N),  #
            layout=SHARED_LAYOUT_B)
    else:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(  #
            base=b_ptr,  #
            shape=(N, K),  #
            strides=(stride_bn, stride_bk),  #
            block_shape=(BLOCK_N, BLOCK_K),  #
            layout=SHARED_LAYOUT_B)

    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    scheduler = PersistentTileScheduler.initialize(M, N, BLOCK_M, BLOCK_N)
    num_tiles = scheduler.get_num_tiles()
    producer = 0

    pid_m, pid_n = scheduler.get_tile(0)
    off_am = pid_m * BLOCK_M
    off_bn = pid_n * BLOCK_N
    for i in ttgl.static_range(NUM_BUFFERS - 1):
        producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                               TRANSPOSE_B)

    for tile_idx in range(num_tiles):
        pid_m_next, pid_n_next = scheduler.get_tile(tile_idx + 1)
        off_am_next = pid_m_next * BLOCK_M
        off_bn_next = pid_n_next * BLOCK_N

        consumer = 0
        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

        for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
            producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                   TRANSPOSE_B)
            consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                               accumulator, (NUM_BUFFERS - 1) * 2, NUM_BUFFERS, TRANSPOSE_B)

        producer = 0
        for i in ttgl.static_range(NUM_BUFFERS - 1):
            if tile_idx + 1 < num_tiles:
                producer = issue_loads(producer, a_desc, b_desc, off_am_next, off_bn_next, a_buffer, b_buffer, BLOCK_K,
                                       NUM_BUFFERS, TRANSPOSE_B)
            consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                               accumulator, (NUM_BUFFERS - 2 - i) * 2, NUM_BUFFERS, TRANSPOSE_B)

        offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
        offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)

        pid_m, pid_n = pid_m_next, pid_n_next
        off_am, off_bn = off_am_next, off_bn_next


@gluon.jit
def gemm_tdm_pipelined_kernel(a_ptr, b_ptr, c_ptr,  #
                              M, N, K,  #
                              stride_am, stride_ak,  #
                              stride_bk, stride_bn,  #
                              stride_cm, stride_cn,  #
                              BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,  #
                              NUM_BUFFERS: ttgl.constexpr,  #
                              TRANSPOSE_B: ttgl.constexpr,  #
                              NUM_WARPS: ttgl.constexpr):
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")

    WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(3, True, [NUM_WARPS // 2, 2], [16, 16, 32])
    shared_layouts: ttgl.constexpr = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B)
    SHARED_LAYOUT_A: ttgl.constexpr = shared_layouts[0]
    SHARED_LAYOUT_B: ttgl.constexpr = shared_layouts[1]
    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(  #
        base=a_ptr + pid_m * BLOCK_M * stride_am,  #
        shape=(M, K),  #
        strides=(stride_am, stride_ak),  #
        block_shape=(BLOCK_M, BLOCK_K),  #
        layout=SHARED_LAYOUT_A)
    if not TRANSPOSE_B:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(  #
            base=b_ptr + pid_n * BLOCK_N * stride_bn,  #
            shape=(K, N),  #
            strides=(stride_bk, stride_bn),  #
            block_shape=(BLOCK_K, BLOCK_N),  #
            layout=SHARED_LAYOUT_B)
    else:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(  #
            base=b_ptr + pid_n * BLOCK_N * stride_bn,  #
            shape=(N, K),  #
            strides=(stride_bn, stride_bk),  #
            block_shape=(BLOCK_N, BLOCK_K),  #
            layout=SHARED_LAYOUT_B)

    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    producer = 0
    consumer = 0
    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

    for _ in ttgl.static_range(NUM_BUFFERS - 1):
        producer = issue_loads(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B)

    for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        producer = issue_loads(producer, a_desc, b_desc, 0, 0, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B)
        consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                           accumulator, (NUM_BUFFERS - 1) * 2, NUM_BUFFERS, TRANSPOSE_B)

    for i in ttgl.static_range(NUM_BUFFERS - 1):
        consumer, accumulator = issue_wmma(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                           accumulator, (NUM_BUFFERS - 2 - i) * 2, NUM_BUFFERS, TRANSPOSE_B)

    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(32, 32, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [2, 4])
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.parametrize("PERSISTENT", [False, True])
@pytest.mark.parametrize("PREFETCH", [False, True])
@pytest.mark.parametrize("M,N,K", [(256, 256, 512), (250, 250, 510)])
@pytest.mark.parametrize("num_warps", [4, 8])
def test_runtime_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, PERSISTENT, PREFETCH, M, N, K,
                                    num_warps):
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        pytest.skip("Skip tests where K/BLOCK_K < NUM_BUFFERS")

    torch.manual_seed(42)

    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)
    if TRANSPOSE_B:
        b = b.T.contiguous()
    c = torch.zeros((M, N), dtype=torch.float32)
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = (b.stride(0), b.stride(1)) if not TRANSPOSE_B else (b.stride(1), b.stride(0))
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    a_device = a.cuda()
    b_device = b.cuda()
    c_device = c.cuda()

    if not PERSISTENT:
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
        gemm_tdm_pipelined_kernel[grid](
            a_device, b_device, c_device,  #
            M, N, K,  #
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,  #
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
            NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B, NUM_WARPS=num_warps,  #
            num_warps=num_warps, waves_per_eu=num_warps // 4)
    else:
        # num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        # NOTE: Explicitly set num_sms to small number to ensure that each CU will compute multiple tiles.
        num_sms = 8
        grid = (min(num_sms, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), 1)
        if PREFETCH:
            persistent_gemm_tdm_pipelined_with_prefetch_kernel[grid](
                a_device, b_device, c_device,  #
                M, N, K,  #
                stride_am, stride_ak,  #
                stride_bk, stride_bn,  #
                stride_cm, stride_cn,  #
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
                NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B, NUM_WARPS=num_warps,  #
                num_warps=num_warps, waves_per_eu=num_warps // 4)
        else:
            persistent_gemm_tdm_pipelined_kernel[grid](
                a_device, b_device, c_device,  #
                M, N, K,  #
                stride_am, stride_ak,  #
                stride_bk, stride_bn,  #
                stride_cm, stride_cn,  #
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
                NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B, NUM_WARPS=num_warps,  #
                num_warps=num_warps, waves_per_eu=num_warps // 4)

    c_triton = c_device.cpu()
    c_torch = a.to(torch.float32) @ (b.to(torch.float32) if not TRANSPOSE_B else b.T.to(torch.float32))
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    M, N, K = 256, 256, 1024
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 128
    NUM_BUFFERS = 2
    NUM_WARPS = 4
    TRANSPOSE_B = True
    PERSISTENT = True
    PREFETCH = False
    print(
        f"({M=}, {N=}, {K=}), ({BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}), {NUM_BUFFERS=}, {TRANSPOSE_B=}, {PERSISTENT=}, {PREFETCH=}, {NUM_WARPS=}"
    )
    test_runtime_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, PERSISTENT, PREFETCH, M, N, K,
                                    NUM_WARPS)
    NUM_WARPS = 8
    print(
        f"({M=}, {N=}, {K=}), ({BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}), {NUM_BUFFERS=}, {TRANSPOSE_B=}, {PERSISTENT=}, {PREFETCH=}, {NUM_WARPS=}"
    )
    test_runtime_gemm_tdm_pipelined(BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, PERSISTENT, PREFETCH, M, N, K,
                                    NUM_WARPS)
