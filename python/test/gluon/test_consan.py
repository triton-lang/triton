import os
import torch
import pytest
import triton
from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia import blackwell
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia import ampere
from triton.experimental.gluon.language.nvidia.blackwell import mbarrier, tma
from triton._internal_testing import is_cuda
import multiprocessing
import tempfile
from typing import Optional

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass  # start method already set


@pytest.fixture
def run_wrapper():
    # Use DISABLE_SUBPROCESS to run the tests in the main process
    # (useful for debugging but assert in any test will make all the tests fail)
    return not os.environ.get("DISABLE_SUBPROCESS")


class ProcessResult:

    def __init__(self, exc, driver_stderr_output):
        self.exc = exc
        self.driver_stderr_output = driver_stderr_output


def target(client_fn, queue: multiprocessing.Queue, args, kwargs):
    # Prepare temp file for capturing low-level stderr
    with tempfile.TemporaryFile(mode="w+") as tmp_stderr:
        saved_stderr_fd = os.dup(2)
        os.dup2(tmp_stderr.fileno(), 2)  # Redirect fd 2 to tmp_stderr
        exc = None

        try:
            client_fn(*args, **kwargs)
        except Exception as e:
            exc = e
        finally:
            # Restore original stderr
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)

            # Read driver stderr
            tmp_stderr.seek(0)
            driver_stderr_output = tmp_stderr.read()
            queue.put(ProcessResult(exc, driver_stderr_output))


def run_in_process(client_fn, args=(), kwargs={}):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=target, args=(client_fn, queue, args, kwargs))
    p.start()
    p.join()
    result = queue.get()
    return result


@gluon.jit
def async_tma_kernel(input_desc, XBLOCK: ttgl.constexpr, FAILURE: ttgl.constexpr):
    smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], input_desc.layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    mbarrier.expect(bar, XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
    tma.async_copy_global_to_shared(input_desc, [0, 0], bar, smem)
    tma.async_copy_global_to_shared(input_desc, [0, 0], bar, smem, pred=FAILURE)
    mbarrier.wait(bar, 0)

    tma.async_copy_global_to_shared(input_desc, [0, 0], bar, smem, pred=(not FAILURE))
    mbarrier.wait(bar, 0, pred=(not FAILURE))

    mbarrier.invalidate(bar)

    tma.async_copy_shared_to_global(input_desc, [0, 0], smem)
    tma.store_wait(0)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_async_tma_kernel(FAILURE, device, run_wrapper):
    if run_wrapper:
        result = run_in_process(test_async_tma_kernel, (FAILURE, device, False))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        return

    knobs.compilation.enable_experimental_consan = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    XBLOCK = 128
    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK, XBLOCK], shared_layout)
    async_tma_kernel[(1, )](input_desc, XBLOCK, FAILURE=FAILURE, num_warps=4)


@gluon.jit
def tma_interleave_kernel(input_desc, XBLOCK: ttgl.constexpr, FAILURE: ttgl.constexpr):
    smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK, XBLOCK], input_desc.layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], mbarrier.MBarrierLayout())
    mbarrier.init(bar.index(0), count=1)
    mbarrier.init(bar.index(1), count=1)

    mbarrier.expect(bar.index(0), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
    mbarrier.expect(bar.index(1), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
    tma.async_copy_global_to_shared(input_desc, [0, 0], bar.index(0), smem.index(0))
    tma.async_copy_global_to_shared(input_desc, [0, 0], bar.index(1), smem.index(1))

    mbarrier.wait(bar.index(0), 0)
    if not FAILURE:
        mbarrier.wait(bar.index(1), 0)

    mbarrier.expect(bar.index(0), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
    mbarrier.expect(bar.index(1), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
    tma.async_copy_global_to_shared(input_desc, [0, 0], bar.index(0), smem.index(0))
    tma.async_copy_global_to_shared(input_desc, [0, 0], bar.index(1), smem.index(1))

    mbarrier.wait(bar.index(0), 1)
    mbarrier.wait(bar.index(1), 1)

    mbarrier.invalidate(bar.index(0))
    mbarrier.invalidate(bar.index(1))

    tma.async_copy_shared_to_global(input_desc, [0, 0], smem.index(0))
    tma.store_wait(0)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_tma_interleave_kernel(FAILURE, device, run_wrapper):
    if run_wrapper:
        result = run_in_process(test_tma_interleave_kernel, (FAILURE, device, False))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Barrier is being reused while still tracking writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    knobs.compilation.enable_experimental_consan = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    XBLOCK = 128
    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK, XBLOCK], shared_layout)
    tma_interleave_kernel[(1, )](input_desc, XBLOCK, FAILURE=FAILURE, num_warps=4)


@gluon.jit
def async_copy_kernel(input, XBLOCK: ttgl.constexpr, FAILURE: ttgl.constexpr):
    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK, XBLOCK], smem_layout)
    blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                        warps_per_cta=[4, 1], order=[0, 1])
    offs_m = ttgl.arange(0, XBLOCK, layout=ttgl.SliceLayout(dim=1, parent=blocked_layout))[:, None]
    offs_n = ttgl.arange(0, XBLOCK, layout=ttgl.SliceLayout(dim=0, parent=blocked_layout))[None, :]
    offs = offs_m * XBLOCK + offs_n
    ampere.async_copy.async_copy_global_to_shared(smem.index(0), input + offs)
    ampere.async_copy.commit_group()

    ampere.async_copy.async_copy_global_to_shared(smem.index(1), input + offs)
    ampere.async_copy.commit_group()
    ampere.async_copy.wait_group(2 if FAILURE else 1)

    ampere.async_copy.async_copy_global_to_shared(smem.index(0), input + offs)
    ampere.async_copy.commit_group()
    ampere.async_copy.wait_group(0)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires ampere or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_async_copy(FAILURE, device, run_wrapper):
    if run_wrapper:
        result = run_in_process(test_async_copy, (FAILURE, device, False))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Accessing buffer with pending access. Pending access type: async_copy_global_to_shared" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    knobs.compilation.enable_experimental_consan = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    XBLOCK = 128
    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    async_copy_kernel[(1, )](input, XBLOCK, FAILURE=FAILURE, num_warps=4)


@gluon.jit
def tcgen5_mma_kernel(input_desc, XBLOCK: ttgl.constexpr, FAILURE: ttgl.constexpr, MEM_ACCESS_KIND: ttgl.constexpr):
    acc_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([XBLOCK, XBLOCK], unpacked=True, cta_split_num=[1, 1])
    blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                        warps_per_cta=[4, 1], order=[0, 1])
    smemA = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], input_desc.layout)
    smemB = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], input_desc.layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc = blackwell.allocate_tensor_memory(ttgl.float32, [XBLOCK, XBLOCK], acc_layout)
    mbarrier.init(bar.index(0), count=1)
    mbarrier.init(bar.index(1), count=1)

    blackwell.tcgen05_mma(smemA, smemB.permute([1, 0]), acc)
    blackwell.tcgen05_commit(bar.index(0))

    if not FAILURE:
        mbarrier.wait(bar.index(0), 0)

    if MEM_ACCESS_KIND == "tma_cp":
        mbarrier.expect(bar.index(1), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar.index(1), smemA)
        mbarrier.wait(bar.index(1), 0)
    elif MEM_ACCESS_KIND == "local_store":
        smemA.store(ttgl.full([XBLOCK, XBLOCK], 42, ttgl.float16, blocked_layout))
    elif MEM_ACCESS_KIND == "tmem_load":
        res = acc.load(blocked_layout)
        smemAcc = ttgl.allocate_shared_memory(ttgl.float32, [XBLOCK, XBLOCK], input_desc.layout, res)
        tma.async_copy_shared_to_global(input_desc, [0, 0], smemAcc)
        tma.store_wait(0)
    elif MEM_ACCESS_KIND == "tmem_store":
        acc.store(ttgl.full([XBLOCK, XBLOCK], 42, ttgl.float32, blocked_layout))

    mbarrier.invalidate(bar.index(0))
    mbarrier.invalidate(bar.index(1))


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
@pytest.mark.parametrize("MEM_ACCESS_KIND", ["tma_cp", "local_store", "tmem_load", "tmem_store"])
def test_tcgen5_mma(FAILURE, MEM_ACCESS_KIND, device, run_wrapper):
    if run_wrapper:
        result = run_in_process(test_tcgen5_mma, (FAILURE, MEM_ACCESS_KIND, device, False))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            if MEM_ACCESS_KIND == "tma_cp":
                # shmem operands are being read by the tcgen05_mma
                assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
            elif MEM_ACCESS_KIND in ["tmem_load", "tmem_store"]:
                # tmem is being written by the tcgen05_mma
                assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    knobs.compilation.enable_experimental_consan = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    XBLOCK = 128
    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK, XBLOCK], shared_layout)
    tcgen5_mma_kernel[(1, )](input_desc, XBLOCK, FAILURE=FAILURE, MEM_ACCESS_KIND=MEM_ACCESS_KIND, num_warps=4)


@gluon.jit
def warpgroup_mma_kernel(input, XBLOCK: ttgl.constexpr, FAILURE: ttgl.constexpr):
    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    smemA = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], smem_layout)
    smemB = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], smem_layout)

    blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                        warps_per_cta=[4, 1], order=[0, 1])

    acc_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                             instr_shape=[16, 32, 16])
    acc = ttgl.zeros([XBLOCK, XBLOCK], ttgl.float16, acc_layout)
    acc = hopper.warpgroup_mma(smemA, smemB, acc, is_async=True)
    if FAILURE:
        smemA.store(ttgl.full([XBLOCK, XBLOCK], 42, ttgl.float16, blocked_layout))
    hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
    smemA.store(ttgl.full([XBLOCK, XBLOCK], 42, ttgl.float16, blocked_layout))


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_warpgroup_mma(FAILURE, device, run_wrapper):
    if run_wrapper:
        result = run_in_process(test_warpgroup_mma, (FAILURE, device, False))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Accessing buffer with pending access. Pending access type: warpgroup_mma operand read" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    knobs.compilation.enable_experimental_consan = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    XBLOCK = 128
    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    warpgroup_mma_kernel[(1, )](input, XBLOCK, FAILURE=FAILURE)


@gluon.jit
def warpgroup_mma_kernel2(input, XBLOCK: ttgl.constexpr, FAILURE: ttgl.constexpr):
    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    smemA = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], smem_layout)
    smemB = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], smem_layout)

    blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                        warps_per_cta=[4, 1], order=[0, 1])

    acc_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                             instr_shape=[16, 32, 16])
    acc = ttgl.zeros([XBLOCK, XBLOCK], ttgl.float16, acc_layout)
    acc = hopper.warpgroup_mma(smemA, smemB, acc, is_async=True)
    acc = hopper.warpgroup_mma(smemA, smemB, acc, is_async=True)
    hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc])
    if FAILURE:
        smemA.store(ttgl.full([XBLOCK, XBLOCK], 42, ttgl.float16, blocked_layout))
    hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
    smemA.store(ttgl.full([XBLOCK, XBLOCK], 42, ttgl.float16, blocked_layout))


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_warpgroup_mma2(FAILURE, device, run_wrapper):
    if run_wrapper:
        result = run_in_process(test_warpgroup_mma2, (FAILURE, device, False))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Accessing buffer with pending access. Pending access type: warpgroup_mma operand read" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    knobs.compilation.enable_experimental_consan = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    XBLOCK = 128
    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    warpgroup_mma_kernel[(1, )](input, XBLOCK, FAILURE=FAILURE)


@gluon.jit
def tcgen5_mma_multibar_kernel(input_desc, XBLOCK: ttgl.constexpr, BUF_IDX: ttgl.constexpr, BAR_IDX: ttgl.constexpr):
    acc_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([XBLOCK, XBLOCK], unpacked=True, cta_split_num=[1, 1])
    blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                        warps_per_cta=[4, 1], order=[0, 1])
    smemA = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], input_desc.layout)
    smemB = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], input_desc.layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [4, 1], mbarrier.MBarrierLayout())
    acc = blackwell.allocate_tensor_memory(ttgl.float32, [2, XBLOCK, XBLOCK], acc_layout)
    for i in range(4):
        mbarrier.init(bar.index(i), count=1)

    blackwell.tcgen05_mma(smemA, smemB.permute([1, 0]), acc.index(0), mbarriers=[bar.index(0),
                                                                                 bar.index(1)],
                          mbarrier_preds=[False, True])
    blackwell.tcgen05_mma(smemA, smemB.permute([1, 0]), acc.index(1), mbarriers=[bar.index(2)])
    blackwell.tcgen05_commit(bar.index(3))

    mbarrier.wait(bar.index(BAR_IDX), 0)

    acc.index(BUF_IDX).store(ttgl.full([XBLOCK, XBLOCK], 42, ttgl.float32, blocked_layout))

    for i in range(4):
        mbarrier.invalidate(bar.index(i))


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("BUF_IDX", [0, 1])
@pytest.mark.parametrize("BAR_IDX", [0, 1, 2, 3])
def test_tcgen5_mma_multibar(BUF_IDX, BAR_IDX, device, run_wrapper):
    if BAR_IDX == 0:
        pytest.skip("Skipping due to wait on false-predicated barrier - not supported yet")
    if run_wrapper:
        result = run_in_process(test_tcgen5_mma_multibar, (BUF_IDX, BAR_IDX, device, False))
        if BAR_IDX // 2 < BUF_IDX:
            assert "device-side assert" in str(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    knobs.compilation.enable_experimental_consan = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    XBLOCK = 128

    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK, XBLOCK], shared_layout)
    tcgen5_mma_multibar_kernel[(1, )](input_desc, XBLOCK, BUF_IDX, BAR_IDX, num_warps=4)


@gluon.jit
def inc_mod(x, mod):
    return (x + 1) % mod


@gluon.jit
def multibuffered_loop_tma_kernel(input_desc, XBLOCK: ttgl.constexpr, FAILURE: ttgl.constexpr):
    num_buffers: ttgl.constexpr = 2 if FAILURE else 3
    num_mma_stages: ttgl.constexpr = 2

    acc_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([XBLOCK, XBLOCK], unpacked=True, cta_split_num=[1, 1])
    blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                        warps_per_cta=[4, 1], order=[0, 1])
    zero = ttgl.zeros([XBLOCK, XBLOCK], ttgl.float32, blocked_layout)

    smemA = ttgl.allocate_shared_memory(ttgl.float16, [num_buffers, XBLOCK, XBLOCK], input_desc.layout)
    smemB = ttgl.allocate_shared_memory(ttgl.float16, [num_buffers, XBLOCK, XBLOCK], input_desc.layout)
    barLoadA = ttgl.allocate_shared_memory(ttgl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    barLoadB = ttgl.allocate_shared_memory(ttgl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    barMMA = ttgl.allocate_shared_memory(ttgl.int64, [num_mma_stages, 1], mbarrier.MBarrierLayout())
    acc = blackwell.allocate_tensor_memory(ttgl.float32, [XBLOCK, XBLOCK], acc_layout, zero)
    for i in range(num_buffers):
        mbarrier.init(barLoadA.index(i), count=1)
        mbarrier.init(barLoadB.index(i), count=1)

    for i in range(num_mma_stages):
        mbarrier.init(barMMA.index(i), count=1)

    phase = 0
    mma_phase = 0
    ins_id = 0
    ext_id = 0
    mma_id = 0
    wait_id = 0

    # ins_id = 0
    mbarrier.expect(barLoadA.index(ins_id), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
    tma.async_copy_global_to_shared(input_desc, [0, 0], barLoadA.index(ins_id), smemA.index(ins_id))

    mbarrier.expect(barLoadB.index(ins_id), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
    tma.async_copy_global_to_shared(input_desc, [0, 0], barLoadB.index(ins_id), smemB.index(ins_id))
    ins_id = inc_mod(ins_id, num_buffers)

    # ins_id = 1
    mbarrier.expect(barLoadA.index(ins_id), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
    tma.async_copy_global_to_shared(input_desc, [0, 0], barLoadA.index(ins_id), smemA.index(ins_id))

    mbarrier.expect(barLoadB.index(ins_id), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
    tma.async_copy_global_to_shared(input_desc, [0, 0], barLoadB.index(ins_id), smemB.index(ins_id))
    ins_id = inc_mod(ins_id, num_buffers)

    mbarrier.wait(barLoadA.index(ext_id), phase)
    mbarrier.wait(barLoadB.index(ext_id), phase)

    blackwell.tcgen05_mma(smemA.index(ext_id), smemB.index(ext_id), acc, mbarriers=[barMMA.index(mma_id)])
    ext_id = inc_mod(ext_id, num_buffers)
    mma_id = inc_mod(mma_id, num_mma_stages)

    # ins_id = 2
    ub = 10
    for i in range(ub):
        if i < ub - 2:
            mbarrier.expect(barLoadA.index(ins_id), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
            tma.async_copy_global_to_shared(input_desc, [0, 0], barLoadA.index(ins_id), smemA.index(ins_id))

            mbarrier.expect(barLoadB.index(ins_id), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
            tma.async_copy_global_to_shared(input_desc, [0, 0], barLoadB.index(ins_id), smemB.index(ins_id))
            ins_id = inc_mod(ins_id, num_buffers)

        if i < ub - 1:
            mbarrier.wait(barLoadA.index(ext_id), phase)
            mbarrier.wait(barLoadB.index(ext_id), phase)

            blackwell.tcgen05_mma(smemA.index(ext_id), smemB.index(ext_id), acc, mbarriers=[barMMA.index(mma_id)])
            mma_id = inc_mod(mma_id, num_mma_stages)

        mbarrier.wait(barMMA.index(wait_id), mma_phase)
        wait_id = inc_mod(wait_id, num_mma_stages)
        if wait_id == 0:
            mma_phase = (mma_phase + 1) % 2
        ext_id = inc_mod(ext_id, num_buffers)
        if ext_id == 0:
            phase = (phase + 1) % 2

    for i in range(num_buffers):
        mbarrier.invalidate(barLoadA.index(i))
        mbarrier.invalidate(barLoadB.index(i))

    for i in range(num_mma_stages):
        mbarrier.invalidate(barMMA.index(i))


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_multibuffered_loop(FAILURE, device, run_wrapper):
    if run_wrapper:
        result = run_in_process(test_multibuffered_loop, (FAILURE, device, False))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    knobs.compilation.enable_experimental_consan = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    XBLOCK = 128
    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK, XBLOCK], shared_layout)
    multibuffered_loop_tma_kernel[(1, )](input_desc, XBLOCK, FAILURE=FAILURE, num_warps=4)


@gluon.jit
def multibuffered_loop_wgmma_kernel(input_desc, XBLOCK: ttgl.constexpr, FAILURE: ttgl.constexpr):
    num_buffers: ttgl.constexpr = 2 if FAILURE else 3

    mma_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                             instr_shape=[16, 32, 16])
    acc = hopper.warpgroup_mma_init(ttgl.zeros([XBLOCK, XBLOCK], ttgl.float32, mma_layout))

    smemA = ttgl.allocate_shared_memory(ttgl.float16, [num_buffers, XBLOCK, XBLOCK], input_desc.layout)
    smemB = ttgl.allocate_shared_memory(ttgl.float16, [num_buffers, XBLOCK, XBLOCK], input_desc.layout)
    barLoadA = ttgl.allocate_shared_memory(ttgl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    barLoadB = ttgl.allocate_shared_memory(ttgl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in range(num_buffers):
        mbarrier.init(barLoadA.index(i), count=1)
        mbarrier.init(barLoadB.index(i), count=1)

    phase = 0
    ins_id = 0
    ext_id = 0

    # ins_id = 0
    mbarrier.expect(barLoadA.index(ins_id), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
    tma.async_copy_global_to_shared(input_desc, [0, 0], barLoadA.index(ins_id), smemA.index(ins_id))

    mbarrier.expect(barLoadB.index(ins_id), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
    tma.async_copy_global_to_shared(input_desc, [0, 0], barLoadB.index(ins_id), smemB.index(ins_id))
    ins_id = inc_mod(ins_id, num_buffers)

    # ins_id = 1
    ub = 10
    for i in range(ub):
        if i < ub - 1:
            mbarrier.expect(barLoadA.index(ins_id), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
            tma.async_copy_global_to_shared(input_desc, [0, 0], barLoadA.index(ins_id), smemA.index(ins_id))

            mbarrier.expect(barLoadB.index(ins_id), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
            tma.async_copy_global_to_shared(input_desc, [0, 0], barLoadB.index(ins_id), smemB.index(ins_id))
            ins_id = inc_mod(ins_id, num_buffers)

        mbarrier.wait(barLoadA.index(ext_id), phase)
        mbarrier.wait(barLoadB.index(ext_id), phase)

        acc = hopper.warpgroup_mma(smemA.index(ext_id), smemB.index(ext_id), acc, is_async=True)
        hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc])
        ext_id = inc_mod(ext_id, num_buffers)
        if ext_id == 0:
            phase = (phase + 1) % 2
    hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])

    for i in range(num_buffers):
        mbarrier.invalidate(barLoadA.index(i))
        mbarrier.invalidate(barLoadB.index(i))


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_multibuffered_wgmma_loop(FAILURE, device, run_wrapper):
    if run_wrapper:
        result = run_in_process(test_multibuffered_wgmma_loop, (FAILURE, device, False))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Accessing buffer with pending access. Pending access type: warpgroup_mma operand read" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    knobs.compilation.enable_experimental_consan = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    XBLOCK = 128
    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK, XBLOCK], shared_layout)
    multibuffered_loop_wgmma_kernel[(1, )](input_desc, XBLOCK, FAILURE=FAILURE, num_warps=4)
