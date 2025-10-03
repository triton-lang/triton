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


# Use the same block size for all tests
XBLOCK = ttgl.constexpr(128)


@gluon.jit
def failing_kernel(input):
    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], smem_layout)
    blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                        warps_per_cta=[4, 1], order=[0, 1])
    offs_m = ttgl.arange(0, XBLOCK, layout=ttgl.SliceLayout(dim=1, parent=blocked_layout))[:, None]
    offs_n = ttgl.arange(0, XBLOCK, layout=ttgl.SliceLayout(dim=0, parent=blocked_layout))[None, :]
    offs = offs_m * XBLOCK + offs_n
    ampere.async_copy.async_copy_global_to_shared(smem, input + offs)
    ampere.async_copy.commit_group()

    ampere.async_copy.async_copy_global_to_shared(smem, input + offs)
    ampere.async_copy.commit_group()
    ampere.async_copy.wait_group(0)


def run_failing_kernel(device, enable_consan, mode):
    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    if enable_consan:
        if mode == "env":
            os.environ["TRITON_INSTRUMENTATION_MODE"] = "consan"
            knobs.refresh_knobs()
        elif mode == "knob":
            knobs.compilation.instrumentation_mode = "consan"

    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    failing_kernel[(1, )](input)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_cache_miss_knob(device, monkeypatch):
    # First run without consan
    run_in_process(run_failing_kernel, (device, False, "knob"))

    # Then run with consan and assert that if fails
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    result = run_in_process(run_failing_kernel, (device, True, "knob"))
    assert "device-side assert" in str(result.exc)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_cache_miss_env(device, monkeypatch):
    # First run without consan
    run_in_process(run_failing_kernel, (device, False, "env"))

    # Then run with consan and assert that if fails
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    result = run_in_process(run_failing_kernel, (device, True, "env"))
    assert "device-side assert" in str(result.exc)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_async_tma_kernel(FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_async_tma_kernel, (FAILURE, device, False, monkeypatch))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def kernel(input_desc, out, FAILURE: ttgl.constexpr):
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], input_desc.layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)

        mbarrier.expect(bar, XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar, smem)
        mbarrier.wait(bar, 0, pred=(not FAILURE))
        val = smem.load(blocked_layout)
        mbarrier.wait(bar, 0, pred=FAILURE)
        mbarrier.invalidate(bar)

        out_m = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK + out_n
        ttgl.store(out_ptr, val)

    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    output = torch.empty((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK.value, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, output, FAILURE=FAILURE, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_tma_interleave_kernel(FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_tma_interleave_kernel, (FAILURE, device, False, monkeypatch))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    @gluon.jit
    def kernel(input_desc, out, FAILURE: ttgl.constexpr):
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

        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1])
        out_m = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK + out_n
        ttgl.store(out_ptr, smem.index(0).load(blocked_layout))
        ttgl.store(out_ptr, smem.index(1).load(blocked_layout))

        mbarrier.invalidate(bar.index(0))
        mbarrier.invalidate(bar.index(1))

        tma.async_copy_shared_to_global(input_desc, [0, 0], smem.index(0))
        tma.store_wait(0)

    triton.set_allocator(alloc_fn)
    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    output = torch.empty((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK.value, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, output, FAILURE=FAILURE, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires ampere or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_async_copy(FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_async_copy, (FAILURE, device, False, monkeypatch))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Accessing buffer with pending access. Pending access type: async_copy_global_to_shared" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def kernel(input, FAILURE: ttgl.constexpr):
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

    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    kernel[(1, )](input, FAILURE=FAILURE, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
@pytest.mark.parametrize("MEM_ACCESS_KIND", ["tma_cp", "local_store", "tmem_load", "tmem_store"])
def test_tcgen5_mma(FAILURE, MEM_ACCESS_KIND, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_tcgen5_mma, (FAILURE, MEM_ACCESS_KIND, device, False, monkeypatch))
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

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def kernel(input_desc, FAILURE: ttgl.constexpr, MEM_ACCESS_KIND: ttgl.constexpr):
        acc_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([XBLOCK, XBLOCK], col_stride=1, cta_split_num=[1, 1])
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

    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK.value, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, FAILURE=FAILURE, MEM_ACCESS_KIND=MEM_ACCESS_KIND, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_warpgroup_mma(FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_warpgroup_mma, (FAILURE, device, False, monkeypatch))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Accessing buffer with pending access. Pending access type: warpgroup_mma operand read" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def kernel(input, FAILURE: ttgl.constexpr):
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

    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    kernel[(1, )](input, FAILURE=FAILURE)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_warpgroup_mma2(FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_warpgroup_mma2, (FAILURE, device, False, monkeypatch))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Accessing buffer with pending access. Pending access type: warpgroup_mma operand read" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def kernel(input, FAILURE: ttgl.constexpr):
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

    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    kernel[(1, )](input, FAILURE=FAILURE)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("BUF_IDX", [0, 1])
@pytest.mark.parametrize("BAR_IDX", [0, 1, 2, 3])
def test_tcgen5_mma_multibar(BUF_IDX, BAR_IDX, device, run_wrapper, monkeypatch):
    if BAR_IDX == 0:
        pytest.skip("Skipping due to wait on false-predicated barrier - not supported yet")
    if run_wrapper:
        result = run_in_process(test_tcgen5_mma_multibar, (BUF_IDX, BAR_IDX, device, False, monkeypatch))
        if BAR_IDX // 2 < BUF_IDX:
            assert "device-side assert" in str(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def kernel(input_desc, BUF_IDX: ttgl.constexpr, BAR_IDX: ttgl.constexpr):
        acc_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([XBLOCK, XBLOCK], col_stride=1, cta_split_num=[1, 1])
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

    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK.value, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, BUF_IDX, BAR_IDX, num_warps=4)


@gluon.jit
def inc_mod(x, mod):
    return (x + 1) % mod


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_multibuffered_loop(FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_multibuffered_loop, (FAILURE, device, False, monkeypatch))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def kernel(input_desc, FAILURE: ttgl.constexpr):
        num_buffers: ttgl.constexpr = 2 if FAILURE else 3
        num_mma_stages: ttgl.constexpr = 2

        acc_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([XBLOCK, XBLOCK], col_stride=1, cta_split_num=[1, 1])
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

    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK.value, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, FAILURE=FAILURE, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_multibuffered_wgmma_loop(FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_multibuffered_wgmma_loop, (FAILURE, device, False, monkeypatch))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Accessing buffer with pending access. Pending access type: warpgroup_mma operand read" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def kernel(input_desc, FAILURE: ttgl.constexpr):
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

    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK.value, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, FAILURE=FAILURE, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_store_wait_load(FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_ws_store_wait_load, (FAILURE, device, False, monkeypatch))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_default(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        mbarrier.wait(bar.index(0), phase=0, pred=(not FAILURE))
        val = smem.index(0).load(layout)
        smem.index(1).store(val)
        mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def ws_1(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        smem.index(0).store(ttgl.arange(0, XBLOCK, layout).to(ttgl.float16))
        mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def ws_kernel(output, FAILURE: ttgl.constexpr):
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], mbarrier.MBarrierLayout())
        for i in range(2):
            mbarrier.init(bar.index(i), count=1)
        ttgl.warp_specialize((smem, bar, FAILURE, blocked_layout), ws_default, (smem, bar, FAILURE, blocked_layout),
                             [ws_1], [4], [32])
        mbarrier.wait(bar.index(1), phase=0)
        val = smem.index(0).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, XBLOCK, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK, ), device=device, dtype=torch.float16)
    ws_kernel[(1, )](output, FAILURE=FAILURE, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_load_wait_store(FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_ws_load_wait_store, (FAILURE, device, False, monkeypatch))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_default(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        mbarrier.wait(bar.index(0), phase=0, pred=(not FAILURE))
        smem.index(0).store(ttgl.arange(0, XBLOCK, layout).to(ttgl.float16))
        mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def ws_1(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        mbarrier.arrive(bar.index(0), count=1)
        smem.index(1).store(val)  # dummy store to make sure the load is executed

    @gluon.jit
    def ws_kernel(output, FAILURE: ttgl.constexpr):
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], mbarrier.MBarrierLayout())
        for i in range(2):
            mbarrier.init(bar.index(i), count=1)
        ttgl.warp_specialize((smem, bar, FAILURE, blocked_layout), ws_default, (smem, bar, FAILURE, blocked_layout),
                             [ws_1], [4], [32])
        mbarrier.wait(bar.index(1), phase=0)
        val = smem.index(0).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, XBLOCK, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK, ), device=device, dtype=torch.float16)
    ws_kernel[(1, )](output, FAILURE=FAILURE, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("MISSING_BAR", ["none", "1", "2"])
def test_ws_two_loads_two_bars(MISSING_BAR, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_ws_two_loads_two_bars, (MISSING_BAR, device, False, monkeypatch))
        if MISSING_BAR != "none":
            assert "device-side assert" in str(result.exc)
            assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_default(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        mbarrier.arrive(bar.index(0), count=1)
        smem.index(1).store(val)  # dummy store to make sure the load is executed

    @gluon.jit
    def ws_1(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        mbarrier.arrive(bar.index(1), count=1)
        smem.index(2).store(val)  # dummy store to make sure the load is executed

    @gluon.jit
    def ws_2(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        if MISSING_BAR != "1":
            mbarrier.wait(bar.index(0), phase=0)
        if MISSING_BAR != "2":
            mbarrier.wait(bar.index(1), phase=0)
        smem.index(0).store(ttgl.arange(0, XBLOCK, layout).to(ttgl.float16))
        mbarrier.arrive(bar.index(2), count=1)

    @gluon.jit
    def kernel(output, MISSING_BAR: ttgl.constexpr):
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, XBLOCK], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [3, 1], mbarrier.MBarrierLayout())
        for i in range(3):
            mbarrier.init(bar.index(i), count=1)
        ttgl.warp_specialize((smem, bar, MISSING_BAR, blocked_layout), ws_default,
                             (smem, bar, MISSING_BAR, blocked_layout), [ws_1, ws_2], [4, 4], [32, 32])
        mbarrier.wait(bar.index(2), phase=0)
        val = smem.index(0).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, XBLOCK, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK, ), device=device, dtype=torch.float16)
    kernel[(1, )](output, MISSING_BAR=MISSING_BAR, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_two_loads_one_bar(FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_ws_two_loads_one_bar, (FAILURE, device, False, monkeypatch))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_default(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        mbarrier.arrive(bar.index(0), count=1)
        smem.index(1).store(val)  # dummy store to make sure the load is executed

    @gluon.jit
    def ws_1(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        mbarrier.arrive(bar.index(0), count=1)
        smem.index(2).store(val)  # dummy store to make sure the load is executed

    @gluon.jit
    def ws_2(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        mbarrier.wait(bar.index(0), phase=0, pred=(not FAILURE))
        smem.index(0).store(ttgl.arange(0, XBLOCK, layout).to(ttgl.float16))
        mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def kernel(output, FAILURE: ttgl.constexpr):
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, XBLOCK], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], mbarrier.MBarrierLayout())
        mbarrier.init(bar.index(0), count=2)
        mbarrier.init(bar.index(1), count=1)
        ttgl.warp_specialize((smem, bar, FAILURE, blocked_layout), ws_default, (smem, bar, FAILURE, blocked_layout),
                             [ws_1, ws_2], [4, 4], [32, 32])
        mbarrier.wait(bar.index(1), phase=0)
        val = smem.index(0).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, XBLOCK, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK, ), device=device, dtype=torch.float16)
    kernel[(1, )](output, FAILURE=FAILURE, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("MISSING_BAR", ["none", "0", "1", "2", "3"])
def test_ws_two_loads_two_bars_loop(MISSING_BAR, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_ws_two_loads_two_bars_loop, (MISSING_BAR, device, False, monkeypatch))
        if MISSING_BAR != "none":
            assert "device-side assert" in str(result.exc)
            if MISSING_BAR in ["0", "1"]:
                assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
            elif MISSING_BAR in ["2", "3"]:
                assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_default(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        acc = ttgl.zeros([XBLOCK], ttgl.float16, layout)
        phase = 0
        for _ in range(10):
            if MISSING_BAR != "2":
                mbarrier.wait(bar.index(2), phase=phase)
            phase = (phase + 1) % 2
            val = smem.index(0).load(layout)
            mbarrier.arrive(bar.index(0), count=1)
            acc = acc + val
        smem.index(1).store(acc)  # dummy store to make sure the load is executed

    @gluon.jit
    def ws_1(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        acc = ttgl.zeros([XBLOCK], ttgl.float16, layout)
        phase = 0
        for _ in range(10):
            if MISSING_BAR != "3":
                mbarrier.wait(bar.index(3), phase=phase)
            phase = (phase + 1) % 2
            val = smem.index(0).load(layout)
            mbarrier.arrive(bar.index(1), count=1)
            acc = acc + val
        smem.index(2).store(acc)  # dummy store to make sure the load is executed

    @gluon.jit
    def ws_2(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        phase = 0
        for _ in range(10):
            if MISSING_BAR != "0":
                mbarrier.wait(bar.index(0), phase=phase)
            if MISSING_BAR != "1":
                mbarrier.wait(bar.index(1), phase=phase)
            phase = (phase + 1) % 2
            smem.index(0).store(ttgl.arange(0, XBLOCK, layout).to(ttgl.float16))
            mbarrier.arrive(bar.index(2), count=1)
            mbarrier.arrive(bar.index(3), count=1)

    @gluon.jit
    def kernel(output, MISSING_BAR: ttgl.constexpr):
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, XBLOCK], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [4, 1], mbarrier.MBarrierLayout())
        for i in range(4):
            mbarrier.init(bar.index(i), count=1)

        mbarrier.arrive(bar.index(2), count=1)
        mbarrier.arrive(bar.index(3), count=1)

        ttgl.warp_specialize((smem, bar, MISSING_BAR, blocked_layout), ws_default,
                             (smem, bar, MISSING_BAR, blocked_layout), [ws_1, ws_2], [4, 4], [32, 32])

    output = torch.empty((XBLOCK, ), device=device, dtype=torch.float16)
    kernel[(1, )](output, MISSING_BAR=MISSING_BAR, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_load_ordering(FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_ws_load_ordering, (FAILURE, device, False, monkeypatch))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_default(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        phase = 0
        for _ in range(10):
            mbarrier.wait(bar.index(2), phase=phase)
            phase = (phase + 1) % 2
            smem.index(0).store(ttgl.arange(0, XBLOCK, layout).to(ttgl.float16))
            mbarrier.arrive(bar.index(0), count=1)
            smem.index(1).store(ttgl.arange(0, XBLOCK, layout).to(ttgl.float16))
            mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def ws_1(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        acc = ttgl.zeros([XBLOCK], ttgl.float16, layout)
        phase = 0
        for _ in range(10):
            mbarrier.wait(bar.index(0), phase=phase)
            val = smem.index(1 if FAILURE else 0).load(layout)
            mbarrier.wait(bar.index(1), phase=phase)
            phase = (phase + 1) % 2
            mbarrier.arrive(bar.index(2), count=1)
            acc = acc + val
        smem.index(2).store(acc)  # dummy store to make sure the load is executed

    @gluon.jit
    def kernel(output, FAILURE: ttgl.constexpr):
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, XBLOCK], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [3, 1], mbarrier.MBarrierLayout())
        for i in range(3):
            mbarrier.init(bar.index(i), count=1)

        mbarrier.arrive(bar.index(2), count=1)

        ttgl.warp_specialize((smem, bar, FAILURE, blocked_layout), ws_default, (smem, bar, FAILURE, blocked_layout),
                             [ws_1], [4], [32])

    output = torch.empty((XBLOCK, ), device=device, dtype=torch.float16)
    kernel[(1, )](output, FAILURE=FAILURE, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("MISSING_BAR", ["none", "T2", "T3"])
def test_ws_two_producers_two_consumers(MISSING_BAR, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_ws_two_producers_two_consumers, (MISSING_BAR, device, False, monkeypatch))
        if MISSING_BAR != "none":
            assert "device-side assert" in str(result.exc)
            assert "Buffer being accessed has outstanding" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_default(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        phase = 0
        for _ in range(10):
            mbarrier.wait(bar.index(2), phase=phase)
            phase = (phase + 1) % 2
            smem.index(0).store(ttgl.arange(0, XBLOCK, layout).to(ttgl.float16))
            mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def ws_1(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        phase = 0
        for _ in range(10):
            mbarrier.wait(bar.index(3), phase=phase)
            phase = (phase + 1) % 2
            smem.index(1).store(ttgl.arange(0, XBLOCK, layout).to(ttgl.float16))
            mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def ws_2(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        acc = ttgl.zeros([XBLOCK], ttgl.float16, layout)
        phase = 0
        for _ in range(10):
            if MISSING_BAR != "T2":
                mbarrier.wait(bar.index(0), phase=phase)
            phase = (phase + 1) % 2
            val = smem.index(0).load(layout)
            mbarrier.arrive(bar.index(2), count=1)
            mbarrier.arrive(bar.index(3), count=1)
            acc = acc + val
        smem.index(2).store(acc)  # dummy store to make sure the load is executed

    @gluon.jit
    def ws_3(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        acc = ttgl.zeros([XBLOCK], ttgl.float16, layout)
        phase = 0
        for _ in range(10):
            if MISSING_BAR != "T3":
                mbarrier.wait(bar.index(0), phase=phase)
            phase = (phase + 1) % 2
            val = smem.index(1).load(layout)
            mbarrier.arrive(bar.index(2), count=1)
            mbarrier.arrive(bar.index(3), count=1)
            acc = acc + val
        smem.index(3).store(acc)  # dummy store to make sure the load is executed

    @gluon.jit
    def kernel(output, MISSING_BAR: ttgl.constexpr):
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [4, XBLOCK], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [4, 1], mbarrier.MBarrierLayout())
        for i in range(4):
            mbarrier.init(bar.index(i), count=2)

        mbarrier.arrive(bar.index(2), count=2)
        mbarrier.arrive(bar.index(3), count=2)

        ttgl.warp_specialize((smem, bar, MISSING_BAR, blocked_layout), ws_default,
                             (smem, bar, MISSING_BAR, blocked_layout), [ws_1, ws_2, ws_3], [4, 4, 4], [32, 32, 32])

    output = torch.empty((XBLOCK, ), device=device, dtype=torch.float16)
    kernel[(1, )](output, MISSING_BAR=MISSING_BAR, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("MISSING_BAR", ["none", "1", "2"])
def test_ws_different_warp_sizes(MISSING_BAR, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_ws_different_warp_sizes, (MISSING_BAR, device, False, monkeypatch))
        if MISSING_BAR != "none":
            assert "device-side assert" in str(result.exc)
            assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_default(smem, bar, MISSING_BAR: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4],
                                                    order=[0])
        val = smem.index(0).load(layout)
        mbarrier.arrive(bar.index(0), count=1)
        smem.index(1).store(val)  # dummy store to make sure the load is executed

    @gluon.jit
    def ws_1(smem, bar, MISSING_BAR: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[2],
                                                    order=[0])
        val = smem.index(0).load(layout)
        mbarrier.arrive(bar.index(1), count=1)
        smem.index(2).store(val)  # dummy store to make sure the load is executed

    @gluon.jit
    def ws_2(smem, bar, MISSING_BAR: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[8],
                                                    order=[0])
        if MISSING_BAR != "1":
            mbarrier.wait(bar.index(0), phase=0)
        if MISSING_BAR != "2":
            mbarrier.wait(bar.index(1), phase=0)
        smem.index(0).store(ttgl.arange(0, XBLOCK, layout).to(ttgl.float16))
        mbarrier.arrive(bar.index(2), count=1)

    @gluon.jit
    def kernel(output, MISSING_BAR: ttgl.constexpr):
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, XBLOCK], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [3, 1], mbarrier.MBarrierLayout())
        for i in range(3):
            mbarrier.init(bar.index(i), count=1)
        ttgl.warp_specialize((smem, bar, MISSING_BAR), ws_default, (smem, bar, MISSING_BAR), [ws_1, ws_2], [2, 8],
                             [32, 32])
        mbarrier.wait(bar.index(2), phase=0)
        val = smem.index(0).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, XBLOCK, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK, ), device=device, dtype=torch.float16)
    kernel[(1, )](output, MISSING_BAR=MISSING_BAR, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_async_copy_commits(FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_ws_async_copy_commits, (FAILURE, device, False, monkeypatch))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_prog(input, smem, FAILURE: ttgl.constexpr, blocked_layout: ttgl.constexpr, BASE: ttgl.constexpr):
        # Two-buffer ping-pong within a partition: buffers BASE and BASE+1
        offs = ttgl.arange(0, XBLOCK, layout=blocked_layout)

        acc = ttgl.zeros([XBLOCK], ttgl.float16, blocked_layout)

        # Prime pipeline
        ampere.async_copy.async_copy_global_to_shared(smem.index(BASE + 0), input + offs)
        ampere.async_copy.commit_group()

        for i in range(1, 10):
            dst = (i % 2)
            src = ((i - 1) % 2)
            if i < 9:
                ampere.async_copy.async_copy_global_to_shared(smem.index(BASE + dst), input + offs)
                ampere.async_copy.commit_group()
                ampere.async_copy.wait_group(1)
            else:
                ampere.async_copy.wait_group(0)

            # Load from last completed buffer. In failure mode for BASE==2 (ws_1), read other partition's buffers (0/1)
            load_base = 0 if (FAILURE and BASE == 2) else BASE
            acc = acc + smem.index(load_base + src).load(blocked_layout)
        smem.index(BASE).store(acc)

    @gluon.jit
    def kernel(input, FAILURE: ttgl.constexpr):
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        # 4 buffers total: ws_default uses 0/1; ws_1 uses 2/3
        smem = ttgl.allocate_shared_memory(ttgl.float16, [4, XBLOCK], smem_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[XBLOCK], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        ttgl.warp_specialize((input, smem, FAILURE, blocked_layout, 0), ws_prog,
                             (input, smem, FAILURE, blocked_layout, 2), [ws_prog], [4], [32])

    input = torch.randn((XBLOCK, ), device=device, dtype=torch.float16)
    kernel[(1, )](input, FAILURE=FAILURE, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_async_copy_wait_visibility(FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_ws_async_copy_wait_visibility, (FAILURE, device, False, monkeypatch))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert (("Buffer being accessed has outstanding writes" in result.driver_stderr_output)
                    or ("Accessing buffer with pending access. Pending access type: async_copy_global_to_shared"
                        in result.driver_stderr_output))
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_default(input, smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        offs = ttgl.arange(0, XBLOCK, layout)
        ampere.async_copy.async_copy_global_to_shared(smem.index(0), input + offs)
        ampere.async_copy.commit_group()
        ampere.async_copy.async_copy_global_to_shared(smem.index(1), input + offs)
        ampere.async_copy.commit_group()
        ampere.async_copy.wait_group(1)
        mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def ws_1(input, smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        mbarrier.wait(bar.index(0), phase=0)
        val = smem.index(1 if FAILURE else 0).load(layout)
        smem.index(0).store(val)  # keep load

    @gluon.jit
    def kernel(input, FAILURE: ttgl.constexpr):
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[XBLOCK], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], mbarrier.MBarrierLayout())
        mbarrier.init(bar.index(0), count=1)
        ttgl.warp_specialize((input, smem, bar, FAILURE, blocked_layout), ws_default,
                             (input, smem, bar, FAILURE, blocked_layout), [ws_1], [4], [32])

    input = torch.randn((XBLOCK, ), device=device, dtype=torch.float16)
    kernel[(1, )](input, FAILURE=FAILURE, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_wgmma_wait_visibility(FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_ws_wgmma_wait_visibility, (FAILURE, device, False, monkeypatch))
        if FAILURE:
            assert "device-side assert" in str(result.exc)
            assert "Accessing buffer with pending access. Pending access type: warpgroup_mma operand read" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_default(smem, bar, FAILURE: ttgl.constexpr, blocked_layout: ttgl.constexpr, mma_layout: ttgl.constexpr):
        acc = ttgl.zeros([XBLOCK, XBLOCK], ttgl.float16, mma_layout)
        # Issue two async MMAs on two different buffers
        acc = hopper.warpgroup_mma(smem.index(0), smem.index(0), acc, is_async=True)
        acc = hopper.warpgroup_mma(smem.index(1), smem.index(1), acc, is_async=True)
        # Wait until only 1 outstanding remains
        hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc])
        # Signal to consumer
        mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def ws_1(smem, bar, FAILURE: ttgl.constexpr, blocked_layout: ttgl.constexpr):
        mbarrier.wait(bar.index(0), phase=0)
        val = ttgl.full([XBLOCK, XBLOCK], 42, ttgl.float16, blocked_layout)
        smem.index(1 if FAILURE else 0).store(val)

    @gluon.jit
    def kernel(FAILURE: ttgl.constexpr):
        smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1])
        mma_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                                 instr_shape=[16, 32, 16])
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK, XBLOCK], smem_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], mbarrier.MBarrierLayout())
        mbarrier.init(bar.index(0), count=1)
        ttgl.warp_specialize((smem, bar, FAILURE, blocked_layout, mma_layout), ws_default,
                             (smem, bar, FAILURE, blocked_layout), [ws_1], [4], [32])

    kernel[(1, )](FAILURE=FAILURE, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_deadlock_two_partitions(device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_deadlock_two_partitions, (device, False, monkeypatch))
        assert "device-side assert" in str(result.exc)
        assert "Deadlock detected" in result.driver_stderr_output
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_default(bar):
        mbarrier.wait(bar.index(0), phase=0)

    @gluon.jit
    def ws_1(bar):
        mbarrier.wait(bar.index(1), phase=0)

    @gluon.jit
    def kernel():
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], mbarrier.MBarrierLayout())
        mbarrier.init(bar.index(0), count=1)
        mbarrier.init(bar.index(1), count=1)
        ttgl.warp_specialize((bar, ), ws_default, (bar, ), [ws_1], [4], [32])

    kernel[(1, )](num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_deadlock_overarrival(device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_deadlock_overarrival, (device, False, monkeypatch))
        assert "device-side assert" in str(result.exc)
        assert "Deadlock detected" in result.driver_stderr_output
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def kernel():
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], mbarrier.MBarrierLayout())
        mbarrier.init(bar.index(0), count=1)
        mbarrier.init(bar.index(1), count=1)

        mbarrier.arrive(bar.index(0), count=1)
        mbarrier.arrive(bar.index(0), count=1)
        mbarrier.wait(bar.index(0), phase=0)

    kernel[(1, )](num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_deadlock_underarrival(device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_deadlock_underarrival, (device, False, monkeypatch))
        assert "device-side assert" in str(result.exc)
        assert "Deadlock detected" in result.driver_stderr_output
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_default(bar):
        mbarrier.arrive(bar.index(1), count=1)
        mbarrier.wait(bar.index(0), phase=0)

    @gluon.jit
    def ws_1(bar):
        mbarrier.arrive(bar.index(0), count=1)
        mbarrier.wait(bar.index(1), phase=0)

    @gluon.jit
    def kernel():
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], mbarrier.MBarrierLayout())
        mbarrier.init(bar.index(0), count=2)
        mbarrier.init(bar.index(1), count=2)
        ttgl.warp_specialize((bar, ), ws_default, (bar, ), [ws_1], [4], [32])

    kernel[(1, )](num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_deadlock_different_phases(device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_deadlock_different_phases, (device, False, monkeypatch))
        assert result.exc is None
        assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_default(bar):
        mbarrier.wait(bar.index(0), phase=0)
        mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def ws_1(bar):
        mbarrier.wait(bar.index(0), phase=1)

    @gluon.jit
    def kernel():
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1, 1], mbarrier.MBarrierLayout())
        mbarrier.init(bar.index(0), count=1)
        mbarrier.arrive(bar.index(0), count=1)
        ttgl.warp_specialize((bar, ), ws_default, (bar, ), [ws_1], [4], [32])

    kernel[(1, )](num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_deadlock_exempt_when_tma_signals(device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_deadlock_exempt_when_tma_signals, (device, False, monkeypatch))
        assert result.exc is None
        assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_default(input_desc, smem, bar):
        mbarrier.expect(bar.index(0), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar.index(0), smem.index(0))
        mbarrier.wait(bar.index(0), phase=0)

    @gluon.jit
    def ws_1(input_desc, smem, bar):
        mbarrier.expect(bar.index(1), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar.index(1), smem.index(1))
        mbarrier.wait(bar.index(1), phase=0)

    @gluon.jit
    def kernel(input_desc):
        shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK, XBLOCK], shared_layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], mbarrier.MBarrierLayout())
        mbarrier.init(bar.index(0), count=1)
        mbarrier.init(bar.index(1), count=1)
        ttgl.warp_specialize((input_desc, smem, bar), ws_default, (input_desc, smem, bar), [ws_1], [4], [32])

    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK.value, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_barrier_underflow(device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_barrier_underflow, (device, False, monkeypatch))
        assert "device-side assert" in str(result.exc)
        assert "Barrier arrive underflow: current count would become negative" in result.driver_stderr_output
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    # ConSan requires a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    @gluon.jit
    def ws_default(bar):
        mbarrier.arrive(bar.index(1), count=2)
        mbarrier.wait(bar.index(0), phase=0)

    @gluon.jit
    def ws_1(bar):
        mbarrier.wait(bar.index(1), phase=0)

    @gluon.jit
    def kernel():
        bar = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], mbarrier.MBarrierLayout())
        mbarrier.init(bar.index(0), count=1)
        mbarrier.init(bar.index(1), count=1)
        ttgl.warp_specialize((bar, ), ws_default, (bar, ), [ws_1], [4], [32])

    kernel[(1, )](num_warps=4)
