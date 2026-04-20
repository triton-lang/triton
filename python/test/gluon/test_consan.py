import os
import tempfile
import torch
import pytest
from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia import blackwell
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia import ampere
from triton.experimental.gluon.language.nvidia.blackwell import allocate_tensor_memory, clc, mbarrier, tma
from triton._internal_testing import is_cuda, run_in_process


@pytest.fixture
def run_wrapper():
    # Use DISABLE_SUBPROCESS to run the tests in the main process
    # (useful for debugging but assert in any test will make all the tests fail)
    return not os.environ.get("DISABLE_SUBPROCESS")


@pytest.fixture(params=[1, 2, 4], ids=lambda num_ctas: f"{num_ctas}ctas")
def num_ctas(request):
    return request.param


def assert_expected_cuda_failure(exc):
    assert exc is not None
    assert any(msg in str(exc) for msg in ["device-side assert", "unspecified launch failure"]), str(exc)


@gluon.constexpr_function
def mma_cga_layout(num_ctas, op_idx, two_cta=False):
    num_ctas = getattr(num_ctas, "value", num_ctas)
    op_idx = getattr(op_idx, "value", op_idx)
    two_cta = getattr(two_cta, "value", two_cta)
    assert op_idx in (0, 1, 2)
    # For now, but the code above is generic really
    assert num_ctas <= 4
    log2_num_ctas = num_ctas.bit_length() - 1
    cga_layout = [[1, 0], [0, 1]][:log2_num_ctas]
    if op_idx == 2 or not cga_layout:
        return tuple(tuple(b) for b in cga_layout)

    # 2CTA performs an outer product so bases are [1, 0] and [0, 1].
    assert cga_layout[0] == [1, 0]
    first = (1, 0) if op_idx == 0 else ((0, 1) if two_cta else (0, 0))
    result = [first]
    # Broadcast along K (the reduction dimension). We multiply by 2 for
    # op_idx == 1, as we have added the (0, 1) basis.
    for b in cga_layout[1:]:
        if op_idx == 0:
            result.append((b[0], 0))
        else:
            mul = 2 if two_cta else 1
            result.append((0, mul * b[1]))
    return tuple(result)


@gluon.constexpr_function
def mma_block_m(num_ctas):
    num_ctas = getattr(num_ctas, "value", num_ctas)
    return 256 if num_ctas > 1 else 128


@gluon.constexpr_function
def mma_block_n(num_ctas):
    num_ctas = getattr(num_ctas, "value", num_ctas)
    return 256 if num_ctas == 4 else 128


@gluon.constexpr_function
def default_cga_layout(num_ctas, rank, dim=0):
    num_ctas = getattr(num_ctas, "value", num_ctas)
    if num_ctas == 1:
        return []
    assert 0 <= dim < rank
    return [[0] * dim + [1 << i] + [0] * (rank - dim - 1) for i in range(num_ctas.bit_length() - 1)]


@gluon.constexpr_function
def multicast_cga_layout(num_ctas, rank):
    num_ctas = getattr(num_ctas, "value", num_ctas)
    if num_ctas == 1:
        return []
    return [[0] * rank for _ in range(num_ctas.bit_length() - 1)]


# Use the same block size for all tests
XBLOCK = ttgl.constexpr(128)


@gluon.jit
def failing_kernel(input):
    cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                                         cga_layout=cga_layout)
    smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], smem_layout)
    blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                        warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
    offs_m = ttgl.arange(0, XBLOCK, layout=ttgl.SliceLayout(dim=1, parent=blocked_layout))[:, None]
    offs_n = ttgl.arange(0, XBLOCK, layout=ttgl.SliceLayout(dim=0, parent=blocked_layout))[None, :]
    offs = offs_m * XBLOCK + offs_n
    ampere.async_copy.async_copy_global_to_shared(smem, input + offs)
    ampere.async_copy.commit_group()

    ampere.async_copy.async_copy_global_to_shared(smem, input + offs)
    ampere.async_copy.commit_group()
    ampere.async_copy.wait_group(0)


def run_failing_kernel(device, enable_consan, mode, num_ctas):
    if enable_consan:
        if mode == "env":
            os.environ["TRITON_INSTRUMENTATION_MODE"] = "consan"
            knobs.refresh_knobs()
        elif mode == "knob":
            knobs.compilation.instrumentation_mode = "consan"

    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    failing_kernel[(1, )](input, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_cache_miss_knob(device, monkeypatch, num_ctas):
    # First run without consan
    run_in_process(run_failing_kernel, (device, False, "knob", num_ctas))

    # Then run with consan and assert that if fails
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    result = run_in_process(run_failing_kernel, (device, True, "knob", num_ctas))
    assert result.exc is not None
    assert any(msg in str(result.exc) for msg in ["device-side assert", "unspecified launch failure"])


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_cache_miss_env(device, monkeypatch, num_ctas):
    # First run without consan
    run_in_process(run_failing_kernel, (device, False, "env", num_ctas))

    # Then run with consan and assert that if fails
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    result = run_in_process(run_failing_kernel, (device, True, "env", num_ctas))
    assert result.exc is not None
    assert any(msg in str(result.exc) for msg in ["device-side assert", "unspecified launch failure"])


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_consan_uses_profile_scratch(device, fresh_knobs, num_ctas):
    with knobs.cache.scope(), knobs.runtime.scope():
        knobs.cache.dir = tempfile.mkdtemp(prefix="triton-cache-")
        fresh_knobs.compilation.instrumentation_mode = "consan"
        input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
        compiled = failing_kernel.warmup(input, grid=(1, ), num_ctas=num_ctas)
        assert compiled.metadata.profile_scratch_size > 0
        assert compiled.metadata.global_scratch_size == 0


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_async_tma_kernel(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_async_tma_kernel, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input_desc, out, FAILURE: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], input_desc.layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        mbarrier.expect(bar, input_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar, smem)
        mbarrier.wait(bar, 0, pred=(not FAILURE), deps=[smem])
        val = smem.load(blocked_layout)
        mbarrier.wait(bar, 0, pred=FAILURE, deps=[smem])
        mbarrier.invalidate(bar)

        out_m = ttgl.arange(0, block_m, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK + out_n
        ttgl.store(out_ptr, val)

    block_m = XBLOCK.value * num_ctas
    input = torch.randn((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    output = torch.empty((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=default_cga_layout(num_ctas, 2))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [block_m, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, output, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_async_tma_multicast_kernel(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if num_ctas == 1:
        pytest.skip("Need at least 2 CTAs for multicast in this test")
    if run_wrapper:
        result = run_in_process(test_async_tma_multicast_kernel, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input_desc, out, FAILURE: ttgl.constexpr):
        cga_layout: ttgl.constexpr = multicast_cga_layout(ttgl.num_ctas(), 2)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], input_desc.layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        mbarrier.expect(bar, input_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar, smem, multicast=True)
        mbarrier.wait(bar, 0, pred=(not FAILURE), deps=[smem])
        val = smem.load(blocked_layout)
        mbarrier.wait(bar, 0, pred=FAILURE, deps=[smem])
        mbarrier.invalidate(bar)

        out_m = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK + out_n
        ttgl.store(out_ptr, val)

    input = torch.randn((XBLOCK.value, XBLOCK.value), device=device, dtype=torch.float16)
    output = torch.empty((XBLOCK.value, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=multicast_cga_layout(num_ctas, 2))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK.value, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, output, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_clc_result_visibility(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_clc_result_visibility, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(out, FAILURE: ttgl.constexpr):
        cga_layout: ttgl.constexpr = multicast_cga_layout(ttgl.num_ctas(), 1)
        layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[0], cga_layout=cga_layout)
        clc_result = ttgl.allocate_shared_memory(ttgl.int64, [2], layout)
        clc_bar = mbarrier.allocate_mbarrier()
        mbarrier.init(clc_bar, count=1)

        clc.try_cancel(clc_result, clc_bar)
        mbarrier.expect(clc_bar, 16)
        mbarrier.wait(clc_bar, 0, pred=(not FAILURE))
        response = clc.load_result(clc_result)
        mbarrier.wait(clc_bar, 0, pred=FAILURE)
        mbarrier.invalidate(clc_bar)

        ttgl.store(out + ttgl.program_id(0), response.is_canceled())

    output = torch.empty((1, ), device=device, dtype=torch.bool)
    kernel[(1, )](output, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
def test_async_tma_multicast_kernel_reuse(device, run_wrapper, monkeypatch, num_ctas):
    if num_ctas == 1:
        pytest.skip("Need at least 2 CTAs for multicast in this test")
    if run_wrapper:
        result = run_in_process(test_async_tma_multicast_kernel_reuse, (device, False, monkeypatch, num_ctas))
        assert result.exc is None
        assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input_desc, out):
        cga_layout: ttgl.constexpr = multicast_cga_layout(ttgl.num_ctas(), 2)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], input_desc.layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        for phase in ttgl.static_range(2):
            mbarrier.expect(bar, input_desc.nbytes_per_cta)
            ttgl.barrier(cluster=True)
            tma.async_copy_global_to_shared(input_desc, [0, 0], bar, smem, multicast=True)
            mbarrier.wait(bar, phase % 2, deps=[smem])
            ttgl.barrier(cluster=True)
        val = smem.load(blocked_layout)
        mbarrier.invalidate(bar)

        out_m = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK + out_n
        ttgl.store(out_ptr, val)

    input = torch.randn((XBLOCK.value, XBLOCK.value), device=device, dtype=torch.float16)
    output = torch.empty((XBLOCK.value, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=multicast_cga_layout(num_ctas, 2))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK.value, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, output, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
def test_async_tma_multicast_kernel_local_store_race(device, run_wrapper, monkeypatch, num_ctas):
    if num_ctas == 1:
        pytest.skip("Need at least 2 CTAs for multicast in this test")
    if run_wrapper:
        result = run_in_process(test_async_tma_multicast_kernel_local_store_race,
                                (device, False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input_desc, out):
        cga_layout: ttgl.constexpr = multicast_cga_layout(ttgl.num_ctas(), 2)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], input_desc.layout)

        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        mbarrier.expect(bar, input_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar, smem, multicast=True)
        ttgl.barrier(cluster=True)
        smem.store(ttgl.full([XBLOCK, XBLOCK], 1, ttgl.float16, blocked_layout))
        mbarrier.wait(bar, 0, deps=[smem])
        val = smem.load(blocked_layout)
        mbarrier.invalidate(bar)

        out_m = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK + out_n
        ttgl.store(out_ptr, val)

    input = torch.randn((XBLOCK.value, XBLOCK.value), device=device, dtype=torch.float16)
    output = torch.empty((XBLOCK.value, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=multicast_cga_layout(num_ctas, 2))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK.value, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, output, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_async_tma_kernel_2bufs_1bar(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_async_tma_kernel_2bufs_1bar, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(a_desc, b_desc, out, FAILURE: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        a_smem = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], a_desc.layout)
        b_smem = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], b_desc.layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        mbarrier.expect(bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(a_desc, [0, 0], bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [0, 0], bar, b_smem)
        mbarrier.wait(bar, 0, pred=(not FAILURE), deps=[a_smem, b_smem])
        val = a_smem.load(blocked_layout)
        val = val + b_smem.load(blocked_layout)
        mbarrier.wait(bar, 0, pred=FAILURE, deps=[a_smem, b_smem])
        mbarrier.invalidate(bar)

        out_m = ttgl.arange(0, block_m, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK + out_n
        ttgl.store(out_ptr, val)

    block_m = XBLOCK.value * num_ctas
    a = torch.randn((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    b = torch.randn((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    output = torch.empty((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=default_cga_layout(num_ctas, 2))
    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(a, [block_m, XBLOCK.value], shared_layout)
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(b, [block_m, XBLOCK.value], shared_layout)
    kernel[(1, )](a_desc, b_desc, output, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("EXPECT_DELTA", [-16, 16], ids=["under", "over"])
def test_async_tma_expect_bytes_mismatch(EXPECT_DELTA, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_async_tma_expect_bytes_mismatch,
                                (EXPECT_DELTA, device, False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Deadlock detected" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input_desc, out, EXPECT_DELTA: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], input_desc.layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        mbarrier.expect(bar, input_desc.nbytes_per_cta + EXPECT_DELTA)
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar, smem)
        mbarrier.wait(bar, 0, deps=[smem])
        val = smem.load(blocked_layout)
        mbarrier.invalidate(bar)

        out_m = ttgl.arange(0, block_m, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK + out_n
        ttgl.store(out_ptr, val)

    block_m = XBLOCK.value * num_ctas
    input = torch.randn((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    output = torch.empty((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=default_cga_layout(num_ctas, 2))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [block_m, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, output, EXPECT_DELTA=EXPECT_DELTA, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_tma_interleave_kernel(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_tma_interleave_kernel, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input_desc, out, FAILURE: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, block_m, XBLOCK], input_desc.layout)
        bar = mbarrier.allocate_mbarrier(batch=2)
        mbarrier.init(bar.index(0), count=1)
        mbarrier.init(bar.index(1), count=1)
        mbarrier.expect(bar.index(0), input_desc.nbytes_per_cta)
        mbarrier.expect(bar.index(1), input_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar.index(0), smem.index(0))
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar.index(1), smem.index(1))

        mbarrier.wait(bar.index(0), 0, deps=[smem.index(0)])
        if not FAILURE:
            mbarrier.wait(bar.index(1), 0, deps=[smem.index(1)])

        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        out_m = ttgl.arange(0, block_m, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK + out_n
        ttgl.store(out_ptr, smem.index(0).load(blocked_layout))
        ttgl.store(out_ptr, smem.index(1).load(blocked_layout))

        mbarrier.invalidate(bar.index(0))
        mbarrier.invalidate(bar.index(1))

        tma.async_copy_shared_to_global(input_desc, [0, 0], smem.index(0))
        tma.store_wait(0)

    block_m = XBLOCK.value * num_ctas
    input = torch.randn((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    output = torch.empty((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=default_cga_layout(num_ctas, 2))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [block_m, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, output, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_tma_wait_tracks_only_waited_barrier(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_tma_wait_tracks_only_waited_barrier,
                                (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input_desc, out, FAILURE: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, block_m, XBLOCK], input_desc.layout)
        bar = mbarrier.allocate_mbarrier(batch=2)
        mbarrier.init(bar.index(0), count=1)
        mbarrier.init(bar.index(1), count=1)

        mbarrier.expect(bar.index(0), input_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar.index(0), smem.index(0))
        mbarrier.expect(bar.index(1), input_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar.index(1), smem.index(1))

        mbarrier.wait(bar.index(1), 0)

        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        if FAILURE:
            val = smem.index(0).load(blocked_layout)
        else:
            val = smem.index(1).load(blocked_layout)
        out_m = ttgl.arange(0, block_m, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, blocked_layout))[None, :]
        out_ptr = out + out_m * XBLOCK + out_n
        ttgl.store(out_ptr, val)

        if not FAILURE:
            mbarrier.wait(bar.index(0), 0)

        mbarrier.invalidate(bar.index(0))
        mbarrier.invalidate(bar.index(1))

    block_m = XBLOCK.value * num_ctas
    input = torch.randn((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    output = torch.empty((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=default_cga_layout(num_ctas, 2))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [block_m, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, output, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires ampere or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_async_copy(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_async_copy, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Accessing buffer with pending access. Pending access type: async_copy_global_to_shared" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input, FAILURE: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                                             cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, block_m, XBLOCK], smem_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        offs_m = ttgl.arange(0, block_m, layout=ttgl.SliceLayout(dim=1, parent=blocked_layout))[:, None]
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

    input = torch.randn((XBLOCK.value * num_ctas, XBLOCK.value), device=device, dtype=torch.float16)
    kernel[(1, )](input, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires ampere or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_tma_store(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_tma_store, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Accessing buffer with pending access. Pending access type: async_copy_shared_to_global" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(output_desc, FAILURE: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                                             cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, block_m, XBLOCK], smem_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        val = ttgl.full([block_m, XBLOCK], 42, ttgl.float16, blocked_layout)
        tma.async_copy_shared_to_global(output_desc, [0, 0], smem.index(0))
        tma.async_copy_shared_to_global(output_desc, [0, 0], smem.index(1))
        tma.store_wait(pendings=1)
        smem.index(0).store(val)
        if not FAILURE:
            tma.store_wait(pendings=0)
        smem.index(1).store(val)

    block_m = XBLOCK.value * num_ctas
    output = torch.empty((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=default_cga_layout(num_ctas, 2))
    output_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(output, [block_m, XBLOCK.value], shared_layout)
    kernel[(1, )](output_desc, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
@pytest.mark.parametrize("MEM_ACCESS_KIND", ["tma_cp", "local_store", "tmem_load", "tmem_store"])
@pytest.mark.parametrize("TWO_CTAS", [False, True])
def test_tcgen5_mma(FAILURE, MEM_ACCESS_KIND, TWO_CTAS, device, run_wrapper, monkeypatch, num_ctas):
    if TWO_CTAS and num_ctas == 1:
        pytest.skip("Need at least 2 CTAs for 2CTA mode in this test")
    if run_wrapper:
        result = run_in_process(test_tcgen5_mma,
                                (FAILURE, MEM_ACCESS_KIND, TWO_CTAS, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
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

    @gluon.jit
    def kernel(input_desc, output_desc, FAILURE: ttgl.constexpr, MEM_ACCESS_KIND: ttgl.constexpr,
               TWO_CTAS: ttgl.constexpr):
        block_m: ttgl.constexpr = mma_block_m(ttgl.num_ctas())
        block_n: ttgl.constexpr = mma_block_n(ttgl.num_ctas())
        acc_layout: ttgl.constexpr = blackwell.TensorMemoryLayout(
            [XBLOCK, XBLOCK],
            col_stride=1,
            cga_layout=mma_cga_layout(ttgl.num_ctas(), 2, TWO_CTAS),
            two_ctas=TWO_CTAS,
        )
        smem_a_blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1], warps_per_cta=[4, 1], order=[0, 1],
            cga_layout=mma_cga_layout(ttgl.num_ctas(), 0, TWO_CTAS))
        acc_blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                                warps_per_cta=[4, 1], order=[0, 1],
                                                                cga_layout=acc_layout.cga_layout)
        smemA = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], input_desc.layout)
        smemB = ttgl.allocate_shared_memory(
            ttgl.float16,
            [XBLOCK, block_n],
            ttgl.NVMMASharedLayout.get_default_for([XBLOCK, block_n], ttgl.float16,
                                                   cga_layout=mma_cga_layout(ttgl.num_ctas(), 1, TWO_CTAS)),
        )
        mma_bar = mbarrier.allocate_mbarrier()
        acc = blackwell.allocate_tensor_memory(ttgl.float32, [block_m, block_n], acc_layout)
        mbarrier.init(mma_bar, count=1)
        if MEM_ACCESS_KIND == "tma_cp":
            tma_bar = mbarrier.allocate_mbarrier(two_ctas=TWO_CTAS)
            mbarrier.init(tma_bar, count=1)

        blackwell.tcgen05_mma(smemA, smemB, acc)
        blackwell.tcgen05_commit(mma_bar)

        if not FAILURE:
            mbarrier.wait(mma_bar, 0)

        if MEM_ACCESS_KIND == "tma_cp":
            mbarrier.expect(tma_bar, input_desc.nbytes_per_cta)
            tma.async_copy_global_to_shared(input_desc, [0, 0], tma_bar, smemA)
            mbarrier.wait(tma_bar, 0)
            mbarrier.invalidate(tma_bar)
        elif MEM_ACCESS_KIND == "local_store":
            smemA.store(ttgl.full([block_m, XBLOCK], 42, ttgl.float16, smem_a_blocked_layout))
        elif MEM_ACCESS_KIND == "tmem_load":
            res = acc.load(acc_blocked_layout)
            smemAcc = ttgl.allocate_shared_memory(
                input_desc.dtype, [block_m, block_n],
                ttgl.NVMMASharedLayout.get_default_for([block_m, block_n], input_desc.dtype,
                                                       cga_layout=acc_layout.cga_layout), res.to(input_desc.dtype))
            tma.async_copy_shared_to_global(output_desc, [0, 0], smemAcc)
            tma.store_wait(0)
        elif MEM_ACCESS_KIND == "tmem_store":
            acc.store(ttgl.full([block_m, block_n], 42, ttgl.float32, acc_blocked_layout))

        mbarrier.invalidate(mma_bar)

    block_m = mma_block_m(num_ctas)
    block_n = mma_block_n(num_ctas)
    input = torch.randn((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout.get_default_for([block_m, XBLOCK.value], ttgl.float16,
                                                           cga_layout=mma_cga_layout(num_ctas, 0, TWO_CTAS))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [block_m, XBLOCK.value], shared_layout)
    output = torch.empty((block_m, block_n), device=device, dtype=torch.float16)
    output_layout = ttgl.NVMMASharedLayout.get_default_for([block_m, block_n], ttgl.float16,
                                                           cga_layout=mma_cga_layout(num_ctas, 2, TWO_CTAS))
    output_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(output, [block_m, block_n], output_layout)
    kernel[(1, )](input_desc, output_desc, FAILURE=FAILURE, MEM_ACCESS_KIND=MEM_ACCESS_KIND, TWO_CTAS=TWO_CTAS,
                  num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
@pytest.mark.parametrize("MEM_ACCESS_KIND", ["local_store", "tmem_load"])
def test_tcgen5_copy(FAILURE, MEM_ACCESS_KIND, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_tcgen5_copy, (FAILURE, MEM_ACCESS_KIND, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            if MEM_ACCESS_KIND == "local_store":
                assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
            else:
                assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input, output, FAILURE: ttgl.constexpr, MEM_ACCESS_KIND: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        tmem_layout: ttgl.constexpr = blackwell.TensorMemoryLayout((128, XBLOCK), col_stride=1, cga_layout=cga_layout)
        tmem = blackwell.allocate_tensor_memory(ttgl.int32, [block_m, XBLOCK], tmem_layout)
        reg_layout: ttgl.constexpr = tmem.get_reg_layout()
        offs_m = ttgl.arange(0, block_m, ttgl.SliceLayout(1, reg_layout))[:, None]
        offs_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, reg_layout))[None, :]
        offs = offs_m * XBLOCK + offs_n
        val = ttgl.load(input + offs)
        smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2,
                                                             cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.int32, [block_m, XBLOCK], smem_layout)
        smem.store(val)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        blackwell.tcgen05_copy(smem, tmem)
        blackwell.tcgen05_commit(bar)
        if not FAILURE:
            mbarrier.wait(bar, 0)
        if MEM_ACCESS_KIND == "local_store":
            smem.store(ttgl.zeros([block_m, XBLOCK], ttgl.int32, reg_layout))
        else:
            val = tmem.load(reg_layout)
            ttgl.store(output + offs, val)
        if FAILURE:
            mbarrier.wait(bar, 0)
        mbarrier.invalidate(bar)

    input = torch.arange(XBLOCK.value * XBLOCK.value * num_ctas, device=device,
                         dtype=torch.int32).reshape(XBLOCK.value * num_ctas, XBLOCK.value)
    output = torch.empty_like(input)
    kernel[(1, )](input, output, FAILURE=FAILURE, MEM_ACCESS_KIND=MEM_ACCESS_KIND, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_warpgroup_mma(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_warpgroup_mma, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Accessing buffer with pending access. Pending access type: warpgroup_mma operand read" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input, FAILURE: ttgl.constexpr):
        block_m: ttgl.constexpr = mma_block_m(ttgl.num_ctas())
        block_n: ttgl.constexpr = mma_block_n(ttgl.num_ctas())
        cga_layout_a: ttgl.constexpr = mma_cga_layout(ttgl.num_ctas(), 0)
        cga_layout_b: ttgl.constexpr = mma_cga_layout(ttgl.num_ctas(), 1)
        cga_layout_c: ttgl.constexpr = mma_cga_layout(ttgl.num_ctas(), 2)
        smem_layout_a: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                                               cga_layout=cga_layout_a)
        smem_layout_b: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                                               cga_layout=cga_layout_b)
        smemA = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], smem_layout_a)
        smemB = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, block_n], smem_layout_b)

        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout_a)

        acc_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                                 instr_shape=[16, 32, 16], cga_layout=cga_layout_c)
        acc = ttgl.zeros([block_m, block_n], ttgl.float16, acc_layout)
        acc = hopper.warpgroup_mma(smemA, smemB, acc, is_async=True)
        if FAILURE:
            smemA.store(ttgl.full([block_m, XBLOCK], 42, ttgl.float16, blocked_layout))
        hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
        smemA.store(ttgl.full([block_m, XBLOCK], 42, ttgl.float16, blocked_layout))

    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    kernel[(1, )](input, FAILURE=FAILURE, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_warpgroup_mma2(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_warpgroup_mma2, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Accessing buffer with pending access. Pending access type: warpgroup_mma operand read" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input, FAILURE: ttgl.constexpr):
        block_m: ttgl.constexpr = mma_block_m(ttgl.num_ctas())
        block_n: ttgl.constexpr = mma_block_n(ttgl.num_ctas())
        cga_layout_a: ttgl.constexpr = mma_cga_layout(ttgl.num_ctas(), 0)
        cga_layout_b: ttgl.constexpr = mma_cga_layout(ttgl.num_ctas(), 1)
        cga_layout_c: ttgl.constexpr = mma_cga_layout(ttgl.num_ctas(), 2)
        smem_layout_a: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                                               cga_layout=cga_layout_a)
        smem_layout_b: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                                               cga_layout=cga_layout_b)
        smemA = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], smem_layout_a)
        smemB = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, block_n], smem_layout_b)

        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout_a)

        acc_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                                 instr_shape=[16, 32, 16], cga_layout=cga_layout_c)
        acc = ttgl.zeros([block_m, block_n], ttgl.float16, acc_layout)
        acc = hopper.warpgroup_mma(smemA, smemB, acc, is_async=True)
        acc = hopper.warpgroup_mma(smemA, smemB, acc, is_async=True)
        hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc])
        if FAILURE:
            smemA.store(ttgl.full([block_m, XBLOCK], 42, ttgl.float16, blocked_layout))
        hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
        smemA.store(ttgl.full([block_m, XBLOCK], 42, ttgl.float16, blocked_layout))

    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    kernel[(1, )](input, FAILURE=FAILURE, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("BUF_IDX", [0, 1])
@pytest.mark.parametrize("BAR_IDX", [0, 1, 2, 3])
def test_tcgen5_mma_multibar(BUF_IDX, BAR_IDX, device, run_wrapper, monkeypatch, num_ctas):
    if BAR_IDX == 0:
        pytest.skip("Skipping due to wait on false-predicated barrier - not supported yet")
    if run_wrapper:
        result = run_in_process(test_tcgen5_mma_multibar, (BUF_IDX, BAR_IDX, device, False, monkeypatch, num_ctas))
        if BAR_IDX // 2 < BUF_IDX:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input_desc, BUF_IDX: ttgl.constexpr, BAR_IDX: ttgl.constexpr):
        block_m: ttgl.constexpr = mma_block_m(ttgl.num_ctas())
        block_n: ttgl.constexpr = mma_block_n(ttgl.num_ctas())
        acc_layout: ttgl.constexpr = blackwell.TensorMemoryLayout(
            [XBLOCK, XBLOCK],
            col_stride=1,
            cga_layout=mma_cga_layout(ttgl.num_ctas(), 2),
        )
        acc_blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                                warps_per_cta=[4, 1], order=[0, 1],
                                                                cga_layout=acc_layout.cga_layout)
        smemA = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], input_desc.layout)
        smemB = ttgl.allocate_shared_memory(
            ttgl.float16,
            [XBLOCK, block_n],
            ttgl.NVMMASharedLayout.get_default_for([XBLOCK, block_n], ttgl.float16,
                                                   cga_layout=mma_cga_layout(ttgl.num_ctas(), 1)),
        )
        bar = mbarrier.allocate_mbarrier(batch=4)
        acc = blackwell.allocate_tensor_memory(ttgl.float32, [2, block_m, block_n], acc_layout)
        for i in range(4):
            mbarrier.init(bar.index(i), count=1)

        blackwell.tcgen05_mma(smemA, smemB, acc.index(0), mbarriers=[bar.index(0), bar.index(1)],
                              mbarrier_preds=[False, True])
        blackwell.tcgen05_mma(smemA, smemB, acc.index(1), mbarriers=[bar.index(2)])
        blackwell.tcgen05_commit(bar.index(3))

        mbarrier.wait(bar.index(BAR_IDX), 0)

        store_shape: ttgl.constexpr = [block_m, block_n]
        acc.index(BUF_IDX).store(ttgl.full(store_shape, 42, ttgl.float32, acc_blocked_layout))

        for i in range(4):
            mbarrier.invalidate(bar.index(i))

    block_m = mma_block_m(num_ctas)
    input = torch.randn((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout.get_default_for([block_m, XBLOCK.value], ttgl.float16,
                                                           cga_layout=mma_cga_layout(num_ctas, 0))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [block_m, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, BUF_IDX, BAR_IDX, num_warps=4, num_ctas=num_ctas)


@gluon.jit
def inc_mod(x, mod):
    return (x + 1) % mod


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_multibuffered_loop(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_multibuffered_loop, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(a_desc, b_desc, FAILURE: ttgl.constexpr):
        num_buffers: ttgl.constexpr = 2 if FAILURE else 3
        num_mma_stages: ttgl.constexpr = 2
        block_m: ttgl.constexpr = mma_block_m(ttgl.num_ctas())
        block_n: ttgl.constexpr = mma_block_n(ttgl.num_ctas())

        acc_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([XBLOCK, XBLOCK], col_stride=1,
                                                                  cga_layout=mma_cga_layout(ttgl.num_ctas(), 2))
        zero_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                         warps_per_cta=[4, 1], order=[0, 1],
                                                         cga_layout=mma_cga_layout(ttgl.num_ctas(), 2))
        zero = ttgl.zeros([block_m, block_n], ttgl.float32, zero_layout)
        b_smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([XBLOCK, block_n], ttgl.float16,
                                                                               cga_layout=mma_cga_layout(
                                                                                   ttgl.num_ctas(), 1))

        smemA = ttgl.allocate_shared_memory(ttgl.float16, [num_buffers, block_m, XBLOCK], a_desc.layout)
        smemB = ttgl.allocate_shared_memory(ttgl.float16, [num_buffers, XBLOCK, block_n], b_smem_layout)
        barLoadA = mbarrier.allocate_mbarrier(batch=num_buffers)
        barLoadB = mbarrier.allocate_mbarrier(batch=num_buffers)
        barMMA = mbarrier.allocate_mbarrier(batch=num_mma_stages)
        acc = blackwell.allocate_tensor_memory(ttgl.float32, [block_m, block_n], acc_layout, zero)
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
        mbarrier.expect(barLoadA.index(ins_id), a_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(a_desc, [0, 0], barLoadA.index(ins_id), smemA.index(ins_id))

        mbarrier.expect(barLoadB.index(ins_id), b_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(b_desc, [0, 0], barLoadB.index(ins_id), smemB.index(ins_id))
        ins_id = inc_mod(ins_id, num_buffers)

        # ins_id = 1
        mbarrier.expect(barLoadA.index(ins_id), a_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(a_desc, [0, 0], barLoadA.index(ins_id), smemA.index(ins_id))

        mbarrier.expect(barLoadB.index(ins_id), b_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(b_desc, [0, 0], barLoadB.index(ins_id), smemB.index(ins_id))
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
                mbarrier.expect(barLoadA.index(ins_id), a_desc.nbytes_per_cta)
                tma.async_copy_global_to_shared(a_desc, [0, 0], barLoadA.index(ins_id), smemA.index(ins_id))

                mbarrier.expect(barLoadB.index(ins_id), b_desc.nbytes_per_cta)
                tma.async_copy_global_to_shared(b_desc, [0, 0], barLoadB.index(ins_id), smemB.index(ins_id))
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

    block_m = mma_block_m(num_ctas)
    block_n = mma_block_n(num_ctas)
    input = torch.randn((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(
        input, [block_m, XBLOCK.value],
        ttgl.NVMMASharedLayout.get_default_for([block_m, XBLOCK.value], ttgl.float16,
                                               cga_layout=mma_cga_layout(num_ctas, 0)))
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(
        input, [XBLOCK.value, block_n],
        ttgl.NVMMASharedLayout.get_default_for([XBLOCK.value, block_n], ttgl.float16,
                                               cga_layout=mma_cga_layout(num_ctas, 1)))
    kernel[(1, )](a_desc, b_desc, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_tma_tcgen05_mma_multicast_loop(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if num_ctas == 1:
        pytest.skip("Need at least 2 CTAs for 2CTA mode in this test")
    if run_wrapper:
        result = run_in_process(test_tma_tcgen05_mma_multicast_loop, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(a_desc, b_desc, FAILURE: ttgl.constexpr):
        num_k_tiles: ttgl.constexpr = 1 if FAILURE else 4
        block_m: ttgl.constexpr = mma_block_m(ttgl.num_ctas())
        block_n: ttgl.constexpr = mma_block_n(ttgl.num_ctas())
        acc_layout: ttgl.constexpr = blackwell.TensorMemoryLayout(
            [XBLOCK, XBLOCK],
            col_stride=1,
            cga_layout=mma_cga_layout(ttgl.num_ctas(), 2, True),
            two_ctas=True,
        )
        smemA = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], a_desc.layout)
        smemB = ttgl.allocate_shared_memory(
            ttgl.float16,
            [XBLOCK, block_n],
            ttgl.NVMMASharedLayout.get_default_for([XBLOCK, block_n], ttgl.float16,
                                                   cga_layout=mma_cga_layout(ttgl.num_ctas(), 1, True)),
        )
        tma_bar = mbarrier.allocate_mbarrier(two_ctas=True)
        mbarrier.init(tma_bar, count=1)
        mma_bar = mbarrier.allocate_mbarrier()
        mma_bar_count: ttgl.constexpr = blackwell.tcgen05_mma_barrier_count([smemA, smemB], True)
        mbarrier.init(mma_bar, count=mma_bar_count)
        acc = blackwell.allocate_tensor_memory(ttgl.float32, [block_m, block_n], acc_layout)

        phase_tma = 0
        phase_mma = 0
        for k in range(num_k_tiles):
            offs_k = k * XBLOCK
            mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
            tma.async_copy_global_to_shared(a_desc, [0, offs_k], tma_bar, smemA, multicast=True)
            tma.async_copy_global_to_shared(b_desc, [offs_k, 0], tma_bar, smemB, multicast=True)
            if not FAILURE:
                mbarrier.wait(tma_bar, phase_tma, deps=[smemA, smemB])
            blackwell.tcgen05_mma(smemA, smemB, acc, use_acc=k != 0, multicast=True, mbarriers=[mma_bar])
            mbarrier.wait(mma_bar, phase_mma, deps=[smemA, smemB])
            phase_tma = (phase_tma + 1) % 2
            phase_mma = (phase_mma + 1) % 2

        mbarrier.invalidate(tma_bar)
        mbarrier.invalidate(mma_bar)

    block_m = mma_block_m(num_ctas)
    block_n = mma_block_n(num_ctas)
    num_k_tiles = 1 if FAILURE else 4
    a = torch.randn((block_m, XBLOCK.value * num_k_tiles), device=device, dtype=torch.float16)
    b = torch.randn((XBLOCK.value * num_k_tiles, block_n), device=device, dtype=torch.float16)
    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(
        a, [block_m, XBLOCK.value],
        ttgl.NVMMASharedLayout.get_default_for([block_m, XBLOCK.value], ttgl.float16,
                                               cga_layout=mma_cga_layout(num_ctas, 0, True)))
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(
        b, [XBLOCK.value, block_n],
        ttgl.NVMMASharedLayout.get_default_for([XBLOCK.value, block_n], ttgl.float16,
                                               cga_layout=mma_cga_layout(num_ctas, 1, True)))
    kernel[(1, )](a_desc, b_desc, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_multibuffered_wgmma_loop(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_multibuffered_wgmma_loop, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Accessing buffer with pending access. Pending access type: warpgroup_mma operand read" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(a_desc, b_desc, FAILURE: ttgl.constexpr):
        num_buffers: ttgl.constexpr = 2 if FAILURE else 3
        block_m: ttgl.constexpr = mma_block_m(ttgl.num_ctas())
        block_n: ttgl.constexpr = mma_block_n(ttgl.num_ctas())

        cga_layout_c: ttgl.constexpr = mma_cga_layout(ttgl.num_ctas(), 2)
        mma_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                                 instr_shape=[16, 32, 16], cga_layout=cga_layout_c)
        acc = hopper.warpgroup_mma_init(ttgl.zeros([block_m, block_n], ttgl.float32, mma_layout))

        smemA = ttgl.allocate_shared_memory(ttgl.float16, [num_buffers, block_m, XBLOCK], a_desc.layout)
        smemB = ttgl.allocate_shared_memory(ttgl.float16, [num_buffers, XBLOCK, block_n], b_desc.layout)
        barLoadA = mbarrier.allocate_mbarrier(batch=num_buffers)
        barLoadB = mbarrier.allocate_mbarrier(batch=num_buffers)
        for i in range(num_buffers):
            mbarrier.init(barLoadA.index(i), count=1)
            mbarrier.init(barLoadB.index(i), count=1)

        phase = 0
        ins_id = 0
        ext_id = 0

        # ins_id = 0
        mbarrier.expect(barLoadA.index(ins_id), a_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(a_desc, [0, 0], barLoadA.index(ins_id), smemA.index(ins_id))

        mbarrier.expect(barLoadB.index(ins_id), b_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(b_desc, [0, 0], barLoadB.index(ins_id), smemB.index(ins_id))
        ins_id = inc_mod(ins_id, num_buffers)

        # ins_id = 1
        ub = 10
        for i in range(ub):
            if i < ub - 1:
                mbarrier.expect(barLoadA.index(ins_id), a_desc.nbytes_per_cta)
                tma.async_copy_global_to_shared(a_desc, [0, 0], barLoadA.index(ins_id), smemA.index(ins_id))

                mbarrier.expect(barLoadB.index(ins_id), b_desc.nbytes_per_cta)
                tma.async_copy_global_to_shared(b_desc, [0, 0], barLoadB.index(ins_id), smemB.index(ins_id))
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

    block_m = mma_block_m(num_ctas)
    block_n = mma_block_n(num_ctas)
    input_a = torch.randn((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    input_b = torch.randn((XBLOCK.value, block_n), device=device, dtype=torch.float16)
    shared_layout_a = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                             cga_layout=mma_cga_layout(num_ctas, 0))
    shared_layout_b = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                             cga_layout=mma_cga_layout(num_ctas, 1))
    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input_a, [block_m, XBLOCK.value], shared_layout_a)
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input_b, [XBLOCK.value, block_n], shared_layout_b)
    kernel[(1, )](a_desc, b_desc, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_store_wait_load(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_ws_store_wait_load, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def ws_default(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        mbarrier.wait(bar.index(0), phase=0, pred=(not FAILURE))
        val = smem.index(0).load(layout)
        smem.index(1).store(val)
        mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def ws_1(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        smem.index(0).store(ttgl.arange(0, block_x, layout).to(ttgl.float16))
        mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def ws_kernel(output, FAILURE: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 1)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0],
                                                                cga_layout=cga_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, block_x], smem_layout)
        bar = mbarrier.allocate_mbarrier(batch=2)
        for i in range(2):
            mbarrier.init(bar.index(i), count=1)
        ttgl.warp_specialize([
            (ws_default, (smem, bar, FAILURE, blocked_layout)),
            (ws_1, (smem, bar, FAILURE, blocked_layout)),
        ], [4], [32])
        mbarrier.wait(bar.index(1), phase=0)
        val = smem.index(0).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, block_x, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK.value * num_ctas, ), device=device, dtype=torch.float16)
    ws_kernel[(1, )](output, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_load_wait_store(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_ws_load_wait_store, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def ws_default(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        mbarrier.wait(bar.index(0), phase=0, pred=(not FAILURE))
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        smem.index(0).store(ttgl.arange(0, block_x, layout).to(ttgl.float16))
        mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def ws_1(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        val = smem.index(0).load(layout)
        mbarrier.arrive(bar.index(0), count=1)
        smem.index(1).store(val)  # dummy store to make sure the load is executed

    @gluon.jit
    def ws_kernel(output, FAILURE: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 1)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0],
                                                                cga_layout=cga_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, block_x], smem_layout)
        bar = mbarrier.allocate_mbarrier(batch=2)
        for i in range(2):
            mbarrier.init(bar.index(i), count=1)
        ttgl.warp_specialize([
            (ws_default, (smem, bar, FAILURE, blocked_layout)),
            (ws_1, (smem, bar, FAILURE, blocked_layout)),
        ], [4], [32])
        mbarrier.wait(bar.index(1), phase=0)
        val = smem.index(0).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, block_x, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK.value * num_ctas, ), device=device, dtype=torch.float16)
    ws_kernel[(1, )](output, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("MISSING_BAR", ["none", "1", "2"])
def test_ws_two_loads_two_bars(MISSING_BAR, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_ws_two_loads_two_bars, (MISSING_BAR, device, False, monkeypatch, num_ctas))
        if MISSING_BAR != "none":
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

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
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        smem.index(0).store(ttgl.arange(0, block_x, layout).to(ttgl.float16))
        mbarrier.arrive(bar.index(2), count=1)

    @gluon.jit
    def kernel(output, MISSING_BAR: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 1)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0],
                                                                cga_layout=cga_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, block_x], smem_layout)
        bar = mbarrier.allocate_mbarrier(batch=3)
        for i in range(3):
            mbarrier.init(bar.index(i), count=1)
        ttgl.warp_specialize([
            (ws_default, (smem, bar, MISSING_BAR, blocked_layout)),
            (ws_1, (smem, bar, MISSING_BAR, blocked_layout)),
            (ws_2, (smem, bar, MISSING_BAR, blocked_layout)),
        ], [4, 4], [32, 32])
        mbarrier.wait(bar.index(2), phase=0)
        val = smem.index(0).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, block_x, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK.value * num_ctas, ), device=device, dtype=torch.float16)
    kernel[(1, )](output, MISSING_BAR=MISSING_BAR, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_two_loads_one_bar(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_ws_two_loads_one_bar, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

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
        mbarrier.wait(bar.index(0), phase=0, pred=(not FAILURE), deps=[smem.index(0)])
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        smem.index(0).store(ttgl.arange(0, block_x, layout).to(ttgl.float16))
        mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def kernel(output, FAILURE: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 1)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0],
                                                                cga_layout=cga_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, block_x], smem_layout)
        bar = mbarrier.allocate_mbarrier(batch=2)
        mbarrier.init(bar.index(0), count=2)
        mbarrier.init(bar.index(1), count=1)
        ttgl.warp_specialize([
            (ws_default, (smem, bar, FAILURE, blocked_layout)),
            (ws_1, (smem, bar, FAILURE, blocked_layout)),
            (ws_2, (smem, bar, FAILURE, blocked_layout)),
        ], [4, 4], [32, 32])
        mbarrier.wait(bar.index(1), phase=0, deps=[smem.index(0)])
        val = smem.index(0).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, block_x, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK.value * num_ctas, ), device=device, dtype=torch.float16)
    kernel[(1, )](output, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("MISSING_BAR", ["none", "0", "1", "2", "3"])
def test_ws_two_loads_two_bars_loop(MISSING_BAR, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_ws_two_loads_two_bars_loop, (MISSING_BAR, device, False, monkeypatch, num_ctas))
        if MISSING_BAR != "none":
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def ws_default(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        acc = ttgl.zeros([block_x], ttgl.float16, layout)
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
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        acc = ttgl.zeros([block_x], ttgl.float16, layout)
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
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        phase = 0
        for _ in range(10):
            if MISSING_BAR != "0":
                mbarrier.wait(bar.index(0), phase=phase)
            if MISSING_BAR != "1":
                mbarrier.wait(bar.index(1), phase=phase)
            phase = (phase + 1) % 2
            smem.index(0).store(ttgl.arange(0, block_x, layout).to(ttgl.float16))
            mbarrier.arrive(bar.index(2), count=1)
            mbarrier.arrive(bar.index(3), count=1)

    @gluon.jit
    def kernel(output, MISSING_BAR: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 1)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0],
                                                                cga_layout=cga_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, block_x], smem_layout)
        bar = mbarrier.allocate_mbarrier(batch=4)
        for i in range(4):
            mbarrier.init(bar.index(i), count=1)

        mbarrier.arrive(bar.index(2), count=1)
        mbarrier.arrive(bar.index(3), count=1)

        ttgl.warp_specialize([
            (ws_default, (smem, bar, MISSING_BAR, blocked_layout)),
            (ws_1, (smem, bar, MISSING_BAR, blocked_layout)),
            (ws_2, (smem, bar, MISSING_BAR, blocked_layout)),
        ], [4, 4], [32, 32])

    output = torch.empty((XBLOCK.value * num_ctas, ), device=device, dtype=torch.float16)
    kernel[(1, )](output, MISSING_BAR=MISSING_BAR, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_load_ordering(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_ws_load_ordering, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def ws_default(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        phase = 0
        for _ in range(10):
            mbarrier.wait(bar.index(2), phase=phase)
            phase = (phase + 1) % 2
            smem.index(0).store(ttgl.arange(0, block_x, layout).to(ttgl.float16))
            mbarrier.arrive(bar.index(0), count=1)
            smem.index(1).store(ttgl.arange(0, block_x, layout).to(ttgl.float16))
            mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def ws_1(smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        acc = ttgl.zeros([block_x], ttgl.float16, layout)
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
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 1)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0],
                                                                cga_layout=cga_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, block_x], smem_layout)
        bar = mbarrier.allocate_mbarrier(batch=3)
        for i in range(3):
            mbarrier.init(bar.index(i), count=1)

        mbarrier.arrive(bar.index(2), count=1)

        ttgl.warp_specialize([
            (ws_default, (smem, bar, FAILURE, blocked_layout)),
            (ws_1, (smem, bar, FAILURE, blocked_layout)),
        ], [4], [32])

    output = torch.empty((XBLOCK.value * num_ctas, ), device=device, dtype=torch.float16)
    kernel[(1, )](output, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("MISSING_BAR", ["none", "T2", "T3"])
def test_ws_two_producers_two_consumers(MISSING_BAR, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_ws_two_producers_two_consumers,
                                (MISSING_BAR, device, False, monkeypatch, num_ctas))
        if MISSING_BAR != "none":
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def ws_default(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        phase = 0
        for _ in range(10):
            mbarrier.wait(bar.index(2), phase=phase)
            phase = (phase + 1) % 2
            smem.index(0).store(ttgl.arange(0, block_x, layout).to(ttgl.float16))
            mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def ws_1(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        phase = 0
        for _ in range(10):
            mbarrier.wait(bar.index(3), phase=phase)
            phase = (phase + 1) % 2
            smem.index(1).store(ttgl.arange(0, block_x, layout).to(ttgl.float16))
            mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def ws_2(smem, bar, MISSING_BAR: ttgl.constexpr, layout: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        acc = ttgl.zeros([block_x], ttgl.float16, layout)
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
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        acc = ttgl.zeros([block_x], ttgl.float16, layout)
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
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 1)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0],
                                                                cga_layout=cga_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [4, block_x], smem_layout)
        bar = mbarrier.allocate_mbarrier(batch=4)
        for i in range(4):
            mbarrier.init(bar.index(i), count=2)

        mbarrier.arrive(bar.index(2), count=2)
        mbarrier.arrive(bar.index(3), count=2)

        ttgl.warp_specialize([
            (ws_default, (smem, bar, MISSING_BAR, blocked_layout)),
            (ws_1, (smem, bar, MISSING_BAR, blocked_layout)),
            (ws_2, (smem, bar, MISSING_BAR, blocked_layout)),
            (ws_3, (smem, bar, MISSING_BAR, blocked_layout)),
        ], [4, 4, 4], [32, 32, 32])

    output = torch.empty((XBLOCK.value * num_ctas, ), device=device, dtype=torch.float16)
    kernel[(1, )](output, MISSING_BAR=MISSING_BAR, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("MISSING_BAR", ["none", "1", "2"])
def test_ws_different_warp_sizes(MISSING_BAR, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_ws_different_warp_sizes, (MISSING_BAR, device, False, monkeypatch, num_ctas))
        if MISSING_BAR != "none":
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def ws_default(smem, bar, MISSING_BAR: ttgl.constexpr):
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 1)
        layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4],
                                                    order=[0], cga_layout=cga_layout)
        val = smem.index(0).load(layout)
        mbarrier.arrive(bar.index(0), count=1)
        smem.index(1).store(val)  # dummy store to make sure the load is executed

    @gluon.jit
    def ws_1(smem, bar, MISSING_BAR: ttgl.constexpr):
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 1)
        layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[2],
                                                    order=[0], cga_layout=cga_layout)
        val = smem.index(0).load(layout)
        mbarrier.arrive(bar.index(1), count=1)
        smem.index(2).store(val)  # dummy store to make sure the load is executed

    @gluon.jit
    def ws_2(smem, bar, MISSING_BAR: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 1)
        layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[8],
                                                    order=[0], cga_layout=cga_layout)
        if MISSING_BAR != "1":
            mbarrier.wait(bar.index(0), phase=0)
        if MISSING_BAR != "2":
            mbarrier.wait(bar.index(1), phase=0)
        smem.index(0).store(ttgl.arange(0, block_x, layout).to(ttgl.float16))
        mbarrier.arrive(bar.index(2), count=1)

    @gluon.jit
    def kernel(output, MISSING_BAR: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 1)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0],
                                                                cga_layout=cga_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [3, block_x], smem_layout)
        bar = mbarrier.allocate_mbarrier(batch=3)
        for i in range(3):
            mbarrier.init(bar.index(i), count=1)
        ttgl.warp_specialize([
            (ws_default, (smem, bar, MISSING_BAR)),
            (ws_1, (smem, bar, MISSING_BAR)),
            (ws_2, (smem, bar, MISSING_BAR)),
        ], [2, 8], [32, 32])
        mbarrier.wait(bar.index(2), phase=0)
        val = smem.index(0).load(blocked_layout)
        output_ptrs = output + ttgl.arange(0, block_x, blocked_layout)
        ttgl.store(output_ptrs, val)

    output = torch.empty((XBLOCK.value * num_ctas, ), device=device, dtype=torch.float16)
    kernel[(1, )](output, MISSING_BAR=MISSING_BAR, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_async_copy_commits(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_ws_async_copy_commits, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert any(msg in result.driver_stderr_output for msg in [
                "Buffer being accessed has outstanding writes",
                "Buffer being accessed has outstanding reads",
                "Accessing buffer with pending access. Pending access type: async_copy_global_to_shared",
            ])
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def ws_prog(input, smem, FAILURE: ttgl.constexpr, blocked_layout: ttgl.constexpr, BASE: ttgl.constexpr):
        # Two-buffer ping-pong within a partition: buffers BASE and BASE+1
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        offs = ttgl.arange(0, block_x, layout=blocked_layout)

        acc = ttgl.zeros([block_x], ttgl.float16, blocked_layout)

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
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 1)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0],
                                                                cga_layout=cga_layout)
        # 4 buffers total: ws_default uses 0/1; ws_1 uses 2/3
        smem = ttgl.allocate_shared_memory(ttgl.float16, [4, block_x], smem_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[block_x], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0], cga_layout=cga_layout)
        ttgl.warp_specialize([
            (ws_prog, (input, smem, FAILURE, blocked_layout, 0)),
            (ws_prog, (input, smem, FAILURE, blocked_layout, 2)),
        ], [4], [32])

    input = torch.randn((XBLOCK.value * num_ctas, ), device=device, dtype=torch.float16)
    kernel[(1, )](input, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_async_copy_wait_visibility(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_ws_async_copy_wait_visibility, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
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

    @gluon.jit
    def ws_default(input, smem, bar, FAILURE: ttgl.constexpr, layout: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        offs = ttgl.arange(0, block_x, layout)
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
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 1)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0],
                                                                cga_layout=cga_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[block_x], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, block_x], smem_layout)
        bar = mbarrier.allocate_mbarrier(batch=1)
        mbarrier.init(bar.index(0), count=1)
        ttgl.warp_specialize([
            (ws_default, (input, smem, bar, FAILURE, blocked_layout)),
            (ws_1, (input, smem, bar, FAILURE, blocked_layout)),
        ], [4], [32])

    input = torch.randn((XBLOCK.value * num_ctas, ), device=device, dtype=torch.float16)
    kernel[(1, )](input, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 9, reason="Requires hopper")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_wgmma_wait_visibility(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_ws_wgmma_wait_visibility, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Accessing buffer with pending access. Pending access type: warpgroup_mma operand read" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def ws_default(smemA, smemB, bar, FAILURE: ttgl.constexpr, blocked_layout: ttgl.constexpr,
                   mma_layout: ttgl.constexpr):
        block_m: ttgl.constexpr = mma_block_m(ttgl.num_ctas())
        block_n: ttgl.constexpr = mma_block_n(ttgl.num_ctas())
        acc = ttgl.zeros([block_m, block_n], ttgl.float16, mma_layout)
        # Issue two async MMAs on two different buffers
        acc = hopper.warpgroup_mma(smemA.index(0), smemB.index(0), acc, is_async=True)
        acc = hopper.warpgroup_mma(smemA.index(1), smemB.index(1), acc, is_async=True)
        # Wait until only 1 outstanding remains
        hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc])
        # Signal to consumer
        mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def ws_1(smemA, smemB, bar, FAILURE: ttgl.constexpr, blocked_layout: ttgl.constexpr):
        block_m: ttgl.constexpr = mma_block_m(ttgl.num_ctas())
        mbarrier.wait(bar.index(0), phase=0)
        val = ttgl.full([block_m, XBLOCK], 42, ttgl.float16, blocked_layout)
        smemA.index(1 if FAILURE else 0).store(val)

    @gluon.jit
    def kernel(FAILURE: ttgl.constexpr):
        block_m: ttgl.constexpr = mma_block_m(ttgl.num_ctas())
        block_n: ttgl.constexpr = mma_block_n(ttgl.num_ctas())
        cga_layout_a: ttgl.constexpr = mma_cga_layout(ttgl.num_ctas(), 0)
        cga_layout_b: ttgl.constexpr = mma_cga_layout(ttgl.num_ctas(), 1)
        cga_layout_c: ttgl.constexpr = mma_cga_layout(ttgl.num_ctas(), 2)
        smem_layout_a: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                                               cga_layout=cga_layout_a)
        smem_layout_b: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                                               cga_layout=cga_layout_b)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout_a)
        mma_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                                 instr_shape=[16, 32, 16], cga_layout=cga_layout_c)
        smemA = ttgl.allocate_shared_memory(ttgl.float16, [2, block_m, XBLOCK], smem_layout_a)
        smemB = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK, block_n], smem_layout_b)
        bar = mbarrier.allocate_mbarrier(batch=1)
        mbarrier.init(bar.index(0), count=1)
        ttgl.warp_specialize([
            (ws_default, (smemA, smemB, bar, FAILURE, blocked_layout, mma_layout)),
            (ws_1, (smemA, smemB, bar, FAILURE, blocked_layout)),
        ], [4], [32])

    kernel[(1, )](FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_deadlock_two_partitions(device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_deadlock_two_partitions, (device, False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Deadlock detected" in result.driver_stderr_output
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def ws_default(bar):
        mbarrier.wait(bar.index(0), phase=0)

    @gluon.jit
    def ws_1(bar):
        mbarrier.wait(bar.index(1), phase=0)

    @gluon.jit
    def kernel():
        bar = mbarrier.allocate_mbarrier(batch=2)
        mbarrier.init(bar.index(0), count=1)
        mbarrier.init(bar.index(1), count=1)
        ttgl.warp_specialize([
            (ws_default, (bar, )),
            (ws_1, (bar, )),
        ], [4], [32])

    kernel[(1, )](num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_deadlock_overarrival(device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_deadlock_overarrival, (device, False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Deadlock detected" in result.driver_stderr_output
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel():
        bar = mbarrier.allocate_mbarrier(batch=2)
        mbarrier.init(bar.index(0), count=1)
        mbarrier.init(bar.index(1), count=1)

        mbarrier.arrive(bar.index(0), count=1)
        mbarrier.arrive(bar.index(0), count=1)
        mbarrier.wait(bar.index(0), phase=0)

    kernel[(1, )](num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_deadlock_underarrival(device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_deadlock_underarrival, (device, False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Deadlock detected" in result.driver_stderr_output
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

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
        bar = mbarrier.allocate_mbarrier(batch=2)
        mbarrier.init(bar.index(0), count=2)
        mbarrier.init(bar.index(1), count=2)
        ttgl.warp_specialize([
            (ws_default, (bar, )),
            (ws_1, (bar, )),
        ], [4], [32])

    kernel[(1, )](num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_deadlock_different_phases(device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_deadlock_different_phases, (device, False, monkeypatch, num_ctas))
        assert result.exc is None
        assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def ws_default(bar):
        mbarrier.wait(bar.index(0), phase=0)
        mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def ws_1(bar):
        mbarrier.wait(bar.index(0), phase=1)

    @gluon.jit
    def kernel():
        bar = mbarrier.allocate_mbarrier(batch=1)
        mbarrier.init(bar.index(0), count=1)
        mbarrier.arrive(bar.index(0), count=1)
        ttgl.warp_specialize([
            (ws_default, (bar, )),
            (ws_1, (bar, )),
        ], [4], [32])

    kernel[(1, )](num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_deadlock_exempt_when_tma_signals(device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_deadlock_exempt_when_tma_signals, (device, False, monkeypatch, num_ctas))
        assert result.exc is None
        assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def ws_default(input_desc, smem, bar):
        mbarrier.expect(bar.index(0), input_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar.index(0), smem.index(0))
        mbarrier.wait(bar.index(0), phase=0)

    @gluon.jit
    def ws_1(input_desc, smem, bar):
        mbarrier.expect(bar.index(1), input_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar.index(1), smem.index(1))
        mbarrier.wait(bar.index(1), phase=0)

    @gluon.jit
    def kernel(input_desc):
        shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                                               cga_layout=default_cga_layout(ttgl.num_ctas(), 2))
        smem = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK, XBLOCK], shared_layout)
        bar = mbarrier.allocate_mbarrier(batch=2)
        mbarrier.init(bar.index(0), count=1)
        mbarrier.init(bar.index(1), count=1)
        ttgl.warp_specialize([
            (ws_default, (input_desc, smem, bar)),
            (ws_1, (input_desc, smem, bar)),
        ], [4], [32])

    input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=default_cga_layout(num_ctas, 2))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK.value, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_barrier_underflow(device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_barrier_underflow, (device, False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Barrier arrive underflow" in result.driver_stderr_output
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def ws_default(bar):
        mbarrier.arrive(bar.index(1), count=2)
        mbarrier.wait(bar.index(0), phase=0)

    @gluon.jit
    def ws_1(bar):
        mbarrier.wait(bar.index(1), phase=0)

    @gluon.jit
    def kernel():
        bar = mbarrier.allocate_mbarrier(batch=2)
        mbarrier.init(bar.index(0), count=1)
        mbarrier.init(bar.index(1), count=1)
        ttgl.warp_specialize([
            (ws_default, (bar, )),
            (ws_1, (bar, )),
        ], [4], [32])

    kernel[(1, )](num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("WITH_INVALIDATE", [False, True])
def test_barrier_reinit_requires_invalidate(WITH_INVALIDATE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_barrier_reinit_requires_invalidate,
                                (WITH_INVALIDATE, device, False, monkeypatch, num_ctas))
        if WITH_INVALIDATE:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        else:
            assert_expected_cuda_failure(result.exc)
            assert "Barrier re-initialized without prior invalidation" in result.driver_stderr_output
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(WITH_INVALIDATE: ttgl.constexpr):
        bar = mbarrier.allocate_mbarrier(batch=1)
        mbarrier.init(bar.index(0), count=1)
        if WITH_INVALIDATE:
            mbarrier.invalidate(bar.index(0))
        mbarrier.init(bar.index(0), count=1)
        mbarrier.invalidate(bar.index(0))

    kernel[(1, )](WITH_INVALIDATE=WITH_INVALIDATE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("USE_KIND", ["wait", "arrive", "invalidate", "expect"])
def test_barrier_use_without_init(USE_KIND, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_barrier_use_without_init, (USE_KIND, device, False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Barrier used before initialization or after invalidation" in result.driver_stderr_output
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(USE_KIND: ttgl.constexpr):
        bar = mbarrier.allocate_mbarrier(batch=1)
        if USE_KIND == "wait":
            mbarrier.wait(bar.index(0), phase=0)
        elif USE_KIND == "arrive":
            mbarrier.arrive(bar.index(0), count=1)
        elif USE_KIND == "invalidate":
            mbarrier.invalidate(bar.index(0))
        elif USE_KIND == "expect":
            mbarrier.expect(bar.index(0), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)

    kernel[(1, )](USE_KIND=USE_KIND, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("USE_KIND", ["wait", "arrive", "expect"])
def test_barrier_use_after_invalidate(USE_KIND, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_barrier_use_after_invalidate, (USE_KIND, device, False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Barrier used before initialization or after invalidation" in result.driver_stderr_output
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(USE_KIND: ttgl.constexpr):
        bar = mbarrier.allocate_mbarrier(batch=1)
        mbarrier.init(bar.index(0), count=1)
        mbarrier.invalidate(bar.index(0))
        if USE_KIND == "wait":
            mbarrier.wait(bar.index(0), phase=0)
        elif USE_KIND == "arrive":
            mbarrier.arrive(bar.index(0), count=1)
        elif USE_KIND == "expect":
            mbarrier.expect(bar.index(0), XBLOCK * XBLOCK * ttgl.float16.primitive_bitwidth // 8)

    kernel[(1, )](USE_KIND=USE_KIND, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("MISSING_BAR", [True, False])
@pytest.mark.parametrize("OVERLAP", [True, False])
def test_aliasing_shared_visibility_outstanding_write(MISSING_BAR, OVERLAP, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_aliasing_shared_visibility_outstanding_write,
                                (MISSING_BAR, OVERLAP, device, False, monkeypatch, num_ctas))
        if MISSING_BAR and OVERLAP:
            assert result.exc is not None
            assert_expected_cuda_failure(result.exc)
            # The race can be reported from either side depending on timing.
            assert "Buffer being accessed has outstanding" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def writer(alias0: ttgl.constexpr, bar: ttgl.constexpr, OVERLAP: ttgl.constexpr, blocked_layout: ttgl.constexpr,
               blocked_layout_wide: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        if OVERLAP:
            vals = ttgl.full([block_m, XBLOCK * 2], 42.0, ttgl.float16, blocked_layout_wide)
        else:
            vals = ttgl.full([block_m, XBLOCK], 42.0, ttgl.float16, blocked_layout)
        alias0.store(vals)
        mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def reader(alias1: ttgl.constexpr, dummy: ttgl.constexpr, bar: ttgl.constexpr, MISSING_BAR: ttgl.constexpr,
               blocked_layout: ttgl.constexpr):
        if not MISSING_BAR:
            mbarrier.wait(bar.index(0), phase=0)
        val = alias1.load(blocked_layout)
        dummy.store(val)  # keep the load alive

    @gluon.jit
    def kernel(MISSING_BAR: ttgl.constexpr, OVERLAP: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0, 1],
                                                                cga_layout=cga_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        blocked_layout_wide: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[2, XBLOCK], threads_per_warp=[32, 1],
                                                                 warps_per_cta=[4, 1], order=[0,
                                                                                              1], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK * 2], smem_layout)
        smem2 = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], smem_layout)
        bar = mbarrier.allocate_mbarrier(batch=1)
        mbarrier.init(bar.index(0), count=1)
        alias0 = smem if OVERLAP else smem.slice(0, XBLOCK, dim=1)
        alias1 = smem.slice(XBLOCK, XBLOCK, dim=1)

        ttgl.warp_specialize([(writer, (alias0, bar, OVERLAP, blocked_layout, blocked_layout_wide)),
                              (reader, (alias1, smem2, bar, MISSING_BAR, blocked_layout))], [4], [32])

    kernel[(1, )](MISSING_BAR=MISSING_BAR, OVERLAP=OVERLAP, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_aliasing_tensor_visibility_outstanding_read(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_aliasing_tensor_visibility_outstanding_read,
                                (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert result.exc is not None
            assert_expected_cuda_failure(result.exc)
            # outstanding reads or writes depends on the timing of the operations.
            assert "Buffer being accessed has outstanding" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def reader(alias0: ttgl.constexpr, smem: ttgl.constexpr, blocked_layout_read: ttgl.constexpr, bar: ttgl.constexpr):
        val = alias0.load(blocked_layout_read)
        smem.store(val)  # keep the load alive
        mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def writer(alias1: ttgl.constexpr, bar: ttgl.constexpr, FAILURE: ttgl.constexpr,
               blocked_layout_write: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        if not FAILURE:
            mbarrier.wait(bar.index(0), phase=0)
        alias1.store(ttgl.zeros([block_m, XBLOCK // 2], ttgl.float32, blocked_layout_write))

    @gluon.jit
    def kernel(FAILURE: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0, 1],
                                                                cga_layout=cga_layout)
        blocked_layout_read: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                                 warps_per_cta=[4, 1], order=[0,
                                                                                              1], cga_layout=cga_layout)
        blocked_layout_write: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK // 2],
                                                                  threads_per_warp=[32, 1], warps_per_cta=[4, 1],
                                                                  order=[0, 1], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float32, [block_m, XBLOCK], smem_layout)
        tmem_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([XBLOCK, XBLOCK * 2], col_stride=1,
                                                                   cga_layout=cga_layout)
        tmem = blackwell.allocate_tensor_memory(ttgl.float32, [block_m, XBLOCK * 2], tmem_layout)
        bar = mbarrier.allocate_mbarrier(batch=1)
        mbarrier.init(bar.index(0), count=1)
        alias0 = tmem.slice(0, XBLOCK)
        # Second half of the tmem
        alias1 = tmem.slice(XBLOCK // 2, XBLOCK // 2)

        ttgl.warp_specialize([(reader, (alias0, smem, blocked_layout_read, bar)),
                              (writer, (alias1, bar, FAILURE, blocked_layout_write))], [4], [32])

    kernel[(1, )](FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("MISSING_WAIT", [True, False])
@pytest.mark.parametrize("OVERLAP", [True, False])
def test_aliasing_commit_tracking(MISSING_WAIT, OVERLAP, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_aliasing_commit_tracking,
                                (MISSING_WAIT, OVERLAP, device, False, monkeypatch, num_ctas))
        if MISSING_WAIT and OVERLAP:
            assert result.exc is not None
            assert_expected_cuda_failure(result.exc)
            assert "Accessing buffer with pending access. Pending access type: async_copy_global_to_shared" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def producer(input, alias0, bar, MISSING_WAIT: ttgl.constexpr, OVERLAP: ttgl.constexpr,
                 blocked_layout: ttgl.constexpr, blocked_layout_wide: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        layout: ttgl.constexpr = blocked_layout_wide if OVERLAP else blocked_layout
        SIZE_N: ttgl.constexpr = XBLOCK * 2 if OVERLAP else XBLOCK
        offs_m = ttgl.arange(0, block_m, layout=ttgl.SliceLayout(dim=1, parent=layout))[:, None]
        offs_n = ttgl.arange(0, SIZE_N, layout=ttgl.SliceLayout(dim=0, parent=layout))[None, :]
        offs = offs_m * (XBLOCK * 2) + offs_n
        ampere.async_copy.async_copy_global_to_shared(alias0, input + offs)
        ampere.async_copy.commit_group()
        if not MISSING_WAIT:
            ampere.async_copy.wait_group(0)
        mbarrier.arrive(bar.index(0), count=1)

    @gluon.jit
    def consumer(alias1, bar, blocked_layout: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        mbarrier.wait(bar.index(0), phase=0)
        alias1.store(ttgl.zeros([block_m, XBLOCK], ttgl.float32, blocked_layout))

    @gluon.jit
    def kernel(input, MISSING_WAIT: ttgl.constexpr, OVERLAP: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0, 1],
                                                                cga_layout=cga_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        blocked_layout_wide: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[2, XBLOCK], threads_per_warp=[32, 1],
                                                                 warps_per_cta=[4, 1], order=[0,
                                                                                              1], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float32, [block_m, XBLOCK * 2], smem_layout)
        bar = mbarrier.allocate_mbarrier(batch=1)
        mbarrier.init(bar.index(0), count=1)

        alias0 = smem if OVERLAP else smem.slice(0, XBLOCK, dim=1)
        alias1 = smem.slice(XBLOCK, XBLOCK, dim=1)

        ttgl.warp_specialize([(producer,
                               (input, alias0, bar, MISSING_WAIT, OVERLAP, blocked_layout, blocked_layout_wide)),
                              (consumer, (alias1, bar, blocked_layout))], [4], [32])

    input = torch.randn((XBLOCK.value * num_ctas, XBLOCK.value * 2), device=device, dtype=torch.float32)
    kernel[(1, )](input, MISSING_WAIT=MISSING_WAIT, OVERLAP=OVERLAP, num_warps=4, num_ctas=num_ctas)


@gluon.jit
def async_copy_mma_write_after_read_kernel(a_ptr, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                           BLOCK_K: ttgl.constexpr):
    blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[32, 1],
                                                        warps_per_cta=[ttgl.num_warps(), 1], order=[0, 1],
                                                        cga_layout=mma_cga_layout(ttgl.num_ctas(), 0))
    a_smem = ttgl.allocate_shared_memory(
        ttgl.float16,
        [BLOCK_M, BLOCK_K],
        ttgl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], ttgl.float16,
                                               cga_layout=mma_cga_layout(ttgl.num_ctas(), 0)),
    )
    b_smem = ttgl.allocate_shared_memory(
        ttgl.float16,
        [BLOCK_K, BLOCK_N],
        ttgl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], ttgl.float16,
                                               cga_layout=mma_cga_layout(ttgl.num_ctas(), 1)),
    )

    bar = mbarrier.allocate_mbarrier()
    tmem_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([XBLOCK, XBLOCK], col_stride=1,
                                                               cga_layout=mma_cga_layout(ttgl.num_ctas(), 2))
    tmem = allocate_tensor_memory(ttgl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    mbarrier.init(bar, count=1)
    blackwell.tcgen05_mma(a_smem, b_smem, tmem, use_acc=False)
    offs_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, blocked_layout))[:, None]
    offs_k = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, blocked_layout))[None, :]
    offs = offs_m * BLOCK_K + offs_k
    ampere.async_copy.async_copy_global_to_shared(a_smem, a_ptr + offs)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
def test_mma_read_async_copy_write(run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_mma_read_async_copy_write, (False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    BLOCK_M = mma_block_m(num_ctas)
    BLOCK_N = mma_block_n(num_ctas)
    BLOCK_K = XBLOCK.value
    A = torch.randn((BLOCK_M, BLOCK_K), device="cuda", dtype=torch.float16)
    async_copy_mma_write_after_read_kernel[(1, )](A, BLOCK_M, BLOCK_N, BLOCK_K, num_ctas=num_ctas)


@gluon.jit
def load_local_alloc_mma_write_after_read_kernel(a_ptr, K, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                                                 BLOCK_K: ttgl.constexpr):
    blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[32, 1],
                                                        warps_per_cta=[ttgl.num_warps(), 1], order=[0, 1],
                                                        cga_layout=mma_cga_layout(ttgl.num_ctas(), 0))
    a_smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], ttgl.float16,
                                                                           cga_layout=mma_cga_layout(
                                                                               ttgl.num_ctas(), 0))
    b_smem = ttgl.allocate_shared_memory(
        ttgl.float16,
        [BLOCK_K, BLOCK_N],
        ttgl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], ttgl.float16,
                                               cga_layout=mma_cga_layout(ttgl.num_ctas(), 1)),
    )

    bar = mbarrier.allocate_mbarrier()
    tmem_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([XBLOCK, XBLOCK], col_stride=1,
                                                               cga_layout=mma_cga_layout(ttgl.num_ctas(), 2))
    tmem = allocate_tensor_memory(ttgl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    mbarrier.init(bar, count=1)

    offs_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, blocked_layout))[:, None]
    offs_k = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, blocked_layout))[None, :]

    use_acc = False
    for k in range(0, K, BLOCK_K):
        a_value = ttgl.load(a_ptr + offs_m * K + offs_k + k)

        a_smem = ttgl.allocate_shared_memory(ttgl.float16, [BLOCK_M, BLOCK_K], a_smem_layout, a_value)
        blackwell.tcgen05_mma(a_smem, b_smem, tmem, use_acc=use_acc)
        use_acc = True
    blackwell.tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
def test_mma_read_local_alloc_write(run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_mma_read_local_alloc_write, (False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    K = 512
    BLOCK_M = mma_block_m(num_ctas)
    BLOCK_N = mma_block_n(num_ctas)
    BLOCK_K = 64
    A = torch.randn((BLOCK_M, K), device="cuda", dtype=torch.float16)
    load_local_alloc_mma_write_after_read_kernel[(1, )](A, K, BLOCK_M, BLOCK_N, BLOCK_K, num_ctas=num_ctas)
