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
from triton.experimental.gluon.language.nvidia import rubin
from triton.experimental.gluon.language.nvidia.blackwell import allocate_tensor_memory, clc, mbarrier, tma
from triton._internal_testing import is_compile_warmup, is_cuda, is_rubin, run_in_process

pytestmark = [pytest.mark.enable_warmup(min_capability=9), pytest.mark.usefixtures("process_pool")]


@pytest.fixture(autouse=True)
def isolated_consan_knobs():
    with knobs.compilation.scope(), knobs.runtime.scope():
        yield


@pytest.fixture
def run_wrapper(request):
    # Use DISABLE_SUBPROCESS to run the tests in the main process
    # (useful for debugging but assert in any test will make all the tests fail)
    return not request.config.getoption("--warmup-only", default=False) and not os.environ.get("DISABLE_SUBPROCESS")


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
    ampere.async_copy.async_load(smem, input + offs)
    ampere.async_copy.commit_group()

    ampere.async_copy.async_load(smem, input + offs)
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
def test_cache_miss_knob(device, monkeypatch, num_ctas, run_wrapper):
    # First run without consan
    run_in_process(run_failing_kernel, (device, False, "knob", num_ctas))

    # Then run with consan and assert that if fails
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    result = run_in_process(run_failing_kernel, (device, True, "knob", num_ctas))
    if run_wrapper:
        assert result.exc is not None
        assert any(msg in str(result.exc) for msg in ["device-side assert", "unspecified launch failure"])


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_cache_miss_env(device, monkeypatch, num_ctas, run_wrapper):
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "")
    # First run without consan
    run_in_process(run_failing_kernel, (device, False, "env", num_ctas))

    # Then run with consan and assert that if fails
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    result = run_in_process(run_failing_kernel, (device, True, "env", num_ctas))
    if run_wrapper:
        assert result.exc is not None
        assert any(msg in str(result.exc) for msg in ["device-side assert", "unspecified launch failure"])


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.disable_warmup(reason="compiles into an intentionally isolated temporary cache")
def test_consan_uses_profile_scratch(device, fresh_knobs, num_ctas):
    with knobs.cache.scope(), knobs.runtime.scope():
        knobs.cache.dir = tempfile.mkdtemp(prefix="triton-cache-")
        fresh_knobs.compilation.instrumentation_mode = "consan"
        input = torch.randn((XBLOCK, XBLOCK), device=device, dtype=torch.float16)
        compiled = failing_kernel.warmup(input, grid=(1, ), num_ctas=num_ctas)
        assert compiled.metadata.profile_scratch_size > 0
        assert compiled.metadata.global_scratch_size == 0


@gluon.jit(noinline=True)
def _consan_noinline_convert_layout(input, output):
    src_layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])
    dst_layout: ttgl.constexpr = ttgl.SliceLayout(1, ttgl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0]))
    src_offsets = ttgl.arange(0, 128, layout=src_layout)
    dst_offsets = ttgl.arange(0, 128, layout=dst_layout)
    values = ttgl.load(input + src_offsets)
    ttgl.store(output + dst_offsets, ttgl.convert_layout(values, dst_layout))


@gluon.jit(noinline=True)
def _consan_noinline_forward_convert_layout(input, output):
    _consan_noinline_convert_layout(input, output)


@gluon.jit
def _consan_noinline_convert_layout_kernel(input, output, sentinel_output):
    layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])
    shared_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [0])
    offsets = ttgl.arange(0, 128, layout=layout)
    sentinel = offsets + 4096
    caller_allocation = ttgl.allocate_shared_memory(ttgl.int32, [128], shared_layout, sentinel)

    _consan_noinline_forward_convert_layout(input, output)
    ttgl.store(sentinel_output + offsets, caller_allocation.load(layout))


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires Hopper or newer")
def test_consan_noinline_convert_layout_scratch(device, fresh_knobs):
    fresh_knobs.compilation.instrumentation_mode = "consan"
    values = torch.arange(128, device=device, dtype=torch.int32)
    output = torch.empty_like(values)
    sentinel_output = torch.empty_like(values)
    compiled = _consan_noinline_convert_layout_kernel[(1, )](values, output, sentinel_output, num_warps=4)
    if is_compile_warmup():
        return
    assert compiled.metadata.shared >= 2 * values.numel() * values.element_size()
    assert compiled.asm["ttgir"].count("noinline = true") >= 2
    torch.testing.assert_close(output, values)
    torch.testing.assert_close(sentinel_output, values + 4096)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("MEMORY_KIND", ["shared", "tensor"])
def test_consan_initializes_allocations_with_nan(MEMORY_KIND, device, num_ctas):
    knobs.compilation.instrumentation_mode = "consan"

    @gluon.jit
    def kernel(output, MEMORY_KIND: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        reg_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                        warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        offs_m = ttgl.arange(0, block_m, ttgl.SliceLayout(1, reg_layout))[:, None]
        offs_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, reg_layout))[None, :]
        offs = offs_m * XBLOCK + offs_n
        if MEMORY_KIND == "shared":
            memory_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2,
                                                                   cga_layout=cga_layout)
            alloc = ttgl.allocate_shared_memory(ttgl.float32, [block_m, XBLOCK], memory_layout)
            value = alloc.load(reg_layout)
        else:
            memory_layout: ttgl.constexpr = blackwell.TensorMemoryLayout((XBLOCK, XBLOCK), col_stride=1,
                                                                         cga_layout=cga_layout)
            alloc = blackwell.allocate_tensor_memory(ttgl.float32, [block_m, XBLOCK], memory_layout)
            value = alloc.load(reg_layout)
        ttgl.store(output + offs, value)

    output = torch.empty((XBLOCK.value * num_ctas, XBLOCK.value), device=device, dtype=torch.float32)
    kernel[(1, )](output, MEMORY_KIND=MEMORY_KIND, num_warps=4, num_ctas=num_ctas)
    assert torch.isnan(output).all()


@pytest.mark.skipif(not is_rubin(), reason="Requires Rubin")
def test_mbarrier_arrive_multicast_completion(device, monkeypatch):
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(out):
        bar = rubin.mbarrier.allocate_mbarrier()
        rubin.mbarrier.init(bar, count=2)
        rubin.mbarrier.arrive(bar, multicast_cta=1)
        rubin.mbarrier.wait(bar, 0)
        rubin.mbarrier.invalidate(bar)
        ttgl.store(out + ttgl.program_id(0), ttgl.program_id(0))

    output = torch.empty(2, device=device, dtype=torch.int32)
    kernel[(2, )](output, num_ctas=2, num_warps=4)
    torch.testing.assert_close(output, torch.arange(2, device=device, dtype=torch.int32))


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
        tma.async_load(input_desc, [0, 0], bar, smem)
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
@pytest.mark.parametrize("BLOCK", [64, 128], ids=["redundant-threads", "full-cta"])
@pytest.mark.parametrize("EXPECT_DELTA", [0, 4], ids=["match", "mismatch"])
def test_async_shared_store_expect_bytes(BLOCK, EXPECT_DELTA, device, run_wrapper, monkeypatch, num_ctas):
    if num_ctas == 1:
        pytest.skip("st.async.shared requires at least 2 CTAs")
    if run_wrapper and EXPECT_DELTA:
        result = run_in_process(test_async_shared_store_expect_bytes,
                                (BLOCK, EXPECT_DELTA, device, False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Deadlock detected" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(out, EXPECT_DELTA: ttgl.constexpr, BLOCK: ttgl.constexpr):
        cga_layout: ttgl.constexpr = multicast_cga_layout(ttgl.num_ctas(), 1)
        layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0], cga_layout=cga_layout)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[0], cga_layout=cga_layout)
        offsets = ttgl.arange(0, BLOCK, layout=layout)
        values = offsets.to(ttgl.int32)
        smem = ttgl.allocate_shared_memory(ttgl.int32, [BLOCK], smem_layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        mbarrier.expect(bar, smem.nbytes_per_cta + EXPECT_DELTA)
        hopper.async_store(smem, values, bar)
        mbarrier.wait(bar, 0, deps=[smem])
        result = smem.load(layout)
        mbarrier.invalidate(bar)
        ttgl.store(out + offsets, result)

    output = torch.empty((BLOCK, ), device=device, dtype=torch.int32)
    kernel[(1, )](output, EXPECT_DELTA=EXPECT_DELTA, BLOCK=BLOCK, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("WAIT", [True, False], ids=["wait", "no-wait"])
def test_async_shared_store_completion(WAIT, device, run_wrapper, monkeypatch, num_ctas):
    if num_ctas == 1:
        pytest.skip("st.async.shared requires at least 2 CTAs")
    if run_wrapper and not WAIT:
        result = run_in_process(test_async_shared_store_completion, (WAIT, device, False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(out, WAIT: ttgl.constexpr):
        cga_layout: ttgl.constexpr = multicast_cga_layout(ttgl.num_ctas(), 1)
        layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0], cga_layout=cga_layout)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[0], cga_layout=cga_layout)
        offsets = ttgl.arange(0, XBLOCK, layout=layout)
        smem = ttgl.allocate_shared_memory(ttgl.int32, [XBLOCK], smem_layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        mbarrier.expect(bar, smem.nbytes_per_cta)
        hopper.async_store(smem, offsets, bar)
        if WAIT:
            mbarrier.wait(bar, 0, deps=[smem])
        result = smem.load(layout)
        if not WAIT:
            mbarrier.wait(bar, 0, deps=[smem])
        mbarrier.invalidate(bar)
        ttgl.store(out + offsets, result)

    output = torch.empty((XBLOCK.value, ), device=device, dtype=torch.int32)
    kernel[(1, )](output, WAIT=WAIT, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("EXPECT_DELTA", [0, 4], ids=["match", "mismatch"])
def test_async_shared_store_split_recipients(EXPECT_DELTA, device, run_wrapper, monkeypatch, num_ctas):
    if num_ctas == 1:
        pytest.skip("st.async.shared requires at least 2 CTAs")
    if run_wrapper and EXPECT_DELTA:
        result = run_in_process(test_async_shared_store_split_recipients,
                                (EXPECT_DELTA, device, False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Deadlock detected" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(out, EXPECT_DELTA: ttgl.constexpr):
        num_ctas: ttgl.constexpr = ttgl.num_ctas()
        source_cga: ttgl.constexpr = multicast_cga_layout(num_ctas, 1)
        target_cga: ttgl.constexpr = default_cga_layout(num_ctas, 1)
        source_layout: ttgl.constexpr = ttgl.BlockedLayout([num_ctas], [32], [4], [0], cga_layout=source_cga)
        target_layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0], cga_layout=target_cga)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[0], cga_layout=target_cga)
        offsets = ttgl.arange(0, XBLOCK * num_ctas, layout=source_layout)
        smem = ttgl.allocate_shared_memory(ttgl.int32, [XBLOCK * num_ctas], smem_layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        mbarrier.expect(bar, smem.nbytes_per_cta + EXPECT_DELTA)
        cta_rank = ttgl.inline_asm_elementwise("mov.u32 $0, %cluster_ctarank;", "=r", [], dtype=ttgl.int32,
                                               is_pure=True, pack=1)
        if cta_rank == 0:
            hopper.async_store(smem, offsets, bar)
        mbarrier.wait(bar, 0, deps=[smem])
        result = smem.load(target_layout)
        mbarrier.invalidate(bar)
        output_offsets = ttgl.arange(0, XBLOCK * num_ctas, layout=target_layout)
        ttgl.store(out + output_offsets, result)

    output = torch.empty((XBLOCK.value * num_ctas, ), device=device, dtype=torch.int32)
    kernel[(1, )](output, EXPECT_DELTA=EXPECT_DELTA, num_warps=4, num_ctas=num_ctas)
    torch.testing.assert_close(output, torch.arange(XBLOCK.value * num_ctas, device=device, dtype=torch.int32))


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FENCE", [False, True], ids=["missing", "present"])
def test_async_shared_store_proxy_handoff(FENCE, device, run_wrapper, monkeypatch, num_ctas):
    if num_ctas == 1:
        pytest.skip("st.async.shared requires at least 2 CTAs")
    if run_wrapper and not FENCE:
        result = run_in_process(test_async_shared_store_proxy_handoff, (FENCE, device, False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Async shared-memory access is missing fence_async_shared" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def producer(smem: ttgl.constexpr, bar: ttgl.constexpr, layout: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        mbarrier.expect(bar, smem.nbytes_per_cta)
        hopper.async_store(smem, ttgl.full([block_m, XBLOCK], 42.0, ttgl.float16, layout), bar)

    @gluon.jit
    def consumer(output_desc, smem: ttgl.constexpr, bar: ttgl.constexpr, FENCE: ttgl.constexpr):
        mbarrier.wait(bar, phase=0, deps=[smem])
        if FENCE:
            hopper.fence_async_shared()
        tma.async_store(output_desc, [0, 0], smem)
        tma.store_wait(0)

    @gluon.jit
    def kernel(output_desc, FENCE: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                                             cga_layout=cga_layout)
        layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                    warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], smem_layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        ttgl.warp_specialize([(producer, (smem, bar, layout)), (consumer, (output_desc, smem, bar, FENCE))], [4], [32])

    block_m = XBLOCK.value * num_ctas
    output = torch.empty((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=default_cga_layout(num_ctas, 2))
    output_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(output, [block_m, XBLOCK.value], shared_layout)
    kernel[(1, )](output_desc, FENCE=FENCE, num_warps=4, num_ctas=num_ctas)
    torch.testing.assert_close(output, torch.full_like(output, 42.0))


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_async_tma_multicast_kernel(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if num_ctas == 1:
        pytest.skip("Need at least 2 CTAs for multicast in this test")
    if run_wrapper:
        result = run_in_process(test_async_tma_multicast_kernel, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert any(msg in result.driver_stderr_output for msg in [
                "Buffer being accessed has outstanding writes",
                "Buffer being accessed has outstanding reads",
            ])
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
        tma.async_load(input_desc, [0, 0], bar, smem, multicast=True)
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
def test_collapsed_wait_does_not_publish_peer_cta(device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_collapsed_wait_does_not_publish_peer_cta, (device, False, monkeypatch))
        assert_expected_cuda_failure(result.exc)
        assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(a_desc, b_desc):
        blocked_a: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1], ((1, 0), ))
        smem_a = ttgl.allocate_shared_memory(ttgl.float16, [256, 128], a_desc.layout)
        smem_b = ttgl.allocate_shared_memory(
            ttgl.float16, [128, 128],
            ttgl.NVMMASharedLayout.get_default_for([128, 128], ttgl.float16, cga_layout=((0, 1), )))
        tma_bar = mbarrier.allocate_mbarrier(two_ctas=True)
        mbarrier.init(tma_bar, count=1)
        mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
        tma.async_load(a_desc, [0, 0], tma_bar, smem_a)
        tma.async_load(b_desc, [0, 0], tma_bar, smem_b)
        mbarrier.wait(tma_bar, 0, deps=[smem_a, smem_b])
        val = smem_a.load(blocked_a)
        smem_a.store(val)

        acc_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([128, 128], col_stride=1, cga_layout=((1, 0), ),
                                                                  two_ctas=True)
        acc = blackwell.allocate_tensor_memory(ttgl.float32, [256, 128], acc_layout)
        blackwell.tcgen05_mma(smem_a, smem_b, acc, use_acc=False)

    a = torch.randn((256, 128), device=device, dtype=torch.float16)
    b = torch.randn((128, 128), device=device, dtype=torch.float16)
    a_layout = ttgl.NVMMASharedLayout.get_default_for([256, 128], ttgl.float16, cga_layout=((1, 0), ))
    b_layout = ttgl.NVMMASharedLayout.get_default_for([128, 128], ttgl.float16, cga_layout=((0, 1), ))
    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(a, [256, 128], a_layout)
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(b, [128, 128], b_layout)
    kernel[(1, )](a_desc, b_desc, num_warps=4, num_ctas=2)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_clc_result_visibility(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_clc_result_visibility, (FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert any(msg in result.driver_stderr_output for msg in [
                "Buffer being accessed has outstanding writes",
                "Buffer being accessed has outstanding reads",
            ])
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
        mbarrier.expect(clc_bar, 16, from_cta=0x0)
        mbarrier.wait(clc_bar, 0, pred=(not FAILURE))
        response = clc.load_result(clc_result)
        mbarrier.wait(clc_bar, 0, pred=FAILURE)
        mbarrier.invalidate(clc_bar)

        ttgl.store(out + ttgl.program_id(0), response.is_canceled())

    output = torch.empty((1, ), device=device, dtype=torch.bool)
    kernel[(1, )](output, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell")
def test_clc_double_try_cancel_result_overwrite(device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_clc_double_try_cancel_result_overwrite, (device, False, monkeypatch))
        assert result.exc is None
        assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel():
        cga_layout: ttgl.constexpr = multicast_cga_layout(ttgl.num_ctas(), 1)
        layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[0], cga_layout=cga_layout)
        result = ttgl.allocate_shared_memory(ttgl.int64, [2], layout)
        bars = mbarrier.allocate_mbarrier(batch=2)
        mbarrier.init(bars.index(0), count=1)
        mbarrier.init(bars.index(1), count=1)

        mbarrier.expect(bars.index(0), 16, from_cta=0x0)
        clc.try_cancel(result, bars.index(0))
        mbarrier.expect(bars.index(1), 16, from_cta=0x0)
        clc.try_cancel(result, bars.index(1))

        mbarrier.wait(bars.index(0), 0)
        mbarrier.wait(bars.index(1), 0)

    kernel[(1, )](num_warps=4, num_ctas=2)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell")
def test_clc_result_reuse_after_cluster_barrier(device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_clc_result_reuse_after_cluster_barrier, (device, False, monkeypatch))
        assert result.exc is None
        assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(out):
        cga_layout: ttgl.constexpr = [[0]]
        layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[0], cga_layout=cga_layout)
        clc_result = ttgl.allocate_shared_memory(ttgl.int64, [2], layout)
        clc_bar = mbarrier.allocate_mbarrier()
        mbarrier.init(clc_bar, count=1)

        mbarrier.expect(clc_bar, 16, from_cta=0x0)
        clc.try_cancel(clc_result, clc_bar)
        mbarrier.wait(clc_bar, 0)
        first = clc.load_result(clc_result)
        ttgl.barrier(cluster=True)
        # A CTA fence must not hide the need for a cluster fence before the
        # next multi-CTA CLC request.
        hopper.fence_async_shared()

        mbarrier.expect(clc_bar, 16, from_cta=0x0)
        clc.try_cancel(clc_result, clc_bar)
        mbarrier.wait(clc_bar, 1)
        second = clc.load_result(clc_result)
        ttgl.store(out + ttgl.program_id(0), first.is_canceled() | second.is_canceled())

    output = torch.empty((1, ), device=device, dtype=torch.bool)
    kernel[(1, )](output, num_warps=4, num_ctas=2)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell")
@pytest.mark.parametrize("SYNCHRONIZED", [False, True], ids=["local-expect", "from-cta0-expect"])
def test_clc_slot_reuse_from_cta(SYNCHRONIZED, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_clc_slot_reuse_from_cta, (SYNCHRONIZED, device, False, monkeypatch))
        if SYNCHRONIZED:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        else:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def consumer(slot, sink, consumed, layout: ttgl.constexpr):
        value = slot.load(layout)
        sink.store(value)
        mbarrier.arrive(consumed, count=1)

    @gluon.jit
    def clc_partition(slot, result, clc_bar, consumed, SYNCHRONIZED: ttgl.constexpr, layout: ttgl.constexpr):
        mbarrier.wait(consumed, 0, deps=[slot])
        if SYNCHRONIZED:
            mbarrier.expect(clc_bar, 16, from_cta=0x0)
        else:
            mbarrier.expect(clc_bar, 16)
        clc.try_cancel(result, clc_bar)
        mbarrier.wait(clc_bar, 0)
        clc.load_result(result)
        slot.store(ttgl.full([1], 1, ttgl.int64, layout=layout))

    @gluon.jit
    def kernel(SYNCHRONIZED: ttgl.constexpr):
        cga_layout: ttgl.constexpr = multicast_cga_layout(ttgl.num_ctas(), 1)
        shared_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[0], cga_layout=cga_layout)
        layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0], cga_layout=cga_layout)
        slot = ttgl.allocate_shared_memory(ttgl.int64, [1], shared_layout)
        sink = ttgl.allocate_shared_memory(ttgl.int64, [1], shared_layout)
        result = ttgl.allocate_shared_memory(ttgl.int64, [2], shared_layout)
        clc_bar = mbarrier.allocate_mbarrier()
        consumed = mbarrier.allocate_mbarrier(two_ctas=True)
        mbarrier.init(clc_bar, count=1)
        mbarrier.init(consumed, count=1)
        slot.store(ttgl.full([1], 0, ttgl.int64, layout=layout))
        ttgl.warp_specialize([
            (consumer, (slot, sink, consumed, layout)),
            (clc_partition, (slot, result, clc_bar, consumed, SYNCHRONIZED, layout)),
        ], [4], [32])

    kernel[(1, )](SYNCHRONIZED=SYNCHRONIZED, num_warps=4, num_ctas=2)


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
        val = ttgl.full([XBLOCK, XBLOCK], 0, ttgl.float16, blocked_layout)
        for phase in ttgl.static_range(2):
            mbarrier.expect(bar, input_desc.nbytes_per_cta)
            ttgl.barrier(cluster=True)
            # A CTA fence after the cluster handoff cannot cover peer-CTA reads.
            hopper.fence_async_shared()
            tma.async_load(input_desc, [0, 0], bar, smem, multicast=True)
            mbarrier.wait(bar, phase % 2, deps=[smem])
            val += smem.load(blocked_layout)
            ttgl.barrier(cluster=True)
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
        tma.async_load(input_desc, [0, 0], bar, smem, multicast=True)
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
def test_cluster_barrier_does_not_publish_later_read(device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_cluster_barrier_does_not_publish_later_read, (device, False, monkeypatch))
        assert_expected_cuda_failure(result.exc)
        assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input_desc):
        cga_layout: ttgl.constexpr = multicast_cga_layout(ttgl.num_ctas(), 2)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], input_desc.layout)
        sink = ttgl.allocate_shared_memory(ttgl.float16, [2, XBLOCK, XBLOCK], input_desc.layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)

        ttgl.barrier(cluster=True)
        val = smem.load(blocked_layout)
        # Keep the post-barrier read live long enough to exercise delayed publication from another CTA.
        for i in ttgl.static_range(2):
            sink.index(i).store(val)

        mbarrier.expect(bar, input_desc.nbytes_per_cta)
        tma.async_load(input_desc, [0, 0], bar, smem, multicast=True)
        mbarrier.wait(bar, 0, deps=[smem])

    input = torch.randn((XBLOCK.value, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=multicast_cga_layout(4, 2))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK.value, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, num_warps=4, num_ctas=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
@pytest.mark.parametrize("PUBLISHED", [True, False], ids=["published-reader", "remote-reader"])
def test_remote_shared_load_reader_visibility(PUBLISHED, FAILURE, device, run_wrapper, monkeypatch):
    if FAILURE and run_wrapper:
        result = run_in_process(test_remote_shared_load_reader_visibility,
                                (PUBLISHED, FAILURE, device, False, monkeypatch))
        assert_expected_cuda_failure(result.exc)
        assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(out, PUBLISHED: ttgl.constexpr, FAILURE: ttgl.constexpr):
        alloc_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0], cga_layout=((1, 0), ))
        tile_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0], cga_layout=((0, 0), ))
        shared_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0], cga_layout=((1, 0), ))
        parent = ttgl.allocate_shared_memory(ttgl.int32, [4, 32], shared_layout, value=ttgl.full([4, 32], 0, ttgl.int32,
                                                                                                 alloc_layout))
        smem = parent.slice(2, 2, dim=0)
        ttgl.barrier(cluster=True)

        cta = ttgl.inline_asm_elementwise("mov.u32 $0, %cluster_ctarank;", "=r", [], dtype=ttgl.int32, is_pure=True,
                                          pack=1)
        if cta == 0:
            ttgl.store(out + cta, ttgl.sum(ttgl.sum(smem.load(tile_layout), axis=1), axis=0))
        if PUBLISHED:
            ttgl.barrier(cluster=True)
            if cta == 1:
                ttgl.store(out + cta, ttgl.sum(ttgl.sum(smem.load(tile_layout), axis=1), axis=0))
        if FAILURE:
            hopper.cluster.barrier(relaxed=True)
        else:
            ttgl.barrier(cluster=True)
        writer: ttgl.constexpr = 0 if PUBLISHED else 1
        if cta == writer:
            smem.store(ttgl.full([2, 32], 1, ttgl.int32, tile_layout))
        ttgl.barrier(cluster=True)

    out = torch.zeros(2, device=device, dtype=torch.int32)
    kernel[(1, )](out, PUBLISHED=PUBLISHED, FAILURE=FAILURE, num_warps=4, num_ctas=2)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("FINISHED", [True, False])
def test_cluster_barrier_publishes_only_observed_tensor_reads(FINISHED, device, run_wrapper, monkeypatch):
    if not FINISHED and run_wrapper:
        result = run_in_process(test_cluster_barrier_publishes_only_observed_tensor_reads,
                                (FINISHED, device, False, monkeypatch))
        assert_expected_cuda_failure(result.exc)
        assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input_desc, out, FINISHED: ttgl.constexpr):
        cga_layout: ttgl.constexpr = ((0, 0), )
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0], cga_layout=cga_layout)
        shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(128, 32, rank=2, cga_layout=cga_layout)
        tensor_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([128, 128], col_stride=1, cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.int32, [128, 128], shared_layout)
        tensor = blackwell.allocate_tensor_memory(ttgl.int32, [128, 128], tensor_layout)
        tensor_barrier = mbarrier.allocate_mbarrier()
        tma_barrier = mbarrier.allocate_mbarrier()
        mbarrier.init(tensor_barrier, count=1)
        mbarrier.init(tma_barrier, count=1)

        if not FINISHED:
            ttgl.store(out, ttgl.sum(ttgl.sum(smem.load(blocked_layout), axis=1), axis=0))
            hopper.fence_async_shared()
        blackwell.tcgen05_copy(smem, tensor)
        if FINISHED:
            blackwell.tcgen05_commit(tensor_barrier)
            mbarrier.wait(tensor_barrier, phase=0, deps=[smem])

        ttgl.barrier(cluster=True)
        hopper.fence_async_shared(cluster=True)
        mbarrier.expect(tma_barrier, input_desc.nbytes_per_cta)
        tma.async_load(input_desc, [0, 0], tma_barrier, smem, multicast=True)
        mbarrier.wait(tma_barrier, phase=0, deps=[smem])
        if not FINISHED:
            blackwell.tcgen05_commit(tensor_barrier)
            mbarrier.wait(tensor_barrier, phase=0, deps=[smem])
        mbarrier.invalidate(tma_barrier)
        mbarrier.invalidate(tensor_barrier)

    values = torch.zeros((128, 128), device=device, dtype=torch.int32)
    out = torch.empty(1, device=device, dtype=torch.int32)
    shared_layout = ttgl.NVMMASharedLayout(128, 32, rank=2, cga_layout=((0, 0), ))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(values, [128, 128], shared_layout)
    kernel[(1, )](input_desc, out, FINISHED=FINISHED, num_warps=4, num_ctas=2)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize(
    "OP,FAILURE",
    [
        pytest.param("gather", True, id="gather-missing-barrier"),
        pytest.param("gather", False, id="gather-synchronized"),
        pytest.param("scatter", False, id="scatter-synchronized"),
        pytest.param("atomic", False, id="atomic-synchronized"),
    ],
)
def test_local_indexed_cross_cta_visibility(OP, FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_local_indexed_cross_cta_visibility, (OP, FAILURE, device, False, monkeypatch))
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
    def kernel(inp, out, layout: ttgl.constexpr, OP: ttgl.constexpr, FAILURE: ttgl.constexpr):
        shared_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[1, 0], cga_layout=[[0, 1]])
        rows = ttgl.arange(0, 2, layout=ttgl.SliceLayout(1, layout))
        cols = ttgl.arange(0, 32, layout=ttgl.SliceLayout(0, layout))
        offsets = rows[:, None] * 32 + cols[None, :]
        values = ttgl.load(inp + offsets)
        smem = ttgl.allocate_shared_memory(ttgl.int32, [2, 32], shared_layout)
        peer_cols = (cols ^ 16)[None, :] + rows[:, None] * 0
        # Finish ConSan's untracked poison initialization before peer DSM access.
        ttgl.barrier(cluster=True)

        if FAILURE:
            result = smem.gather(peer_cols, axis=1)
            ttgl.store(out + offsets, result)
            # Order every peer read before the stores without publishing it.
            hopper.cluster.barrier(relaxed=True)
            smem.store(values)
        else:
            smem.store(values)
            ttgl.barrier(cluster=True)
            if OP == "gather":
                result = smem.gather(peer_cols, axis=1)
                # Order peer DSM reads and keep allocations alive until they finish.
                ttgl.barrier(cluster=True)
            elif OP == "scatter":
                smem.scatter(values, peer_cols, axis=1)
                ttgl.barrier(cluster=True)
                result = smem.load(layout)
            else:
                smem.atomic_scatter_add(values, peer_cols, axis=1)
                ttgl.barrier(cluster=True)
                result = smem.load(layout)
            ttgl.store(out + offsets, result)

    inp = torch.arange(64, dtype=torch.int32, device=device).reshape(2, 32)
    out = torch.empty_like(inp)
    layout = ttgl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0], cga_layout=[[0, 1]])
    kernel[(1, )](inp, out, layout, OP=OP, FAILURE=FAILURE, num_warps=4, num_ctas=2)
    if not FAILURE:
        peer = inp.reshape(2, 2, 16).flip(1).reshape(2, 32)
        expected = inp + peer if OP == "atomic" else peer
        torch.testing.assert_close(out, expected)


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
        tma.async_load(a_desc, [0, 0], bar, a_smem)
        tma.async_load(b_desc, [0, 0], bar, b_smem)
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
        tma.async_load(input_desc, [0, 0], bar, smem)
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
        tma.async_load(input_desc, [0, 0], bar.index(0), smem.index(0))
        tma.async_load(input_desc, [0, 0], bar.index(1), smem.index(1))

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

        hopper.fence_async_shared()
        tma.async_store(input_desc, [0, 0], smem.index(0))
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
        tma.async_load(input_desc, [0, 0], bar.index(0), smem.index(0))
        mbarrier.expect(bar.index(1), input_desc.nbytes_per_cta)
        tma.async_load(input_desc, [0, 0], bar.index(1), smem.index(1))

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


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_tma_wait_tracks_only_requested_phase(FAILURE, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_tma_wait_tracks_only_requested_phase,
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
    def producer(input_desc, smem, bar):
        mbarrier.arrive(bar.index(0), count=1)
        mbarrier.wait(bar.index(0), phase=0)
        mbarrier.expect(bar.index(0), input_desc.nbytes_per_cta)
        tma.async_load(input_desc, [0, 0], bar.index(0), smem)
        mbarrier.arrive(bar.index(1), count=1)
        mbarrier.wait(bar.index(0), phase=1, deps=[smem])

    @gluon.jit
    def consumer(smem, bar, out, FAILURE: ttgl.constexpr, blocked_layout: ttgl.constexpr):
        mbarrier.arrive(bar.index(0), count=1)
        mbarrier.wait(bar.index(1), phase=0)
        mbarrier.wait(bar.index(0), phase=0)

        if not FAILURE:
            mbarrier.arrive(bar.index(0), count=1)
            mbarrier.wait(bar.index(0), phase=1, deps=[smem])

        val = smem.load(blocked_layout)
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        out_m = ttgl.arange(0, block_m, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, blocked_layout))[None, :]
        ttgl.store(out + out_m * XBLOCK + out_n, val)

        if FAILURE:
            mbarrier.arrive(bar.index(0), count=1)
            mbarrier.wait(bar.index(0), phase=1, deps=[smem])

    @gluon.jit
    def kernel(input_desc, out, FAILURE: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], input_desc.layout)
        bar = mbarrier.allocate_mbarrier(batch=2)
        mbarrier.init(bar.index(0), count=2)
        mbarrier.init(bar.index(1), count=1)
        ttgl.warp_specialize([
            (producer, (input_desc, smem, bar)),
            (consumer, (smem, bar, out, FAILURE, blocked_layout)),
        ], [4], [32])

    block_m = XBLOCK.value * num_ctas
    input = torch.randn((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    output = torch.empty_like(input)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=default_cga_layout(num_ctas, 2))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [block_m, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, output, FAILURE=FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("WAIT_LATEST", [True, False])
def test_tma_wait_does_not_publish_overwritten_row(WAIT_LATEST, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_tma_wait_does_not_publish_overwritten_row,
                                (WAIT_LATEST, device, False, monkeypatch, num_ctas))
        if WAIT_LATEST:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        else:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input_desc, out, WAIT_LATEST: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], input_desc.layout)
        bar = mbarrier.allocate_mbarrier(batch=2)
        mbarrier.init(bar.index(0), count=1)
        mbarrier.init(bar.index(1), count=1)

        mbarrier.expect(bar.index(0), input_desc.nbytes_per_cta)
        tma.async_load(input_desc, [0, 0], bar.index(0), smem)
        mbarrier.wait(bar.index(0), 0)
        mbarrier.expect(bar.index(1), input_desc.nbytes_per_cta)
        tma.async_load(input_desc, [0, 0], bar.index(1), smem)

        if WAIT_LATEST:
            mbarrier.wait(bar.index(1), 0)
        else:
            mbarrier.wait(bar.index(0), 0)

        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        val = smem.load(blocked_layout)
        out_m = ttgl.arange(0, block_m, ttgl.SliceLayout(1, blocked_layout))[:, None]
        out_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, blocked_layout))[None, :]
        ttgl.store(out + out_m * XBLOCK + out_n, val)

        if not WAIT_LATEST:
            mbarrier.wait(bar.index(1), 0)

        mbarrier.invalidate(bar.index(0))
        mbarrier.invalidate(bar.index(1))

    block_m = XBLOCK.value * num_ctas
    input = torch.randn((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    output = torch.empty_like(input)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=default_cga_layout(num_ctas, 2))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [block_m, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, output, WAIT_LATEST=WAIT_LATEST, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("OP", ["load", "store"])
@pytest.mark.parametrize("SYNCHRONIZED", [True, False])
def test_tma_overlapping_operations(OP, SYNCHRONIZED, device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_tma_overlapping_operations,
                                (OP, SYNCHRONIZED, device, False, monkeypatch, num_ctas))
        if SYNCHRONIZED:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        else:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input_desc, output_desc, OP: ttgl.constexpr, SYNCHRONIZED: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        smem = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], input_desc.layout)
        bars = mbarrier.allocate_mbarrier(batch=2)
        mbarrier.init(bars.index(0), count=1)
        if OP == "load":
            mbarrier.init(bars.index(1), count=1)

        mbarrier.expect(bars.index(0), input_desc.nbytes_per_cta)
        tma.async_load(input_desc, [0, 0], bars.index(0), smem)
        if SYNCHRONIZED:
            mbarrier.wait(bars.index(0), 0)

        if OP == "load":
            mbarrier.expect(bars.index(1), input_desc.nbytes_per_cta)
            tma.async_load(input_desc, [0, 0], bars.index(1), smem)
            mbarrier.wait(bars.index(1), 0)

        tma.async_store(output_desc, [0, 0], smem)
        tma.store_wait(0)
        if not SYNCHRONIZED:
            mbarrier.wait(bars.index(0), 0)
        mbarrier.invalidate(bars.index(0))
        if OP == "load":
            mbarrier.invalidate(bars.index(1))

    block_m = XBLOCK.value * num_ctas
    input = torch.randn((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    output = torch.empty_like(input)
    layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                    cga_layout=default_cga_layout(num_ctas, 2))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [block_m, XBLOCK.value], layout)
    output_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(output, [block_m, XBLOCK.value], layout)
    kernel[(1, )](input_desc, output_desc, OP=OP, SYNCHRONIZED=SYNCHRONIZED, num_warps=4, num_ctas=num_ctas)
    if SYNCHRONIZED:
        torch.testing.assert_close(output, input)


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
        ampere.async_copy.async_load(smem.index(0), input + offs)
        ampere.async_copy.commit_group()

        ampere.async_copy.async_load(smem.index(1), input + offs)
        ampere.async_copy.commit_group()
        ampere.async_copy.wait_group(2 if FAILURE else 1)

        ampere.async_copy.async_load(smem.index(0), input + offs)
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
        tma.async_store(output_desc, [0, 0], smem.index(0))
        tma.async_store(output_desc, [0, 0], smem.index(1))
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


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires Hopper or newer")
@pytest.mark.parametrize("FAILURE", [
    pytest.param(True, id="reused-pending-source"),
    pytest.param(False, id="stable-source"),
])
def test_tma_store_convert_layout_scratch(FAILURE, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_tma_store_convert_layout_scratch, (FAILURE, device, False, monkeypatch))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            expected_error = "Accessing buffer with pending access. Pending access type: async_copy_shared_to_global"
            assert expected_error in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(output_desc, input, sink, iterations, FAILURE: ttgl.constexpr):
        src_layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])
        dst_layout: ttgl.constexpr = ttgl.SliceLayout(1, ttgl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0]))
        src_offsets = ttgl.arange(0, 128, layout=src_layout)
        dst_offsets = ttgl.arange(0, 128, layout=dst_layout)
        persistent_page = ttgl.allocate_shared_memory(ttgl.int32, [128], output_desc.layout)
        if not FAILURE:
            stable_source_page = ttgl.allocate_shared_memory(ttgl.int32, [128], output_desc.layout)

        for iteration in range(iterations):
            base = iteration * 128

            # On iteration 1, scratch aliases iteration 0's deallocated source
            # while its asynchronous TMA read can still be pending.
            value = ttgl.load(input + base + src_offsets)
            converted = ttgl.convert_layout(value, dst_layout)
            ttgl.store(sink + base + dst_offsets, converted)

            # Keep an unrelated store pending so wait(1) does not complete the
            # source transfer below.
            tma.store_wait(1)
            persistent_page.store(value)
            hopper.fence_async_shared()
            tma.async_store(output_desc, [2 * base], persistent_page)

            tma.store_wait(1)
            if FAILURE:
                source_page = ttgl.allocate_shared_memory(ttgl.int32, [128], output_desc.layout, value)
            else:
                stable_source_page.store(value)
                source_page = stable_source_page
            hopper.fence_async_shared()
            tma.async_store(output_desc, [2 * base + 128], source_page)
            if FAILURE:
                source_page._keep_alive()

        tma.store_wait(0)
        persistent_page._keep_alive()
        if not FAILURE:
            stable_source_page._keep_alive()

    iterations = 2
    output = torch.empty(iterations * 2 * 128, dtype=torch.int32, device=device)
    input = torch.arange(iterations * 128, dtype=torch.int32, device=device)
    sink = torch.empty_like(input)
    shared_layout = ttgl.NVMMASharedLayout.get_default_for([128], ttgl.int32)
    output_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(output, [128], shared_layout)
    kernel[(1, )](output_desc, input, sink, iterations, FAILURE=FAILURE, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability() != (10, 3), reason="Requires sm103 K96")
@pytest.mark.parametrize("vec", [16, 32])
@pytest.mark.parametrize(
    "case",
    ["correct", "unused_data", "unused_scale", "unused_next", "next_wait", "reuse", "scale_overwrite", "completion"])
def test_tcgen05_mma_scaled_k96_dependencies(vec, case, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_tcgen05_mma_scaled_k96_dependencies, (vec, case, False, monkeypatch))
        if case in ("correct", "unused_data", "unused_scale", "unused_next"):
            assert result.exc is None
            assert result.driver_stderr_output == ""
        else:
            assert_expected_cuda_failure(result.exc)
            if case == "next_wait":
                assert "ANext" in result.driver_stderr_output or "BNext" in result.driver_stderr_output
            else:
                assert "outstanding reads" in result.driver_stderr_output
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    @gluon.jit
    def kernel(a_desc, b_desc, out, VEC: ttgl.constexpr, CASE: ttgl.constexpr):
        a0 = ttgl.allocate_shared_memory(ttgl.uint8, a_desc.block_shape, a_desc.layout)
        b0 = ttgl.allocate_shared_memory(ttgl.uint8, b_desc.block_shape, b_desc.layout)
        a1 = ttgl.allocate_shared_memory(ttgl.uint8, a_desc.block_shape, a_desc.layout)
        b1 = ttgl.allocate_shared_memory(ttgl.uint8, b_desc.block_shape, b_desc.layout)
        ready0 = mbarrier.allocate_mbarrier(two_ctas=True)
        ready1 = mbarrier.allocate_mbarrier(two_ctas=True)
        done = mbarrier.allocate_mbarrier()
        mbarrier.init(ready0, count=1)
        mbarrier.init(ready1, count=1)
        mbarrier.init(done, count=1)
        BYTES: ttgl.constexpr = a_desc.nbytes_per_cta + b_desc.nbytes_per_cta
        mbarrier.expect(ready0, BYTES)
        tma.async_load(a_desc, [0, 0], ready0, a0)
        tma.async_load(b_desc, [0, 0], ready0, b0)
        mbarrier.expect(ready1, BYTES)
        tma.async_load(a_desc, [0, 128], ready1, a1)
        tma.async_load(b_desc, [0, 128], ready1, b1)
        mbarrier.wait(ready0, 0)
        if CASE != "next_wait" and CASE != "unused_next":
            mbarrier.wait(ready1, 0)
        # The two-CTA waits run on the even CTA. Generic writes below also
        # execute on the odd CTA, so publish completed TMA writes to it first.
        hopper.cluster.barrier()
        SCALE_DTYPE: ttgl.constexpr = ttgl.float8e4nv if VEC == 16 else ttgl.uint8
        sa = allocate_tensor_memory(SCALE_DTYPE, [256, 512 // VEC], blackwell.TensorMemoryScalesLayout([[1, 0]]))
        sb = allocate_tensor_memory(SCALE_DTYPE, [256, 512 // VEC], blackwell.TensorMemoryScalesLayout([[0, 0]]))
        sa.store(ttgl.full(sa.shape, 1 if VEC == 16 else 127, SCALE_DTYPE, sa.get_reg_layout()))
        sb.store(ttgl.full(sb.shape, 1 if VEC == 16 else 127, SCALE_DTYPE, sb.get_reg_layout()))
        hopper.cluster.barrier()
        acc = allocate_tensor_memory(
            ttgl.float32, [256, 256],
            blackwell.TensorMemoryLayout([128, 256], col_stride=1, cga_layout=[[1, 0]], two_ctas=True))
        WINDOW: ttgl.constexpr = (0, 192) if CASE == "unused_next" else (192, 288)
        if CASE == "completion":
            blackwell.tcgen05_mma_scaled(a0, b0.permute((1, 0)), acc, sa, sb, "e2m1", "e2m1", use_acc=False,
                                         k_range=WINDOW, instruction_k=96, a_next=a1, b_next=b1.permute(
                                             (1, 0)), scale_block_size=VEC, mbarriers=[], is_async=True)
            # A generic arrival does not complete tensor-core accesses.
            mbarrier.arrive(done)
        else:
            blackwell.tcgen05_mma_scaled(a0, b0.permute((1, 0)), acc, sa, sb, "e2m1", "e2m1", use_acc=False,
                                         k_range=WINDOW, instruction_k=96, a_next=a1, b_next=b1.permute(
                                             (1, 0)), scale_block_size=VEC, mbarriers=[done])
        if CASE == "reuse":
            a1.store(
                ttgl.full(a1.shape, 0, ttgl.uint8,
                          ttgl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0], cga_layout=[[1, 0]])))
        if CASE == "scale_overwrite":
            sa.store(ttgl.full(sa.shape, 0, SCALE_DTYPE, sa.get_reg_layout()))
        if CASE == "unused_data":
            unused = a1.slice(32, 32, dim=1)
            unused.store(
                ttgl.full(unused.shape, 0, ttgl.uint8,
                          ttgl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0], cga_layout=[[1, 0]])))
        if CASE == "unused_scale":
            unused_scale = sa.slice(384 // VEC, 4)
            unused_scale.store(ttgl.full(unused_scale.shape, 0, SCALE_DTYPE, unused_scale.get_reg_layout()))
        mbarrier.wait(done, 0)
        if CASE == "unused_next":
            mbarrier.wait(ready1, 0)
        if CASE == "completion":
            a1.store(
                ttgl.full(a1.shape, 0, ttgl.uint8,
                          ttgl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0], cga_layout=[[1, 0]])))
        value = acc.load()
        layout: ttgl.constexpr = acc.get_reg_layout()
        rows = ttgl.arange(0, 256, layout=ttgl.SliceLayout(1, layout))[:, None]
        cols = ttgl.arange(0, 256, layout=ttgl.SliceLayout(0, layout))[None, :]
        ttgl.store(out + rows * 256 + cols, value)
        mbarrier.invalidate(ready0)
        mbarrier.invalidate(ready1)
        mbarrier.invalidate(done)

    data = torch.zeros((256, 256), device="cuda", dtype=torch.uint8)
    layout = ttgl.NVMMASharedLayout(128, 8, cga_layout=[[1, 0]])
    desc = TensorDescriptor.from_tensor(data, [256, 128], layout)
    out = torch.empty((256, 256), device="cuda", dtype=torch.float32)
    kernel[(1, )](desc, desc, out, vec, case, num_ctas=2)
    torch.testing.assert_close(out, torch.zeros_like(out))


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability() != (10, 3), reason="Requires sm103 K96")
@pytest.mark.parametrize("fmt, buffers", [
    ("mxfp4", 5),
    ("mxfp4", 6),
    ("nvfp4", 4),
    ("nvfp4", 5),
    ("nvfp4", 6),
])
@pytest.mark.parametrize("clc_scheduler", [False, True])
def test_tcgen05_mma_scaled_k96_pipeline(fmt, buffers, clc_scheduler, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_tcgen05_mma_scaled_k96_pipeline, (fmt, buffers, clc_scheduler, False, monkeypatch))
        assert result.exc is None
        assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    knobs.refresh_knobs()
    from importlib import import_module
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/gluon"))
    example = import_module("07-pure-k96-matmul")
    torch.manual_seed(123)
    a, b, sa, sb, expected = example.make_problem(3968, 4096, 4608, fmt)
    scheduler = example.SCHEDULER_CLC if clc_scheduler else example.SCHEDULER_SPS
    # The six-slot NVFP4 ring stages scales independently in five slots and
    # reuses retired input storage for its final SPS output tile.
    epilogue = 32 if fmt == "mxfp4" else (16 if buffers == 6 else 64)
    for _ in range(2):
        actual = example.matmul(a, b, sa, sb, buffers=buffers, epilogue=epilogue, scheduler=scheduler,
                                out_dtype=torch.float16)
        torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=1e-3)


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
                assert (("Buffer being accessed has outstanding reads" in result.driver_stderr_output) or
                        (TWO_CTAS
                         and "Barrier used before initialization or after invalidation" in result.driver_stderr_output))
            elif MEM_ACCESS_KIND in ["tmem_load", "tmem_store"]:
                # The tcgen05_mma is writing tmem and its pending completion is
                # reading the barrier storage. Either conflict may report first.
                assert (("Buffer being accessed has outstanding writes" in result.driver_stderr_output)
                        or ("Buffer being accessed has outstanding reads" in result.driver_stderr_output) or
                        (TWO_CTAS
                         and "Barrier used before initialization or after invalidation" in result.driver_stderr_output))
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
            tma_bar = mbarrier.allocate_mbarrier()
            mbarrier.init(tma_bar, count=1)

        blackwell.tcgen05_mma(smemA, smemB, acc)
        blackwell.tcgen05_commit(mma_bar)

        if not FAILURE:
            mbarrier.wait(mma_bar, 0)

        if MEM_ACCESS_KIND == "tma_cp":
            mbarrier.expect(tma_bar, input_desc.nbytes_per_cta)
            tma.async_load(input_desc, [0, 0], tma_bar, smemA)
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
            tma.async_store(output_desc, [0, 0], smemAcc)
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
        # An earlier synchronous reader must not impersonate the tensor core.
        ttgl.store(output + offs, smem.load(reg_layout))
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


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("MODE", ["stale-observer", "stale-snapshot", "synchronized"])
def test_reader_generation(MODE, device, run_wrapper, monkeypatch):
    if MODE != "synchronized" and run_wrapper:
        result = run_in_process(test_reader_generation, (MODE, device, False, monkeypatch))
        assert_expected_cuda_failure(result.exc)
        assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(MODE: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
        shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(128, 32, rank=2)
        tensor_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([128, 128], col_stride=1)
        smem = ttgl.allocate_shared_memory(ttgl.int32, [128, 128], shared_layout)
        tensor = blackwell.allocate_tensor_memory(ttgl.int32, [128, 128], tensor_layout)
        barrier = mbarrier.allocate_mbarrier()
        mbarrier.init(barrier, count=1)
        zeros = ttgl.full([128, 128], 0, ttgl.int32, layout)
        smem.store(zeros)

        blackwell.tcgen05_copy(smem, tensor)
        blackwell.tcgen05_commit(barrier)
        if MODE != "stale-snapshot":
            mbarrier.wait(barrier, phase=0)

        blackwell.tcgen05_copy(smem, tensor)
        if MODE == "stale-snapshot":
            # Waiting on the first read must not publish the second read.
            mbarrier.wait(barrier, phase=0)
        if MODE == "synchronized":
            blackwell.tcgen05_commit(barrier)
            mbarrier.wait(barrier, phase=1)

        smem.store(zeros)
        mbarrier.invalidate(barrier)

    kernel[(1, )](MODE=MODE, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("TC_COMMIT", [False, True], ids=["ordinary-arrive", "tc-commit"])
def test_reader_visibility_across_partitions(TC_COMMIT, device, run_wrapper, monkeypatch):
    if not TC_COMMIT and run_wrapper:
        result = run_in_process(test_reader_visibility_across_partitions, (TC_COMMIT, device, False, monkeypatch))
        assert_expected_cuda_failure(result.exc)
        assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def first_reader(smem, tensor, barriers):
        blackwell.tcgen05_copy(smem, tensor)
        blackwell.tcgen05_commit(barriers.index(0))

    @gluon.jit
    def second_reader(smem, tensor, barriers, TC_COMMIT: ttgl.constexpr):
        mbarrier.wait(barriers.index(0), phase=0)
        blackwell.tcgen05_copy(smem, tensor)
        if TC_COMMIT:
            blackwell.tcgen05_commit(barriers.index(1))
        else:
            mbarrier.arrive(barriers.index(1), count=1)

    @gluon.jit
    def writer(smem, barriers, layout: ttgl.constexpr):
        mbarrier.wait(barriers.index(1), phase=0)
        smem.store(ttgl.zeros([128, 128], ttgl.int32, layout))

    @gluon.jit
    def kernel(TC_COMMIT: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
        shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(128, 32, rank=2)
        tensor_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([128, 128], col_stride=1)
        smem = ttgl.allocate_shared_memory(ttgl.int32, [128, 128], shared_layout)
        first_tensor = blackwell.allocate_tensor_memory(ttgl.int32, [128, 128], tensor_layout)
        second_tensor = blackwell.allocate_tensor_memory(ttgl.int32, [128, 128], tensor_layout)
        barriers = mbarrier.allocate_mbarrier(batch=2)
        mbarrier.init(barriers.index(0), count=1)
        mbarrier.init(barriers.index(1), count=1)
        smem.store(ttgl.zeros([128, 128], ttgl.int32, layout))

        ttgl.warp_specialize([
            (first_reader, (smem, first_tensor, barriers)),
            (second_reader, (smem, second_tensor, barriers, TC_COMMIT)),
            (writer, (smem, barriers, layout)),
        ], [4, 4], [32, 32])

        mbarrier.invalidate(barriers.index(0))
        mbarrier.invalidate(barriers.index(1))

    kernel[(1, )](TC_COMMIT=TC_COMMIT, num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
def test_ws_join_write_visibility(device, monkeypatch, num_ctas):
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def default_partition():
        pass

    @gluon.jit
    def writer(smem, layout: ttgl.constexpr):
        smem.store(ttgl.full([XBLOCK * ttgl.num_ctas()], 7, ttgl.int32, layout))

    @gluon.jit
    def kernel(output):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 1)
        shared_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [0], cga_layout)
        layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.int32, [block_x], shared_layout)

        ttgl.warp_specialize([(default_partition, ()), (writer, (smem, layout))], [4], [32])

        offsets = ttgl.arange(0, block_x, layout)
        ttgl.store(output + offsets, smem.load(layout))

    output = torch.empty((XBLOCK.value * num_ctas, ), device=device, dtype=torch.int32)
    kernel[(1, )](output, num_warps=4, num_ctas=num_ctas)
    torch.testing.assert_close(output, torch.full_like(output, 7))


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FENCE_LOCATION", ["producer", "consumer"])
def test_ws_join_publishes_to_async_peer(FENCE_LOCATION, device, monkeypatch):
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def default_partition():
        pass

    @gluon.jit
    def producer(smem, layout: ttgl.constexpr, FENCE_LOCATION: ttgl.constexpr):
        smem.store(ttgl.full([XBLOCK, XBLOCK], 42, ttgl.float16, layout))
        if FENCE_LOCATION == "producer":
            hopper.fence_async_shared()

    @gluon.jit
    def kernel(output_desc, FENCE_LOCATION: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout([1, XBLOCK], [32, 1], [4, 1], [0, 1])
        shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(128, 16, rank=2)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], shared_layout)
        ttgl.warp_specialize([
            (default_partition, ()),
            (producer, (smem, layout, FENCE_LOCATION)),
        ], [4], [32])
        if FENCE_LOCATION == "consumer":
            hopper.fence_async_shared()
        tma.async_store(output_desc, [0, 0], smem)
        tma.store_wait(0)

    output = torch.empty((XBLOCK.value, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(128, 16, rank=2)
    output_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(output, [XBLOCK.value, XBLOCK.value], shared_layout)
    kernel[(1, )](output_desc, FENCE_LOCATION=FENCE_LOCATION, num_warps=4)
    torch.testing.assert_close(output, torch.full_like(output, 42))


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FENCE", [False, True], ids=["missing", "present"])
def test_ws_join_publishes_proxy_generation(FENCE, device, run_wrapper, monkeypatch):
    if not FENCE and run_wrapper:
        result = run_in_process(test_ws_join_publishes_proxy_generation, (FENCE, device, False, monkeypatch))
        assert_expected_cuda_failure(result.exc)
        assert "Async shared-memory access is missing fence_async_shared" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def default_partition():
        pass

    @gluon.jit
    def reader(smem, sink, layout: ttgl.constexpr):
        offsets_m = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(1, layout))[:, None]
        offsets_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, layout))[None, :]
        ttgl.store(sink + offsets_m * XBLOCK + offsets_n, smem.load(layout))

    @gluon.jit
    def kernel(output_desc, sink, FENCE: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout([1, XBLOCK], [32, 1], [4, 1], [0, 1])
        shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(128, 16, rank=2)
        initial = ttgl.full([XBLOCK, XBLOCK], 42, ttgl.float16, layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], shared_layout, initial)
        hopper.fence_async_shared()
        ttgl.warp_specialize([
            (default_partition, ()),
            (reader, (smem, sink, layout)),
        ], [4], [32])
        if FENCE:
            hopper.fence_async_shared()
        tma.async_store(output_desc, [0, 0], smem)
        tma.store_wait(0)

    output = torch.empty((XBLOCK.value, XBLOCK.value), device=device, dtype=torch.float16)
    sink = torch.empty_like(output)
    shared_layout = ttgl.NVMMASharedLayout(128, 16, rank=2)
    output_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(output, [XBLOCK.value, XBLOCK.value], shared_layout)
    kernel[(1, )](output_desc, sink, FENCE=FENCE, num_warps=4)
    torch.testing.assert_close(output, torch.full_like(output, 42))


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("WAIT", [False, True], ids=["pending", "completed"])
def test_ws_join_async_write_visibility(WAIT, device, run_wrapper, monkeypatch):
    if not WAIT and run_wrapper:
        result = run_in_process(test_ws_join_async_write_visibility, (WAIT, device, False, monkeypatch))
        assert_expected_cuda_failure(result.exc)
        assert "Buffer being accessed has outstanding writes" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def default_partition():
        pass

    @gluon.jit
    def producer(input_desc, smem, bar, WAIT: ttgl.constexpr):
        mbarrier.expect(bar, input_desc.nbytes_per_cta)
        tma.async_load(input_desc, [0, 0], bar, smem)
        if WAIT:
            mbarrier.wait(bar, phase=0, deps=[smem])

    @gluon.jit
    def kernel(input_desc, output, WAIT: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], input_desc.layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)

        ttgl.warp_specialize([(default_partition, ()), (producer, (input_desc, smem, bar, WAIT))], [4], [32])

        offsets_m = ttgl.arange(0, block_m, ttgl.SliceLayout(1, layout))[:, None]
        offsets_n = ttgl.arange(0, XBLOCK, ttgl.SliceLayout(0, layout))[None, :]
        ttgl.store(output + offsets_m * XBLOCK + offsets_n, smem.load(layout))
        if WAIT:
            mbarrier.invalidate(bar)

    block_m = XBLOCK.value
    input = torch.randn((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    output = torch.empty_like(input)
    shared_layout = ttgl.NVMMASharedLayout(128, 16, rank=2)
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [block_m, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, output, WAIT=WAIT, num_warps=4, num_ctas=1)
    torch.testing.assert_close(output, input)


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

        # Waiting a completion barrier publishes the tensor-core frontier that
        # existed when that barrier was attached. Later barriers may still
        # receive deferred completions and cannot be invalidated yet.
        for i in range(BAR_IDX + 1):
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
        tma.async_load(a_desc, [0, 0], barLoadA.index(ins_id), smemA.index(ins_id))

        mbarrier.expect(barLoadB.index(ins_id), b_desc.nbytes_per_cta)
        tma.async_load(b_desc, [0, 0], barLoadB.index(ins_id), smemB.index(ins_id))
        ins_id = inc_mod(ins_id, num_buffers)

        # ins_id = 1
        mbarrier.expect(barLoadA.index(ins_id), a_desc.nbytes_per_cta)
        tma.async_load(a_desc, [0, 0], barLoadA.index(ins_id), smemA.index(ins_id))

        mbarrier.expect(barLoadB.index(ins_id), b_desc.nbytes_per_cta)
        tma.async_load(b_desc, [0, 0], barLoadB.index(ins_id), smemB.index(ins_id))
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
                tma.async_load(a_desc, [0, 0], barLoadA.index(ins_id), smemA.index(ins_id))

                mbarrier.expect(barLoadB.index(ins_id), b_desc.nbytes_per_cta)
                tma.async_load(b_desc, [0, 0], barLoadB.index(ins_id), smemB.index(ins_id))
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
        acc = blackwell.allocate_tensor_memory(ttgl.float32, [block_m, block_n], acc_layout)
        tma_bar = mbarrier.allocate_mbarrier(two_ctas=True)
        mbarrier.init(tma_bar, count=1)
        mma_bar = mbarrier.allocate_mbarrier()
        mma_bar_count: ttgl.constexpr = blackwell.tcgen05_mma_barrier_count([smemA, smemB], True,
                                                                            acc.type.layout.two_ctas)
        mbarrier.init(mma_bar, count=mma_bar_count)

        phase_tma = 0
        phase_mma = 0
        for k in range(num_k_tiles):
            offs_k = k * XBLOCK
            mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
            tma.async_load(a_desc, [0, offs_k], tma_bar, smemA, multicast=True)
            tma.async_load(b_desc, [offs_k, 0], tma_bar, smemB, multicast=True)
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


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
def test_tma_tcgen05_mma_missing_multicast(device, run_wrapper, monkeypatch, num_ctas):
    if num_ctas != 4:
        pytest.skip("Need 4 CTAs to exercise the missing tcgen05_mma multicast race")
    if run_wrapper:
        result = run_in_process(test_tma_tcgen05_mma_missing_multicast, (device, False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Buffer being accessed has outstanding" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(a_desc, b_desc):
        num_k_tiles: ttgl.constexpr = 4
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
        acc = blackwell.allocate_tensor_memory(ttgl.float32, [block_m, block_n], acc_layout)
        tma_bar = mbarrier.allocate_mbarrier(two_ctas=True)
        mbarrier.init(tma_bar, count=1)
        mma_bar = mbarrier.allocate_mbarrier()
        mbarrier.init(mma_bar, count=blackwell.tcgen05_mma_barrier_count([smemA, smemB], False,
                                                                         acc.type.layout.two_ctas))

        phase_tma = 0
        phase_mma = 0
        for k in range(num_k_tiles):
            offs_k = k * XBLOCK
            mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
            tma.async_load(a_desc, [0, offs_k], tma_bar, smemA, multicast=True)
            tma.async_load(b_desc, [offs_k, 0], tma_bar, smemB, multicast=True)
            mbarrier.wait(tma_bar, phase_tma, deps=[smemA, smemB])

            # Missing multicast=True is the bug under test. The next iteration
            # reuses smemA/smemB after a local completion wait.
            blackwell.tcgen05_mma(smemA, smemB, acc, use_acc=k != 0, mbarriers=[mma_bar])
            mbarrier.wait(mma_bar, phase_mma, deps=[smemA, smemB])
            phase_tma = (phase_tma + 1) % 2
            phase_mma = (phase_mma + 1) % 2

        mbarrier.invalidate(tma_bar)
        mbarrier.invalidate(mma_bar)

    block_m = mma_block_m(num_ctas)
    block_n = mma_block_n(num_ctas)
    num_k_tiles = 4
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
    kernel[(1, )](a_desc, b_desc, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
@pytest.mark.parametrize("OVERCOUNTED", [False, True])
def test_tcgen5_commit_multicast_barrier_count(OVERCOUNTED, device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_tcgen5_commit_multicast_barrier_count, (OVERCOUNTED, device, False, monkeypatch))
        if OVERCOUNTED:
            assert_expected_cuda_failure(result.exc)
            assert "Deadlock detected" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(a_desc, b_desc, OVERCOUNTED: ttgl.constexpr):
        block_m: ttgl.constexpr = 256
        block_n: ttgl.constexpr = 128
        smem_a = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], a_desc.layout)
        smem_b = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, block_n], b_desc.layout)
        acc_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([XBLOCK, XBLOCK], col_stride=1, cga_layout=((1, 0), ),
                                                                  two_ctas=True)
        acc = allocate_tensor_memory(ttgl.float32, [block_m, block_n], acc_layout)

        tma_bar = mbarrier.allocate_mbarrier(two_ctas=True)
        commit_bar = mbarrier.allocate_mbarrier()
        count: ttgl.constexpr = blackwell.tcgen05_mma_barrier_count([smem_a, smem_b], True, acc.type.layout.two_ctas)
        mbarrier.init(tma_bar, count=1)
        mbarrier.init(commit_bar, count=count + OVERCOUNTED)
        mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
        tma.async_load(a_desc, [0, 0], tma_bar, smem_a, multicast=True)
        tma.async_load(b_desc, [0, 0], tma_bar, smem_b, multicast=True)
        mbarrier.wait(tma_bar, 0, deps=[smem_a, smem_b])
        mbarrier.invalidate(tma_bar)

        blackwell.tcgen05_mma(smem_a, smem_b, acc, use_acc=False, multicast=True)
        blackwell.tcgen05_commit(commit_bar, descs=[smem_a, smem_b])
        mbarrier.wait(commit_bar, 0)

    a = torch.randn((256, XBLOCK.value), device=device, dtype=torch.float16)
    b = torch.randn((XBLOCK.value, 128), device=device, dtype=torch.float16)
    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(
        a, [256, XBLOCK.value],
        ttgl.NVMMASharedLayout.get_default_for([256, XBLOCK.value], ttgl.float16, cga_layout=((1, 0), )))
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(
        b, [XBLOCK.value, 128],
        ttgl.NVMMASharedLayout.get_default_for([XBLOCK.value, 128], ttgl.float16, cga_layout=((0, 1), )))
    kernel[(1, )](a_desc, b_desc, OVERCOUNTED=OVERCOUNTED, num_warps=4, num_ctas=2)


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
        tma.async_load(a_desc, [0, 0], barLoadA.index(ins_id), smemA.index(ins_id))

        mbarrier.expect(barLoadB.index(ins_id), b_desc.nbytes_per_cta)
        tma.async_load(b_desc, [0, 0], barLoadB.index(ins_id), smemB.index(ins_id))
        ins_id = inc_mod(ins_id, num_buffers)

        # ins_id = 1
        ub = 10
        for i in range(ub):
            if i < ub - 1:
                mbarrier.expect(barLoadA.index(ins_id), a_desc.nbytes_per_cta)
                tma.async_load(a_desc, [0, 0], barLoadA.index(ins_id), smemA.index(ins_id))

                mbarrier.expect(barLoadB.index(ins_id), b_desc.nbytes_per_cta)
                tma.async_load(b_desc, [0, 0], barLoadB.index(ins_id), smemB.index(ins_id))
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
@pytest.mark.parametrize(
    "FENCE_LOCATION",
    ["none", "producer_after_arrive", "producer_after_arrive_cluster_barrier", "producer", "consumer"])
def test_fence_async_shared_across_warp_specialize(FENCE_LOCATION, device, run_wrapper, monkeypatch, num_ctas):
    if FENCE_LOCATION == "producer_after_arrive_cluster_barrier" and num_ctas == 1:
        pytest.skip("Need multiple CTAs for a cluster barrier")
    if run_wrapper:
        result = run_in_process(test_fence_async_shared_across_warp_specialize,
                                (FENCE_LOCATION, device, False, monkeypatch, num_ctas))
        if FENCE_LOCATION in ("none", "producer_after_arrive", "producer_after_arrive_cluster_barrier"):
            assert_expected_cuda_failure(result.exc)
            assert "Async shared-memory access is missing fence_async_shared" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def producer(smem: ttgl.constexpr, bar: ttgl.constexpr, ready, FENCE_LOCATION: ttgl.constexpr,
                 layout: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        smem.store(ttgl.full([block_m, XBLOCK], 42.0, ttgl.float16, layout))
        if FENCE_LOCATION == "producer":
            hopper.fence_async_shared()
        mbarrier.arrive(bar, count=1)
        if FENCE_LOCATION == "producer_after_arrive":
            hopper.fence_async_shared()
        if FENCE_LOCATION == "producer_after_arrive_cluster_barrier":
            hopper.fence_async_shared()
            # Order every producer fence before publishing a control-only
            # global signal. The relaxed barrier does not publish ConSan state.
            hopper.cluster.barrier(relaxed=True)
            row_layout: ttgl.constexpr = ttgl.SliceLayout(1, layout)
            rows = ttgl.arange(0, block_m, row_layout)
            ttgl.store(ready + rows, 1, mask=rows == 0, cache_modifier=".wt")

    @gluon.jit
    def consumer(output_desc, smem: ttgl.constexpr, bar: ttgl.constexpr, ready, FENCE_LOCATION: ttgl.constexpr):
        mbarrier.wait(bar, phase=0)
        if FENCE_LOCATION == "producer_after_arrive_cluster_barrier":
            flag = ttgl.load(ready, volatile=True)
            while flag == 0:
                flag = ttgl.load(ready, volatile=True)
            ttgl.barrier(cluster=True)
        if FENCE_LOCATION == "consumer":
            hopper.fence_async_shared()
        tma.async_store(output_desc, [0, 0], smem)
        tma.store_wait(0)

    @gluon.jit
    def kernel(output_desc, ready, FENCE_LOCATION: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                                             cga_layout=cga_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], smem_layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        ttgl.warp_specialize([
            (producer, (smem, bar, ready, FENCE_LOCATION, blocked_layout)),
            (consumer, (output_desc, smem, bar, ready, FENCE_LOCATION)),
        ], [4], [32])

    block_m = XBLOCK.value * num_ctas
    output = torch.empty((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=default_cga_layout(num_ctas, 2))
    output_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(output, [block_m, XBLOCK.value], shared_layout)
    ready = torch.zeros((1, ), device=device, dtype=torch.int32)
    kernel[(1, )](output_desc, ready, FENCE_LOCATION=FENCE_LOCATION, num_warps=4, num_ctas=num_ctas)
    torch.testing.assert_close(output, torch.full_like(output, 42.0))


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
@pytest.mark.parametrize("SYNCHRONIZED", [False, True], ids=["missing-local-handoff", "local-handoff"])
def test_ws_cluster_barrier_does_not_replace_local_handoff(SYNCHRONIZED, device, run_wrapper, monkeypatch, num_ctas):
    if num_ctas == 1:
        pytest.skip("Need multiple CTAs for a cluster barrier")
    if run_wrapper:
        result = run_in_process(test_ws_cluster_barrier_does_not_replace_local_handoff,
                                (SYNCHRONIZED, device, False, monkeypatch, num_ctas))
        if SYNCHRONIZED:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        else:
            assert_expected_cuda_failure(result.exc)
            assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def consumer(smem, consumed, ready, output, layout: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        offsets = ttgl.arange(0, block_x, layout)
        val = smem.load(layout)
        ttgl.store(output + offsets, val)
        # Order every consumer load before publishing a control-only global
        # signal. The relaxed barrier does not publish ConSan state.
        hopper.cluster.barrier(relaxed=True)
        ttgl.store(ready + offsets, 1, mask=offsets == 0, cache_modifier=".wt")
        mbarrier.arrive(consumed, count=1)

    @gluon.jit
    def producer(smem, consumed, ready, SYNCHRONIZED: ttgl.constexpr, layout: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        flag = ttgl.load(ready, volatile=True)
        while flag == 0:
            flag = ttgl.load(ready, volatile=True)
        if SYNCHRONIZED:
            mbarrier.wait(consumed, phase=0, deps=[smem])
        ttgl.barrier(cluster=True)
        smem.store(ttgl.full([block_x], 2, ttgl.float32, layout))

    @gluon.jit
    def kernel(output, ready, SYNCHRONIZED: ttgl.constexpr):
        block_x: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 1)
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0],
                                                                cga_layout=cga_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                                                            warps_per_cta=[4], order=[0], cga_layout=cga_layout)
        smem = ttgl.allocate_shared_memory(ttgl.float32, [block_x], smem_layout)
        consumed = mbarrier.allocate_mbarrier()
        mbarrier.init(consumed, count=1)
        smem.store(ttgl.full([block_x], 1, ttgl.float32, blocked_layout))
        ttgl.warp_specialize([
            (consumer, (smem, consumed, ready, output, blocked_layout)),
            (producer, (smem, consumed, ready, SYNCHRONIZED, blocked_layout)),
        ], [4], [32])

    output = torch.empty((XBLOCK.value * num_ctas, ), device=device, dtype=torch.float32)
    ready = torch.zeros((1, ), device=device, dtype=torch.int32)
    kernel[(1, )](output, ready, SYNCHRONIZED=SYNCHRONIZED, num_warps=4, num_ctas=num_ctas)


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
    if num_ctas == 4 and MISSING_BAR == "3" and "H100" in torch.cuda.get_device_name(device):
        # PTXAS 12.9 can clobber the ConSan lock pointer across a device call.
        pytest.skip("PTXAS miscompiles the ConSan lock release on H100")
    if run_wrapper:
        result = run_in_process(test_ws_two_loads_two_bars_loop, (MISSING_BAR, device, False, monkeypatch, num_ctas))
        if MISSING_BAR != "none":
            assert_expected_cuda_failure(result.exc)
            expected = ["Buffer being accessed has outstanding"]
            # If the partition with the missing producer wait runs ahead and
            # retires first, the same broken protocol can be reported as an
            # mbarrier deadlock instead of an overlapping access.
            if MISSING_BAR in ("2", "3"):
                expected.append("Deadlock detected")
            assert any(msg in result.driver_stderr_output for msg in expected)
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
        ampere.async_copy.async_load(smem.index(BASE + 0), input + offs)
        ampere.async_copy.commit_group()

        for i in range(1, 10):
            dst = (i % 2)
            src = ((i - 1) % 2)
            if i < 9:
                ampere.async_copy.async_load(smem.index(BASE + dst), input + offs)
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
        ampere.async_copy.async_load(smem.index(0), input + offs)
        ampere.async_copy.commit_group()
        ampere.async_copy.async_load(smem.index(1), input + offs)
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
@pytest.mark.parametrize("EXPLICIT_BARRIER", [False, True], ids=["terminal-barrier", "explicit-barrier"])
@pytest.mark.parametrize("DEFAULT_WARPS", [4, 8], ids=["four-default-warps", "eight-default-warps"])
def test_cluster_barrier_warp_specialized_phase_snapshot(EXPLICIT_BARRIER, DEFAULT_WARPS, device, run_wrapper,
                                                         monkeypatch, num_ctas):
    if num_ctas == 1:
        pytest.skip("cluster barriers require multiple CTAs")
    if run_wrapper:
        result = run_in_process(test_cluster_barrier_warp_specialized_phase_snapshot,
                                (EXPLICIT_BARRIER, DEFAULT_WARPS, device, False, monkeypatch, num_ctas))
        assert result.exc is None
        assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def wait_then_signal(bar):
        mbarrier.wait(bar.index(0), phase=0)
        mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def signal_then_wait(bar):
        mbarrier.arrive(bar.index(0), count=1)
        mbarrier.wait(bar.index(1), phase=0)

    @gluon.jit
    def kernel(output, EXPLICIT_BARRIER: ttgl.constexpr):
        bar = mbarrier.allocate_mbarrier(batch=2)
        mbarrier.init(bar.index(0), count=1)
        mbarrier.init(bar.index(1), count=1)
        ttgl.warp_specialize([
            (wait_then_signal, (bar, )),
            (signal_then_wait, (bar, )),
        ], [4], [32])
        if EXPLICIT_BARRIER:
            ttgl.barrier(cluster=True)
        pid = ttgl.program_id(0)
        ttgl.store(output + pid, pid)

    output = torch.empty((num_ctas * 16, ), device=device, dtype=torch.int32)
    for _ in range(4):
        kernel[(num_ctas * 16, )](output, EXPLICIT_BARRIER, num_warps=DEFAULT_WARPS, num_ctas=num_ctas)
    torch.testing.assert_close(output, torch.arange(num_ctas * 16, device=device, dtype=torch.int32))


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
def test_deadlock_with_padded_warp_specialize_partition(device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_deadlock_with_padded_warp_specialize_partition,
                                (device, False, monkeypatch, num_ctas))
        assert_expected_cuda_failure(result.exc)
        assert "Deadlock detected" in result.driver_stderr_output
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def wait_forever(bar):
        mbarrier.wait(bar, phase=0)

    @gluon.jit
    def done(bar):
        pass

    @gluon.jit
    def kernel():
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        ttgl.warp_specialize([
            (done, (bar, )),
            (wait_forever, (bar, )),
            (wait_forever, (bar, )),
            (wait_forever, (bar, )),
        ], [1, 1, 1], [32, 32, 32])

    kernel[(1, )](num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("TWO_CTAS,CLUSTER_BARRIER,FAILURE", [
    pytest.param(False, False, True, id="single-cta-mbarrier"),
    pytest.param(True, False, True, id="two-cta-mbarrier"),
    pytest.param(True, True, True, id="cluster-deadlock"),
    pytest.param(True, True, False, id="cluster-no-deadlock"),
])
def test_deadlock_after_other_partition_returns(TWO_CTAS, CLUSTER_BARRIER, FAILURE, device, run_wrapper, monkeypatch,
                                                num_ctas):
    if TWO_CTAS and num_ctas == 1:
        pytest.skip("two-CTA barriers require at least two CTAs")
    if run_wrapper:
        result = run_in_process(test_deadlock_after_other_partition_returns,
                                (TWO_CTAS, CLUSTER_BARRIER, FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Deadlock detected" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return
    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def done(bar):
        pass

    @gluon.jit
    def wait_forever(bar, FAILURE: ttgl.constexpr):
        if FAILURE:
            mbarrier.wait(bar.index(0), phase=0)

    @gluon.jit
    def complete_and_return(bar):
        mbarrier.arrive(bar.index(1), count=1)

    @gluon.jit
    def kernel(TWO_CTAS: ttgl.constexpr, CLUSTER_BARRIER: ttgl.constexpr, FAILURE: ttgl.constexpr):
        bar = mbarrier.allocate_mbarrier(batch=2, two_ctas=TWO_CTAS)
        mbarrier.init(bar.index(0), count=1)
        mbarrier.init(bar.index(1), count=1)
        ttgl.warp_specialize([
            (done, (bar, )),
            (wait_forever, (bar, FAILURE)),
            (complete_and_return, (bar, )),
        ], [4, 4], [32, 32])
        if CLUSTER_BARRIER:
            for _ in range(2):
                ttgl.barrier(cluster=True)

    kernel[(1, )](TWO_CTAS, CLUSTER_BARRIER, FAILURE, num_warps=4, num_ctas=num_ctas)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("CLUSTER_PARTITION", [False, True], ids=["default-region", "partition-region"])
@pytest.mark.parametrize("FAILURE", [True, False], ids=["deadlock", "no-deadlock"])
def test_deadlock_user_cluster_barrier_inside_warp_specialize(CLUSTER_PARTITION, FAILURE, device, run_wrapper,
                                                              monkeypatch, num_ctas):
    if num_ctas == 1:
        pytest.skip("two-CTA barriers require at least two CTAs")
    if run_wrapper:
        result = run_in_process(test_deadlock_user_cluster_barrier_inside_warp_specialize,
                                (CLUSTER_PARTITION, FAILURE, device, False, monkeypatch, num_ctas))
        if FAILURE:
            assert_expected_cuda_failure(result.exc)
            assert "Deadlock detected" in result.driver_stderr_output
        else:
            assert result.exc is None
            assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def wait_local(local_bar, paired_bar):
        mbarrier.wait(local_bar, phase=0)

    @gluon.jit
    def wait_paired_then_cluster(local_bar, paired_bar):
        ttgl.barrier(cluster=True)
        mbarrier.wait(paired_bar, phase=0)
        ttgl.barrier(cluster=True)

    @gluon.jit
    def kernel(FAILURE: ttgl.constexpr, CLUSTER_PARTITION: ttgl.constexpr):
        local_bar = mbarrier.allocate_mbarrier()
        paired_bar = mbarrier.allocate_mbarrier(two_ctas=True)
        mbarrier.init(local_bar, count=1)
        mbarrier.init(paired_bar, count=1)
        if not FAILURE:
            mbarrier.arrive(local_bar, count=1)
            mbarrier.arrive(paired_bar, count=1)
        if CLUSTER_PARTITION:
            ttgl.warp_specialize([
                (wait_local, (local_bar, paired_bar)),
                (wait_paired_then_cluster, (local_bar, paired_bar)),
            ], [4], [32])
        else:
            ttgl.warp_specialize([
                (wait_paired_then_cluster, (local_bar, paired_bar)),
                (wait_local, (local_bar, paired_bar)),
            ], [4], [32])

    compiled = kernel[(1, )](FAILURE, CLUSTER_PARTITION, num_warps=4, num_ctas=num_ctas)
    if not FAILURE and not is_compile_warmup():
        assert compiled.asm["ptx"].count("mbarrier.arrive.release.cluster.shared::cluster") >= 2


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
        tma.async_load(input_desc, [0, 0], bar.index(0), smem.index(0))
        mbarrier.wait(bar.index(0), phase=0)

    @gluon.jit
    def ws_1(input_desc, smem, bar):
        mbarrier.expect(bar.index(1), input_desc.nbytes_per_cta)
        tma.async_load(input_desc, [0, 0], bar.index(1), smem.index(1))
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


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("ACCESS", ["generic", "async-copy", "convert-layout"])
@pytest.mark.parametrize("INVALIDATE", [False, True], ids=["live", "invalidated"])
def test_payload_reuse_requires_barrier_invalidation(ACCESS, INVALIDATE, device, run_wrapper, monkeypatch):
    if run_wrapper and not INVALIDATE:
        result = run_in_process(test_payload_reuse_requires_barrier_invalidation,
                                (ACCESS, INVALIDATE, device, False, monkeypatch))
        assert_expected_cuda_failure(result.exc)
        assert "Shared memory reused before barrier invalidation" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(source, output, ACCESS: ttgl.constexpr, INVALIDATE: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [0])
        barrier = mbarrier.allocate_mbarrier()
        mbarrier.init(barrier, count=1)
        if INVALIDATE:
            mbarrier.invalidate(barrier)

        offsets = ttgl.arange(0, XBLOCK, layout=layout)
        if ACCESS == "async-copy":
            smem = ttgl.allocate_shared_memory(ttgl.int32, [XBLOCK], smem_layout)
            ampere.async_copy.async_load(smem, source + offsets)
            ampere.async_copy.commit_group()
            ampere.async_copy.wait_group(0)
        elif ACCESS == "convert-layout":
            dst_layout: ttgl.constexpr = ttgl.SliceLayout(1, ttgl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0]))
            dst_offsets = ttgl.arange(0, XBLOCK, layout=dst_layout)
            values = ttgl.load(source + offsets)
            ttgl.store(output + dst_offsets, ttgl.convert_layout(values, dst_layout))
        else:
            values = ttgl.full([XBLOCK], 7, ttgl.int32, layout)
            smem = ttgl.allocate_shared_memory(ttgl.int32, [XBLOCK], smem_layout, values)
        if ACCESS != "convert-layout":
            ttgl.store(output + offsets, smem.load(layout))

    source = torch.arange(XBLOCK.value, device=device, dtype=torch.int32)
    output = torch.empty_like(source)
    compiled = kernel.warmup(source, output, ACCESS=ACCESS, INVALIDATE=INVALIDATE, grid=(1, ), num_warps=4, num_ctas=1)
    assert compiled.metadata.shared == XBLOCK.value * source.element_size()
    kernel[(1, )](source, output, ACCESS=ACCESS, INVALIDATE=INVALIDATE, num_warps=4, num_ctas=1)
    expected = torch.full_like(source, 7) if ACCESS == "generic" else source
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("WAIT", [False, True], ids=["outstanding-copy", "waited-copy"])
def test_barrier_init_is_generic_write(WAIT, device, run_wrapper, monkeypatch):
    if run_wrapper and not WAIT:
        result = run_in_process(test_barrier_init_is_generic_write, (WAIT, device, False, monkeypatch))
        assert_expected_cuda_failure(result.exc)
        assert "Pending access type: async_copy_global_to_shared" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(source, WAIT: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0])
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [0])
        offsets = ttgl.arange(0, XBLOCK, layout=layout)
        smem = ttgl.allocate_shared_memory(ttgl.int32, [XBLOCK], smem_layout)
        ampere.async_copy.async_load(smem, source + offsets)
        ampere.async_copy.commit_group()
        if WAIT:
            ampere.async_copy.wait_group(0)
        barrier = mbarrier.allocate_mbarrier()
        mbarrier.init(barrier, count=1)
        if not WAIT:
            ampere.async_copy.wait_group(0)
        mbarrier.invalidate(barrier)

    source = torch.arange(XBLOCK.value, device=device, dtype=torch.int32)
    compiled = kernel.warmup(source, WAIT=WAIT, grid=(1, ), num_warps=4, num_ctas=1)
    assert compiled.metadata.shared == XBLOCK.value * source.element_size()
    kernel[(1, )](source, WAIT=WAIT, num_warps=4, num_ctas=1)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("INITIALIZE", [False, True], ids=["uninitialized", "initialized"])
def test_async_copy_mbarrier_arrive_requires_init(INITIALIZE, device, run_wrapper, monkeypatch):
    if run_wrapper and not INITIALIZE:
        result = run_in_process(test_async_copy_mbarrier_arrive_requires_init, (INITIALIZE, device, False, monkeypatch))
        assert_expected_cuda_failure(result.exc)
        assert "Barrier used before initialization or after invalidation" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(INITIALIZE: ttgl.constexpr):
        barrier = mbarrier.allocate_mbarrier()
        if INITIALIZE:
            mbarrier.init(barrier, count=1)
        ampere.async_copy.mbarrier_arrive(barrier, increment_count=False)

    kernel[(1, )](INITIALIZE=INITIALIZE, num_warps=4, num_ctas=1)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
@pytest.mark.parametrize("WAIT", [False, True], ids=["outstanding-completion", "waited-completion"])
def test_barrier_invalidate_requires_completion_wait(WAIT, device, run_wrapper, monkeypatch):
    if run_wrapper and not WAIT:
        result = run_in_process(test_barrier_invalidate_requires_completion_wait, (WAIT, device, False, monkeypatch))
        assert_expected_cuda_failure(result.exc)
        assert "Buffer being accessed has outstanding reads" in result.driver_stderr_output
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def kernel(input_desc, WAIT: ttgl.constexpr):
        smem = ttgl.allocate_shared_memory(ttgl.float16, [XBLOCK, XBLOCK], input_desc.layout)
        barrier = mbarrier.allocate_mbarrier()
        mbarrier.init(barrier, count=1)
        mbarrier.expect(barrier, input_desc.nbytes_per_cta)
        tma.async_load(input_desc, [0, 0], barrier, smem)
        if WAIT:
            mbarrier.wait(barrier, 0, deps=[smem])
        mbarrier.invalidate(barrier)

    source = torch.randn((XBLOCK.value, XBLOCK.value), device=device, dtype=torch.float16)
    smem_layout = ttgl.NVMMASharedLayout(128, 16, rank=2, cga_layout=[])
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(source, [XBLOCK.value, XBLOCK.value], smem_layout)
    kernel[(1, )](input_desc, WAIT=WAIT, num_warps=4, num_ctas=1)


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


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper")
def test_aliasing_tma_overwrite_clears_stale_write_visibility(device, run_wrapper, monkeypatch, num_ctas):
    if run_wrapper:
        result = run_in_process(test_aliasing_tma_overwrite_clears_stale_write_visibility,
                                (device, False, monkeypatch, num_ctas))
        assert result.exc is None
        assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def writer(full: ttgl.constexpr, tail: ttgl.constexpr, input_desc, bar: ttgl.constexpr,
               blocked_layout_wide: ttgl.constexpr):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        vals = ttgl.full([block_m, XBLOCK * 2], 42.0, ttgl.float16, blocked_layout_wide)
        full.store(vals)
        mbarrier.expect(bar.index(0), input_desc.nbytes_per_cta)
        tma.async_load(input_desc, [0, 0], bar.index(0), tail)

    @gluon.jit
    def reader(tail: ttgl.constexpr, dummy: ttgl.constexpr, bar: ttgl.constexpr, blocked_layout: ttgl.constexpr):
        mbarrier.wait(bar.index(0), phase=0)
        val = tail.load(blocked_layout)
        dummy.store(val)

    @gluon.jit
    def kernel(input_desc):
        block_m: ttgl.constexpr = XBLOCK * ttgl.num_ctas()
        cga_layout: ttgl.constexpr = default_cga_layout(ttgl.num_ctas(), 2)
        smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                                             cga_layout=cga_layout)
        blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, XBLOCK], threads_per_warp=[32, 1],
                                                            warps_per_cta=[4, 1], order=[0, 1], cga_layout=cga_layout)
        blocked_layout_wide: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[2, XBLOCK], threads_per_warp=[32, 1],
                                                                 warps_per_cta=[4, 1], order=[0,
                                                                                              1], cga_layout=cga_layout)
        full = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK * 2], smem_layout)
        tail = full.slice(XBLOCK, XBLOCK, dim=1)
        dummy = ttgl.allocate_shared_memory(ttgl.float16, [block_m, XBLOCK], smem_layout)
        bar = mbarrier.allocate_mbarrier(batch=1)
        mbarrier.init(bar.index(0), count=1)
        ttgl.warp_specialize([(writer, (full, tail, input_desc, bar, blocked_layout_wide)),
                              (reader, (tail, dummy, bar, blocked_layout))], [4], [32])

    block_m = XBLOCK.value * num_ctas
    input = torch.randn((block_m, XBLOCK.value), device=device, dtype=torch.float16)
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2,
                                           cga_layout=default_cga_layout(num_ctas, 2))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [block_m, XBLOCK.value], shared_layout)
    kernel[(1, )](input_desc, num_warps=4, num_ctas=num_ctas)


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


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
def test_disjoint_noncontiguous_tmem_subslices(device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_disjoint_noncontiguous_tmem_subslices, (device, False, monkeypatch))
        assert result.exc is None
        assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def reader(src: ttgl.constexpr, sink: ttgl.constexpr, layout: ttgl.constexpr):
        sink.store(src.load(layout))

    @gluon.jit
    def writer(dst: ttgl.constexpr, layout: ttgl.constexpr):
        dst.store(ttgl.zeros([256, 64], ttgl.float32, layout))

    @gluon.jit
    def kernel():
        tmem_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([128, 128], col_stride=1, cga_layout=())
        tmem = blackwell.allocate_tensor_memory(ttgl.float32, [2, 256, 128], tmem_layout)
        page0 = tmem.index(0)
        page1 = tmem.index(1)
        page0_slab1 = page0.slice(64, 64)
        page1_slab0 = page1.slice(0, 64)
        layout: ttgl.constexpr = page0_slab1.get_reg_layout()
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0, 1])
        sink = ttgl.allocate_shared_memory(ttgl.float32, [256, 64], smem_layout)

        ttgl.warp_specialize([(reader, (page0_slab1, sink, layout)), (writer, (page1_slab0, layout))], [4], [32])

    kernel[(1, )](num_warps=4)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 10, reason="Requires blackwell or newer")
def test_same_page_tmem_slabs_with_full_page_descriptor_collision(device, run_wrapper, monkeypatch):
    if run_wrapper:
        result = run_in_process(test_same_page_tmem_slabs_with_full_page_descriptor_collision,
                                (device, False, monkeypatch))
        assert result.exc is None
        assert result.driver_stderr_output == ""
        return

    monkeypatch.setenv("TRITON_INSTRUMENTATION_MODE", "consan")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    knobs.refresh_knobs()

    @gluon.jit
    def reader(src: ttgl.constexpr, sink: ttgl.constexpr, layout: ttgl.constexpr):
        sink.store(src.load(layout))

    @gluon.jit
    def writer(dst: ttgl.constexpr, layout: ttgl.constexpr):
        dst.store(ttgl.zeros([256, 64], ttgl.float32, layout))

    @gluon.jit
    def kernel():
        tmem_layout: ttgl.constexpr = blackwell.TensorMemoryLayout([128, 128], col_stride=1, cga_layout=())
        zero_layout: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 128], threads_per_warp=[32, 1],
                                                         warps_per_cta=[4, 1], order=[0, 1])
        zero = ttgl.zeros([256, 128], ttgl.float32, zero_layout)
        # Initializing the full page makes both the full-page region and slab 0
        # appear in ConSan's registry with the same runtime descriptor key.
        tmem = blackwell.allocate_tensor_memory(ttgl.float32, [256, 128], tmem_layout, zero)
        slab0 = tmem.slice(0, 64)
        slab1 = tmem.slice(64, 64)
        layout: ttgl.constexpr = slab0.get_reg_layout()
        smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0, 1])
        sink = ttgl.allocate_shared_memory(ttgl.float32, [256, 64], smem_layout)

        ttgl.warp_specialize([(reader, (slab0, sink, layout)), (writer, (slab1, layout))], [4], [32])

    kernel[(1, )](num_warps=4)


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
        ampere.async_copy.async_load(alias0, input + offs)
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
    ampere.async_copy.async_load(a_smem, a_ptr + offs)


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
