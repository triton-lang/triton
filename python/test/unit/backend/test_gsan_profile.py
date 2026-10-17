from contextlib import contextmanager, ExitStack
from functools import lru_cache

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_blackwell, is_cuda, is_hopper_or_newer, run_in_process
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gsan import create_mem_pool as create_race_mem_pool
from triton.experimental.gsan.profile import AllocationProfile, is_installed
from triton.tools.tensor_descriptor import TensorDescriptor


@pytest.fixture(autouse=True)
def gsan_allocations():
    # Workers import this module before startup, so defer CUDA checks until setup.
    if not is_cuda():
        pytest.skip("NVIDIA-only allocation profiler")
    if not is_installed():
        pytest.skip("requires startup installation; run scripts/validate_gsan_profile.sh test")


@triton.jit
def _fill(output, N: tl.constexpr, STEP: tl.constexpr = 1, BLOCK: tl.constexpr = 1024):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(output + offset * STEP, 17, offset < N)


@triton.jit
def _copy(source, output, N: tl.constexpr, BLOCK: tl.constexpr = 1024):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + offset, offset < N, 0)
    tl.store(output + offset, value, offset < N)


@triton.jit
def _wait_and_fill(output, gate):
    while tl.load(gate, volatile=True) == 0:
        pass
    offsets = tl.arange(0, 1024)
    tl.store(output + offsets, 17)


@triton.jit
def _atomic(output):
    tl.atomic_add(output, 1)


@triton.jit
def _adjacent_ctas(output):
    tl.store(output + tl.program_id(0), 17)


@triton.jit
def _tma_copy(source, output):
    value = source.load([0, 0])
    output.store([0, 0], value)


@triton.jit
def _tma_indexed(source, output):
    rows = tl.arange(0, 8) * 2
    values = source.gather(rows, 0)
    output.scatter(values, rows, 0)


@triton.jit
def _tma_reduce(output):
    output.atomic_add([0, 0], tl.full((8, 32), 1, tl.int32))


@gluon.jit
def _ws_partition(output, start):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = gl.arange(0, 128, layout=layout)
    gl.store(output + start + offsets, 17)


@gluon.jit
def _ws_fill(output):
    gl.warp_specialize([(_ws_partition, (output, 0)), (_ws_partition, (output, 128))], [4], [24])


def _allocation(profile, name):
    report = profile.report()
    assert report["warnings"] == []
    return next(row for row in report["allocations"] if name in row["names"])


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize("count", [257, 1024])
def test_gsan_profile_access_granularity_and_write_once(dtype, count, fresh_knobs):
    with AllocationProfile() as profile:
        source = torch.empty(count, dtype=dtype, device="cuda")
        output = torch.empty_like(source)
        profile.track(source, name="source")
        profile.track(output, name="output")
        kernel = _fill[(1, )](source, count)
        _copy[(1, )](source, output, count)
    shadow_atomics = [line for line in kernel.asm["ptx"].splitlines() if ".or.b32" in line]
    assert shadow_atomics and all(".sys." in line for line in shadow_atomics)
    torch.testing.assert_close(output, torch.full_like(output, 17))
    for name in ("source", "output"):
        row = _allocation(profile, name)
        assert row["write_once_candidate"]
        # At 1024 elements / 128 threads the byte kernel has eight-byte
        # per-thread accesses, even though the allocation is 16-byte aligned.
        expected = source.element_size() if count == 257 else min(16, 8 * source.element_size())
        assert row["max_granularity"] == expected
        assert row["bytes_written"] == count * source.element_size()
        assert row["cells_read"] == (triton.cdiv(source.nbytes, 16) if name == "source" else 0)
        assert row["recommended_write_once_shadow_bytes"] < row["default_gsan_shadow_bytes"]
    assert triton.knobs.compilation.instrumentation_mode == ""


@pytest.mark.parametrize("count", [1, 16, 16400])
@pytest.mark.parametrize("reads", [1, 2])
def test_gsan_profile_cells_read_counts_distinct_cells(count, reads, fresh_knobs):
    with AllocationProfile() as profile:
        source = torch.empty(count + 512, dtype=torch.uint8, device="cuda")
        output = torch.empty(count, dtype=torch.uint8, device="cuda")
        expected = triton.cdiv(15 + count, 16)
        profile.track(source, name="source")
        profile.track(output, name="output")
        _fill[(triton.cdiv(source.numel(), 1024), )](source, source.numel())
        # Start at the last byte of a cell, leaving both untouched bytes and
        # whole untouched cells in the registered storage. The large case
        # also requires combining multiple summary-kernel output rows.
        for _ in range(reads):
            _copy[(triton.cdiv(count, 1024), )](source[15:], output, count)
    assert _allocation(profile, "source")["cells_read"] == expected
    assert _allocation(profile, "output")["cells_read"] == 0
    torch.testing.assert_close(output, torch.full_like(output, 17))


def test_gsan_profile_partial_cell_writes_are_not_byte_rewrites(fresh_knobs):
    with AllocationProfile() as profile:
        output = torch.empty(32, dtype=torch.uint8, device="cuda")
        profile.track(output, name="output")
        _fill[(1, )](output, 16, STEP=2)
        _fill[(1, )](output[1:], 16, STEP=2)
    row = _allocation(profile, "output")
    assert row["write_once_candidate"]
    assert row["max_granularity"] == 1
    assert row["bytes_written"] == 32
    torch.testing.assert_close(output, torch.full_like(output, 17))


def test_gsan_profile_adjacent_ctas_do_not_report_repeated_writes(fresh_knobs):
    with AllocationProfile() as profile:
        output = torch.empty(32, dtype=torch.uint8, device="cuda")
        profile.track(output, name="output")
        _adjacent_ctas[(32, )](output)
    row = _allocation(profile, "output")
    assert row["write_once_candidate"]
    assert row["max_granularity"] == 1
    assert row["bytes_written"] == 32
    torch.testing.assert_close(output, torch.full_like(output, 17))


def test_gsan_profile_aliases_and_repeated_writes(fresh_knobs):
    with AllocationProfile() as profile:
        output = torch.empty(1024, dtype=torch.int32, device="cuda")
        view = output[4:8]
        profile.track(output, name="output")
        profile.track(view, name="view")
        _fill[(1, )](output, 1024)
        _fill[(1, )](view, 4)
    row = _allocation(profile, "view")
    assert row["names"] == ["output", "view"]
    assert not row["write_once_candidate"]
    assert row["cells_with_repeated_writes"] == 1
    assert "repeated writes" in row["reasons"]
    assert row["bytes_written"] == output.numel() * output.element_size()


def test_gsan_profile_read_before_observed_initialization(fresh_knobs):
    with AllocationProfile() as profile:
        source = torch.ones(1024, device="cuda")
        output = torch.empty_like(source)
        profile.track(source, name="source")
        profile.track(output, name="output")
        _copy[(1, )](source, output, 1024)
    row = _allocation(profile, "source")
    assert not row["write_once_candidate"]
    assert "reads before an observed write" in row["reasons"]
    assert row["cells_read_before_write"] == 256
    assert row["cells_read"] == 256
    assert _allocation(profile, "output")["write_once_candidate"]
    torch.testing.assert_close(output, source)


def test_gsan_profile_atomics_disqualify_write_once(fresh_knobs):
    with AllocationProfile() as profile:
        output = torch.zeros(1, dtype=torch.int32, device="cuda")
        profile.track(output, name="output")
        _atomic[(1, )](output)
    row = _allocation(profile, "output")
    assert not row["write_once_candidate"]
    assert row["max_granularity"] == 4
    assert "atomic accesses" in row["reasons"]
    assert row["cells_read"] == 0
    assert output.item() == 1


@pytest.mark.parametrize("register_unmanaged", [False, True])
def test_gsan_profile_unmanaged_memory_is_ignored(register_unmanaged, fresh_knobs):
    source = torch.empty(32, dtype=torch.int32, device="cuda")
    with AllocationProfile() as profile:
        output = torch.empty_like(source)
        if register_unmanaged:
            assert profile.track(source, name="ignored") is source
        profile.track(output, name="output")
        _fill[(1, )](source, 32)
        _copy[(1, )](source, output, 32)
    report = profile.report()
    assert report["warnings"] == []
    assert len(report["allocations"]) == 1
    assert report["allocations"][0]["names"] == ["output"]
    assert report["allocations"][0]["write_once_candidate"]
    torch.testing.assert_close(output, torch.full_like(output, 17))


def test_gsan_profile_unregistered_managed_memory_uses_shadow(fresh_knobs):
    from triton.experimental.gsan import _profile_allocator as _allocator
    from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr
    with AllocationProfile() as profile:
        source = torch.empty(32, dtype=torch.int32, device="cuda")
        output = torch.empty_like(source)
        profile.track(output, name="output")
        _fill[(1, )](source, 32)
        _copy[(1, )](source, output, 32)
    # Registration selects report rows; it is not an instrumentation filter.
    _, _, shadow = _allocator.get_allocation_profile_layout(source.data_ptr())
    cell = uint8_cuda_tensor_from_ptr(shadow, 4, source.device.index).view(torch.int32).item()
    from triton._C.libtriton import gsan_profile as layout
    assert cell & layout.WRITTEN_MASK == layout.WRITTEN_MASK
    assert cell & layout.READ
    assert cell & layout.WRITE
    assert _allocation(profile, "output")["write_once_candidate"]


@pytest.mark.parametrize("granularity", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("write_once", [False, True])
def test_gsan_profile_ignores_race_pools(granularity, write_once, fresh_knobs):
    from triton.experimental.gsan import _allocator as race_allocator
    from triton.experimental.gsan import _profile_allocator as _allocator
    from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr
    race_pool = create_race_mem_pool(shadow_granularity=granularity, write_once=write_once)
    with torch.cuda.use_mem_pool(race_pool):
        source = torch.empty(64, dtype=torch.uint8, device="cuda")
    _, _, shadow, shadow_size = race_allocator.export_allocation_memhandle_regions(source.data_ptr())
    metadata = uint8_cuda_tensor_from_ptr(shadow, shadow_size, source.device.index)
    metadata.fill_(0xA5)
    assert _allocator.get_allocation_profile_layout(source.data_ptr()) is None
    with pytest.raises(ValueError, match="live profiling allocation"):
        _allocator.export_allocation_handles(source.data_ptr(), _allocator.ShareableHandleType.POSIX_FILE_DESCRIPTOR)
    with AllocationProfile() as profile:
        output = torch.empty_like(source)
        profile.track(source, name="ignored")
        profile.track(output, name="output")
        _adjacent_ctas[(64, )](source)
        _copy[(1, )](source, output, 64)
    assert torch.all(metadata == 0xA5)
    report = profile.report()
    assert not report["warnings"]
    assert len(report["allocations"]) == 1
    row = report["allocations"][0]
    assert row["names"] == ["output"]
    assert row["write_once_candidate"]
    assert row["bytes_written"] == 64
    assert row["profiling_metadata_bytes"] == 16
    torch.testing.assert_close(output, torch.full_like(output, 17))


def test_gsan_profile_pool_has_compact_metadata(fresh_knobs):
    from triton.experimental.gsan import _allocator as race_allocator, _profile_allocator as allocator, profile
    with AllocationProfile():
        output = torch.empty(1024, dtype=torch.int32, device="cuda")
        base, size, shadow, shadow_size = allocator.export_allocation_memhandle_regions(output.data_ptr())
        assert profile.AllocationProfile is AllocationProfile
        assert shadow_size == size // 4
        layout = allocator.get_allocation_profile_layout(output.data_ptr())
        assert layout == (base, size, shadow + (output.data_ptr() - base) // 16 * 4)
        next_cell = allocator.get_allocation_profile_layout(output.data_ptr() + 16)
        assert next_cell == (base, size, layout[2] + 4)
        with pytest.raises(RuntimeError, match="gsanExportAllocationMemhandleRegions failed"):
            race_allocator.export_allocation_memhandle_regions(output.data_ptr())


def _check_allocator_lifetime_is_independent():
    import torch
    from triton.experimental.gsan import _allocator as race, _profile_allocator as profile

    device = torch.cuda.current_device()
    pointer = profile.malloc(4096, device)
    assert profile.has_live_allocations()
    assert not race.has_live_allocations()
    race.configure(rng_seed=1234)
    rank = race.get_device_rank(device)
    try:
        race.get_runtime_state_layout(rank)
    except RuntimeError as error:
        assert "has not been mapped" in str(error)
    else:
        raise AssertionError("profiling allocated race-checker clocks")
    race.reset()
    assert profile.get_allocation_profile_layout(pointer) is not None
    profile.free_allocation(pointer, device)
    profile.reset()

    pointer = race.gsan_malloc(4096, device)
    profile.configure(handle_type=profile.ShareableHandleType.POSIX_FILE_DESCRIPTOR)
    assert not profile.has_live_allocations()
    assert profile.get_allocation_profile_layout(pointer) is None
    profile.reset()
    assert race.has_live_allocations()
    race.free_allocation(pointer, device)
    race.reset()


def test_gsan_profile_allocator_lifetime_is_independent():
    # Fresh state proves that neither allocator freezes the other's config,
    # owns the other's pointers, or prevents the other's reset.
    result = run_in_process(_check_allocator_lifetime_is_independent)
    assert result.exc is None, result


@pytest.mark.parametrize("operation", ["tensor", "atomic", "tma", "tma_indexed", "tma_reduce"])
def test_gsan_profile_pool_is_ignored_by_race_checker(operation, fresh_knobs, with_allocator):
    from triton.experimental.gsan import _profile_allocator as _allocator
    from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr
    if operation.startswith("tma") and not is_hopper_or_newer():
        pytest.skip("TMA requires Hopper or newer")
    if operation == "tma_indexed" and not is_blackwell():
        pytest.skip("TMA gather/scatter requires Blackwell")
    # A race allocation initializes the per-device clocks for the GSan launch.
    race_pool = create_race_mem_pool()
    with torch.cuda.use_mem_pool(race_pool):
        runtime_owner = torch.empty(1, device="cuda")
    with AllocationProfile():
        source = torch.full((16, 32), 17, dtype=torch.int32, device="cuda")
        output = torch.zeros_like(source)
    assert _allocator.get_allocation_profile_layout(runtime_owner.data_ptr()) is None
    metadata = []
    for tensor in (source, output):
        _, _, shadow, size = _allocator.export_allocation_memhandle_regions(tensor.data_ptr())
        cells = uint8_cuda_tensor_from_ptr(shadow, size, tensor.device.index)
        cells.fill_(0xA5)
        metadata.append(cells)
    triton.knobs.compilation.instrumentation_mode = "gsan"
    if operation == "tensor":
        _copy[(1, )](source, output, output.numel())
    elif operation == "atomic":
        _atomic[(1, )](output)
    elif operation == "tma":
        _tma_copy[(1, )](TensorDescriptor.from_tensor(source, [8, 32]), TensorDescriptor.from_tensor(output, [8, 32]))
    elif operation == "tma_indexed":
        _tma_indexed[(1, )](TensorDescriptor.from_tensor(source, [1, 32]),
                            TensorDescriptor.from_tensor(output, [1, 32]))
    else:
        _tma_reduce[(1, )](TensorDescriptor.from_tensor(output, [8, 32]))
    torch.cuda.synchronize()
    for cells in metadata:
        assert torch.all(cells == 0xA5)
    expected = torch.zeros_like(output)
    if operation == "tensor":
        expected.fill_(17)
    elif operation == "atomic":
        expected[0, 0] = 1
    elif operation == "tma_indexed":
        expected[::2] = 17
    else:
        expected[:8] = 1 if operation == "tma_reduce" else 17
    torch.testing.assert_close(output, expected)


def test_gsan_profile_unaccessed_and_empty_storage(fresh_knobs):
    with AllocationProfile() as profile:
        profile.track(torch.empty(0, device="cuda"), name="empty")
        profile.track(torch.empty(16, device="cuda"), name="unaccessed")
    for row in profile.report()["allocations"]:
        assert "observed" not in row
        assert row["cells_read"] == 0
        assert row["max_granularity"] is None
        assert not row["write_once_candidate"]


@pytest.mark.skipif("not is_hopper_or_newer()", reason="TMA requires Hopper or newer")
@pytest.mark.parametrize("num_ctas", [1, 2])
def test_gsan_profile_tma_observes_whole_rows(num_ctas, fresh_knobs, with_allocator):
    with AllocationProfile() as profile:
        source = torch.empty((32, 64), device="cuda")
        output = torch.empty_like(source)
        source_desc = TensorDescriptor.from_tensor(source, [32, 64])
        output_desc = TensorDescriptor.from_tensor(output, [32, 64])
        profile.track(source, name="source")
        profile.track(output, name="output")
        _fill[(2, )](source, source.numel())
        kernel = _tma_copy[(1, )](source_desc, output_desc, num_ctas=num_ctas)
    assert "__triton_gsan_load_tensor_desc" in kernel.asm["llir"]
    torch.testing.assert_close(output, torch.full_like(output, 17))
    for name in ("source", "output"):
        row = _allocation(profile, name)
        assert row["max_granularity"] == 16
        assert row["write_once_candidate"]
        assert row["bytes_written"] == source.numel() * source.element_size()


@pytest.mark.parametrize("replays", [1, 2])
def test_gsan_profile_cuda_graph_replay_detects_overwrite(replays, fresh_knobs):
    with AllocationProfile() as profile:
        output = torch.empty(1024, device="cuda")
        profile.track(output, name="output")
        kernel = _fill.warmup(output, 1024, grid=(1, ))
        kernel._init_handles()
        profile.prepare()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _fill[(1, )](output, 1024)
        for _ in range(replays):
            graph.replay()
    assert _allocation(profile, "output")["write_once_candidate"] == (replays == 1)
    torch.testing.assert_close(output, torch.full_like(output, 17))


@pytest.mark.skipif("not is_blackwell()", reason="TMA gather/scatter requires Blackwell")
def test_gsan_profile_tma_gather_scatter(fresh_knobs, with_allocator):
    with AllocationProfile() as profile:
        source = torch.empty((16, 32), dtype=torch.int32, device="cuda")
        output = torch.empty_like(source)
        source_desc = TensorDescriptor.from_tensor(source, [1, 32])
        output_desc = TensorDescriptor.from_tensor(output, [1, 32])
        profile.track(source, name="source")
        profile.track(output, name="output")
        _fill[(1, )](source, source.numel())
        _tma_indexed[(1, )](source_desc, output_desc)
    row = _allocation(profile, "output")
    assert row["write_once_candidate"]
    assert row["max_granularity"] == 16
    assert row["bytes_written"] == 8 * 32 * 4
    torch.testing.assert_close(output[::2], torch.full_like(output[::2], 17))


@pytest.mark.skipif("not is_hopper_or_newer()", reason="TMA requires Hopper or newer")
def test_gsan_profile_tma_reduction_is_atomic(fresh_knobs, with_allocator):
    with AllocationProfile() as profile:
        output = torch.zeros((8, 32), dtype=torch.int32, device="cuda")
        desc = TensorDescriptor.from_tensor(output, [8, 32])
        profile.track(output, name="output")
        _tma_reduce[(1, )](desc)
    row = _allocation(profile, "output")
    assert row["max_granularity"] == 4
    assert not row["write_once_candidate"]
    assert "atomic accesses" in row["reasons"]
    assert row["cells_read"] == 0
    torch.testing.assert_close(output, torch.ones_like(output))


@pytest.mark.skipif("not is_hopper_or_newer()", reason="warp specialization requires Hopper or newer")
def test_gsan_profile_warp_specialized_partitions(fresh_knobs, with_allocator):
    with AllocationProfile() as profile:
        output = torch.empty(256, dtype=torch.int32, device="cuda")
        profile.track(output, name="output")
        _ws_fill[(1, )](output)
    row = _allocation(profile, "output")
    assert row["write_once_candidate"]
    assert row["bytes_written"] == 1024
    torch.testing.assert_close(output, torch.full_like(output, 17))


def test_gsan_profile_registration_and_context_guards(fresh_knobs):
    with AllocationProfile() as profile:
        output = torch.empty(16, device="cuda")
        profile.track(output, name="output")
        with pytest.raises(RuntimeError, match="nested"):
            with AllocationProfile():
                pass
        with pytest.raises(RuntimeError, match="another AllocationProfile"):
            AllocationProfile().prepare()
        with pytest.raises(ValueError, match="overlapping"):
            profile.track_pointer(output.data_ptr() + 16, 16, owner=output, name="overlap")
        profile.prepare()
        profile.track(output, name="late")
        with pytest.raises(RuntimeError, match="before prepare"):
            profile.track(output, name="shared-late", allocation_id="late")
    triton.knobs.compilation.instrumentation_mode = "gsan-profile"
    with pytest.raises(RuntimeError, match="active AllocationProfile"):
        _fill[(1, )](output, 16)


@pytest.mark.parametrize("count", [1024, 1001])
def test_gsan_profile_caching_allocator_reuses_retired_storage(count, fresh_knobs):
    import weakref
    with AllocationProfile() as profile:

        addresses = []
        readback = torch.empty(count, dtype=torch.int32, device="cuda")
        read_counts = [count, 17, 1]
        for i, read_count in enumerate(read_counts):
            output = torch.empty(count, dtype=torch.int32, device="cuda")
            addresses.append(output.data_ptr())
            storage_ref = weakref.ref(output.untyped_storage())
            profile.track(output, name=f"lifetime{i}")
            _fill[(1, )](output, count)
            _copy[(1, )](output, readback, read_count)
            del output
            assert storage_ref() is None
    assert len(set(addresses)) == 1, "test must exercise actual cache reuse"
    rows = profile.report()["allocations"]
    assert len(rows) == 3
    assert all(row["write_once_candidate"] for row in rows)
    assert all(row["bytes_written"] == count * 4 for row in rows)
    assert all(row["cells_with_repeated_writes"] == 0 for row in rows)
    assert all(row["cells_read_before_write"] == 0 for row in rows)
    assert [row["cells_read"] for row in rows] == [triton.cdiv(n * 4, 16) for n in read_counts]
    assert profile.report() == profile.report()


def test_gsan_profile_reuse_does_not_inherit_initialization(fresh_knobs):
    import weakref
    with AllocationProfile() as profile:

        output = torch.empty(1024, dtype=torch.int32, device="cuda")
        profile.track(output, name="output")
        first = torch.empty_like(output)
        profile.track(first, name="first")
        address = first.data_ptr()
        _fill[(1, )](first, 1024)
        first_ref = weakref.ref(first)
        del first
        assert first_ref() is None
        second = torch.empty_like(output)
        assert second.data_ptr() == address
        profile.track(second, name="second")
        _copy[(1, )](second, output, 1024)
        del second
    assert _allocation(profile, "first")["write_once_candidate"]
    row = _allocation(profile, "second")
    assert row["bytes_written"] == 0
    assert row["cells_read_before_write"] == 256
    assert row["cells_read"] == 256
    assert not row["write_once_candidate"]


def test_gsan_profile_reuse_after_untracked_storage(fresh_knobs):
    with AllocationProfile() as profile:
        output = torch.empty(1024, dtype=torch.int32, device="cuda")
        profile.track(output, name="output")
        temporary = torch.empty_like(output)
        address = temporary.data_ptr()
        _fill[(1, )](temporary, 1024)
        del temporary
        temporary = torch.empty_like(output)
        assert temporary.data_ptr() == address
        profile.track(temporary, name="new")
        _copy[(1, )](temporary, output, 1024)
        del temporary
    assert _allocation(profile, "new")["cells_read_before_write"] == 256


def test_gsan_profile_retirement_waits_for_storage_aliases(fresh_knobs):
    import weakref
    with AllocationProfile() as profile:

        tensor = torch.empty(1024, dtype=torch.int32, device="cuda")
        alias = tensor.detach()
        view = alias[:512]
        storage_ref = weakref.ref(tensor.untyped_storage())
        profile.track(tensor, name="tensor")
        profile.track(view, name="view")
        _fill[(1, )](tensor, 1024)
        del tensor, alias
        assert storage_ref() is not None
        _fill[(1, )](view, 512)
        del view
        assert storage_ref() is None
    rows = profile.report()["allocations"]
    assert len(rows) == 1
    assert rows[0]["names"] == ["tensor", "view"]
    assert rows[0]["cells_with_repeated_writes"] == 128


def test_gsan_profile_retirement_waits_for_cuda_streams(fresh_knobs):
    with AllocationProfile() as profile:
        stream = torch.cuda.Stream()
        output = torch.empty(1024, dtype=torch.int32, device="cuda")
        profile.track(output, name="output")
        profile.prepare()
        _fill.warmup(output, 1024, grid=(1, ))
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10_000_000)
            _fill[(1, )](output, 1024)
            output.record_stream(stream)
        del output
    row = _allocation(profile, "output")
    assert row["write_once_candidate"]
    assert row["bytes_written"] == 4096


def test_gsan_profile_retirement_on_another_host_thread(fresh_knobs):
    from concurrent.futures import ThreadPoolExecutor
    import weakref
    with AllocationProfile() as profile:

        holders = [torch.empty(1024, dtype=torch.int32, device="cuda")]
        storage_ref = weakref.ref(holders[0].untyped_storage())
        profile.track(holders[0], name="output")
        _fill[(1, )](holders[0], 1024)
        with ThreadPoolExecutor(1) as executor:
            executor.submit(holders.clear).result(timeout=30)
        assert storage_ref() is None
        assert triton.knobs.compilation.instrumentation_mode == "gsan-profile"
    row = _allocation(profile, "output")
    assert row["write_once_candidate"]
    assert row["bytes_written"] == 4096


@pytest.mark.parametrize("side_stream", [False, True])
def test_gsan_profile_retirement_does_not_wait_for_gpu(side_stream, fresh_knobs):
    from concurrent.futures import ThreadPoolExecutor
    with AllocationProfile() as profile:

        gate = torch.zeros((), dtype=torch.int32, device="cuda")
        work = torch.cuda.Stream()
        allocation_stream = torch.cuda.current_stream() if side_stream else work
        with torch.cuda.stream(allocation_stream):
            holders = [torch.empty(1024, dtype=torch.int32, device="cuda")]
        address = holders[0].data_ptr()
        profile.track(holders[0], name="output")
        profile.prepare()
        _wait_and_fill.warmup(holders[0], gate, grid=(1, ))._init_handles()
        _fill.warmup(holders[0], 1024, grid=(1, ))._init_handles()
        with torch.cuda.stream(work):
            _wait_and_fill[(1, )](holders[0], gate)
            if side_stream:
                holders[0].record_stream(work)
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(holders.clear)
            try:
                # The GPU cannot complete until this host thread opens the gate.
                # A device/stream synchronization in deallocation times out here.
                future.result(timeout=5)
                if not side_stream:
                    with torch.cuda.stream(work):
                        replacement = torch.empty(1024, dtype=torch.int32, device="cuda")
                        assert replacement.data_ptr() == address
                        _fill[(1, )](replacement, 1024)
                        _fill[(1, )](replacement, 1024)
            finally:
                gate.fill_(1)
    row = _allocation(profile, "output")
    assert row["write_once_candidate"]
    assert row["bytes_written"] == 4096
    assert row["cells_with_repeated_writes"] == 0


def test_gsan_profile_retirement_before_prepare(fresh_knobs):
    with AllocationProfile() as profile:
        output = torch.empty(32, device="cuda")
        profile.track(output, name="output")
        del output
    row = _allocation(profile, "output")
    assert "observed" not in row
    assert row["cells_read"] == 0
    assert row["bytes_written"] == 0


def test_gsan_profile_failed_allocator_reset_preserves_retirement(fresh_knobs):
    from triton.experimental.gsan import profile as api

    with AllocationProfile():
        output = torch.empty(1024, dtype=torch.int32, device="cuda")
    address = output.data_ptr()
    with pytest.raises(RuntimeError, match="live allocations"):
        api.reset()
    with AllocationProfile() as profile:
        profile.track(output, name="first")
        _fill[(1, )](output, 1024)
        del output
        output = torch.empty(1024, dtype=torch.int32, device="cuda")
        assert output.data_ptr() == address
        profile.track(output, name="second")
        _fill[(1, )](output, 1024)
        _fill[(1, )](output, 1024)
        del output
    assert _allocation(profile, "first")["write_once_candidate"]
    assert _allocation(profile, "second")["cells_with_repeated_writes"] == 256


def test_gsan_profile_finished_context_releases_storage(fresh_knobs):
    import weakref
    with AllocationProfile() as first:

        output = torch.empty(32, device="cuda")
        storage_ref = weakref.ref(output.untyped_storage())
        first.track(output, name="first")
        _fill[(1, )](output, 32)
    with AllocationProfile() as second:
        second.track(output, name="second")
        _fill[(1, )](output, 32)
        del output
    assert storage_ref() is None
    assert _allocation(first, "first")["write_once_candidate"]
    assert _allocation(second, "second")["write_once_candidate"]


@pytest.mark.parametrize("owner_kind", ["pointer", "cycle", "autograd"])
def test_gsan_profile_retirement_follows_actual_owner(owner_kind, fresh_knobs):
    import gc
    import weakref
    with AllocationProfile() as profile:

        output = torch.empty(32, device="cuda", requires_grad=owner_kind == "autograd")
        storage_ref = weakref.ref(output.untyped_storage())
        if owner_kind == "pointer":
            profile.track_pointer(output.data_ptr(), output.nbytes, owner=output, name="output")
        else:
            profile.track(output, name="output")
        _fill[(1, )](output, 32)
        if owner_kind == "cycle":
            output.cycle = output
        elif owner_kind == "autograd":
            saved = output.square()
        del output
        if owner_kind == "autograd":
            assert storage_ref() is not None
            del saved
        gc.collect()
        assert storage_ref() is None
    assert _allocation(profile, "output")["write_once_candidate"]


def test_gsan_profile_retirement_failure_is_reported(monkeypatch, fresh_knobs):

    def fail(*args):
        raise RuntimeError("simulated summary failure")

    with monkeypatch.context() as patch:
        with AllocationProfile() as profile:
            output = torch.empty(32, device="cuda")
            profile.track(output, name="output")
            _fill[(1, )](output, 32)
            del output
            patch.setattr(type(profile._hooks), "finalize", fail)
    row = _allocation(profile, "output")
    assert not row["write_once_candidate"]
    assert row["max_granularity"] is None
    assert any("simulated summary failure" in reason for reason in row["reasons"])


def test_gsan_profile_retirement_during_capture_is_reported(fresh_knobs):
    with AllocationProfile() as profile:
        output = torch.empty(32, device="cuda")
        profile.track(output, name="output")
        profile.prepare()
        kernel = _fill.warmup(output, 32, grid=(1, ))
        kernel._init_handles()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _fill[(1, )](output, 32)
            del output
        graph.replay()
    row = _allocation(profile, "output")
    assert not row["write_once_candidate"]
    assert row["max_granularity"] is None
    assert any("capture" in reason for reason in row["reasons"])


@pytest.mark.parametrize("dtype", [torch.float16, torch.int32, torch.int64])
def test_gsan_profile_atomic_object_width(dtype, fresh_knobs):
    with AllocationProfile() as profile:
        output = torch.zeros(32, dtype=dtype, device="cuda")
        profile.track(output, name="output")
        _atomic[(1, )](output)
    row = _allocation(profile, "output")
    assert row["max_granularity"] == output.element_size()
    assert not row["write_once_candidate"]


@pytest.mark.parametrize("operation", ["load", "store"])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.int32, torch.float64])
def test_gsan_profile_vectorized_atomic_object_width(operation, dtype, fresh_knobs):

    @gluon.jit
    def kernel(source, output, STORE: gl.constexpr):
        offsets = gl.arange(0, 256, layout=gl.BlockedLayout([8], [32], [1], [0]))
        if STORE:
            values = gl.load(source + offsets)
            gl.atomic_store(output + offsets, values, sem="relaxed")
        else:
            values = gl.atomic_load(source + offsets, sem="relaxed")
            gl.store(output + offsets, values)

    ordinary = torch.full((256, ), 17 if operation == "store" else 0, dtype=dtype, device="cuda")
    with AllocationProfile() as profile:
        target = torch.full_like(ordinary, 0 if operation == "store" else 17)
        profile.track(target, name="atomic")
        source, output = (ordinary, target) if operation == "store" else (target, ordinary)
        compiled = kernel[(1, )](source, output, operation == "store", num_warps=1)
    row = _allocation(profile, "atomic")
    assert row["max_granularity"] == target.element_size()
    assert not row["write_once_candidate"]
    assert "atomic accesses" in row["reasons"]
    assert row["cells_read"] == 0
    opcode = "st" if operation == "store" else "ld"
    suffix = "b64" if dtype == torch.float64 else "v2.b32"
    assert f"{opcode}.relaxed.gpu.global.{suffix}" in compiled.asm["ptx"]
    torch.testing.assert_close(output, torch.full_like(output, 17))


@pytest.mark.parametrize("address,nbytes", [(-16, 4), (16, -1), (1, 4), (0, 4), (2**63, 0), (2**63 - 16, 32), (16.0, 4),
                                            (16, True)])
def test_gsan_profile_invalid_range(address, nbytes, fresh_knobs):
    with AllocationProfile() as profile:
        with pytest.raises(ValueError, match="storage"):
            profile.track_pointer(address, nbytes, owner=object(), name="invalid")


def test_gsan_profile_exception_suppresses_recommendations(fresh_knobs):
    with pytest.raises(RuntimeError, match="workload failed"):
        with AllocationProfile() as profile:
            output = torch.empty(32, device="cuda")
            profile.track(output, name="output")
            _fill[(1, )](output, 32)
            raise RuntimeError("workload failed")
    report = profile.report()
    assert report["warnings"] == ["profile context exited with an exception"]
    assert not report["allocations"][0]["write_once_candidate"]
    assert report["allocations"][0]["max_granularity"] is None
    assert triton.knobs.compilation.instrumentation_mode == ""
    with pytest.raises(RuntimeError, match="re-entered"):
        with profile:
            pass
    with AllocationProfile():
        pass


def test_gsan_profile_report_requires_context_exit(fresh_knobs):
    profile = AllocationProfile()
    with pytest.raises(RuntimeError, match="after the AllocationProfile context exits"):
        profile.report()
    assert not profile._prepared
    with profile:
        output = torch.empty(32, device="cuda")
        profile.track(output, name="output")
        with pytest.raises(RuntimeError, match="after the AllocationProfile context exits"):
            profile.report()
        assert not profile._prepared
        _fill[(1, )](output, 32)
        with pytest.raises(RuntimeError, match="after the AllocationProfile context exits"):
            profile.report()
        assert triton.knobs.compilation.instrumentation_mode == "gsan-profile"
        _fill[(1, )](output, 32)
    row = _allocation(profile, "output")
    assert not row["write_once_candidate"]
    assert row["cells_with_repeated_writes"] == 8
    report = profile.report()
    del output
    assert profile.report() == report


def test_gsan_profile_stream_publication(fresh_knobs):
    with AllocationProfile() as profile:
        output = torch.empty(1024, device="cuda")
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            profile.track(output, name="output")
        profile.prepare()
        with torch.cuda.stream(consumer):
            _fill[(1, )](output, 1024)
    assert _allocation(profile, "output")["write_once_candidate"]
    torch.testing.assert_close(output, torch.full_like(output, 17))


def test_gsan_profile_thread_guard(fresh_knobs):
    from concurrent.futures import ThreadPoolExecutor
    with AllocationProfile() as profile:
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(profile.prepare)
            with pytest.raises(RuntimeError, match="creating host thread"):
                future.result()


@pytest.mark.skipif("torch.cuda.device_count() < 2", reason="requires two CUDA devices")
def test_gsan_profile_multiple_devices(fresh_knobs):
    with AllocationProfile(devices=[0, 1]) as profile:
        outputs = []
        for i in range(2):
            with torch.cuda.device(i):
                outputs.append(torch.empty(32, device=f"cuda:{i}"))
        for i, output in enumerate(outputs):
            profile.track(output, name=f"output{i}")
        for i, output in enumerate(outputs):
            with torch.cuda.device(i):
                _fill[(1, )](output, 32)
    for i, output in enumerate(outputs):
        row = _allocation(profile, f"output{i}")
        assert row["write_once_candidate"]
        assert row["mappings"] == [{"device": i, "address": output.data_ptr(), "rank": 0}]
        torch.testing.assert_close(output, torch.full_like(output, 17))


@pytest.mark.skipif("torch.cuda.device_count() < 2", reason="requires two CUDA devices")
@pytest.mark.parametrize("overlap", [False, True])
def test_gsan_profile_peer_writes_share_shadow(overlap, fresh_knobs):
    from contextlib import ExitStack
    import os
    from triton.experimental.gsan import _profile_allocator as _allocator
    from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr

    if not torch.cuda.can_device_access_peer(1, 0):
        pytest.skip("requires peer access")
    with ExitStack() as stack:
        ptr = _allocator.malloc(4096, 0)
        stack.callback(_allocator.free_allocation, ptr, 0)
        real, shadow, size = _allocator.export_allocation_handles(ptr,
                                                                  _allocator.ShareableHandleType.POSIX_FILE_DESCRIPTOR)
        stack.callback(os.close, real)
        stack.callback(os.close, shadow)
        with torch.cuda.device(1):
            peer = _allocator.import_allocation_handles(real, shadow, size, 1,
                                                        _allocator.ShareableHandleType.POSIX_FILE_DESCRIPTOR)

        def free_peer():
            with torch.cuda.device(1):
                _allocator.free_allocation(peer, 1)

        stack.callback(free_peer)
        output = uint8_cuda_tensor_from_ptr(ptr, 4096, 0).view(torch.int32)
        alias = uint8_cuda_tensor_from_ptr(peer, 4096, 1).view(torch.int32)
        with AllocationProfile(devices=[0, 1]) as profile:
            profile.track(output, name="owner", allocation_id="shared")
            profile.track(alias, name="peer", allocation_id="shared")
            profile.prepare()
            with torch.cuda.device(0):
                _fill[(1, )](output, 512)
                torch.cuda.synchronize()
            with torch.cuda.device(1):
                _fill[(1, )](alias if overlap else alias[512:], 512)
        report = profile.report()
        assert not report["warnings"]
        assert len(report["allocations"]) == 1
        row = report["allocations"][0]
        assert row["write_once_candidate"] == (not overlap)
        assert row["bytes_written"] == (512 if overlap else 1024) * 4
        assert row["cells_with_repeated_writes"] == (128 if overlap else 0)
        assert row["profiling_metadata_bytes"] == 1024
        count = 512 if overlap else 1024
        torch.testing.assert_close(output[:count], torch.full_like(output[:count], 17))


def _distributed_checks(rank, world_size, init_method, backend, *, shared=False):
    from datetime import timedelta
    import os
    import torch.distributed as dist

    device = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(device)
    timeout = timedelta(seconds=180)
    store, _, _ = next(dist.rendezvous(init_method, rank=rank, world_size=world_size, timeout=timeout))
    # torchrun reuses its rendezvous store across pytest cases. Isolate each
    # process group from stale connection keys left by the previous case.
    store = dist.PrefixStore(f"gsan-profile/{backend}/{shared}/", store)
    dist.init_process_group(backend, rank=rank, world_size=world_size, store=store, timeout=timeout)
    import socket
    from triton.experimental.gsan import _profile_allocator as _allocator
    hosts = [None] * world_size
    dist.all_gather_object(hosts, socket.gethostname())
    handle_type = (_allocator.ShareableHandleType.POSIX_FILE_DESCRIPTOR
                   if len(set(hosts)) == 1 else _allocator.ShareableHandleType.FABRIC)
    _configure_profile_test_allocator(handle_type)
    try:
        if shared:
            _distributed_shared_checks(rank, world_size)
            return
        with AllocationProfile(group=dist.group.WORLD) as profile:
            output = torch.empty(32, dtype=torch.int32, device="cuda")
            # Identical names/virtual addresses on different ranks must not
            # merge unrelated storage.
            profile.track(output, name="rank-local")
            _fill[(1, )](output, 32)
        report = profile.report()
        assert not report["warnings"]
        assert len(report["allocations"]) == world_size
        assert all(row["write_once_candidate"] for row in report["allocations"])
        assert {row["mappings"][0]["rank"] for row in report["allocations"]} == set(range(world_size))
        # An unmanaged access on one rank must not suppress any rank's results.
        ordinary = torch.empty_like(output)
        with AllocationProfile(group=dist.group.WORLD) as profile:
            profile.track(output, name="known")
            profile.prepare()
            if rank == 0:
                _fill[(1, )](ordinary, 32)
            _fill[(1, )](output, 32)
        report = profile.report()
        assert report["warnings"] == []
        assert all(row["write_once_candidate"] for row in report["allocations"])
        # Rank-local allocations may come and go independently after the
        # initial collective prepare. Their retired reports must survive reuse.
        with AllocationProfile(group=dist.group.WORLD) as profile:
            profile.prepare()
            for i in range(rank + 1):
                temporary = torch.empty(1024, dtype=torch.int32, device="cuda")
                profile.track(temporary, name=f"temporary/{rank}/{i}")
                _fill[(1, )](temporary, 1024)
                del temporary
        rows = profile.report()["allocations"]
        assert len(rows) == world_size * (world_size + 1) // 2
        assert all(row["write_once_candidate"] and row["bytes_written"] == 4096 for row in rows)
        # Collective validation must fail on every rank before data collectives.
        with AllocationProfile(group=dist.group.WORLD) as profile:
            profile.track_pointer(output.data_ptr(), 16 * (rank + 1), owner=output, name="bad-size",
                                  allocation_id="shared")
        with pytest.raises(ValueError, match="same size across ranks"):
            profile.report()
        torch.testing.assert_close(output, torch.full_like(output, 17))
    finally:
        dist.destroy_process_group()


@lru_cache(None)
def _configure_profile_test_allocator(handle_type):
    from triton.experimental.gsan import _profile_allocator
    _profile_allocator.configure(handle_type=handle_type)


def _exchange_profile_ipc_handles(own_handle, rank, world_size):
    # Exercise actual SCM_RIGHTS transport even with independent torchrun
    # processes, whose multiprocessing authentication keys need not match.
    import socket
    import struct
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing.reduction import sendfds, recvfds
    import torch.distributed as dist
    import os

    handles = [None] * world_size
    handles[rank] = own_handle
    sizes = [None] * world_size
    dist.all_gather_object(sizes, own_handle[2])
    with tempfile.TemporaryDirectory() as directory, socket.socket(socket.AF_UNIX) as listener:
        path = directory + "/profile.sock"
        listener.bind(path)
        listener.listen(world_size)
        listener.settimeout(60)
        paths = [None] * world_size
        dist.all_gather_object(paths, path)

        def receive():
            for _ in range(world_size - 1):
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(60)
                    peer, = struct.unpack("i", connection.recv(4, socket.MSG_WAITALL))
                    real, shadow = recvfds(connection, 2)
                    handles[peer] = (real, shadow, sizes[peer])

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(receive)
                for peer in range(world_size):
                    if peer == rank:
                        continue
                    with socket.socket(socket.AF_UNIX) as connection:
                        connection.settimeout(60)
                        connection.connect(paths[peer])
                        connection.sendall(struct.pack("i", rank))
                        sendfds(connection, list(own_handle[:2]))
                future.result()
        except BaseException:
            for peer, handle in enumerate(handles):
                if peer != rank and handle is not None:
                    os.close(handle[0])
                    os.close(handle[1])
            raise
    return handles


@contextmanager
def _shared_profile_buffers(rank, world_size, *, cached=False):
    import os
    import socket
    import torch.distributed as dist
    from triton.experimental.gsan import _profile_allocator as allocator
    from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr

    hosts = [None] * world_size
    dist.all_gather_object(hosts, socket.gethostname())
    device = torch.cuda.current_device()
    ipc = len(set(hosts)) == 1
    kind = allocator.ShareableHandleType.POSIX_FILE_DESCRIPTOR if ipc else allocator.ShareableHandleType.FABRIC
    _configure_profile_test_allocator(kind)
    with ExitStack() as stack:
        own_tensor = None
        if cached:
            with AllocationProfile():
                own_tensor = torch.empty(1024, dtype=torch.int32, device="cuda")
            owned = own_tensor.data_ptr()
        else:
            owned = allocator.malloc(4096, device)
            stack.callback(allocator.free_allocation, owned, device)
        base, _, _ = allocator.get_allocation_profile_layout(owned)
        offsets = [None] * world_size
        dist.all_gather_object(offsets, owned - base)
        own_handle = allocator.export_allocation_handles(owned, kind)
        if ipc:
            stack.callback(os.close, own_handle[0])
            stack.callback(os.close, own_handle[1])
            handles = _exchange_profile_ipc_handles(own_handle, rank, world_size)
            for peer, (real, shadow, _) in enumerate(handles):
                if peer != rank:
                    stack.callback(os.close, real)
                    stack.callback(os.close, shadow)
        else:
            handles = [None] * world_size
            dist.all_gather_object(handles, own_handle)
        buffers = []
        for peer, (real, shadow, size) in enumerate(handles):
            pointer = owned
            if peer != rank:
                pointer = allocator.import_allocation_handles(real, shadow, size, device, kind)
                stack.callback(allocator.free_allocation, pointer, device)
                pointer += offsets[peer]
            buffers.append(own_tensor if cached and peer == rank else uint8_cuda_tensor_from_ptr(pointer, 4096, device).
                           view(torch.int32))
        own_tensor = None
        try:
            yield buffers
        finally:
            torch.cuda.synchronize()
            dist.barrier()


def _distributed_shared_checks(rank, world_size):
    import torch.distributed as dist
    with _shared_profile_buffers(rank, world_size) as buffers:
        for overlap in (False, True):
            with AllocationProfile(group=dist.group.WORLD) as profile:
                for i, buffer in enumerate(buffers):
                    profile.track(buffer, name=f"owner{i}", allocation_id=f"allocation/{i}")
                # Each rank writes a disjoint, aligned portion of allocation 0.
                profile.prepare()
                _fill[(1, )](buffers[0][rank * 128:], 128)
                torch.cuda.synchronize()
                dist.barrier()
                # Another rank must see the existing bits through shared shadow.
                if overlap and rank == 1:
                    _fill[(1, )](buffers[0], 128)
            report = profile.report()
            assert not report["warnings"]
            row = next(r for r in report["allocations"] if "owner0" in r["names"])
            assert row["write_once_candidate"] == (not overlap)
            assert row["bytes_written"] == world_size * 128 * 4
            assert row["cells_with_repeated_writes"] == (32 if overlap else 0)
            assert profile.report() == report
            torch.testing.assert_close(buffers[0][:world_size * 128],
                                       torch.full_like(buffers[0][:world_size * 128], 17))
        # Reads on multiple ranks count each shared physical cell only once.
        with AllocationProfile(group=dist.group.WORLD) as profile:
            for i, buffer in enumerate(buffers):
                profile.track(buffer, name=f"owner{i}", allocation_id=f"allocation/{i}")
            profile.prepare()
            if rank == 0:
                _fill[(1, )](buffers[0], 128)
            torch.cuda.synchronize()
            dist.barrier()
            scratch = torch.empty(128, dtype=torch.int32, device="cuda")
            _copy[(1, )](buffers[0], scratch, 128)
            if rank == 1:
                _copy[(1, )](buffers[0], buffers[1], 128)
        row = next(r for r in profile.report()["allocations"] if "owner0" in r["names"])
        assert row["write_once_candidate"]
        assert row["cells_read_before_write"] == 0
        assert row["cells_read"] == 32
        torch.testing.assert_close(buffers[1][:128], torch.full_like(buffers[1][:128], 17))
        # Different GPUs update different bits of the same physical word.
        # Every byte must survive, with no spurious repeated-write flag.
        with AllocationProfile(group=dist.group.WORLD) as profile:
            profile.track(buffers[0], name="interleaved", allocation_id="allocation/0")
            profile.prepare()
            _fill[(1, )](buffers[0].view(torch.uint8)[rank:], 256, STEP=world_size)
        row = _allocation(profile, "interleaved")
        assert row["write_once_candidate"]
        assert row["bytes_written"] == 256 * world_size
        assert row["cells_with_repeated_writes"] == 0

    # Exercise actual PyTorch owners with remote aliases. Dropping an imported
    # view must not reset its owner's shadow; dropping the owner must save it.
    with _shared_profile_buffers(rank, world_size, cached=True) as buffers:
        address = buffers[rank].data_ptr()
        with AllocationProfile(group=dist.group.WORLD) as profile:
            for i, buffer in enumerate(buffers):
                profile.track(buffer, name=f"cached/{i}", allocation_id=f"cached/{i}")
            del buffer
            profile.prepare()
            _fill[(1, )](buffers[(rank + 1) % world_size], 1024)
            for i in range(world_size):
                if i != rank:
                    buffers[i] = None
            torch.cuda.synchronize()
            dist.barrier()  # Remote accesses must finish before owner reuse.
            buffers[rank] = None
            replacement = torch.empty(1024, dtype=torch.int32, device="cuda")
            assert replacement.data_ptr() == address
            profile.track(replacement, name="replacement")
            _fill[(1, )](replacement, 1024)
            _fill[(1, )](replacement, 1024)
            del replacement
        rows = profile.report()["allocations"]
        shared_rows = [row for row in rows if row["allocation_id"] is not None]
        local_rows = [row for row in rows if row["allocation_id"] is None]
        assert len(shared_rows) == len(local_rows) == world_size
        assert all(row["write_once_candidate"] and row["bytes_written"] == 4096 for row in shared_rows)
        assert all(not row["write_once_candidate"] and row["cells_with_repeated_writes"] == 256 for row in local_rows)


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
@pytest.mark.parametrize("shared", [False, True])
def test_gsan_profile_distributed(backend, shared, fresh_knobs):
    import os
    import socket
    import time
    import torch.multiprocessing as mp

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        _distributed_checks(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), "env://", backend, shared=shared)
        return
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices or torchrun across nodes")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.spawn(_distributed_worker, args=(f"tcp://127.0.0.1:{port}", backend, shared), nprocs=2, join=False)
    deadline = time.monotonic() + 240
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("distributed profiler test timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
            process.join()


def _distributed_worker(rank, init_method, backend, shared):
    # Explicit bootstrap: module import leaves CUDA and the allocator untouched.
    from triton.experimental.gsan.profile import install_allocator
    install_allocator()
    _distributed_checks(rank, 2, init_method, backend, shared=shared)


def test_gsan_profile_identity_guards(fresh_knobs):
    with AllocationProfile() as profile:
        output = torch.empty(64, device="cuda")
        profile.track(output, name="output", allocation_id="storage")
        with pytest.raises(ValueError, match="same allocation_id"):
            profile.track(output, name="conflict", allocation_id="different")
        with pytest.raises(ValueError, match="same size"):
            profile.track_pointer(output.data_ptr() + 256, 16, owner=output, name="size-conflict",
                                  allocation_id="storage")
        profile.track(output[4:], name="view", allocation_id="storage")
        _fill[(1, )](output, 64)
    row = _allocation(profile, "view")
    assert row["names"] == ["output", "view"]
    assert row["write_once_candidate"]


@pytest.mark.skipif("torch.cuda.device_count() < 2", reason="requires two CUDA devices")
def test_gsan_profile_wrong_launch_device(fresh_knobs):
    with AllocationProfile(device=0):
        output = torch.empty(16, device="cuda:1")
        with torch.cuda.device(1), pytest.raises(RuntimeError, match="unregistered CUDA device"):
            _fill[(1, )](output, 16)


def test_gsan_profile_direct_allocator_scope_and_operation_outputs(fresh_knobs):
    from triton.experimental.gsan import profile as api
    from triton.experimental.gsan import _profile_allocator as allocator
    from concurrent.futures import ThreadPoolExecutor

    api.install_allocator()  # Repeated installation is harmless after startup.
    ordinary = torch.empty(32, device="cuda")
    assert allocator.get_allocation_profile_layout(ordinary.data_ptr()) is None
    with AllocationProfile() as profile:
        source = torch.empty_like(ordinary)
        source.fill_(1)
        output = source + source
        assert "tensor(" in repr(output)
        assert profile._hooks.owns(source.data_ptr())
        assert profile._hooks.owns(output.data_ptr())
        with ThreadPoolExecutor(1) as executor:
            other_thread = executor.submit(torch.empty_like, ordinary).result()
        assert allocator.get_allocation_profile_layout(other_thread.data_ptr()) is None
        profile.track(source, name="source")
        profile.track(output, name="output")
        _fill[(1, )](source, 32)
        _copy[(1, )](source, output, 32)
    after = torch.empty_like(ordinary)
    assert allocator.get_allocation_profile_layout(after.data_ptr()) is None
    assert _allocation(profile, "output")["write_once_candidate"]


@pytest.mark.parametrize("explicit_stream", [False, True])
@pytest.mark.parametrize("free_inside", [False, True])
def test_gsan_profile_raw_allocations_use_original_allocator(explicit_stream, free_inside, fresh_knobs):
    from triton.experimental.gsan import _profile_allocator as allocator
    from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr

    stream = torch.cuda.Stream() if explicit_stream else torch.cuda.current_stream()
    with ExitStack() as raw_lifetime:
        with AllocationProfile() as profile, torch.cuda.stream(stream):
            pointer = torch.cuda.memory.caching_allocator_alloc(1024, stream=stream if explicit_stream else None)
            raw_lifetime.callback(torch.cuda.memory.caching_allocator_delete, pointer)
            assert not profile._hooks.owns(pointer)
            assert allocator.get_allocation_profile_layout(pointer) is None
            raw = uint8_cuda_tensor_from_ptr(pointer, 1024, torch.cuda.current_device()).view(torch.int32)
            output = torch.empty_like(raw)
            assert profile._hooks.owns(output.data_ptr())
            profile.track(raw, name="raw")
            profile.track_pointer(pointer, 1024, owner=raw, name="raw-pointer")
            profile.track(output, name="tensor")
            _fill[(1, )](raw, 256)
            _fill[(1, )](raw, 256)
            _copy[(1, )](raw, output, 256)
            if free_inside:
                raw_lifetime.close()
        rows = profile.report()["allocations"]
        assert len(rows) == 1
        assert rows[0]["names"] == ["tensor"]
        assert rows[0]["write_once_candidate"]
        assert rows[0]["bytes_written"] == 1024
        torch.testing.assert_close(output, torch.full_like(output, 17))


def _check_import_does_not_install_allocator(cuda_initialized):
    import importlib
    from unittest.mock import patch
    import torch

    # Importing this module and the process helper must leave startup untouched.
    assert not torch.cuda.is_initialized()
    assert not is_installed()
    if cuda_initialized:
        torch.cuda.init()
    original = torch._C._cuda_getAllocator()
    with (
            patch.object(torch._C, "_cuda_changeCurrentAllocator", side_effect=AssertionError("allocator installed")),
            patch.object(torch._C, "_cuda_init", side_effect=AssertionError("CUDA initialized")),
            patch("triton.runtime.build.compile_module_from_file", side_effect=AssertionError("native module loaded")),
    ):
        from triton.experimental.gsan import profile
        # Multiprocessing already imported this test module and its dependencies.
        # Re-execute the profile module under the guards to test its import body.
        importlib.reload(profile)
        assert not profile.is_installed()
        assert torch.cuda.is_initialized() == cuda_initialized
        assert torch._C._cuda_getAllocator() is original
        try:
            profile.AllocationProfile()
        except RuntimeError as error:
            assert "call profile.install_allocator()" in str(error)
        else:
            raise AssertionError("context implicitly installed the allocator")


@pytest.mark.parametrize("cuda_initialized", [False, True])
def test_gsan_profile_import_does_not_install_allocator(cuda_initialized):
    result = run_in_process(_check_import_does_not_install_allocator, args=(cuda_initialized, ))
    assert result.exc is None, result


def _check_native_allocator_registration_is_explicit():
    import torch
    from triton.experimental.gsan import profile, _profile_allocator

    original = torch._C._cuda_getAllocator()
    module = _profile_allocator._load_module()
    assert not hasattr(module, "CUDAAllocator")
    assert not profile.is_installed()
    assert not torch.cuda.is_initialized()
    assert torch._C._cuda_getAllocator() is original
    module._register_cuda_allocator()
    allocator_type = module.CUDAAllocator
    module._register_cuda_allocator()
    assert module.CUDAAllocator is allocator_type
    assert torch._C._cuda_getAllocator() is original
    assert not torch.cuda.is_initialized()
    profile.install_allocator()
    installed = profile.get_allocator()
    assert isinstance(installed, allocator_type)
    assert torch._C._cuda_getAllocator() is installed
    assert not torch.cuda.is_initialized()
    profile.install_allocator()
    assert profile.get_allocator() is installed


def test_gsan_profile_native_allocator_registration_is_explicit():
    result = run_in_process(_check_native_allocator_registration_is_explicit)
    assert result.exc is None, result


def _check_direct_allocator_late_install_rejected():
    import torch
    from triton.experimental.gsan.profile import install_allocator
    torch.cuda.init()
    try:
        install_allocator()
    except RuntimeError as error:
        assert "before PyTorch initializes CUDA" in str(error)
    else:
        raise AssertionError("late installation succeeded")


def test_gsan_profile_direct_allocator_late_install_rejected():
    result = run_in_process(_check_direct_allocator_late_install_rejected)
    assert result.exc is None, result


def test_gsan_profile_direct_allocator_explicit_pool_and_coalescing(fresh_knobs):
    from triton.experimental.gsan import _profile_allocator as allocator

    torch.cuda.empty_cache()
    ordinary_pool = torch.cuda.MemPool()
    with AllocationProfile() as profile:
        with torch.cuda.use_mem_pool(ordinary_pool):
            ordinary = torch.empty(1024, device="cuda")
        assert allocator.get_allocation_profile_layout(ordinary.data_ptr()) is None
        first = torch.empty(1024, dtype=torch.int32, device="cuda")
        second = torch.empty_like(first)
        address = first.data_ptr()
        profile.track(first, name="first")
        profile.track(second, name="second")
        _fill[(1, )](first, 1024)
        _fill[(1, )](second, 1024)
        del first, second
        combined = torch.empty(2048, dtype=torch.int32, device="cuda")
        assert combined.data_ptr() == address
        profile.track(combined, name="combined")
        _fill[(2, )](combined, 2048)
        del combined
    rows = profile.report()["allocations"]
    assert len(rows) == 3
    assert all(row["write_once_candidate"] for row in rows)


def _check_direct_allocator_reset_keeps_installation():
    import torch
    from triton.experimental.gsan import profile
    profile.install_allocator()
    for _ in range(2):
        with profile.AllocationProfile():
            tensor = torch.empty(1024, device="cuda")
        del tensor
        profile.reset()
        assert profile.is_installed()
        assert not profile.has_live_allocations()


def test_gsan_profile_direct_allocator_reset_keeps_installation():
    result = run_in_process(_check_direct_allocator_reset_keeps_installation)
    assert result.exc is None, result


@pytest.mark.skipif("torch.cuda.device_count() < 2", reason="requires two CUDA devices")
def test_gsan_profile_direct_allocator_cross_device_copy(fresh_knobs):
    with AllocationProfile(devices=[0, 1]):
        source = torch.arange(1024, device="cuda:0")
        destination = source.to("cuda:1")
        torch.testing.assert_close(destination.cpu(), source.cpu())


def test_gsan_profile_direct_allocator_late_registration_in_new_context(fresh_knobs):
    with AllocationProfile():
        output = torch.empty(1024, dtype=torch.int32, device="cuda")
        _fill[(1, )](output, 1024)
    with AllocationProfile() as profile:
        profile.prepare()
        profile.track(output, name="output")
        _fill[(1, )](output, 1024)
    assert _allocation(profile, "output")["write_once_candidate"]
