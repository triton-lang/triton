# ruff: noqa: E402
import hip

hip.hip.hipInit(0)

import pytest
import subprocess
import sys
import os

from triton._internal_testing import is_hip_gfx1250


def _run_consan_subprocess(test_name, *args, timeout=120):
    """Run a ConSan test kernel in a subprocess and return captured stderr.

    The subprocess pipe captures all stderr output including messages
    flushed during process exit.
    """
    helper_script = os.path.join(os.path.dirname(__file__), "gfx1250_consan_helper.py")
    proc = subprocess.Popen(
        [sys.executable, helper_script, test_name] + [str(a) for a in args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, stderr = proc.communicate()
    return stderr.decode(errors="replace")


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_store_wait_load(FAILURE):
    """
    Producer stores to smem and signals ready_bar. Consumer conditionally waits
    on ready_bar before reading smem. FAILURE=True skips the wait, triggering
    an error.
    """
    stderr_str = _run_consan_subprocess("ws_store_wait_load_failure", FAILURE)

    if FAILURE:
        assert "Buffer being accessed has outstanding writes" in stderr_str, \
            f"Expected 'Buffer being accessed has outstanding writes' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_load_wait_store(FAILURE):
    """
    Worker reads from smem and signals bar[0]. Default partition conditionally
    waits on bar[0] before writing to smem. FAILURE=True skips the wait,
    triggering an error.
    """
    stderr_str = _run_consan_subprocess("ws_load_wait_store_failure", FAILURE)

    if FAILURE:
        assert "Buffer being accessed has outstanding reads" in stderr_str, \
            f"Expected 'Buffer being accessed has outstanding reads' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
def test_deadlock_two_partitions():
    """
    Verifies that ConSan detects a deadlock when two warp-specialize partitions
    each wait on separate barriers that nobody ever arrives on.
    """
    stderr_str = _run_consan_subprocess("deadlock_two_partitions")

    assert "Deadlock detected" in stderr_str, f"Expected 'Deadlock detected' in stderr, got:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
def test_deadlock_overarrival():
    """
    Verifies that ConSan detects a deadlock caused by over-arrival on an mbarrier.
    Each thread arrives twice but the barrier only expects one arrival per thread.
    """
    stderr_str = _run_consan_subprocess("deadlock_overarrival")

    assert "Deadlock detected" in stderr_str, f"Expected 'Deadlock detected' in stderr, got:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
def test_deadlock_underarrival():
    """
    Verifies that ConSan detects a deadlock caused by under-arrival on mbarriers.
    Two partitions each arrive on the other's barrier but the init count requires
    arrivals from both partitions.
    """
    stderr_str = _run_consan_subprocess("deadlock_underarrival")

    assert "Deadlock detected" in stderr_str, f"Expected 'Deadlock detected' in stderr, got:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
def test_deadlock_different_phases():
    """
    Positive test: verifies ConSan does NOT report a false deadlock when two
    partitions wait on different phases of the same barrier.
    """
    stderr_str = _run_consan_subprocess("deadlock_different_phases")

    assert "device assertion failed" not in stderr_str, \
        f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
def test_barrier_underflow():
    """
    Verifies that ConSan detects a barrier arrive underflow when each thread
    arrives with count=3 on a barrier initialized for count=1 per thread
    (4*WARP_SIZE=128). Since 128 % 3 != 0, the 43rd arrival triggers underflow.
    """
    stderr_str = _run_consan_subprocess("barrier_underflow")

    assert "Barrier arrive underflow" in stderr_str, \
        f"Expected 'Barrier arrive underflow' in stderr, got:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("MISSING_BAR", [True, False])
@pytest.mark.parametrize("OVERLAP", [True, False])
def test_aliasing_shared_visibility(MISSING_BAR, OVERLAP):
    """
    Verifies that ConSan detects unsynchronized access when aliased shared memory
    regions overlap and the reader skips the barrier synchronization.

    Only the MISSING_BAR=True, OVERLAP=True combination should trigger a
    ConSan assertion. All other combinations are clean (no overlap or barrier
    is present).
    """
    stderr_str = _run_consan_subprocess("aliasing_shared_visibility", MISSING_BAR, OVERLAP)

    if MISSING_BAR and OVERLAP:
        has_violation = ("outstanding writes" in stderr_str or "outstanding reads" in stderr_str)
        assert has_violation, \
            f"Expected 'outstanding writes' or 'outstanding reads' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("MISSING_BAR", ["none", "1", "2"])
def test_ws_two_loads_two_bars(MISSING_BAR):
    """
    Verifies that ConSan detects unsynchronized access when two reader
    partitions load from the same buffer and a writer partition skips
    one of the barrier waits.

    MISSING_BAR="1" or "2" should trigger a ConSan assertion.
    "none" should be clean.
    """
    stderr_str = _run_consan_subprocess("ws_two_loads_two_bars", MISSING_BAR)

    if MISSING_BAR != "none":
        has_violation = ("outstanding writes" in stderr_str or "outstanding reads" in stderr_str)
        assert has_violation, \
            f"Expected 'outstanding writes' or 'outstanding reads' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_two_loads_one_bar(FAILURE):
    """
    Verifies that ConSan detects unsynchronized access when two reader
    partitions load from the same buffer via a single shared barrier and
    the writer partition skips the wait.

    FAILURE=True should trigger "Buffer being accessed has outstanding
    reads". False should be clean.
    """
    stderr_str = _run_consan_subprocess("ws_two_loads_one_bar", FAILURE)

    if FAILURE:
        assert "Buffer being accessed has outstanding reads" in stderr_str, \
            f"Expected 'Buffer being accessed has outstanding reads' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("MISSING_BAR", ["none", "0", "1"])
def test_ws_two_loads_two_bars_loop(MISSING_BAR):
    """
    Verifies that ConSan detects unsynchronized access in a looped
    producer-consumer pattern with two reader partitions and one writer
    partition, using barriers and phase tracking.
    """
    stderr_str = _run_consan_subprocess("ws_two_loads_two_bars_loop", MISSING_BAR)

    if MISSING_BAR == "0":
        assert "Buffer being accessed has outstanding reads" in stderr_str, \
            f"Expected 'outstanding reads' in stderr, got:\n{stderr_str}"
    elif MISSING_BAR == "1":
        assert "Buffer being accessed has outstanding reads" in stderr_str, \
            f"Expected 'outstanding reads' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_ws_load_ordering(FAILURE):
    """
    Verifies that ConSan detects unsynchronized access when ws_1 loads
    from smem[1] (protected by bar[1]) after only waiting on bar[0].

    FAILURE=True should trigger "Buffer being accessed has outstanding
    writes". False should be clean.
    """
    stderr_str = _run_consan_subprocess("ws_load_ordering", FAILURE)

    if FAILURE:
        assert "Buffer being accessed has outstanding writes" in stderr_str, \
            f"Expected 'outstanding writes' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("MISSING_BAR", ["none", "1", "2"])
def test_ws_different_warp_sizes(MISSING_BAR):
    """
    Verifies that ConSan detects unsynchronized access when three partitions
    with different warp counts (4, 2, 8) share a buffer.  The writer partition
    (ws_2, 8 warps) skips one of two barrier waits, storing to smem[0] while
    readers still have outstanding reads.
    """
    stderr_str = _run_consan_subprocess("ws_different_warp_sizes", MISSING_BAR)

    if MISSING_BAR != "none":
        assert "Buffer being accessed has outstanding reads" in stderr_str, \
            f"Expected 'Buffer being accessed has outstanding reads' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_async_tdm_kernel(FAILURE):
    """
    Verifies that ConSan detects outstanding writes when shared memory is
    read before a TDM async load (global-to-shared copy) completes.

    FAILURE=True skips the mbarrier.wait before the load, triggering
    an error.
    """
    stderr_str = _run_consan_subprocess("async_tdm_kernel", FAILURE)

    if FAILURE:
        assert "Buffer being accessed has outstanding writes" in stderr_str, \
            f"Expected 'Buffer being accessed has outstanding writes' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_async_tdm_kernel_2bufs_1bar(FAILURE):
    """
    Two TDM async loads sharing one mbarrier. ConSan must detect
    outstanding writes when the wait is skipped (FAILURE=True).
    """
    stderr_str = _run_consan_subprocess("async_tdm_kernel_2bufs_1bar", FAILURE)

    if FAILURE:
        assert "Buffer being accessed has outstanding writes" in stderr_str, \
            f"Expected 'Buffer being accessed has outstanding writes' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_tdm_interleave_kernel(FAILURE):
    """
    Two TDM loads with separate barriers, interleaved access.
    FAILURE=True skips wait on bar[1], so reading smem[1] has outstanding writes.
    """
    stderr_str = _run_consan_subprocess("tdm_interleave_kernel", FAILURE)

    if FAILURE:
        assert "Buffer being accessed has outstanding writes" in stderr_str, \
            f"Expected 'Buffer being accessed has outstanding writes' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_consan_async_copy(FAILURE):
    """
    Async copy with commit_group/wait_group. FAILURE=True uses wait_group(2)
    instead of wait_group(1), so smem[0] is reused before its copy completes.
    """
    stderr_str = _run_consan_subprocess("async_copy_kernel", FAILURE)

    if FAILURE:
        assert "Accessing buffer with pending access" in stderr_str, \
            f"Expected 'Accessing buffer with pending access' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_consan_tdm_store(FAILURE):
    """
    TDM async store with partial wait. FAILURE=True skips the second
    async_wait(0) so smem[1] is written while its async store is still
    reading it.
    """
    stderr_str = _run_consan_subprocess("tdm_store_kernel", FAILURE)

    if FAILURE:
        assert "Accessing buffer with pending access" in stderr_str, \
            f"Expected 'Accessing buffer with pending access' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_consan_tdm_load_no_barrier(FAILURE):
    """
    TDM async load without barrier, using async_wait for completion.
    FAILURE=True reads shared memory before async_wait, triggering
    a pending access violation.
    """
    stderr_str = _run_consan_subprocess("tdm_load_no_barrier_kernel", FAILURE)

    if FAILURE:
        assert "Accessing buffer with pending access" in stderr_str, \
            f"Expected 'Accessing buffer with pending access' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_consan_tdm_two_bufs_one_wait(FAILURE):
    """
    Two TDM async loads (no barriers) to separate buffers and a single
    async_wait. Validates that TDM operations from the same wave are completed in issue order.
    FAILURE=True reads smem before the wait.
    """
    stderr_str = _run_consan_subprocess("tdm_two_bufs_one_wait_kernel", FAILURE)

    if FAILURE:
        assert "Accessing buffer with pending access" in stderr_str, \
            f"Expected 'Accessing buffer with pending access' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_consan_tdm_load_store_combined(FAILURE):
    """
    TDM async load + async store (both without barriers) and a single
    async_wait. FAILURE=True reads shared memory before async_wait,
    triggering a pending access violation.
    """
    stderr_str = _run_consan_subprocess("tdm_load_store_combined_kernel", FAILURE)

    if FAILURE:
        assert "Accessing buffer with pending access" in stderr_str, \
            f"Expected 'Accessing buffer with pending access' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_consan_tdm_cross_partition(FAILURE):
    """
    Two WS partitions both do TDM async load to the same shared memory buffer
    (commit-tracked, no TDM barrier). FAILURE=True skips async_wait in ws_default
    partition, leaving its commits outstanding; ws_1's check detects the
    cross-partition race. FAILURE=False clears commits via async_wait first.
    """
    stderr_str = _run_consan_subprocess("tdm_cross_partition_kernel", FAILURE)

    if FAILURE:
        assert "Accessing buffer with pending access" in stderr_str, \
            f"Expected 'Accessing buffer with pending access' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires GFX1250")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_consan_tdm_cross_partition_load_store(FAILURE):
    """
    ws_default partition does TDM async load (global -> shared), ws_1 does
    TDM async store (shared -> global) on the same buffer.
    FAILURE=True skips async_wait in ws_default, so its access is
    still pending when ws_1 accesses the buffer; ConSan detects
    the cross-partition race.
    """
    stderr_str = _run_consan_subprocess("tdm_cross_partition_load_store_kernel", FAILURE)

    if FAILURE:
        assert "Accessing buffer with pending access" in stderr_str, \
            f"Expected 'Accessing buffer with pending access' in stderr, got:\n{stderr_str}"
    else:
        assert "device assertion failed" not in stderr_str, \
            f"Unexpected ConSan violation in stderr:\n{stderr_str}"
