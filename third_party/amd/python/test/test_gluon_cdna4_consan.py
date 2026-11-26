"""ConSan runtime tests on CDNA4 (gfx950).

Each test runs a scenario from cdna4_consan_helper.py in a child process.
Safe scenarios must produce exact output and no ConSan diagnostic. Unsafe
scenarios must produce the expected ConSan diagnostic.
"""
import os

import pytest

from triton._internal_testing import is_hip_cdna4
from consan_subprocess import run_consan_helper

PENDING_COPY = "Accessing buffer with pending access. Pending access type: async_copy_global_to_shared"


def _run_consan_subprocess(test_name, *args, diagnostic=None):
    """Run a helper scenario and check its ConSan diagnostic (None means no diagnostic)."""
    helper_script = os.path.join(os.path.dirname(__file__), "cdna4_consan_helper.py")
    stderr = run_consan_helper(helper_script, test_name, *args)
    if diagnostic is None:
        assert "device assertion failed" not in stderr, stderr
    else:
        assert diagnostic in stderr, stderr


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("copy_path", ["global", "buffer"])
@pytest.mark.parametrize("num_warps", [1, 4, 8])
@pytest.mark.parametrize("vec", [1, 4], ids=["32bit", "128bit"])
@pytest.mark.parametrize("wait", [True, False], ids=["safe", "missing_wait"])
def test_async_copy_missing_wait(copy_path, num_warps, vec, wait):
    _run_consan_subprocess("async_copy_access", copy_path, "read", wait, vec, num_warps, 2048,
                           diagnostic=None if wait else PENDING_COPY)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("copy_path", ["global", "buffer"])
@pytest.mark.parametrize("buffers", [2, 3])
@pytest.mark.parametrize("mode", ["full", "partial", "off_by_one"])
def test_async_copy_partial_wait_loop(copy_path, buffers, mode):
    _run_consan_subprocess("async_copy_partial_wait_loop", copy_path, mode, buffers,
                           diagnostic=PENDING_COPY if mode == "off_by_one" else None)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("copy_path", ["global", "buffer"])
@pytest.mark.parametrize("mode", ["copy_overwrite", "store_overwrite"])
@pytest.mark.parametrize("wait", [True, False], ids=["safe", "early_reuse"])
def test_async_copy_buffer_reuse(copy_path, mode, wait):
    _run_consan_subprocess("async_copy_access", copy_path, mode, wait, 4, 4, 2048,
                           diagnostic=None if wait else PENDING_COPY)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("copy_path", ["global", "buffer"])
@pytest.mark.parametrize("wait", [True, False], ids=["safe", "missing_wait"])
def test_async_copy_ordinary_shared_access(copy_path, wait):
    _run_consan_subprocess("async_copy_access", copy_path, "ordinary", wait, 4, 4, 2048,
                           diagnostic=None if wait else PENDING_COPY)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("copy_path", ["global", "buffer"])
@pytest.mark.parametrize("vec", [1, 4], ids=["32bit", "128bit"])
def test_async_copy_masked_zero_fill(copy_path, vec):
    # The vector-aligned boundary is required by both CDNA4 copy APIs.
    _run_consan_subprocess("async_copy_access", copy_path, "read", True, vec, 4, 1536)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("staged", [True, False], ids=["scheduled_warning", "ordinary_instrumented"])
def test_consan_warp_pipeline_execution(staged):
    _run_consan_subprocess("warp_pipeline_execution", staged)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("mode", ["pingpong", "single_stage", "disabled"])
def test_consan_pingpong_execution(mode):
    _run_consan_subprocess("pingpong_execution", mode)
