from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_cuda, run_in_process
from triton.experimental.gsan import create_mem_pool

pytestmark = pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")


@triton.jit
def _raw_kernel(ptr, scratch_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(ptr, 1)
    else:
        value = tl.load(ptr)
        tl.store(scratch_ptr, value)


@triton.jit
def _war_kernel(ptr, scratch_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        value = tl.load(ptr)
        tl.store(scratch_ptr, value)
    else:
        tl.store(ptr, 1)


@triton.jit
def _waw_kernel(ptr, scratch_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(ptr, 1)
    else:
        tl.store(ptr, 2)


def _run_case(case: str) -> None:
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        target = torch.zeros(1, dtype=torch.int32, device="cuda")
        scratch = torch.zeros(1, dtype=torch.int32, device="cuda")

    triton.knobs.compilation.instrumentation_mode = "gsan"
    kernel = globals()[f"_{case}_kernel"]
    kernel[(2, )](target, scratch, num_warps=1)


def _run_failure_case(case: str) -> None:
    if torch.cuda.device_count() < 1:
        pytest.skip("requires at least 1 CUDA device")

    result = run_in_process(_run_case, (case, ))
    assert isinstance(result.exc, RuntimeError), (f"case={case} completed without the expected GSan failure\n"
                                                  f"exc={result.exc!r}\n"
                                                  f"driver stderr:\n{result.driver_stderr_output}")
    assert "GSanLibrary.cu" in result.driver_stderr_output
    assert (("clock[write.threadId] >= write.epoch" in result.driver_stderr_output)
            or ("clock[read.threadId] >= read.epoch" in result.driver_stderr_output))


def test_read_after_write():
    _run_failure_case("raw")


def test_write_after_read():
    _run_failure_case("war")


def test_write_after_write():
    _run_failure_case("waw")
