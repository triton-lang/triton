from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest

_SUBPROCESS_SCRIPT = r'''
import sys
import torch
import triton
import triton.language as tl

from triton.experimental.gsan import create_mem_pool


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


def _run_case(case: str) -> int:
    if torch.cuda.device_count() < 1:
        return 100

    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        target = torch.zeros(1, dtype=torch.int32, device="cuda")
        scratch = torch.zeros(1, dtype=torch.int32, device="cuda")

    triton.knobs.compilation.instrumentation_mode = "gsan"
    kernel = globals()[f"_{case}_kernel"]
    kernel[(2, )](target, scratch, num_warps=1)

    try:
        torch.cuda.synchronize()
    except RuntimeError:
        return 0

    print("expected a GSan synchronization failure, but kernel completed")
    return 1


if __name__ == "__main__":
    case = sys.argv[1]
    raise SystemExit(_run_case(case))
'''


def _run_failure_case(case: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_SUBPROCESS_SCRIPT)
        script_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, script_path, case],
            capture_output=True,
            text=True,
            errors="replace",
        )
    finally:
        os.unlink(script_path)
    if proc.returncode == 100:
        pytest.skip("requires CUDA backend")
    assert proc.returncode == 0, (f"case={case} failed with returncode={proc.returncode}\n"
                                  f"stdout:\n{proc.stdout}\n"
                                  f"stderr:\n{proc.stderr}")


def test_read_after_write():
    _run_failure_case("raw")


def test_write_after_read():
    _run_failure_case("war")


def test_write_after_write():
    _run_failure_case("waw")
