import os
import subprocess
import sys
import pytest

import triton


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def test_address_sanitizer():
    if not is_hip():
        return  #not supported on NV backend

    # It is recommended to disable various memory caching strategies both within the ROCm stack and PyTorch
    # This will give the address sanitizer the best chance at finding the memory fault where it originates,
    # otherwise it could be masked by writing past the end of a cached block within a larger allocation.
    os.environ["HSA_DISABLE_FRAGMENT_ALLOCATOR"] = "1"
    os.environ["AMD_PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    os.environ["PYTORCH_NO_HIP_MEMORY_CACHING"] = "1"
    os.environ["TRITON_ENABLE_ASAN"] = "1"

    # HSA_XNACK here is required to set the xnack+ setting for the GPU at runtime.
    # If it is not set and the default xnack setting of the system is xnack-
    # a runtime error something like "No kernel image found" will occur. The system
    # xnack setting can be found through rocminfo. xnack+ is required for ASAN.
    # More information about xnack in general can be found here:
    # https://llvm.org/docs/AMDGPUUsage.html#target-features
    # https://rocm.docs.amd.com/en/docs-6.1.0/conceptual/gpu-memory.html
    os.environ["HSA_XNACK"] = "1"

    # Disable buffer ops given it has builtin support for out of bound access.
    os.environ["AMDGCN_USE_BUFFER_OPS"] = "0"

    os.environ["TRITON_ALWAYS_COMPILE"] = "1"

    # Override the job-level TRITON_DISABLE_LINE_INFO=1
    os.environ["TRITON_DISABLE_LINE_INFO"] = "0"

    helper = os.path.join(os.path.dirname(__file__), "address_sanitizer_helper.py")

    try:
        r = subprocess.run(
            [sys.executable, helper],
            capture_output=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail("ASan helper timed out after {}s\n"
                    "stdout tail: {!r}\n"
                    "stderr tail: {!r}".format(
                        e.timeout,
                        (e.stdout or b"")[-2000:],
                        (e.stderr or b"")[-2000:],
                    ))

    stdout = r.stdout.decode(errors="replace")
    stderr = r.stderr.decode(errors="replace")
    has_begin = "Begin function __asan_report" in stdout
    has_overflow = "heap-buffer-overflow" in stderr

    # Always forward the helper's stderr (ASan report + diagnostics) to the
    # test runner's stderr so that CI logs contain the full report even on
    # passing runs.
    sys.stderr.write(stderr)
    sys.stderr.flush()

    assert has_begin and has_overflow, ("ASan check failed\n"
                                        f"  returncode   = {r.returncode}\n"
                                        f"  stdout_bytes = {len(stdout)}\n"
                                        f"  stderr_bytes = {len(stderr)}\n"
                                        f"  has_begin    = {has_begin}   # 'Begin function __asan_report' in stdout\n"
                                        f"  has_overflow = {has_overflow} # 'heap-buffer-overflow' in stderr\n"
                                        "--- stdout head ---\n"
                                        f"{stdout[:1500]}\n"
                                        "--- stdout tail ---\n"
                                        f"{stdout[-1500:]}\n"
                                        "--- stderr head ---\n"
                                        f"{stderr[:2000]}\n")
