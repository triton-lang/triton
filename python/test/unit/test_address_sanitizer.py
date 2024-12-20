import os
import subprocess

import triton

# It is recommended to disable various memory caching strategies both within the ROCm stack and PyTorch
# This will give the address sanitizer the best chance at finding the memory fault where it originates,
# otherwise it could be masked by writing past the end of a cached block within a larger allocation.
os.environ["HSA_DISABLE_FRAGMENT_ALLOCATOR"] = "1"
os.environ["AMD_PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["PYTORCH_NO_HIP_MEMORY_CACHING"] = "1"
os.environ["TRITON_ENABLE_ASAN"] = "1"

# HSA_XNACKhere is require to set the xnack+ setting for the GPU at runtime
# if this is not set and the default xnack setting of the system is -xnack
# a runtime error something like "No kernel image found" will occur. The system
# xnack setting can be found through rocminfo.
os.environ["HSA_XNACK"] = "1"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def test_address_sanitizer():
    if not is_hip():
        return  #not supported on NV backend
    out = subprocess.Popen(["python", "address_sanitizer_helper.py"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    assert "Begin function __asan_report" in out.stdout.read().decode()
    assert "heap-buffer-overflow" in out.stderr.read().decode()
