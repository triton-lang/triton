import os
import subprocess


# It is recommended to disable various memory caching strategies both within the ROCm stack and PyTorch
# This will give the address sanitizer the best chance at finding the memory fault where it originates, 
# otherwise it could be masked by writing past the end of a cached block within a larger allocation.
os.environ["HSA_DISABLE_FRAGMENT_ALLOCATOR"] = "1"
os.environ["AMD_PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["PYTORCH_NO_HIP_MEMORY_CACHING"] = "1"
os.environ["TRITON_ENABLE_ASAN"] = "1"

# XNACK setting here is require to set the xnack+ setting for the GPU at runtime
# if this is not set and the xnack setting is configured be defult for xnack-
# a runtime error some like "No kernel image found" will occur
os.environ["HSA_XNACK"] = "1"

def test_address_sanitizer():
    out = subprocess.Popen(["python", "address_sanitizer_helper.py"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    assert "Begin function __asan_report" in out.stdout.read().decode()
    assert "heap-buffer-overflow" in out.stderr.read().decode()
