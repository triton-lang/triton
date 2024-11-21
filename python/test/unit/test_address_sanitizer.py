import os
from contextlib import contextmanager
import subprocess
import torch
import triton
import triton.language as tl

os.environ["AMD_PYTORCH_NO_CUDA_MEMORY_CACHING"]="1"
os.environ["PYTORCH_NO_HIP_MEMORY_CACHING"]="1"
os.environ["TRITON_ENABLE_ADDRESS_SANITIZER"]="1"
os.environ["HSA_XNACK"]="1"

def test_address_sanitizer():
    out = subprocess.Popen(["python", "address_sanitizer_helper.py"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    assert "Begin function __asan_report" in out.stdout.read().decode()
    assert "heap-buffer-overflow" in out.stderr.read().decode()
