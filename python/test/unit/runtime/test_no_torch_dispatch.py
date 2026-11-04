import os
from pathlib import Path
import subprocess
import sys
import torch

import pytest
from triton._internal_testing import is_cuda


def test_nvidia_kernel_dispatch_without_torch():
    if not is_cuda() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires CUDA and TMAs")

    env = os.environ.copy()
    env.pop("TRITON_DEFAULT_BACKEND", None)
    # force compilation to ensure there is no torch dependencies in the compiler.
    env["TRITON_ALWAYS_COMPILE"] = "1"

    script_path = Path(__file__).with_name("no_torch_dispatch_example.py")
    proc = subprocess.run([sys.executable, str(script_path)], text=True, capture_output=True, env=env)

    assert proc.returncode == 0, ("Torch-free runtime dispatch subprocess failed.\n"
                                  f"stdout:\n{proc.stdout}\n"
                                  f"stderr:\n{proc.stderr}")
