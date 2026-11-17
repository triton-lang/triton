import os
from pathlib import Path
import subprocess
import sys
import torch

import pytest
from triton._internal_testing import is_cuda, is_hip


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


_HIP_DISCOVERY_SCRIPT = """
import sys

class _ForbidTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch import is forbidden in this test")
        return None

sys.meta_path.insert(0, _ForbidTorch())

from triton.backends import backends

active = sorted(name for name, backend in backends.items() if backend.driver.is_active())
if active != ["amd"]:
    raise RuntimeError(f"Expected only the amd backend to be active, got {active!r}")
if "torch" in sys.modules:
    raise RuntimeError("torch module was imported")
"""


def test_hip_backend_discovery_without_torch():
    # Only backend discovery is checked: HIPDriver construction and kernel
    # launch still depend on torch.
    if not is_hip():
        pytest.skip("Requires an AMD GPU")

    env = os.environ.copy()
    env.pop("TRITON_DEFAULT_BACKEND", None)
    proc = subprocess.run([sys.executable, "-c", _HIP_DISCOVERY_SCRIPT], text=True, capture_output=True, env=env)

    assert proc.returncode == 0, ("Torch-free HIP backend discovery subprocess failed.\n"
                                  f"stdout:\n{proc.stdout}\n"
                                  f"stderr:\n{proc.stderr}")
