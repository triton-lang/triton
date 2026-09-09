import os
from pathlib import Path
import subprocess
import sys
import torch

import pytest
from triton._internal_testing import is_cuda, is_hip


def test_nvidia_kernel_dispatch_without_torch():
    if not is_cuda() and torch.cuda.get_device_capability()[0] >= 9:
        pytest.skip("Requires CUDA and TMAs")

    env = os.environ.copy()
    # force cuda driver to avoid importing torch when checking for other backends.
    env["TRITON_DEFAULT_BACKEND"] = "nvidia"
    # force compilation to ensure there is no torch dependencies in the compiler.
    env["TRITON_ALWAYS_COMPILE"] = "1"

    script_path = Path(__file__).with_name("no_torch_dispatch_example.py")
    proc = subprocess.run([sys.executable, str(script_path)], text=True, capture_output=True, env=env)

    assert proc.returncode == 0, ("Torch-free runtime dispatch subprocess failed.\n"
                                  f"stdout:\n{proc.stdout}\n"
                                  f"stderr:\n{proc.stderr}")


@pytest.mark.skipif(not is_hip(), reason="Requires an AMD GPU")
def test_hip_driver_amdsmi():
    from triton.backends.amd.driver import HIPDriver

    amdsmi = pytest.importorskip("amdsmi")

    amdsmi.amdsmi_init()
    try:
        assert amdsmi.amdsmi_get_processor_handles()
    finally:
        amdsmi.amdsmi_shut_down()
    assert HIPDriver.is_active()


def test_hip_driver_without_amdsmi(monkeypatch):
    from triton.backends.amd.driver import HIPDriver

    monkeypatch.setitem(sys.modules, "amdsmi", None)
    assert HIPDriver.is_active() == (torch.cuda.is_available() and torch.version.hip is not None)
