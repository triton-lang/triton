import os
from pathlib import Path
import subprocess
import sys
import textwrap
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
def test_hip_discovery_without_visible_devices():
    code = textwrap.dedent("""\
        import ctypes
        import sys
        from triton.backends.amd.driver import HIPDriver, _get_path_to_hip_runtime_dylib

        assert "torch" not in sys.modules
        ctypes.CDLL(_get_path_to_hip_runtime_dylib()).hipGetDeviceCount
        assert not HIPDriver.is_active()
        assert "torch" not in sys.modules
    """)
    env = dict(os.environ, HIP_VISIBLE_DEVICES="-1")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_hip_discovery_runtime_unavailable(monkeypatch):
    from triton.backends.amd import driver

    def unavailable():
        raise RuntimeError("HIP runtime unavailable")

    monkeypatch.setattr(driver, "_get_path_to_hip_runtime_dylib", unavailable)
    assert driver.HIPDriver.is_active() == (torch.cuda.is_available() and torch.version.hip is not None)
