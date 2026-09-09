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


@pytest.mark.parametrize("mode", ["missing_runtime", "visible", "hidden"])
def test_hip_discovery_without_torch(mode, tmp_path):
    if mode != "missing_runtime" and not is_hip():
        pytest.skip("Requires an AMD GPU")

    code = textwrap.dedent("""\
        import ctypes
        import sys
        from triton.backends.amd.driver import HIPDriver, _get_path_to_hip_runtime_dylib

        assert "torch" not in sys.modules
        if sys.argv[1] != "missing_runtime":
            ctypes.CDLL(_get_path_to_hip_runtime_dylib()).hipGetDeviceCount
        assert HIPDriver.is_active() == (sys.argv[1] == "visible")
        assert "torch" not in sys.modules
    """)
    env = os.environ.copy()
    if mode == "missing_runtime":
        env["TRITON_LIBHIP_PATH"] = str(tmp_path / "libamdhip64.so")
    elif mode == "hidden":
        env["HIP_VISIBLE_DEVICES"] = "-1"
    proc = subprocess.run([sys.executable, "-c", code, mode], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
