import os
from pathlib import Path
import subprocess
import sys
import textwrap
import torch
from types import SimpleNamespace

import pytest
from triton._internal_testing import is_cuda


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


@pytest.mark.parametrize("cuda_active", [False, True])
def test_backend_discovery_without_torch(cuda_active):
    code = textwrap.dedent("""\
        import builtins
        import importlib
        import sys
        from types import SimpleNamespace

        import triton.language as tl
        from triton.backends.amd import driver as amd_driver
        from triton.backends.compiler import GPUTarget
        from triton.experimental.gluon import language as gl
        from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
        from triton.runtime.jit import MockTensor

        assert "torch" not in sys.modules
        original_import = builtins.__import__
        def no_torch_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("backend discovery must not import Torch")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = no_torch_import

        sys.modules["amdsmi"] = SimpleNamespace(
            amdsmi_init=lambda: None,
            amdsmi_get_processor_handles=lambda: [],
            amdsmi_shut_down=lambda: None,
        )

        cuda_active = sys.argv[1] == "1"
        target = GPUTarget("cuda", 103, 32)
        class CudaDriver:
            @staticmethod
            def is_active():
                return cuda_active
            def get_current_target(self):
                return target

        runtime = importlib.import_module("triton.runtime.driver")
        runtime.backends = {"nvidia": SimpleNamespace(driver=CudaDriver),
                            "amd": SimpleNamespace(driver=amd_driver.HIPDriver)}
        assert tl.target_info.current_target() == (target if cuda_active else None)
        tensor = MockTensor(gl.uint8, (128, 128))
        layout = gl.NVMMASharedLayout(128, 8, fp4_padded=True)
        TensorDescriptor.from_tensor(tensor, [128, 128], layout)
        assert "torch" not in sys.modules
    """)
    env = os.environ.copy()
    env.pop("TRITON_DEFAULT_BACKEND", None)
    proc = subprocess.run([sys.executable, "-c", code, str(int(cuda_active))], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("torch_loaded", [False, True])
@pytest.mark.parametrize("management, available, hip, expected", [
    ("empty", True, "7.0", False),
    ("present", True, "7.0", True),
    ("present", False, "7.0", False),
    ("present", True, None, False),
    ("missing_module", True, "7.0", True),
    ("missing_library", True, "7.0", True),
    ("init_failure", True, "7.0", True),
    ("query_failure", True, "7.0", True),
    ("shutdown_failure", True, "7.0", True),
])
def test_hip_driver_amdsmi_fallback(management, available, hip, expected, torch_loaded, monkeypatch):
    import builtins
    from triton.backends.amd import driver

    class AmdSmiException(Exception):
        pass

    calls = []

    def init():
        calls.append("init")
        if management == "init_failure":
            raise AmdSmiException("initialization failed")

    def get_handles():
        calls.append("query")
        if management == "query_failure":
            raise AmdSmiException("enumeration failed")
        return [] if management in ("empty", "shutdown_failure") else [object()]

    def shutdown():
        calls.append("shutdown")
        if management == "shutdown_failure":
            raise AmdSmiException("shutdown failed")

    amdsmi = SimpleNamespace(amdsmi_init=init, amdsmi_get_processor_handles=get_handles, amdsmi_shut_down=shutdown)
    original_import = builtins.__import__

    def import_dependency(name, *args, **kwargs):
        if name == "amdsmi":
            calls.append("amdsmi")
            if management == "missing_module":
                raise ImportError("amdsmi is not installed")
            if management == "missing_library":
                raise KeyError("libamd_smi.so")
            return amdsmi
        if name == "torch":
            calls.append("torch")
            return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available),
                                   version=SimpleNamespace(hip=hip))
        return original_import(name, *args, **kwargs)

    if not torch_loaded:
        monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.setattr(builtins, "__import__", import_dependency)
    assert driver.HIPDriver.is_active() is expected

    expected_calls = ["amdsmi"]
    if management not in ("missing_module", "missing_library"):
        expected_calls.append("init")
        if management != "init_failure":
            expected_calls.extend(["query", "shutdown"])
    if management != "empty":
        expected_calls.append("torch")
    assert calls == expected_calls
