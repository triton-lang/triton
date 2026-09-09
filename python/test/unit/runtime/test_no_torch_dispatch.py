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
    env.pop("TRITON_DEFAULT_BACKEND", None)
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

        def missing_hip_runtime():
            raise amd_driver._HIPRuntimeNotFoundError("no HIP runtime")
        amd_driver._get_path_to_hip_runtime_dylib = missing_hip_runtime

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
@pytest.mark.parametrize("runtime, available, hip, expected", [
    ("missing", True, "7.0", False),
    ("no_devices", True, "7.0", False),
    ("no_device_status", True, "7.0", False),
    ("unknown_status", True, "7.0", True),
    ("missing_symbol", True, "7.0", True),
    ("found", True, "7.0", True),
    ("configuration_error", True, "7.0", True),
    ("os_error", True, "7.0", True),
    ("subprocess_error", True, "7.0", True),
    ("found", False, "7.0", False),
    ("found", True, None, False),
])
def test_hip_driver_runtime_lookup(runtime, available, hip, expected, torch_loaded, monkeypatch):
    import builtins
    from triton.backends.amd import driver

    lookups = []

    def find_runtime():
        lookups.append(True)
        if runtime == "missing":
            raise driver._HIPRuntimeNotFoundError("no HIP runtime")
        if runtime == "configuration_error":
            raise RuntimeError("invalid HIP runtime configuration")
        if runtime == "os_error":
            raise OSError("cannot search for HIP runtime")
        if runtime == "subprocess_error":
            raise subprocess.CalledProcessError(1, "ldconfig")
        return "libamdhip64.so"

    def get_device_count(pointer):
        if runtime == "unknown_status":
            return 999
        if runtime == "no_device_status":
            return 100
        driver.ctypes.cast(pointer, driver.ctypes.POINTER(driver.ctypes.c_int))[0] = int(runtime != "no_devices")
        return 0

    imports = []
    original_import = builtins.__import__

    def import_torch(name, *args, **kwargs):
        if name == "torch":
            imports.append(name)
            return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available),
                                   version=SimpleNamespace(hip=hip))
        return original_import(name, *args, **kwargs)

    if not torch_loaded:
        monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.setattr(builtins, "__import__", import_torch)
    monkeypatch.setattr(driver, "_get_path_to_hip_runtime_dylib", find_runtime)
    library = SimpleNamespace() if runtime == "missing_symbol" else SimpleNamespace(hipGetDeviceCount=get_device_count)
    monkeypatch.setattr(driver.ctypes, "CDLL", lambda path: library)
    assert driver.HIPDriver.is_active() is expected
    assert lookups == [True]
    assert imports == ([] if runtime in ("missing", "no_devices", "no_device_status") else ["torch"])


def test_hip_runtime_versioned_torch_library(tmp_path, monkeypatch):
    from triton.backends.amd import driver

    library = tmp_path / "torch" / "lib" / "libamdhip64.so.6"
    library.parent.mkdir(parents=True)
    library.touch()
    monkeypatch.setattr(driver.knobs.amd, "libhip_path", None)
    monkeypatch.setitem(sys.modules, "rocm_sdk", None)
    monkeypatch.setattr(driver, "_find_already_mmapped_dylib_on_linux", lambda name: None)
    monkeypatch.setattr(driver, "__file__", str(tmp_path / "backend" / "driver.py"))
    monkeypatch.setattr(driver.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(submodule_search_locations=[str(library.parent.parent)]))
    driver._get_path_to_hip_runtime_dylib.cache_clear()
    try:
        assert driver._get_path_to_hip_runtime_dylib() == str(library)
    finally:
        driver._get_path_to_hip_runtime_dylib.cache_clear()
