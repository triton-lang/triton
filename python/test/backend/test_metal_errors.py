import platform
import shutil
import pytest

from third_party.metal.backend import driver as metal_driver_mod
from third_party.metal.backend import compiler as metal_compiler_mod
from third_party.metal.backend import runtime as metal_runtime_mod

def test_driver_returns_stub_when_runtime_unavailable(monkeypatch):
    # Simulate import-time failure of runtime module by setting runtime_mod to None
    monkeypatch.setattr(metal_driver_mod, "runtime_mod", None)
    drv = metal_driver_mod.MetalDriver()
    handle = drv.load_binary(b"BIN")
    assert getattr(handle, "is_stub", False) is True
    with pytest.raises(RuntimeError) as exc:
        handle.launch_kernel()
    assert "Metal runtime unavailable" in str(exc.value)

def test_compile_handles_reflection_regex_failure(monkeypatch):
    # Ensure xcrun presence passes
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/xcrun")
    # Make _run_metal_compile produce a dummy metallib so compile can read it
    def fake_run(cmd, msg):
        if "-o" in cmd:
            out = cmd[cmd.index("-o") + 1]
            with open(out, "wb") as f:
                f.write(b"DUMMY")
    monkeypatch.setattr(metal_compiler_mod, "_run_metal_compile", fake_run)
    # Force re.compile used inside reflection to raise
    monkeypatch.setattr(metal_compiler_mod, "re", type("R", (), {"compile": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad regex"))}))
    comp = metal_compiler_mod.MetalCompiler()
    # Should not raise even though reflection parsing fails; returns binary and metadata (possibly empty)
    binary, metadata = comp.compile("# dummy", options={}, reflection=True)
    assert isinstance(binary, (bytes, bytearray))
    # If reflection parsing failed we expect metadata to exist but be empty or missing 'kernels'
    assert isinstance(metadata, dict)

def test_launch_kernel_timeout_propagates(monkeypatch):
    # Simulate runtime.bind_library returning a handle whose launch_kernel raises TimeoutError
    class FakeTimeoutHandle:
        def __init__(self, b, metadata=None):
            self.binary_bytes = b
            self.metadata = metadata or {}
            self.is_stub = False
        def launch_kernel(self, *args, **kwargs):
            raise TimeoutError("Command buffer did not complete before timeout.")
    monkeypatch.setattr(metal_runtime_mod, "bind_library", lambda b, metadata=None: FakeTimeoutHandle(b, metadata), raising=False)
    drv = metal_driver_mod.MetalDriver()
    h = drv.load_binary(b"B")
    with pytest.raises(TimeoutError):
        h.launch_kernel(name="k", args=(), grid=(1,1,1), block=(1,1,1), timeout=0.01)

def test_launch_kernel_runtime_error_propagates(monkeypatch):
    # Simulate runtime.bind_library returning a handle whose launch_kernel raises MetalRuntimeError
    class FakeRuntimeErrorHandle:
        def __init__(self, b, metadata=None):
            self.binary_bytes = b
            self.metadata = metadata or {}
            self.is_stub = False
        def launch_kernel(self, *args, **kwargs):
            raise metal_runtime_mod.MetalRuntimeError("GPU execution error: fault")
    monkeypatch.setattr(metal_runtime_mod, "bind_library", lambda b, metadata=None: FakeRuntimeErrorHandle(b, metadata), raising=False)
    drv = metal_driver_mod.MetalDriver()
    h = drv.load_binary(b"B2")
    with pytest.raises(metal_runtime_mod.MetalRuntimeError):
        h.launch_kernel(name="k", args=(), grid=(1,1,1), block=(1,1,1))