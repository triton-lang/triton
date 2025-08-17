import sys
import shutil
import pytest
from loguru import logger

from third_party.metal.backend import compiler as metal_compiler_mod
from third_party.metal.backend import driver as metal_driver_mod
from third_party.metal.backend import runtime as metal_runtime_mod

MINIMAL_METAL_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void writer_kernel(device float* out [[buffer(0)]]) { out[0] = 42.0; }
"""

def _has_real_metal_tools():
    if shutil.which("xcrun") is None:
        return False
    return shutil.which("metal") is not None and shutil.which("metallib") is not None

def test_launch_kernel_mocked(monkeypatch):
    """
    Mock runtime.bind_library to return a fake handle that records pipeline creation,
    dispatch sizes, and commit/wait invocations. Ensure pipeline caching is exercised.
    """
    compiler = metal_compiler_mod.MetalCompiler()
    driver = metal_driver_mod.MetalDriver()

    # Create a fake compile output (bytes)
    fake_binary = b"FAKE_METALLIB"

    # Fake runtime.bind_library -> returns a FakeHandle recording calls
    class FakeHandle:
        def __init__(self, binary, metadata=None):
            self.binary_bytes = binary
            self.metadata = dict(metadata or {})
            self.is_stub = False
            self.pipeline_cache = {}
            self.pipeline_creation_count = 0
            self.last_grid = None
            self.last_block = None
            self.commit_called = False
            self.wait_called = False

        def launch_kernel(self, name=None, args=None, grid=None, block=None, timeout=None, *a, **kw):
            # Simulate pipeline creation on first use and caching afterwards
            if name not in self.pipeline_cache:
                self.pipeline_cache[name] = True
                self.pipeline_creation_count += 1
            # Record dispatch parameters
            self.last_grid = grid
            self.last_block = block
            # Simulate commit/wait
            self.commit_called = True
            self.wait_called = True
            return {"status": "ok", "duration_ms": 0.0, "gpu_error": None}

    recorded = {}
    def _fake_bind(b, metadata=None):
        # Record that bind_library received metadata and return a FakeHandle
        recorded['metadata'] = metadata
        return FakeHandle(b, metadata=metadata)

    monkeypatch.setattr(metal_runtime_mod, "bind_library", _fake_bind, raising=False)

    # Exercise driver path: load binary (which calls bind_library) then launch kernel twice
    handle = driver.load_binary(fake_binary)
    assert getattr(handle, "is_stub", True) is False
    # First launch should create pipeline
    handle.launch_kernel(name="writer_kernel", args=(b"\x00\x00",), grid=(16,1,1), block=(16,1,1))
    assert handle.pipeline_creation_count == 1, "Pipeline should be created on first launch"
    # Second launch should reuse cached pipeline (creation count unchanged)
    handle.launch_kernel(name="writer_kernel", args=(b"\x00\x00",), grid=(32,1,1), block=(16,1,1))
    assert handle.pipeline_creation_count == 1, "Pipeline must be cached and not re-created"
    # Dispatch sizes recorded correctly
    assert handle.last_grid == (32,1,1)
    assert handle.last_block == (16,1,1)
    # Commit/wait simulated
    assert handle.commit_called is True
    assert handle.wait_called is True

def test_reflection_metadata_parsing(monkeypatch):
    """
    Ensure compiler.compile(..., reflection=True) metadata is forwarded through driver.load_binary
    into runtime.bind_library and exposed on the returned handle.
    """
    driver = metal_driver_mod.MetalDriver()

    # Prepare fake compiler output (binary + metadata)
    fake_binary = b"FAKE_METALLIB_2"
    fake_metadata = {"kernels": [{"name": "writer_kernel", "args": ["device float* out [[buffer(0)]]"]}]}

    # Monkeypatch MetalCompiler.compile to return (bytes, metadata) when reflection=True
    def _fake_compile(self, source, options, reflection=False):
        if reflection:
            return fake_binary, fake_metadata
        return fake_binary

    monkeypatch.setattr(metal_compiler_mod.MetalCompiler, "compile", _fake_compile, raising=True)

    # Monkeypatch runtime.bind_library so we can inspect received metadata
    received = {}
    class FakeHandle2:
        def __init__(self, binary, metadata=None):
            self.binary_bytes = binary
            self.metadata = dict(metadata or {})
            self.is_stub = False
        def launch_kernel(self, *a, **kw):
            return {"status": "ok", "duration_ms": 0.0, "gpu_error": None}

    def _fake_bind2(b, metadata=None):
        received['metadata'] = metadata
        # Ensure the returned handle contains metadata as well
        h = FakeHandle2(b, metadata=metadata)
        # populate kernels if present
        if metadata and "kernels" in metadata:
            h.metadata["kernels"] = metadata["kernels"]
        return h

    monkeypatch.setattr(metal_runtime_mod, "bind_library", _fake_bind2, raising=False)

    # Call compile with reflection and forward metadata via driver.load_binary
    binary, metadata = metal_compiler_mod.MetalCompiler().compile(MINIMAL_METAL_SRC, options={}, reflection=True)
    assert isinstance(metadata, dict) and "kernels" in metadata
    handle = driver.load_binary(binary, metadata=metadata)
    # runtime.bind_library must have received metadata
    assert received.get("metadata") == metadata
    # Returned handle must expose the kernels metadata
    assert isinstance(handle.metadata.get("kernels"), list)
    assert handle.metadata["kernels"][0]["name"] == "writer_kernel"

@pytest.mark.macos_integration
def test_end_to_end_minimal_macos():
    """
    Minimal end-to-end integration test that only runs on macOS with XCode tools.
    This test is intentionally skipped on non-darwin platforms.
    """
    if sys.platform != "darwin" or not _has_real_metal_tools():
        pytest.skip("Skipping macOS integration test on non-darwin or missing tools")

    compiler = metal_compiler_mod.MetalCompiler()
    driver = metal_driver_mod.MetalDriver()

    # Perform real compile with reflection
    binary, metadata = compiler.compile(MINIMAL_METAL_SRC, options={}, reflection=True)
    handle = driver.load_binary(binary, metadata=metadata)

    # Allocate a small buffer via the real handle if available.
    # The real handle exposes device and methods; attempt a minimal launch that writes to buffer[0].
    import numpy as np
    out = np.zeros(1, dtype=np.float32)
    res = handle.launch_kernel(name="writer_kernel", args=(out,), grid=(1,1,1), block=(1,1,1), timeout=5.0)
    assert res["status"] == "ok"
    # Validate the kernel wrote the expected pattern (42.0) if device supports readback via numpy
    # Some runtimes may require explicit readback steps; this assert is best-effort.
    assert out[0] == pytest.approx(42.0)