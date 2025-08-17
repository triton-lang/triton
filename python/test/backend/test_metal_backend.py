import os
import sys
import shutil
import ctypes
import subprocess
import pytest
from loguru import logger

from third_party.metal.backend import compiler as metal_compiler_mod
from third_party.metal.backend import driver as metal_driver_mod
from third_party.metal.backend import runtime as metal_runtime_mod

# Minimal Metal source used for compile attempts (very small and harmless).
MINIMAL_METAL_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void empty_kernel(const device float* in [[buffer(0)]]) { }
"""


def _fake_run_metal_compile(cmd, error_msg):
    """
    Fake _run_metal_compile replacement used by tests when metal/metallib tools
    are unavailable. It creates the expected output file for the -o argument
    so MetalCompiler.compile can read it afterwards.
    """
    # Find output path after "-o"
    if "-o" in cmd:
        out_idx = cmd.index("-o") + 1
        if out_idx < len(cmd):
            out_path = cmd[out_idx]
            # Create a small fake metallib/air file depending on extension
            try:
                with open(out_path, "wb") as f:
                    # Small, deterministic bytes blob
                    f.write(b"FAKE_METALLIB_BYTES")
            except Exception as e:
                raise RuntimeError(f"Fake compile failed to write {out_path}: {e}")
    return None


def _has_real_metal_tools():
    # Detect whether xcrun/metal/metallib are present on PATH.
    if shutil.which("xcrun") is None:
        return False
    # Prefer explicit metal/metallib presence if available via xcrun discovery.
    # This is a conservative check; many CI environments will return False.
    return shutil.which("metal") is not None and shutil.which("metallib") is not None


def test_compile_and_load_real_or_mock(monkeypatch, tmp_path):
    """
    Attempt a real compile if Metal tools are available; otherwise mock the
    compiler subprocess helper and simulate a successful compile+link.
    Then call MetalDriver.load_binary and assert we obtain bytes and a non-stub handle
    when the runtime is available (real or mocked-as-success).
    """
    compiler = metal_compiler_mod.MetalCompiler()
    driver = metal_driver_mod.MetalDriver()

    # If real tools are available and running on macOS, try a minimal real compile.
    real_tools = _has_real_metal_tools() and sys.platform == "darwin"

    if real_tools:
        # Perform real compilation (may still fail in constrained CI - guarded above).
        binary = compiler.compile(MINIMAL_METAL_SRC, options={})
        assert isinstance(binary, (bytes, bytearray)), "compile() must return bytes"
        assert len(binary) > 0, "Compiled metallib must be non-empty bytes"
        handle = driver.load_binary(binary)
        # On real macOS with tools present we expect a non-stub handle.
        assert getattr(handle, "is_stub", True) is False, "Expected non-stub handle when real runtime is available"
        assert handle.binary_bytes == binary, "Handle must retain original binary bytes"
        assert "library_size" in handle.metadata, "Handle metadata should contain library_size key"
        assert handle.metadata["library_size"] == len(binary)
    else:
        # No real tools - patch the compiler helper to produce a fake metallib file.
        monkeypatch.setattr(metal_compiler_mod, "_run_metal_compile", _fake_run_metal_compile)
        # To simulate a runtime present for the driver, set sys.platform to darwin
        monkeypatch.setattr(sys, "platform", "darwin", raising=False)
        # Monkeypatch runtime.bind_library to return a fake non-stub handle
        class _FakeHandle:
            def __init__(self, binary):
                self.binary_bytes = binary
                self.metadata = {"platform": "darwin", "library_size": len(binary)}
                self.is_stub = False
            def launch_kernel(self, *args, **kwargs):
                return {"status":"ok","duration_ms":0.0,"gpu_error":None}
        monkeypatch.setattr(metal_runtime_mod, "bind_library", lambda b, metadata=None: _FakeHandle(b), raising=False)

        binary = compiler.compile(MINIMAL_METAL_SRC, options={})
        assert isinstance(binary, (bytes, bytearray)), "compile() must return bytes even when mocked"
        assert len(binary) > 0, "Mocked compiled metallib must be non-empty bytes"
        handle = driver.load_binary(binary)
        # Because we patched runtime.bind_library, driver should return non-stub
        assert getattr(handle, "is_stub", True) is False, "Expected non-stub handle when mocked runtime is available"
        assert handle.binary_bytes == binary, "Handle must retain original binary bytes"
        assert "library_size" in handle.metadata, "Handle metadata should contain library_size key"
        assert handle.metadata["library_size"] == len(binary)


def test_load_binary_stub_fallback(monkeypatch):
    """
    Simulate missing Metal runtime and verify load_binary returns a stub handle.
    Using the stub handle's launch_kernel should raise RuntimeError with a clear message.
    """
    driver = metal_driver_mod.MetalDriver()
    fake_binary = b"\x00\x01FAKE"

    # Ensure platform appears non-Darwin to trigger stub fallback path
    monkeypatch.setattr(sys, "platform", "linux", raising=False)

    handle = driver.load_binary(fake_binary)
    assert handle.is_stub is True, "When runtime unavailable, handle.is_stub must be True"
    assert handle.binary_bytes == fake_binary, "Stub handle must retain provided bytes"
    assert isinstance(handle.metadata, dict), "Stub handle metadata must be a dict"
    assert handle.metadata.get("library_size", None) == len(fake_binary)

    # Using the stub to launch a kernel should raise a RuntimeError with descriptive text
    with pytest.raises(RuntimeError) as exc:
        handle.launch_kernel()
    assert "Metal runtime unavailable" in str(exc.value), "Stub launch must raise RuntimeError explaining missing runtime"


def test_handle_metadata_and_shape(monkeypatch):
    """
    Validate that load_binary always returns a handle exposing binary_bytes, metadata and is_stub.
    The metadata['library_size'] must correctly reflect the binary length.
    This test is deterministic and does not depend on system Metal runtime availability.
    """
    driver = metal_driver_mod.MetalDriver()
    # Provide deterministic fake bytes
    binary = b"TEST_BINARY_BYTES"

    # Force a stable environment: set platform to linux to exercise stub path deterministically
    monkeypatch.setattr(sys, "platform", "linux", raising=False)

    handle = driver.load_binary(binary)
    # Basic shape assertions
    assert hasattr(handle, "binary_bytes"), "Handle must expose binary_bytes attribute"
    assert hasattr(handle, "metadata"), "Handle must expose metadata attribute"
    assert hasattr(handle, "is_stub"), "Handle must expose is_stub attribute"
    assert handle.binary_bytes == binary, "binary_bytes must equal provided input"
    assert isinstance(handle.metadata, dict), "metadata must be a dict"
    assert handle.metadata.get("library_size") == len(binary), "metadata.library_size must equal len(binary)"
    # is_stub should be True because we forced non-darwin platform
    assert handle.is_stub is True