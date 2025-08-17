import os
import shutil
import tempfile
import pytest

from third_party.metal.backend import compiler as metal_compiler_mod

MINIMAL_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* a [[buffer(0)]]) { a[0] = 1.0; }
"""

def test_missing_xcrun_raises():
    # Simulate xcrun not being on PATH -> compiler should raise a RuntimeError early.
    monkey = pytest.MonkeyPatch()
    monkey.setattr(shutil, "which", lambda name: None)
    try:
        comp = metal_compiler_mod.MetalCompiler()
        with pytest.raises(RuntimeError) as exc:
            comp.compile(MINIMAL_SRC, options={})
        assert "xcrun" in str(exc.value) or "xcrun" in repr(exc.value)
    finally:
        monkey.undo()

def test_tempfile_creation_failure_raises(monkeypatch):
    # Simulate NamedTemporaryFile throwing during creation -> compiler surfaces RuntimeError.
    def fake_ntf(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fake_ntf)
    # Ensure xcrun presence check passes
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/xcrun")
    comp = metal_compiler_mod.MetalCompiler()
    with pytest.raises(RuntimeError) as exc:
        comp.compile(MINIMAL_SRC, options={})
    assert "Failed to create temporary source file" in str(exc.value)

def test_compile_propagates_subprocess_errors(monkeypatch, tmp_path):
    # If the underlying _run_metal_compile raises, compiler.compile should raise RuntimeError.
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/xcrun")

    def fake_run(cmd, msg):
        raise RuntimeError("Metal compilation failed: stdout=... stderr=bad")
    monkeypatch.setattr(metal_compiler_mod, "_run_metal_compile", fake_run)

    comp = metal_compiler_mod.MetalCompiler()
    with pytest.raises(RuntimeError) as exc:
        comp.compile(MINIMAL_SRC, options={})

def test_compile_reflection_parses_various_argument_forms(monkeypatch):
    """
    Ensure the conservative reflection parser extracts kernel names and argument metadata
    (raw, type, name, buffer_index, address_space) for a number of tricky declarations.
    """
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/xcrun")

    # Create a fake _run_metal_compile that writes out the expected metallib file so compile() can read it.
    def fake_run(cmd, msg):
        # Find the '-o' output path and write a dummy metallib byte blob there.
        try:
            if "-o" in cmd:
                out = cmd[cmd.index("-o") + 1]
                # Ensure directory exists
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "wb") as f:
                    f.write(b"DUMMY_METALLIB")
        except Exception:
            # make sure failures in test helper surface as test failures
            raise
    monkeypatch.setattr(metal_compiler_mod, "_run_metal_compile", fake_run)

    # Kernel source with varied/edge-case argument declarations:
    src = r'''
    #include <metal_stdlib>
    using namespace metal;
    // simple buffer annotation with spaces
    kernel void writer_kernel(device float *out [[ buffer(0) ]]) { out[0] = 42.0; }

    // missing parameter name (edge case)
    kernel void unnamed_arg_kernel(device uint * [[ buffer(1) ]]) { }

    // threadgroup and constant address spaces
    kernel void tg_const_kernel(threadgroup float *local [[buffer(2)]], constant int count) { }

    // multiple attributes and pointer stars in types
    kernel void complex_kernel(device const float** ptr [[buffer(3)]], device int *arr [[buffer(4)]]) { }
    '''

    comp = metal_compiler_mod.MetalCompiler()
    binary, metadata = comp.compile(src, options={}, reflection=True)

    # Basic binary sanity
    assert isinstance(binary, (bytes, bytearray))
    assert "kernels" in metadata
    kernels = metadata["kernels"]
    # Expect at least the declared kernels
    names = {k["name"] for k in kernels}
    assert "writer_kernel" in names
    assert "unnamed_arg_kernel" in names
    assert "tg_const_kernel" in names
    assert "complex_kernel" in names

    # Find writer_kernel and assert parsing of buffer index and address space
    wk = next(k for k in kernels if k["name"] == "writer_kernel")
    assert len(wk["args"]) == 1
    arg0 = wk["args"][0]
    assert arg0["buffer_index"] == 0
    assert arg0["address_space"] == "device"
    assert arg0["name"] in ("out",)  # parser should capture 'out'

    # unnamed_arg_kernel: param name may be None
    uk = next(k for k in kernels if k["name"] == "unnamed_arg_kernel")
    assert len(uk["args"]) == 1
    assert uk["args"][0]["buffer_index"] == 1
    # name may be None for this edge case
    assert uk["args"][0]["name"] is None or uk["args"][0]["name"] == ""

    # threadgroup + constant inference
    tc = next(k for k in kernels if k["name"] == "tg_const_kernel")
    assert any(a["address_space"] == "threadgroup" or a["address_space"] == "constant" for a in tc["args"])
    # check buffer indices for threadgroup arg
    assert tc["args"][0]["buffer_index"] == 2

    # complex_kernel pointer parsing sanity
    ck = next(k for k in kernels if k["name"] == "complex_kernel")
    assert ck["args"][0]["buffer_index"] == 3
    assert ck["args"][1]["buffer_index"] == 4