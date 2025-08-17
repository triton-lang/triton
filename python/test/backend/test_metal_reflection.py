import shutil
import pytest

from third_party.metal.backend import compiler as metal_compiler_mod

SRC_EDGE_CASES = r'''
#include <metal_stdlib>
using namespace metal;

// unusual spacing and nested attributes
kernel void spaced_kernel(device   float*  out   [[  buffer(0)  ]]) { out[0] = 3.14f; }

// attribute order swapped, pointer with const
kernel void swapped_attr_kernel([[buffer(1)]] constant int *count) { }

// missing name, complex qualifiers
kernel void complex_decl_kernel(device const volatile float* [[ buffer(2) ]]) { }

// no attributes at all (should still parse type/name)
kernel void no_attr_kernel(device int *vals, uint n) { }

// multiple attributes in same token
kernel void multi_attr_kernel(device float* data [[buffer(3)]] [[maybe_unused]]) { }
'''

def test_reflection_parses_edge_cases(monkeypatch):
    # Ensure compiler uses fake xcrun
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/xcrun")
    # Monkeypatch _run_metal_compile to create expected metallib outputs so compile() can proceed.
    def fake_run(cmd, msg):
        if "-o" in cmd:
            out = cmd[cmd.index("-o") + 1]
            with open(out, "wb") as f:
                f.write(b"DUMMY")
    monkeypatch.setattr(metal_compiler_mod, "_run_metal_compile", fake_run)

    comp = metal_compiler_mod.MetalCompiler()
    binary, meta = comp.compile(SRC_EDGE_CASES, options={}, reflection=True)

    assert isinstance(binary, (bytes, bytearray))
    assert "kernels" in meta
    names = {k["name"] for k in meta["kernels"]}
    expected = {"spaced_kernel", "swapped_attr_kernel", "complex_decl_kernel", "no_attr_kernel", "multi_attr_kernel"}
    assert expected.issubset(names)

    # Check some specific parsing outcomes
    sk = next(k for k in meta["kernels"] if k["name"] == "spaced_kernel")
    assert sk["args"][0]["buffer_index"] == 0
    assert sk["args"][0]["address_space"] == "device"

    sa = next(k for k in meta["kernels"] if k["name"] == "swapped_attr_kernel")
    assert sa["args"][0]["buffer_index"] == 1
    # may have captured name or None
    assert sa["args"][0]["type"] is not None

    ca = next(k for k in meta["kernels"] if k["name"] == "complex_decl_kernel")
    assert ca["args"][0]["buffer_index"] == 2

    na = next(k for k in meta["kernels"] if k["name"] == "no_attr_kernel")
    # Two args: first has address_space device, second is probably primitive uint
    assert len(na["args"]) >= 2

    ma = next(k for k in meta["kernels"] if k["name"] == "multi_attr_kernel")
    assert ma["args"][0]["buffer_index"] == 3