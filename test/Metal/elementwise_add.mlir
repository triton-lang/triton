; RUN: python - <<'PY'
; REQUIRES: darwin
import sys, shutil, json
from third_party.metal.backend.compiler import MetalCompiler
# Minimal Metal kernel source for a smoke compile
source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void elementwise_add(device float* a [[ buffer(0) ]],
                            device float* b [[ buffer(1) ]],
                            device float* out [[ buffer(2) ]],
                            uint gid [[ thread_position_in_grid ]]) {
  out[gid] = a[gid] + b[gid];
}
"""
try:
    comp = MetalCompiler()
    # Attempt a real compile; on non-darwin or missing tools this will raise,
    # but lit will skip this test on non-darwin platforms because of REQUIRES.
    binary, reflection = comp.compile(source, {})
    # Signal success and print a small artifact for FileCheck
    print("COMPILE_OK")
    print("KERNEL_NAME:", reflection["kernels"][0]["name"])
except Exception as e:
    # Bubble up error to cause test failure on darwin if compilation truly fails.
    print("COMPILE_FAIL:", e)
    sys.exit(1)
PY
; CHECK: COMPILE_OK
; CHECK: KERNEL_NAME: elementwise_add
