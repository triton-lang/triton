; RUN: python - <<'PY'
; REQUIRES: darwin
import sys, json
from third_party.metal.backend.compiler import MetalCompiler
# Kernel with different argument address spaces and annotations to exercise
# the lightweight reflection parser.
source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void mix(const device float* a [[ buffer(0) ]],
                device float* b [[ buffer(1) ]],
                constant uint &count [[ buffer(2) ]]) {
  // body omitted
}
"""
try:
    comp = MetalCompiler()
    binary, meta = comp.compile(source, {}, reflection=True)
    # Print a deterministic, compact JSON for FileCheck
    print("REFLECTION_OK")
    print(json.dumps(meta, sort_keys=True))
except Exception as e:
    print("REFLECTION_FAIL:", e)
    sys.exit(1)
PY
; CHECK: REFLECTION_OK
; CHECK: "kernels"
; CHECK: "name"
; CHECK: "mix"
; CHECK: "buffer_index"