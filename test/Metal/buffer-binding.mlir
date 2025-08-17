; RUN: python - <<'PY'
; REQUIRES: darwin
import sys, json, re
from third_party.metal.backend.compiler import MetalCompiler
# Test explicit buffer index parsing
source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void bind_test(device float* a [[ buffer(3) ]],
                      device int* meta [[ buffer(7) ]]) {
}
"""
try:
    comp = MetalCompiler()
    _, meta = comp.compile(source, {}, reflection=True)
    print("BINDING_OK")
    # Expect metadata.kernels[0].args to contain buffer_index 3 and 7
    print(json.dumps(meta, sort_keys=True))
except Exception as e:
    print("BINDING_FAIL:", e)
    sys.exit(1)
PY
; CHECK: BINDING_OK
; CHECK: "buffer_index"
; CHECK: 3
; CHECK: 7