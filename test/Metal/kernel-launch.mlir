; RUN: python - <<'PY'
; REQUIRES: darwin
import sys, json, traceback
import numpy as _np
from third_party.metal.backend.compiler import MetalCompiler
from third_party.metal.backend.runtime import bind_library
# Small kernel that writes index into output buffer
source = r"""
#include <metal_stdlib>
using namespace metal;
kernel void write_index(device int* out [[ buffer(0) ]],
                        uint gid [[ thread_position_in_grid ]]) {
  out[gid] = (int)gid;
}
"""
try:
    comp = MetalCompiler()
    metallib, meta = comp.compile(source, {}, reflection=True)
    handle = bind_library(metallib, metadata=meta)
    # If runtime is not fully available bind_library may return a stub
    if getattr(handle, "is_stub", True):
        print("LAUNCH_SKIPPED_STUB")
        sys.exit(0)
    # Prepare a small numpy array as buffer (4 ints)
    arr = _np.zeros(4, dtype=_np.int32)
    # Launch kernel: total threads = 4, threads per threadgroup = 4
    res = handle.launch_kernel(
        name="write_index",
        args=(arr,),
        grid=(4, 1, 1),
        block=(4, 1, 1),
        timeout=5.0,
        explicit_readback=True,
    )
    if res.get("status") != "ok":
        print("LAUNCH_FAIL_STATUS", res)
        sys.exit(1)
    # On success, arr should contain [0,1,2,3]
    print("LAUNCH_OK")
    print(json.dumps(arr.tolist()))
except Exception as e:
    traceback.print_exc()
    print("LAUNCH_FAIL:", e)
    sys.exit(1)
PY
; CHECK: LAUNCH_OK
; CHECK: [0, 1, 2, 3]