; REQUIRES: darwin
; RUN: python - <<'PY'
; import sys
; try:
;     from third_party.metal.backend.compiler import MetalCompiler
;     from third_party.metal.backend.runtime import bind_library, MetalRuntimeError
;     source = r"""
;     #include <metal_stdlib>
;     using namespace metal;
;     kernel void trivial(device float* a [[ buffer(0) ]]) { }
;     """
;     comp = MetalCompiler()
;     bin_bytes, meta = comp.compile(source, {}, reflection=True)
;     try:
;         handle = bind_library(bin_bytes)
;         print("DEVICE_OK")
;     except MetalRuntimeError as e:
;         print("DEVICE_FAIL")
;         print(str(e))
; except Exception as e:
;     print("DEVICE_FAIL")
;     print(str(e))
;     sys.exit(1)
; PY
; CHECK: DEVICE_OK
; CHECK: DEVICE_FAIL