; RUN: python - <<'PY'
; import sys, json
; try:
;     import third_party.metal.backend.runtime as rt
;     names = [
;         "bind_library",
;         "MetalLibraryHandle",
;         "MetalRuntimeError",
;         "KernelNotFoundError",
;         "PipelineCreationError",
;         "ResourceError",
;     ]
;     print("EXPORTS_OK")
;     for n in names:
;         print(f"{n} {hasattr(rt, n)}")
; except Exception as e:
;     print("IMPORT_FAIL:", e)
;     sys.exit(1)
; PY
; CHECK: EXPORTS_OK
; CHECK-LABEL: bind_library True
; CHECK: MetalLibraryHandle True
; CHECK: MetalRuntimeError True