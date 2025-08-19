; RUN: python - <<'PY'
; import sys, json
; try:
;     from third_party.metal.backend.runtime import bind_library
;     handle = bind_library(b"0")
;     if getattr(handle, "is_stub", False):
;         print("STUB_OK")
;         print(json.dumps(handle.metadata, sort_keys=True))
;     else:
;         print("SKIP_NATIVE")
; except Exception as e:
;     print("STUB_FAIL:", e)
;     sys.exit(1)
; PY
; CHECK: STUB_OK
; CHECK: "platform"
; CHECK: "library_size"