; RUN: python - <<'PY'
; import sys
; try:
;     from third_party.metal.backend.runtime import bind_library
;     handle = bind_library(b"0")
;     if getattr(handle, "is_stub", False):
;         try:
;             # Call with minimal/empty args — stub should raise RuntimeError
;             handle.launch_kernel(name="dummy", args=[])
;             print("NO_RAISE")
;         except RuntimeError as e:
;             print("RAISE_OK")
;             print(str(e))
;         except Exception as e:
;             print("RAISE_OTHER:", e)
;     else:
;         print("SKIP_NATIVE")
; except Exception as e:
;     print("TEST_FAIL:", e)
;     sys.exit(1)
; PY
; CHECK: RAISE_OK
; CHECK: Metal runtime unavailable