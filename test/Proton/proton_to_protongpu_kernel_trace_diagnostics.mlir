// RUN: triton-opt --split-input-file -convert-proton-to-protongpu="max-shared-mem-size=32768 kernel-trace-mode=true" --verify-diagnostics %s >/dev/null

module attributes {"ttg.num-warps" = 8 : i32} {
  tt.func @kernel_trace_rejects_records() {
    // expected-error @below {{trace_mode=kernel currently supports launch-level timing only and cannot be combined with proton.record ops}}
    proton.record start "name0"
    tt.return
  }
}
