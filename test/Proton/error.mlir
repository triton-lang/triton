// RUN: triton-opt --split-input-file %s -proton-lowering="max-shared-mem=1024 scratch-mem=512 alignment=128" -verify-diagnostics

module attributes {"ttg.num-warps" = 8 : i32, ttg.shared = 128 : i32} {
  // expected-error @+1 {{Global scratch memory for proton is not large enough}}
  tt.func @insufficient_global_scratch() {
    proton.record() {isStart = true, regionId = 1 : i32}
    tt.return
  }
} // end module
