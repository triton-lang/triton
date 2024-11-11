// RUN: triton-opt --split-input-file %s -cse -canonicalize | FileCheck %s

module {
  // CHECK-LABEL: proton_record
  tt.func @proton_record() {
    // CHECK: proton.record <1, "start", "cycle", "warpgroup">
    // CHECK-NEXT: proton.record <1, "end", "cycle", "warpgroup">
    // CHECK-NEXT: tt.return
    proton.record <1, "start", "cycle", "warpgroup">
    proton.record <1, "end", "cycle", "warpgroup">
    tt.return
  }
} // end module

// -----
