// RUN: triton-opt --split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: proton_record
  tt.func @proton_record() {
    // CHECK: proton.record start "name0"
    // CHECK: proton.record end "name0"
    // CHECK-NEXT: tt.return
    proton.record start "name0"
    proton.record end "name0"
    tt.return
  }
} // end module

// -----
