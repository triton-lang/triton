// RUN: triton-opt --split-input-file --proton-lowering %s | FileCheck %s

module {
// CHECK: module
}

// -----

module {
  // CHECK-LABEL: no_record
  tt.func @no_record() {
    // CHECK: tt.return
    tt.return
  }
}
