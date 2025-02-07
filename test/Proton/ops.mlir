// RUN: triton-opt --split-input-file %s -cse -canonicalize --proton-lowering-pass | FileCheck %s

module {
  tt.func @proton_record() {
    // CHECK: proton.record() {isStart = true, regionId = 1 : i32}
    // CHECK-NEXT: proton.record() {isStart = false, regionId = 1 : i32}
    // CHECK-NEXT: tt.return
    proton.record() {isStart = true, regionId = 1 : i32}
    proton.record() {isStart = false, regionId = 1 : i32}
    tt.return
  }
} // end module

// -----
