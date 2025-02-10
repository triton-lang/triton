// RUN: triton-opt --split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: proton_init_scope
  tt.func @proton_init_scope() {
    // CHECK: proton.init_scope "name0" : i32
    // CHECK-NEXT: tt.return
    %0 = proton.init_scope "name0" : i32
    tt.return
  }
  // CHECK-LABEL: proton_record
  tt.func @proton_record() {
    // CHECK: proton.record start %0
    // CHECK: proton.record end %0
    // CHECK-NEXT: tt.return
    %0 = proton.init_scope "name0" : i32
    proton.record start %0
    proton.record end %0
    tt.return
  }
} // end module

// -----
