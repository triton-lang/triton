// RUN: triton-opt --split-input-file %s -cse -canonicalize --proton-lowering-pass | FileCheck %s

module {
  tt.func @proton_record() {
    // CHECK: %0 = proton.buffer_alloc() {bufferSize = 1024 : i32} : !tt.ptr<i8>
    // CHECK-NEXT: proton.record() {isStart = true, regionId = 1 : i32}
    // CHECK-NEXT: proton.record() {isStart = false, regionId = 1 : i32}

    // CHECK-NEXT: tt.return
    proton.record() {isStart = true,  regionId = 1 : i32}
    proton.record() {isStart = false, regionId = 1 : i32}
    tt.return
  }
} // end module

// -----
