// RUN: triton-opt --split-input-file --allocate-proton-device-buffer %s | FileCheck %s

module {
  tt.func @proton_record() {
    // CHECK: %0 = proton.init_device_buffer() {bufferSize = 1024 : i32}
    // CHECK-NEXT: proton.record() {isStart = true, regionId = 1 : i32}
    // CHECK-NEXT: proton.record() {isStart = false, regionId = 1 : i32}
    // CHECK-NEXT: tt.return
    proton.record() {isStart = true, regionId = 1 : i32}
    proton.record() {isStart = false, regionId = 1 : i32}
    tt.return
  }
} // end module

// -----
