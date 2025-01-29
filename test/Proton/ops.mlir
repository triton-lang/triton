// RUN: triton-opt --split-input-file --allocate-proton-device-buffer %s | FileCheck %s

module {
  tt.func @proton_record() {
    // CHECK: %0 = proton.buffer_alloc() {bufferSize = 1024 : i32} : !tt.ptr<i8>
    // CHECK-NEXT: proton.record() {isStart = true, regionId = 1 : i32}
    // CHECK-NEXT: proton.record() {isStart = false, regionId = 1 : i32}
    // CHECK-NEXT: proton.record() {isStart = true, regionId = 2 : i32, useDeviceBuffer = true}
    // CHECK-NEXT: proton.record() {isStart = false, regionId = 2 : i32, useDeviceBuffer = true}

    // CHECK-NEXT: tt.return
    proton.record() {isStart = true,  regionId = 1 : i32}
    proton.record() {isStart = false, regionId = 1 : i32}
    proton.record() {isStart = true,  regionId = 2 : i32, useDeviceBuffer = true}
    proton.record() {isStart = false, regionId = 2 : i32, useDeviceBuffer = true}
    tt.return
  }
} // end module

// -----
