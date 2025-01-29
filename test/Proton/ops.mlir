// RUN: triton-opt --split-input-file %s -cse -canonicalize | FileCheck %s

module {
  // CHECK-LABEL: proton_record
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

#shared = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
#smem = #ttg.shared_memory
module {
  // CHECK-LABEL: circular_record
  tt.func @circular_record() {
    // CHECK: proton.init
    // CHECK-NEXT: ttg.local_alloc
    // CHECK-NEXT: ttg.global_scratch_alloc{{.*}}nbytes = 231487
    // CHECK-NEXT: proton.circular_record
    // CHECK-NEXT: proton.finalize{{.*}}{size = 231487 : i32}
    // CHECK-NEXT: tt.return
    %0 = proton.init : !tt.ptr<i32>
    %1 = ttg.local_alloc  {allocation.offset = 213016 : i32} : () -> !ttg.memdesc<4096xi32, #shared, #smem, mutable>
    %2 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 231487 : i32, ttg.global_scratch_memory_offset = 384 : i32} : !tt.ptr<i32>
    proton.circular_record %1, %0 {isStart = false, regionId = 1 : i32} : !ttg.memdesc<4096xi32, #shared, #smem, mutable>, !tt.ptr<i32>
    proton.finalize %1, %0, %2 {size = 231487 : i32} : !ttg.memdesc<4096xi32, #shared, #smem, mutable>, !tt.ptr<i32>, !tt.ptr<i32>
    tt.return
  }
} // end module
