// RUN: triton-opt --split-input-file %s -proton-lowering="max-shared-mem=1024 scratch-mem=1024 alignment=32" | FileCheck %s

// CHECK: module attributes{{.*}}ttg.global_scratch_memory_alignment = 32{{.*}}ttg.global_scratch_memory_size = 1024
module attributes {"ttg.num-warps" = 8 : i32, ttg.shared = 512 : i32} {
  // CHECK-LABEL: sufficient_global_scratch_size
  tt.func @sufficient_global_scratch_size() {
    proton.record() {isStart = true, regionId = 1 : i32}
    // CHECK: proton.finalize{{.*}}{size = 1024 : i32}
    // CHECK-NEXT: tt.return
    tt.return
  }
} // end module

// -----

// CHECK: module attributes{{.*}}ttg.global_scratch_memory_alignment = 128{{.*}}ttg.global_scratch_memory_size = 1280
module attributes {ttg.global_scratch_memory_alignment = 128, ttg.global_scratch_memory_size = 150, "ttg.num-warps" = 8 : i32, ttg.shared = 512 : i32} {
  // CHECK-LABEL: unalign_global_scratch_alloc
  tt.func @unalign_global_scratch_alloc() {
    proton.record() {isStart = true, regionId = 1 : i32}
    // CHECK: ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 1024 : i32, ttg.global_scratch_memory_offset = 256 : i32}
    // CHECK: proton.finalize{{.*}}{size = 1024 : i32}
    // CHECK-NEXT: tt.return
    tt.return
  }
} // end module

// -----

// CHECK: module attributes{{.*}}ttg.global_scratch_memory_alignment = 64{{.*}}ttg.global_scratch_memory_size = 1152
module attributes {ttg.global_scratch_memory_alignment = 64, ttg.global_scratch_memory_size = 128, "ttg.num-warps" = 8 : i32, ttg.shared = 512 : i32} {
  // CHECK-LABEL: align_global_scratch_alloc
  tt.func @align_global_scratch_alloc() {
    proton.record() {isStart = true, regionId = 1 : i32}
    // CHECK: %[[ARG0:.*]] = proton.init
    // CHECK-NEXT: %[[ARG1:.*]] = ttg.local_alloc
    // CHECK-NEXT: %[[ARG2:.*]] = ttg.global_scratch_alloc {alignment = 64 : i32, nbytes = 1024 : i32, ttg.global_scratch_memory_offset = 128 : i32} : !tt.ptr<i32>
    // CHECK-NEXT: proton.circular_record %[[ARG1]], %[[ARG0]] {isStart = true, regionId = 1 : i32} : !ttg.memdesc<128xi32, #shared, #smem, mutable>, !tt.ptr<i32>
    // CHECK-NEXT: proton.finalize %[[ARG1]], %[[ARG0]],  %[[ARG2]] {size = 1024 : i32} : !ttg.memdesc<128xi32, #shared, #smem, mutable>, !tt.ptr<i32>, !tt.ptr<i32>
    // CHECK-NEXT: tt.return
    tt.return
  }
} // end module
