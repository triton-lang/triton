// RUN: triton-opt --split-input-file -convert-proton-to-protongpu="max-shared-mem-size=32768" -proton-schedule-buffer-store -canonicalize -cse %s | FileCheck %s

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: simple_record
  // CHECK: %[[SCRATCH:.*]] = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
  // CHECK-NEXT: %[[BUF:.*]] = ttg.local_alloc  : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
  // CHECK-NEXT: %[[SEGMENT:.*]] = proton_gpu.segment_alloc %[[BUF]]
  // CHECK-NEXT: %[[START:.*]] = proton_gpu.read_counter : i32
  // CHECK-NEXT: %[[END:.*]] = proton_gpu.read_counter : i32
  // CHECK-NEXT: proton_gpu.circular_store start %[[SEGMENT]], %[[START]] {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK-NEXT: proton_gpu.circular_store end %[[SEGMENT]], %[[END]] {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: proton_gpu.finalize %[[SEGMENT]], %[[SCRATCH]] : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
  // CHECK-NEXT: tt.return
  tt.func @simple_record() {
    proton.record start "name0"
    proton.record end "name0"
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: simple_record
  // CHECK: %[[SCRATCH:.*]] = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
  // CHECK-NEXT: %[[BUF:.*]] = ttg.local_alloc  : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
  // CHECK-NEXT: %[[SEGMENT:.*]] = proton_gpu.segment_alloc %[[BUF]]
  // CHECK-NEXT: %[[START1:.*]] = proton_gpu.read_counter : i32
  // CHECK-NEXT: %[[START2:.*]] = proton_gpu.read_counter : i32
  // CHECK-NEXT: %[[END2:.*]] = proton_gpu.read_counter : i32
  // CHECK-NEXT: proton_gpu.circular_store start %[[SEGMENT]], %[[START2]] {scopeId = 1 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK-NEXT: proton_gpu.circular_store end %[[SEGMENT]], %[[END2]] {scopeId = 1 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK-NEXT: %[[END1:.*]] = proton_gpu.read_counter : i32
  // CHECK-NEXT: proton_gpu.circular_store start %[[SEGMENT]], %[[START1]] {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK-NEXT: proton_gpu.circular_store end %[[SEGMENT]], %[[END1]] {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: proton_gpu.finalize %[[SEGMENT]], %[[SCRATCH]] : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
  // CHECK-NEXT: tt.return
  tt.func @simple_record() {
    proton.record start "name0"
    proton.record start "name1"
    proton.record end "name1"
    proton.record end "name0"
    tt.return
  }
}
