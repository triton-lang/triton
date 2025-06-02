// RUN: triton-opt --split-input-file --convert-proton-to-protongpu="max-shared-mem-size=32768" --move-proton-stores-to-end -canonicalize -cse %s | FileCheck %s

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: simple_record
  // CHECK: %[[SCRATCH:.*]] = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
  // CHECK: %[[BUF:.*]] = ttg.local_alloc  : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
  // CHECK: %[[SEGMENT:.*]] = proton_gpu.segment_alloc %[[BUF]]
  // CHECK: %[[START:.*]] = proton_gpu.read_counter : i32
  // CHECK: proton_gpu.circular_store start %[[SEGMENT]], %[[START]] {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK: %[[END:.*]] = proton_gpu.read_counter : i32
  // CHECK: proton_gpu.circular_store end %[[SEGMENT]], %[[END]] {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK: gpu.barrier
  // CHECK: proton_gpu.finalize %[[SEGMENT]], %[[SCRATCH]] : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
  // CHECK: tt.return
  tt.func @simple_record() {
    proton.record start "name0"
    proton.record end "name0"
    tt.return
  }
}
