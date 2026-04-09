// RUN: triton-opt --split-input-file -convert-proton-to-protongpu="max-shared-mem-size=32768 kernel-trace-mode=true" -canonicalize -cse %s | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: module attributes
  // CHECK-SAME: ttg.profile_scratch_buffer_unit = 1 : i32
  // CHECK-LABEL: kernel_trace_no_record
  // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32, third_party_allocation} : !tt.ptr<i64>
  // CHECK: proton_gpu.initialize %[[SCRATCH]] : !tt.ptr<i64>
  // CHECK: %[[BUF:.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
  // CHECK: %[[SEGMENT:.*]] = proton_gpu.segment_alloc %[[BUF]]
  // CHECK-NOT: proton_gpu.read_counter
  // CHECK-NOT: proton_gpu.circular_store
  // CHECK: ttg.barrier local|global_read|global_write
  // CHECK: proton_gpu.finalize %[[SEGMENT]], %[[SCRATCH]] : !proton_gpu.segment<4, #smem, warp>, !tt.ptr<i64>
  // CHECK: tt.return
  tt.func @kernel_trace_no_record() {
    tt.return
  }
}
