// RUN: triton-opt %s -allow-unregistered-dialect --tritongpu-global-scratch-memory-allocation --convert-triton-gpu-to-llvm | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @global_scratch_alloc_warpgroup(%arg0: !llvm.ptr<1>)
  tt.func @global_scratch_alloc_warpgroup() {
    // CHECK-NEXT: ttg.warp_specialize(%arg0)
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    // CHECK: partition0(%arg1: !llvm.ptr<1>)
    partition0() num_warps(1) {
      // CHECK-COUNT-2: llvm.getelementptr %arg1
      %0 = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 100 : i32, ttg.global_scratch_memory_offset = 0 : i32} : !tt.ptr<i8>
      %1 = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 100 : i32, ttg.global_scratch_memory_offset = 0 : i32} : !tt.ptr<i8>
      "use"(%0, %1) : (!tt.ptr<i8>, !tt.ptr<i8>) -> ()
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
