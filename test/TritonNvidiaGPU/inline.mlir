// RUN: triton-opt %s -inline | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @inline_ttng_ops
tt.func public @inline_ttng_ops() {
  // CHECK-NEXT: ttg.local_alloc
  // CHECK-NEXT: ttng.init_barrier
  tt.call @function_with_ttng_ops() : () -> ()
  tt.return
}

tt.func private @function_with_ttng_ops() {
  %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
  ttng.init_barrier %0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
  tt.return
}

}
