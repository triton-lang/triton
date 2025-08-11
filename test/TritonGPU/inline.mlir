// RUN: triton-opt %s -inline | FileCheck %s

#smem = #ttg.shared_memory
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// CHECK-LABEL: @inline_in_warp_specialize
tt.func public @inline_in_warp_specialize(%arg0: !ttg.memdesc<1xi32, #shared, #smem, mutable>) {
  ttg.warp_specialize(%arg0)
  default {
    ttg.warp_yield
  }
  // CHECK: partition0
  partition0(%arg1: !ttg.memdesc<1xi32, #shared, #smem, mutable>) num_warps(4) {
    // CHECK-NEXT: %cst = arith.constant dense<1> : tensor<1xi32>
    // CHECK-NEXT: local_store %cst, %arg1
    tt.call @store_1(%arg1) : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
    // CHECK-NEXT: warp_return
    ttg.warp_return
  } : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
  tt.return
}

tt.func private @store_1(%arg0: !ttg.memdesc<1xi32, #shared, #smem, mutable>) {
  %cst = arith.constant dense<1> : tensor<1xi32>
  ttg.local_store %cst, %arg0 : tensor<1xi32> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
  tt.return
}
