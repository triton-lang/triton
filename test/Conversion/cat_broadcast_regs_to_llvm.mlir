// RUN: triton-opt %s --convert-triton-gpu-to-llvm=compute-capability=100 2>&1 | FileCheck %s

// Regression test for tt.cat lowering when the result encoding has broadcasted
// register bits (i.e. the linear layout has zero register bases).
//
// Previously this could crash in packLLElements due to a mismatch between the
// number of values produced by CatOpConversion and the LLVM struct type size.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#linear_bcast = #ttg.linear<{register = [[1], [0], [8], [1024]],
                            lane = [[2], [4], [16], [32], [64]],
                            warp = [[128], [256], [512]],
                            block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @cat_broadcast
  tt.func @cat_broadcast() {
    %c0_i32 = arith.constant 0 : i32
    %lhs = tt.splat %c0_i32 : i32 -> tensor<1024xi32, #blocked>
    %rhs = tt.splat %c0_i32 : i32 -> tensor<1024xi32, #blocked>
    %cat = tt.cat %lhs, %rhs : tensor<1024xi32, #blocked> -> tensor<2048xi32, #linear_bcast>
    tt.return
  }
}
