// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions 2>&1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @for_arg_used_in_nested_for_bound
  tt.func public @for_arg_used_in_nested_for_bound(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    // CHECK: scf.for
    // CHECK-SAME: iter_args(%[[OUTER_ARG0:.*]] = %arg0, %[[OUTER_ARG1:.*]] = %c0_i32)
    %0:2 = scf.for %iv = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%outer_arg0 = %arg0, %outer_arg1 = %c0_i32) -> (i32, i32) : i32 {
      %1 = arith.remsi %outer_arg1, %arg2 : i32
      %2 = arith.addi %1, %c1_i32 : i32
      %3 = arith.muli %2, %c256_i32 : i32
      // CHECK: scf.for %[[INNER_IV:.*]] = %c0_i32 to %3 step %c1_i32
      %inner_result = scf.for %inner_iv = %c0_i32 to %3 step %c1_i32 iter_args(%inner_acc = %outer_arg0) -> (i32) : i32 {
        %sum = arith.addi %inner_acc, %inner_iv : i32
        scf.yield %sum : i32
      }
      %4 = arith.addi %outer_arg0, %c1_i32 : i32
      scf.yield %inner_result, %4 : i32, i32
    }
    tt.return %0#0 : i32
  }
}
