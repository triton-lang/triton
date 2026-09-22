// RUN: triton-opt %s -allow-unregistered-dialect -verify-diagnostics -o /dev/null
// RUN: not triton-opt %s -allow-unregistered-dialect -tritongpu-partition-loops -o /dev/null 2>&1 | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @partition_attrs_are_verified_only_when_consumed(
      %lb: i32, %ub: i32, %step: i32) {
    scf.for %i = %lb to %ub step %step : i32 {
      %0 = arith.addi %i, %i {ttg.partition = array<i32: 1, 0>} : i32
      "use"(%0) {ttg.partition = array<i32: 0, 1>} : (i32) -> ()
    } {ttg.partition.stages = [0, 0], ttg.warp_specialize.tag = 0 : i32,
       ttg.partition = array<i32: 0, 1>}
    tt.return
  }
}

// CHECK: error: 'arith.addi' op partition ids not in sorted order in attribute ttg.partition
