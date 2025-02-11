// RUN: triton-opt %s -split-input-file -tritongpu-test-pipeline-lower-loop -canonicalize | FileCheck %s

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @one_dep
// CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1
// CHECK-DAG: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32
// CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]], %[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
// CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
// CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_VAL:.*]] = ttg.local_load %[[A_EXT]] token %[[A_TOK3]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[RES:.*]] = arith.addf %[[ACC]], %[[A_VAL]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   scf.yield %[[RES]], %[[INS_NEXT]], %[[EXT_NEXT]]

tt.func @one_dep_async(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> tensor<128x32xf16, #A> {
  %init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (tensor<128x32xf16, #A>) {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    %res = arith.addf %acc, %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x32xf16, #A>
    scf.yield %res : tensor<128x32xf16, #A>
  }
  tt.return %loop#0 : tensor<128x32xf16, #A>
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @one_dep_sync
// CHECK: %[[INIT:.*]] = arith.constant dense<0.000000e+00>
// CHECK: scf.for
// CHECK:   tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
tt.func @one_dep_sync(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<1x!tt.ptr<f16>, #A>) -> tensor<1xf16, #A> {
  %init = arith.constant dense<0.00e+00> : tensor<1xf16, #A>
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (tensor<1xf16, #A>) {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x!tt.ptr<f16>, #A>
    %res = arith.addf %acc, %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1xf16, #A>
    scf.yield %res : tensor<1xf16, #A>
  }
  tt.return %loop#0 : tensor<1xf16, #A>
}
}


// Tests to write:
// - load and mma layouts that don't match
// - load with multiple uses
