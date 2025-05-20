// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -triton-test-loop-peeling -canonicalize | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @simple_loop_index
// CHECK: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index) -> f32
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[NUB:.*]] = arith.subi %[[UB]], %[[STEP]]
// CHECK: %[[FOR:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[NUB]] step %[[STEP]]
// CHECK: scf.yield
// CHECK: %[[COND:.*]] = arith.cmpi slt, %[[LB]], %[[UB]]
// CHECK: %[[IF:.*]] = scf.if %[[COND]]
// CHECK:   %[[DEF:.*]] = "def"() : () -> f32
// CHECK:   %[[RES:.*]] = arith.addf %[[FOR]], %[[DEF]] : f32
// CHECK:   scf.yield %[[RES]] : f32
// CHECK: else
// CHECK:   scf.yield %[[CST]] : f32
// CHECK: tt.return %[[IF]] : f32
tt.func @simple_loop_index(%lb : index, %ub : index, %step : index) -> f32 {
  %init = arith.constant 0.00e+00 : f32
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (f32) {
    %a = "def"() : () -> f32
    %res = arith.addf %acc, %a : f32
    scf.yield %res : f32
  } {__test_peel_epilogue_iterations = 1 : i32}

  tt.return %loop#0 : f32
}
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @simple_loop_i32
// CHECK: (%[[LB:.*]]: i32, %[[UB:.*]]: i32, %[[STEP:.*]]: i32) -> f32
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[NUB:.*]] = arith.subi %[[UB]], %[[STEP]]
// CHECK: %[[FOR:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[NUB]] step %[[STEP]]
// CHECK: scf.yield
// CHECK: %[[COND:.*]] = arith.cmpi slt, %[[LB]], %[[UB]]
// CHECK: %[[IF:.*]] = scf.if %[[COND]]
// CHECK:   %[[DEF:.*]] = "def"() : () -> f32
// CHECK:   %[[RES:.*]] = arith.addf %[[FOR]], %[[DEF]] : f32
// CHECK:   scf.yield %[[RES]] : f32
// CHECK: else
// CHECK:   scf.yield %[[CST]] : f32
// CHECK: tt.return %[[IF]] : f32
tt.func @simple_loop_i32(%lb : i32, %ub : i32, %step : i32) -> f32 {
  %init = arith.constant 0.00e+00 : f32
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (f32) : i32 {
    %a = "def"() : () -> f32
    %res = arith.addf %acc, %a : f32
    scf.yield %res : f32
  } {__test_peel_epilogue_iterations = 1 : i32}

  tt.return %loop#0 : f32
}
}
