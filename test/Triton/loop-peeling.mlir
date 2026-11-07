// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -triton-test-loop-peeling -canonicalize | FileCheck %s

// CHECK-LABEL: @unsigned_loop
// CHECK-SAME: %[[LB:.*]]: i64, %[[UB:.*]]: i64, %[[STEP:.*]]: i64
tt.func @unsigned_loop(%lb: i64, %ub: i64, %step: i64) -> i64 {
  %zero = arith.constant 0 : i64
  // CHECK: arith.cmpi ule, %[[STEP]], %[[UB]]
  // CHECK: arith.select
  // CHECK: scf.for unsigned
  // CHECK: arith.divui
  // CHECK: arith.cmpi ult, %[[LB]], %[[UB]]
  // CHECK: scf.if
  %result = scf.for unsigned %i = %lb to %ub step %step iter_args(%acc = %zero) -> i64 : i64 {
    %value = "body"(%i) : (i64) -> i64
    %next = arith.addi %acc, %value : i64
    scf.yield %next : i64
  } {__test_peel_epilogue}
  tt.return %result : i64
}

// -----

// CHECK-LABEL: @unsigned_empty
// CHECK-NOT: "body"
// CHECK: tt.return
tt.func @unsigned_empty() {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  scf.for unsigned %i = %zero to %zero step %one : i32 {
    "body"(%i) : (i32) -> ()
  } {__test_peel_epilogue}
  tt.return
}

// -----

// CHECK-LABEL: @unsigned_single_iteration
// CHECK-NOT: scf.for
// CHECK: "body"
// CHECK-NOT: "body"
// CHECK: tt.return
tt.func @unsigned_single_iteration() {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32
  scf.for unsigned %i = %zero to %one step %two : i32 {
    "body"(%i) : (i32) -> ()
  } {__test_peel_epilogue}
  tt.return
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @simple_loop_i32
// CHECK: (%[[LB:.*]]: i32, %[[UB:.*]]: i32, %[[STEP:.*]]: i32) -> f32
// CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[NUB:.*]] = arith.subi %[[UB]], %[[STEP]]
// CHECK: %[[FOR:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[NUB]] step %[[STEP]]
// CHECK: scf.yield
// CHECK: %[[RANGE:.*]] = arith.subi %[[UB]], %[[LB]]
// CHECK: %[[RANGE_M1:.*]] = arith.subi %[[RANGE]], %[[ONE]]
// CHECK: %[[ITERS_M1:.*]] = arith.divsi %[[RANGE_M1]], %[[STEP]]
// CHECK: %[[DELTA:.*]] = arith.muli %[[ITERS_M1]], %[[STEP]]
// CHECK: %[[LAST_IV:.*]] = arith.addi %[[DELTA]], %[[LB]]
// CHECK: %[[COND:.*]] = arith.cmpi slt, %[[LB]], %[[UB]]
// CHECK: %[[IF:.*]] = scf.if %[[COND]]
// CHECK:   %[[DEF:.*]] = "def"(%[[LAST_IV]]) : (i32) -> f32
// CHECK:   %[[RES:.*]] = arith.addf %[[FOR]], %[[DEF]] : f32
// CHECK:   scf.yield %[[RES]] : f32
// CHECK: else
// CHECK:   scf.yield %[[FOR]] : f32
// CHECK: tt.return %[[IF]] : f32
tt.func @simple_loop_i32(%lb : i32, %ub : i32, %step : i32) -> f32 {
  %init = arith.constant 0.00e+00 : f32
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (f32) : i32 {
    %a = "def"(%iv) : (i32) -> f32
    %res = arith.addf %acc, %a : f32
    scf.yield %res : f32
  } {__test_peel_epilogue}

  tt.return %loop#0 : f32
}
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @simple_loop_i32
// CHECK: (%[[LB:.*]]: i32, %[[UB:.*]]: i32, %[[STEP:.*]]: i32) -> f32
// CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[NUB:.*]] = arith.subi %[[UB]], %[[STEP]]
// CHECK: %[[FOR:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[NUB]] step %[[STEP]]
// CHECK: scf.yield
// CHECK: %[[RANGE:.*]] = arith.subi %[[UB]], %[[LB]]
// CHECK: %[[RANGE_M1:.*]] = arith.subi %[[RANGE]], %[[ONE]]
// CHECK: %[[ITERS_M1:.*]] = arith.divsi %[[RANGE_M1]], %[[STEP]]
// CHECK: %[[DELTA:.*]] = arith.muli %[[ITERS_M1]], %[[STEP]]
// CHECK: %[[LAST_IV:.*]] = arith.addi %[[DELTA]], %[[LB]]
// CHECK: %[[COND:.*]] = arith.cmpi slt, %[[LB]], %[[UB]]
// CHECK: %[[IF:.*]] = scf.if %[[COND]]
// CHECK:   %[[DEF:.*]] = "def"(%[[LAST_IV]]) : (i32) -> f32
// CHECK:   %[[RES:.*]] = arith.addf %[[FOR]], %[[DEF]] : f32
// CHECK:   scf.yield %[[RES]] : f32
// CHECK: else
// CHECK:   scf.yield %[[FOR]] : f32
// CHECK: tt.return %[[IF]] : f32
tt.func @simple_loop_i32(%lb : i32, %ub : i32, %step : i32) -> f32 {
  %init = arith.constant 0.00e+00 : f32
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (f32) : i32 {
    %a = "def"(%iv) : (i32) -> f32
    %res = arith.addf %acc, %a : f32
    scf.yield %res : f32
  } {__test_peel_epilogue}

  tt.return %loop#0 : f32
}
}
