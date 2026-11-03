// RUN: triton-opt %s -triton-loop-unswitch | FileCheck %s

// The condition is computed inside the loop from loop-invariant values: the
// computation is hoisted and the loop is unswitched.
// CHECK-LABEL: @unswitch_cond_computed_in_loop
tt.func @unswitch_cond_computed_in_loop(%lb: index, %ub: index, %step: index, %flag: i32, %x: f32, %y: f32) -> f32 {
  %init = arith.constant 0.0 : f32
  // CHECK: %[[C3:.*]] = arith.constant 3 : i32
  // CHECK: %[[COND:.*]] = arith.cmpi eq, %{{.*}}, %[[C3]]
  // CHECK: scf.if %[[COND]]
  // CHECK: scf.for
  // CHECK-NOT: scf.if
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (f32) {
    %c3 = arith.constant 3 : i32
    %c = arith.cmpi eq, %flag, %c3 : i32
    %v = scf.if %c -> (f32) {
      %a = arith.addf %acc, %x : f32
      scf.yield %a : f32
    } else {
      %m = arith.mulf %acc, %y : f32
      scf.yield %m : f32
    }
    scf.yield %v : f32
  }
  tt.return %r : f32
}

// The condition depends on the induction variable: must not unswitch.
// CHECK-LABEL: @variant_condition_not_unswitched
tt.func @variant_condition_not_unswitched(%lb: index, %ub: index, %step: index, %n: index, %x: f32) -> f32 {
  %init = arith.constant 0.0 : f32
  // CHECK: scf.for
  // CHECK: scf.if
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (f32) {
    %c = arith.cmpi slt, %i, %n : index
    %v = scf.if %c -> (f32) {
      %a = arith.addf %acc, %x : f32
      scf.yield %a : f32
    } else {
      scf.yield %acc : f32
    }
    scf.yield %v : f32
  }
  tt.return %r : f32
}

// If without results or else region: the then"copy" inlines the body, the
// else"copy" simply drops it.
// CHECK-LABEL: @unswitch_if_without_else
tt.func @unswitch_if_without_else(%lb: index, %ub: index, %step: index, %flag: i1, %ptr: !tt.ptr<f32>, %x: f32) {
  // CHECK: scf.if %{{.*}}
  // CHECK: scf.for
  // CHECK-NOT: scf.if
  // CHECK: tt.store
  // CHECK: else
  // CHECK: scf.for
  // CHECK-NOT: tt.store
  scf.for %i = %lb to %ub step %step {
    scf.if %flag {
      tt.store %ptr, %x : !tt.ptr<f32>
      scf.yield
    }
    scf.yield
  }
  tt.return
}
