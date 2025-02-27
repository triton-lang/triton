// RUN: triton-opt --split-input-file %s -triton-hoist-load -canonicalize | FileCheck %s

tt.func @hoist_load_without_mask(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // Check if the load is hoisted
  // CHECK-LABEL: hoist_load_without_mask
  // CHECK: %[[TRIP_COUNT_CMP:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
  // CHECK: %[[SPLAT:.*]] = tt.splat %[[TRIP_COUNT_CMP]]
  // CHECK: tt.load %[[_:.*]], %[[SPLAT]]
  // CHECK: scf.for
  // CHECK-NOT: tt.load
  %1 = scf.for %arg7 = %arg3 to %arg4 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1024xf32>)  : i32 {
    %2 = tt.load %arg0 : tensor<1024x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<1024xf32>
    %4 = arith.addf %arg6, %3 : tensor<1024xf32>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %arg5, %1 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @hoist_load(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // Check if the load is hoisted
  // CHECK-LABEL: hoist_load
  // CHECK: %[[ZERO:.*]] = arith.constant dense<false>
  // CHECK: %[[MASK:.*]] = arith.cmpi
  // CHECK: %[[TRIP_COUNT_CMP:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
  // CHECK: %[[SELECT:.*]] = arith.select %[[TRIP_COUNT_CMP]], %[[MASK]], %[[ZERO]]
  // CHECK: tt.load %[[_:.*]], %[[SELECT]]
  // CHECK: scf.for
  // CHECK-NOT: tt.load
  %0 = arith.cmpi slt, %arg1, %arg2 : tensor<1024xi32>
  %1 = scf.for %arg7 = %arg3 to %arg4 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1024xf32>)  : i32 {
    %2 = tt.load %arg0, %0 : tensor<1024x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<1024xf32>
    %4 = arith.addf %arg6, %3 : tensor<1024xf32>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %arg5, %1, %0 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @hoist_load_with_print_in_loop(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // Check if the load is hoisted when there is a tt.print which has a write side effect
  // CHECK-LABEL: hoist_load_with_print_in_loop
  // CHECK: %[[ZERO:.*]] = arith.constant dense<false>
  // CHECK: %[[MASK:.*]] = arith.cmpi
  // CHECK: %[[TRIP_COUNT_CMP:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
  // CHECK: %[[SELECT:.*]] = arith.select %[[TRIP_COUNT_CMP]], %[[MASK]], %[[ZERO]]
  // CHECK: tt.load %[[_:.*]], %[[SELECT]]
  // CHECK: scf.for
  // CHECK-NOT: tt.load
  %0 = arith.cmpi slt, %arg1, %arg2 : tensor<1024xi32>
  %1 = scf.for %arg7 = %arg3 to %arg4 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1024xf32>)  : i32 {
    %2 = tt.load %arg0, %0 : tensor<1024x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<1024xf32>
    %4 = arith.addf %arg6, %3 : tensor<1024xf32>
    tt.print " x: " {hex = false, isSigned = array<i32: 0>} : %4 : tensor<1024xf32>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %arg5, %1, %0 : tensor<1024x!tt.ptr<f32>>
  tt.return
}
// -----

tt.func @hoist_load_with_assert_in_loop(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // Check if the load is hoisted when there is a tt.assert which has a write side effect
  // CHECK-LABEL: hoist_load_with_assert_in_loop
  // CHECK: %[[ZERO:.*]] = arith.constant dense<false>
  // CHECK: %[[MASK:.*]] = arith.cmpi
  // CHECK: %[[TRIP_COUNT_CMP:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
  // CHECK: %[[SELECT:.*]] = arith.select %[[TRIP_COUNT_CMP]], %[[MASK]], %[[ZERO]]
  // CHECK: tt.load %[[_:.*]], %[[SELECT]]
  // CHECK: scf.for
  // CHECK-NOT: tt.load
  %0 = arith.cmpi slt, %arg1, %arg2 : tensor<1024xi32>
  %cmp = arith.cmpi sge, %arg4, %arg3 : i32
  %1 = scf.for %arg7 = %arg3 to %arg4 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1024xf32>)  : i32 {
    tt.assert %cmp, "cond must be true " : i1
    %2 = tt.load %arg0, %0 : tensor<1024x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<1024xf32>
    %4 = arith.addf %arg6, %3 : tensor<1024xf32>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %arg5, %1, %0 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @cannot_hoist_load_with_store_in_loop(%arg0: tensor<1024x!tt.ptr<f32>>, %arg1: tensor<1024xi32>, %arg2: tensor<1024xi32>, %arg3: i32, %arg4 : i32, %arg5: tensor<1024x!tt.ptr<f32>>, %tmp: tensor<1024x!tt.ptr<f32>>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %c1_i32 = arith.constant 1 : i32
  // Check if the load is not hoisted when there is an op with write side effect that is neither tt.print nor tt.assert
  // CHECK-LABEL: cannot_hoist_load_with_store_in_loop
  // CHECK-NOT: tt.load
  // CHECK: scf.for
  // CHECK: tt.load
  %0 = arith.cmpi slt, %arg1, %arg2 : tensor<1024xi32>
  %cmp = arith.cmpi sge, %arg4, %arg3 : i32
  %1 = scf.for %arg7 = %arg3 to %arg4 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<1024xf32>)  : i32 {
    tt.assert %cmp, "cond must be true " : i1
    %2 = tt.load %arg0, %0 : tensor<1024x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<1024xf32>
    %4 = arith.addf %arg6, %3 : tensor<1024xf32>
    tt.store %tmp, %4, %0 : tensor<1024x!tt.ptr<f32>>
    scf.yield %4 : tensor<1024xf32>
  }
  tt.store %arg5, %1, %0 : tensor<1024x!tt.ptr<f32>>
  tt.return
}
