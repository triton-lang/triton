// RUN: triton-opt %s -inline | FileCheck %s

// Check that AxisInfo hint attributes on a tt.call are transferred to the
// defining op of the corresponding inlined result.

tt.func private @get_ptrs(%ptr: tensor<128x!tt.ptr<f32>>, %offsets: tensor<128xi32>) -> tensor<128x!tt.ptr<f32>> {
  %0 = tt.addptr %ptr, %offsets : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  tt.return %0 : tensor<128x!tt.ptr<f32>>
}

tt.func private @get_offsets(%offsets: tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>) {
  %0 = arith.addi %offsets, %offsets : tensor<128xi32>
  %1 = arith.muli %offsets, %offsets : tensor<128xi32>
  tt.return %0, %1 : tensor<128xi32>, tensor<128xi32>
}

// The call-wide hints do not identify which result they describe.
// CHECK-LABEL: @multi_result_caller
// CHECK-NOT: tt.call
// CHECK: arith.addi %arg0, %arg0 : tensor<128xi32>
// CHECK: arith.muli %arg0, %arg0 : tensor<128xi32>
tt.func public @multi_result_caller(%arg0: tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>) {
  %0:2 = tt.call @get_offsets(%arg0) {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<16> : tensor<1xi32>, tt.divisibility = dense<16> : tensor<1xi32>} : (tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>)
  tt.return %0#0, %0#1 : tensor<128xi32>, tensor<128xi32>
}

tt.func private @pass_through(%a: tensor<128xi32>, %b: tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>) {
  tt.return %a, %b : tensor<128xi32>, tensor<128xi32>
}

// Do not overwrite hints on a value that the callee returns unchanged.
// CHECK-LABEL: @pass_through_caller
// CHECK: arith.addi %arg0, %arg0 {tt.contiguity = dense<2> : tensor<1xi32>} : tensor<128xi32>
// CHECK: arith.muli %arg0, %arg0 : tensor<128xi32>
// CHECK-NOT: tt.call
tt.func public @pass_through_caller(%arg0: tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>) {
  %a = arith.addi %arg0, %arg0 {tt.contiguity = dense<2> : tensor<1xi32>} : tensor<128xi32>
  %b = arith.muli %arg0, %arg0 : tensor<128xi32>
  %0:2 = tt.call @pass_through(%a, %b) {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<16> : tensor<1xi32>, tt.divisibility = dense<16> : tensor<1xi32>} : (tensor<128xi32>, tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>)
  tt.return %0#0, %0#1 : tensor<128xi32>, tensor<128xi32>
}

// CHECK-LABEL: @caller
tt.func public @caller(%arg0: tensor<128x!tt.ptr<f32>>, %arg1: tensor<128xi32>) -> tensor<128x!tt.ptr<f32>> {
  // CHECK-NOT: tt.call
  // CHECK: tt.addptr
  // CHECK-SAME: tt.constancy = dense<1>
  // CHECK-SAME: tt.contiguity = dense<16>
  // CHECK-SAME: tt.divisibility = dense<16>
  %0 = tt.call @get_ptrs(%arg0, %arg1) {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<16> : tensor<1xi32>, tt.divisibility = dense<16> : tensor<1xi32>} : (tensor<128x!tt.ptr<f32>>, tensor<128xi32>) -> tensor<128x!tt.ptr<f32>>
  tt.return %0 : tensor<128x!tt.ptr<f32>>
}
