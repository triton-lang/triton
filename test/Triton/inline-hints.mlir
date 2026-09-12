// RUN: triton-opt %s -inline | FileCheck %s

// Check that AxisInfo hint attributes on a tt.call are transferred to the
// defining op of the corresponding inlined result.

tt.func private @get_ptrs(%ptr: tensor<128x!tt.ptr<f32>>, %offsets: tensor<128xi32>) -> tensor<128x!tt.ptr<f32>> {
  %0 = tt.addptr %ptr, %offsets : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  tt.return %0 : tensor<128x!tt.ptr<f32>>
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
