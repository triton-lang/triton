// RUN: triton-opt %s -split-input-file -canonicalize | FileCheck %s

// CHECK-LABEL: dead_load
tt.func @dead_load(%ptr: tensor<32x128x!tt.ptr<f16>>) {
  %mask = arith.constant dense<true> : tensor<32x128xi1>
  %other = arith.constant dense<0.00e+00> : tensor<32x128xf16>
  // CHECK-NOT: tt.load {{.*}} isVolatile = false
  //     CHECK: tt.load {{.*}} isVolatile = true
  %a = tt.load %ptr, %mask, %other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16>
  %b = tt.load %ptr, %mask, %other {cache = 1 : i32, evict = 1 : i32, isVolatile = true} : tensor<32x128xf16>
  tt.return
}
