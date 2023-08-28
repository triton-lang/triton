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


// CHECK-LABEL: make_range
tt.func @make_range() -> (tensor<128x1xi32>, tensor<1xi32>) {
  // CHECK-DAG: %[[c:.*]] = arith.constant dense<0> : tensor<128x1xi32>
  %a = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32>
  %b = tt.expand_dims %a {axis = 1 : i32} : (tensor<1xi32>) -> tensor<1x1xi32>
  %c = tt.broadcast %b : (tensor<1x1xi32>) -> tensor<128x1xi32>

  // CHECK-DAG: %[[d:.*]] = arith.constant dense<1> : tensor<1xi32>
  %d = tt.make_range {end = 2 : i32, start = 1 : i32} : tensor<1xi32>

  // CHECK-DAG: tt.return %[[c]], %[[d]] : tensor<128x1xi32>, tensor<1xi32>
  tt.return %c, %d : tensor<128x1xi32>, tensor<1xi32>
}
