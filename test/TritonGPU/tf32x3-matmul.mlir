// RUN: triton-opt %s -split-input-file -tritongpu-F32DotTC -canonicalize  | FileCheck %s --check-prefixes=CHECK 

// CHECK:     %[[DOT1:.*]] = tt.dot %[[LHS_LOW:.*]], %[[RHS_HIGH:.*]], %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK:     %[[DOT2:.*]] = tt.dot %[[LHS_HIGH:.*]], %[[RHS_LOW:.*]], %[[DOT1]], inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK:     %[[CMP:.*]] = arith.cmpf uno, %[[DOT2]], %[[DOT2]] : tensor<16x16xf32>
// CHECK:     %[[MASKED:.*]] = arith.select %[[CMP]], %cst, %[[DOT2]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK:     %[[RESULT:.*]] = tt.dot %[[LHS_HIGH]], %[[RHS_HIGH]], %[[MASKED]], inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>

module {
  tt.func @dot_test(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c16_i64 = arith.constant 16: i64
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %0 = tt.make_tensor_ptr %arg0, [%c16_i64, %c16_i64], [%c16_i64, %c16_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %1 = tt.make_tensor_ptr %arg1, [%c16_i64, %c16_i64], [%c16_i64, %c16_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %2 = tt.load %0 : !tt.ptr<tensor<16x16xf32>>
    %3 = tt.load %1 : !tt.ptr<tensor<16x16xf32>>
    %4 = tt.dot %2, %3, %cst, inputPrecision = tf32x3 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %5 = tt.make_tensor_ptr %arg2, [%c16_i64, %c16_i64], [%c16_i64, %c16_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    tt.store %5, %4 : !tt.ptr<tensor<16x16xf32>>
    tt.return
  }
}
