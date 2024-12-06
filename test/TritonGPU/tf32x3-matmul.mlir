// RUN: triton-opt %s -split-input-file -tritongpu-F32DotTC -canonicalize  | FileCheck %s --check-prefixes=CHECK 

// CHECK:     %[[DOT1:.*]] = tt.dot %[[LHS_LOW:.*]], %[[RHS_HIGH:.*]], %cst_1, inputPrecision = tf32 : tensor<32x32xf32> * tensor<32x32xf32> -> tensor<32x32xf32>
// CHECK:     %[[DOT2:.*]] = tt.dot %[[LHS_HIGH:.*]], %[[RHS_LOW:.*]], %[[DOT1]], inputPrecision = tf32 : tensor<32x32xf32> * tensor<32x32xf32> -> tensor<32x32xf32>
// CHECK:     %[[CMP:.*]] = arith.cmpf uno, %[[DOT2]], %[[DOT2]] : tensor<32x32xf32>
// CHECK:     %[[MASKED:.*]] = arith.select %[[CMP]], %cst_1, %[[DOT2]] : tensor<32x32xi1>, tensor<32x32xf32>
// CHECK:     %[[RESULT:.*]] = tt.dot %[[LHS_HIGH]], %[[RHS_HIGH]], %[[MASKED]], inputPrecision = tf32 : tensor<32x32xf32> * tensor<32x32xf32> -> tensor<32x32xf32>

module {
    tt.func @a_1_impl(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i322}) {
    %cst = arith.constant dense<8> : tensor<32x1xi32>
    %cst_0 = arith.constant dense<8> : tensor<1x32xi32>
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c8_i64 = arith.constant 8 : i64
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c8_i32 : i32
    %2 = arith.muli %1, %c8_i32 : i32
    %3 = arith.subi %c1_i32, %2 : i32
    %4 = arith.cmpi slt, %3, %c8_i32 : i32
    %5 = arith.select %4, %3, %c8_i32 : i32
    %6 = arith.remsi %0, %5 : i32
    %7 = arith.addi %2, %6 : i32
    %8 = arith.remsi %0, %c8_i32 : i32
    %9 = arith.divsi %8, %5 : i32
    %10 = arith.muli %7, %c32_i32 : i32
    %11 = tt.make_tensor_ptr %arg0, [%c8_i64, %c8_i64], [%c8_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf32>>
    %12 = tt.advance %11, [%10, %c0_i32] : <tensor<32x32xf32>>
    %13 = arith.muli %9, %c32_i32 : i32
    %14 = tt.make_tensor_ptr %arg1, [%c8_i64, %c8_i64], [%c8_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf32>>
    %15 = tt.advance %14, [%c0_i32, %13] : <tensor<32x32xf32>>
    %16 = tt.load %12 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x32xf32>>
    %17 = tt.load %15 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x32xf32>>
    %18 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %19 = tt.expand_dims %18 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %20 = arith.cmpi slt, %19, %cst_0 : tensor<1x32xi32>
    %21 = tt.broadcast %20 : tensor<1x32xi1> -> tensor<32x32xi1>
    %22 = arith.select %21, %16, %cst_1 : tensor<32x32xi1>, tensor<32x32xf32>
    %23 = tt.expand_dims %18 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %24 = arith.cmpi slt, %23, %cst : tensor<32x1xi32>
    %25 = tt.broadcast %24 : tensor<32x1xi1> -> tensor<32x32xi1>
    %26 = arith.select %25, %17, %cst_1 : tensor<32x32xi1>, tensor<32x32xf32>
    %27 = tt.dot %22, %26, %cst_1, inputPrecision = tf32x3 : tensor<32x32xf32> * tensor<32x32xf32> -> tensor<32x32xf32>
    %28 = tt.make_tensor_ptr %arg2, [%c8_i64, %c8_i64], [%c8_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf32>>
    %29 = tt.advance %28, [%10, %13] : <tensor<32x32xf32>>
    tt.store %29, %27 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x32xf32>>
    tt.return
  }
}

