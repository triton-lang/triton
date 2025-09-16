// RUN: triton-opt %s --triton-rewrite-tensor-descriptor-to-pointer --canonicalize --cse --split-input-file | FileCheck %s --implicit-check-not \!tt.tensordesc

module {
  tt.func public @load(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32) -> (tensor<128x128xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <tensor<128x128xf32>>
    %3 = tt.descriptor_load %0[%arg1, %arg2] : !tt.tensordesc<tensor<128x128xf32>> -> tensor<128x128xf32>
    tt.return %3 : tensor<128x128xf32>
  }
}

// CHECK-LABEL: @load
// CHECK-SAME: %[[ARG0:[^:]*]]
// CHECK-SAME: %[[ARG1:[^:]*]]
// CHECK-SAME: %[[ARG2:[^:]*]]
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-DAG: %[[CST0:.*]] = arith.constant dense<0> : tensor<1x128xi64>
// CHECK-DAG: %[[CST1:.*]] = arith.constant dense<256> : tensor<128x1xi64>
// CHECK-DAG: %[[CST2:.*]] = arith.constant dense<0> : tensor<128x1xi64>
// CHECK-DAG: %[[CST3:.*]] = arith.constant dense<256> : tensor<1x128xi64>

// CHECK-DAG: %[[VAL0:.*]] = arith.extsi %[[ARG1]] : i32 to i64
// CHECK-DAG: %[[VAL1:.*]] = arith.extsi %[[ARG2]] : i32 to i64
// CHECK-DAG: %[[VAL2:.*]] = tt.splat %[[ARG0]] :
// CHECK-DAG: %[[VAL3:.*]] = tt.splat %[[VAL0]] :
// CHECK-DAG: %[[VAL4:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32}
// CHECK-DAG: %[[VAL5:.*]] = arith.extsi %[[VAL4]] :
// CHECK-DAG: %[[VAL6:.*]] = arith.addi %[[VAL3]], %[[VAL5]] :
// CHECK-DAG: %[[VAL7:.*]] = tt.expand_dims %[[VAL6]] {axis = 1 : i32}
// CHECK-DAG: %[[VAL8:.*]] = tt.broadcast %[[VAL7]] : tensor<128x1xi64> -> tensor<128x128xi64>
// CHECK-DAG: %[[VAL9:.*]] = tt.addptr %[[VAL2]], %[[VAL8]] :
// CHECK-DAG: %[[VAL10:.*]] = tt.splat %[[VAL1]] :
// CHECK-DAG: %[[VAL11:.*]] = arith.addi %[[VAL10]], %[[VAL5]] :
// CHECK-DAG: %[[VAL12:.*]] = tt.expand_dims %[[VAL11]] {axis = 0 : i32}
// CHECK-DAG: %[[VAL13:.*]] = arith.muli %[[VAL12]], %[[CST3]] :
// CHECK-DAG: %[[VAL14:.*]] = tt.broadcast %[[VAL13]] : tensor<1x128xi64> -> tensor<128x128xi64>
// CHECK-DAG: %[[VAL15:.*]] = tt.addptr %[[VAL9]], %[[VAL14]] :

// CHECK-DAG: %[[VAL16:.*]] = arith.cmpi sge, %[[VAL7]], %[[CST2]]
// CHECK-DAG: %[[VAL17:.*]] = arith.cmpi slt, %[[VAL7]], %[[CST1]]
// CHECK-DAG: %[[VAL18:.*]] = arith.andi %[[VAL16]], %[[VAL17]]
// CHECK-DAG: %[[VAL19:.*]] = tt.broadcast %[[VAL18]] : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-DAG: %[[VAL20:.*]] = arith.cmpi sge, %[[VAL12]], %[[CST0]]
// CHECK-DAG: %[[VAL21:.*]] = arith.cmpi slt, %[[VAL12]], %[[CST3]]
// CHECK-DAG: %[[VAL22:.*]] = arith.andi %[[VAL20]], %[[VAL21]]
// CHECK-DAG: %[[VAL23:.*]] = tt.broadcast %[[VAL22]] : tensor<1x128xi1> -> tensor<128x128xi1>
// CHECK-DAG: %[[VAL24:.*]] = arith.andi %[[VAL19]], %[[VAL23]]

// CHECK-DAG: %[[VAL25:.*]] = tt.load %[[VAL15]], %[[VAL24]], %[[CST]]
// CHECK: tt.return %[[VAL25]] :

// -----

module {
  tt.func public @store(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: tensor<128x128xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] {order = array<i32: 0>} : <f32>, <tensor<128x128xf32>>
    tt.descriptor_store %0[%arg1, %arg2], %arg3 : !tt.tensordesc<tensor<128x128xf32>>, tensor<128x128xf32>
    tt.return
  }
}

// CHECK-LABEL: @store
// CHECK-SAME: %[[ARG0:[^:]*]]
// CHECK-SAME: %[[ARG1:[^:]*]]
// CHECK-SAME: %[[ARG2:[^:]*]]
// CHECK-SAME: %[[ARG3:[^:]*]]
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<0> : tensor<1x128xi64>
// CHECK-DAG: %[[CST0:.*]] = arith.constant dense<256> : tensor<128x1xi64>
// CHECK-DAG: %[[CST1:.*]] = arith.constant dense<0> : tensor<128x1xi64>
// CHECK-DAG: %[[CST2:.*]] = arith.constant dense<256> : tensor<1x128xi64>

// CHECK-DAG: %[[VAL0:.*]] = arith.extsi %[[ARG1]] : i32 to i64
// CHECK-DAG: %[[VAL1:.*]] = arith.extsi %[[ARG2]] : i32 to i64
// CHECK-DAG: %[[VAL2:.*]] = tt.splat %[[ARG0]] :
// CHECK-DAG: %[[VAL3:.*]] = tt.splat %[[VAL0]] :
// CHECK-DAG: %[[VAL4:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32}
// CHECK-DAG: %[[VAL5:.*]] = arith.extsi %[[VAL4]] :
// CHECK-DAG: %[[VAL6:.*]] = arith.addi %[[VAL3]], %[[VAL5]] :
// CHECK-DAG: %[[VAL7:.*]] = tt.expand_dims %[[VAL6]] {axis = 1 : i32}
// CHECK-DAG: %[[VAL8:.*]] = tt.broadcast %[[VAL7]] : tensor<128x1xi64> -> tensor<128x128xi64>
// CHECK-DAG: %[[VAL9:.*]] = tt.addptr %[[VAL2]], %[[VAL8]] :
// CHECK-DAG: %[[VAL10:.*]] = tt.splat %[[VAL1]] :
// CHECK-DAG: %[[VAL11:.*]] = arith.addi %[[VAL10]], %[[VAL5]] :
// CHECK-DAG: %[[VAL12:.*]] = tt.expand_dims %[[VAL11]] {axis = 0 : i32}
// CHECK-DAG: %[[VAL13:.*]] = arith.muli %[[VAL12]], %[[CST2]] :
// CHECK-DAG: %[[VAL14:.*]] = tt.broadcast %[[VAL13]] : tensor<1x128xi64> -> tensor<128x128xi64>
// CHECK-DAG: %[[VAL15:.*]] = tt.addptr %[[VAL9]], %[[VAL14]] :

// CHECK-DAG: %[[VAL16:.*]] = arith.cmpi sge, %[[VAL7]], %[[CST1]]
// CHECK-DAG: %[[VAL17:.*]] = arith.cmpi slt, %[[VAL7]], %[[CST0]]
// CHECK-DAG: %[[VAL18:.*]] = arith.andi %[[VAL16]], %[[VAL17]]
// CHECK-DAG: %[[VAL19:.*]] = tt.broadcast %[[VAL18]] : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-DAG: %[[VAL20:.*]] = arith.cmpi sge, %[[VAL12]], %[[CST]]
// CHECK-DAG: %[[VAL21:.*]] = arith.cmpi slt, %[[VAL12]], %[[CST2]]
// CHECK-DAG: %[[VAL22:.*]] = arith.andi %[[VAL20]], %[[VAL21]]
// CHECK-DAG: %[[VAL23:.*]] = tt.broadcast %[[VAL22]] : tensor<1x128xi1> -> tensor<128x128xi1>
// CHECK-DAG: %[[VAL24:.*]] = arith.andi %[[VAL19]], %[[VAL23]]

// CHECK: tt.store %[[VAL15]], %[[ARG3]], %[[VAL24]]

// -----

module {
  tt.func public @callee(%tensordesc: !tt.tensordesc<tensor<128x128xf32>>) -> !tt.tensordesc<tensor<128x128xf32>> {
    tt.return %tensordesc : !tt.tensordesc<tensor<128x128xf32>>
  }

  tt.func public @caller(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] {order = array<i32: 0>} : <f32>, <tensor<128x128xf32>>
    %1 = tt.call @callee(%0) : (!tt.tensordesc<tensor<128x128xf32>>) -> !tt.tensordesc<tensor<128x128xf32>>
    tt.return
  }
}

// CHECK-LABEL: @callee
// CHECK-SAME: %[[PTR:[^:]*]]
// CHECK-SAME: %[[SHAPE0:[^:]*]]
// CHECK-SAME: %[[SHAPE1:[^:]*]]
// CHECK-SAME: %[[STRIDE0:[^:]*]]
// CHECK-SAME: %[[STRIDE1:[^:]*]]
// CHECK-NEXT: tt.return %[[PTR]], %[[SHAPE0]], %[[SHAPE1]], %[[STRIDE0]], %[[STRIDE1]]

// CHECK-LABEL: @caller
// CHECK-SAME: %[[PTR:[^:]*]]
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c256:.*]] = arith.constant 256 : i64
// CHECK: %{{.*}}:6 = tt.call @callee(%[[PTR]], %[[c256]], %[[c256]], %[[c256]], %[[c1]], %false)
// CHECK-SAME -> (!tt.ptr<f32>, i64, i64, i64, i64, i1)
