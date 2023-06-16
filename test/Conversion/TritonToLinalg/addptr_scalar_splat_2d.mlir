// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.splat %2 : (!tt.ptr<f32>) -> tensor<128x128x!tt.ptr<f32>>
    // source = %arg1, offset = [%1, 0], size = [128, 128], strides = [0, 0]
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : (tensor<128xi32>) -> tensor<1x128xi32>
    %6 = tt.broadcast %5 : (tensor<1x128xi32>) -> tensor<128x128xi32>
    // offset = [0, 0], size = [128, 128], strides = [0, 1]
    %7 = tt.make_range {end = 384 : i32, start = 128 : i32} : tensor<128xi32>
    // offset = 128, size = 128, strides = 1
    %8 = tt.expand_dims %7 {axis = 1 : i32} : (tensor<128xi32>) -> tensor<128x1xi32>
    %9 = tt.broadcast %8 : (tensor<128x1xi32>) -> tensor<128x128xi32>
    // offset = [128, 0], size = [128, 128], strides = [2, 0]
    %10 = arith.addi %6, %9 : tensor<128x128xi32>
    // offset = [128, 0], size = [128, 128], strides = [2, 1]
    %11 = tt.addptr %3, %10 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
    // source = %arg1, offset = [%1 + 128, 0], size = [128, 128], strides = [2, 1]
    %12 = tt.load %11 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf32>
    %17 = math.exp %12 : tensor<128x128xf32>
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    // source = arg0, offset = %18, size = 1, strides = 0
    %20 = tt.splat %19 : (!tt.ptr<f32>) -> tensor<128x128x!tt.ptr<f32>>
    // source = arg0, offset = [%18, 0], size = [128, 128], strides = [0, 0]
    %21 = tt.addptr %20, %10 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
    // source = %arg0, offset = [%18 + 128, 0], size = [128, 128], strides = [2, 1]
    tt.store %21, %17 : tensor<128x128xf32>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xf32> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: memref<*xf32> {tt.divisibility = 16 : i32}, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32) {
// CHECK:           %[[VAL_8:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_9:.*]] = arith.muli %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[VAL_9]] : i32 to index
// CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_8]] : index
// CHECK:           %[[VAL_12:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_11]]], sizes: [128, 128], strides: [2, 1] : memref<*xf32> to memref<128x128xf32, strided<[2, 1], offset: ?>>
// CHECK:           %[[VAL_13:.*]] = memref.alloc() : memref<128x128xf32>
// CHECK:           memref.copy %[[VAL_12]], %[[VAL_13]] : memref<128x128xf32, strided<[2, 1], offset: ?>> to memref<128x128xf32>
// CHECK:           %[[VAL_14:.*]] = bufferization.to_tensor %[[VAL_13]] restrict writable : memref<128x128xf32>
// CHECK:           %[[VAL_15:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_14]] : tensor<128x128xf32>) outs(%[[VAL_14]] : tensor<128x128xf32>) {
// CHECK:           ^bb0(%[[VAL_16:.*]]: f32, %[[VAL_17:.*]]: f32):
// CHECK:             %[[VAL_18:.*]] = math.exp %[[VAL_16]] : f32
// CHECK:             linalg.yield %[[VAL_18]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           %[[VAL_19:.*]] = arith.muli %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_20:.*]] = arith.index_cast %[[VAL_19]] : i32 to index
// CHECK:           %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_8]] : index
// CHECK:           %[[VAL_22:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_21]]], sizes: [128, 128], strides: [2, 1] : memref<*xf32> to memref<128x128xf32, strided<[2, 1], offset: ?>>
// CHECK:           memref.tensor_store %[[VAL_23:.*]], %[[VAL_22]] : memref<128x128xf32, strided<[2, 1], offset: ?>>
// CHECK:           return
// CHECK:         }
