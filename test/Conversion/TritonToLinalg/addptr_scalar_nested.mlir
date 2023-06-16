// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    // source = arg1, offset = %1, size = 1, strides = 0
    %3 = arith.muli %0, %arg3 : i32
    %4 = tt.addptr %2, %3 : !tt.ptr<f32>, i32
    // source = arg1, offset = %1+%3, size = 1, strides = 0
    %5 = arith.muli %0, %arg4 : i32
    %6 = tt.addptr %4, %5 : !tt.ptr<f32>, i32
    // source = arg1, offset = %1+%3+%5, size = 1, strides = 0
    %7 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // offset = 0, size = 1024, strides = 1
    %8 = tt.splat %6 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    // source = arg1, offset = %1, size = 1024, strides = 0
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // source = arg1, offset = %1+%3+%5, size = 1024, strides = 1
    %10 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32>
    %17 = math.exp %10 : tensor<1024xf32>
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    // source = arg0, offset = %18, size = 1, strides = 0
    %20 = tt.splat %19 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    // source = arg0, offset = %18, size = 1024, strides = 0
    %21 = tt.addptr %20, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // source = arg0, offset = %18, size = 1024, strides = 1
    tt.store %21, %17 : tensor<1024xf32>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xf32> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: memref<*xf32> {tt.divisibility = 16 : i32}, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32) {
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_9:.*]] = arith.muli %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_10:.*]] = arith.muli %[[VAL_5]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_11:.*]] = arith.index_cast %[[VAL_8]] : i32 to index
// CHECK:           %[[VAL_12:.*]] = arith.index_cast %[[VAL_9]] : i32 to index
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : index
// CHECK:           %[[VAL_14:.*]] = arith.index_cast %[[VAL_10]] : i32 to index
// CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] : index
// CHECK:           %[[VAL_16:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_15]]], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK:           %[[VAL_17:.*]] = memref.alloc() : memref<1024xf32>
// CHECK:           memref.copy %[[VAL_16]], %[[VAL_17]] : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
// CHECK:           %[[VAL_18:.*]] = bufferization.to_tensor %[[VAL_17]] restrict writable : memref<1024xf32>
// CHECK:           %[[VAL_19:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%[[VAL_18]] : tensor<1024xf32>) outs(%[[VAL_18]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_20:.*]]: f32, %[[VAL_21:.*]]: f32):
// CHECK:             %[[VAL_22:.*]] = math.exp %[[VAL_20]] : f32
// CHECK:             linalg.yield %[[VAL_22]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           %[[VAL_23:.*]] = arith.muli %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_24:.*]] = arith.index_cast %[[VAL_23]] : i32 to index
// CHECK:           %[[VAL_25:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_24]]], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK:           memref.tensor_store %[[VAL_26:.*]], %[[VAL_25]] : memref<1024xf32, strided<[1], offset: ?>>
// CHECK:           return
// CHECK:         }
