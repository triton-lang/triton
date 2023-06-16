// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    // source = %arg1, offset = %1, size = 1, strides = 0
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // offset = 0, size = 1024, strides = 1
    %4 = tt.splat %2 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    // source = %arg1, offset = %1, size = 1024, strides = 0
    %5 = tt.addptr %4, %3 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // source = %arg1, offset = %1, size = 1024, strides = 1
    %8 = tt.load %5 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32>
    %17 = math.exp %8 : tensor<1024xf32>
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    // source = %arg0, offset = %18, size = 1, strides = 0
    %20 = tt.splat %19 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    // source = %arg0, offset = %18, size = 1024, strides = 0
    %21 = tt.addptr %20, %3 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // source = %arg0, offset = %18, size = 1024, strides = 1
    tt.store %21, %17 : tensor<1024xf32>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xf32> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: memref<*xf32> {tt.divisibility = 16 : i32}, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32) {
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_8]] : i32 to index
// CHECK:           %[[VAL_10:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_9]]], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK:           %[[VAL_11:.*]] = memref.alloc() : memref<1024xf32>
// CHECK:           memref.copy %[[VAL_10]], %[[VAL_11]] : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
// CHECK:           %[[VAL_12:.*]] = bufferization.to_tensor %[[VAL_11]] restrict writable : memref<1024xf32>
// CHECK:           %[[VAL_13:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%[[VAL_12]] : tensor<1024xf32>) outs(%[[VAL_12]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_14:.*]]: f32, %[[VAL_15:.*]]: f32):
// CHECK:             %[[VAL_16:.*]] = math.exp %[[VAL_14]] : f32
// CHECK:             linalg.yield %[[VAL_16]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           %[[VAL_17:.*]] = arith.muli %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_18:.*]] = arith.index_cast %[[VAL_17]] : i32 to index
// CHECK:           %[[VAL_19:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_18]]], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK:           memref.tensor_store %[[VAL_20:.*]], %[[VAL_19]] : memref<1024xf32, strided<[1], offset: ?>>
// CHECK:           return
// CHECK:         }
