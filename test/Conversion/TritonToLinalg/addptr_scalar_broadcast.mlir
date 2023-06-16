// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    // source = arg1, offset = %1, size = 1, strides = 0
    %3 = tt.splat %2 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    // source = arg1, offset = %1, size = 1024, strides = 0
    %4 = tt.expand_dims %3 {axis = 1 : i32} : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024x1x!tt.ptr<f32>>
    // source = arg1, offset = [%1, 0], size = [1024, 1], strides = [0, 0]
    %5 = tt.broadcast %4 : (tensor<1024x1x!tt.ptr<f32>>) -> tensor<1024x1024x!tt.ptr<f32>>
    // source = arg1, offset = [%1, 0], size = [1024, 1024], strides = [0, 0]
    %6 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // offset = 0, size = 1024, strides = 1
    %7 = tt.expand_dims %6 {axis = 0 : i32} : (tensor<1024xi32>) -> tensor<1x1024xi32>
    // offset = [0, 0], size = [1, 1024], strides = [0, 1]
    %8 = tt.broadcast %7 : (tensor<1x1024xi32>) -> tensor<1024x1024xi32>
    // offset = [0, 0], size = [1024, 1024], strides = [0, 1]
    %9 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<1024xi32>
    // offset = 0, size = 1024, strides = 2
    %10 = tt.expand_dims %9 {axis = 1 : i32} : (tensor<1024xi32>) -> tensor<1024x1xi32>
    // offset = [0, 0], size = [1024, 1], strides = [2, 0]
    %11 = tt.broadcast %10 : (tensor<1024x1xi32>) -> tensor<1024x1024xi32>
    // offset = [0, 0], size = [1024, 1024], strides = [2, 0]
    %12 = arith.addi %8, %11 : tensor<1024x1024xi32>
    // offset = [0, 0], size = [1024, 1024], strides = [2, 1]
    %13 = tt.addptr %5, %12 : tensor<1024x1024x!tt.ptr<f32>>, tensor<1024x1024xi32>
    // source = arg1, offset = [pid * %arg2, 0], size = [1024, 1024], strides = [2, 1]
    %14 = tt.load %13 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024x1024xf32>
    %17 = math.exp %14 : tensor<1024x1024xf32>
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    // source = arg0, offset = pid+arg3, size = 1, strides = 0
    %20 = tt.splat %19 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    // source = arg0, offset = pid+arg3, size = 1024, strides = 0
    %21 = tt.expand_dims %20 {axis = 1 : i32} : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024x1x!tt.ptr<f32>>
    // source = arg0, offset = [pid+arg3, 0], size = [1024, 1], strides = [0, 0]
    %22 = tt.broadcast %21 : (tensor<1024x1x!tt.ptr<f32>>) -> tensor<1024x1024x!tt.ptr<f32>>
    // source = arg0, offset = [pid+arg3, 0], size = [1024, 1024], strides = [0, 0]
    %23 = tt.addptr %22, %12 : tensor<1024x1024x!tt.ptr<f32>>, tensor<1024x1024xi32>
    // source = arg0, offset = [pid+arg3, 0], size = [1024, 1024], strides = [2, 1]
    tt.store %23, %17 : tensor<1024x1024xf32>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xf32> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: memref<*xf32> {tt.divisibility = 16 : i32}, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32) {
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_8]] : i32 to index
// CHECK:           %[[VAL_10:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_9]]], sizes: [1024, 1024], strides: [2, 1] : memref<*xf32> to memref<1024x1024xf32, strided<[2, 1], offset: ?>>
// CHECK:           %[[VAL_11:.*]] = memref.alloc() : memref<1024x1024xf32>
// CHECK:           memref.copy %[[VAL_10]], %[[VAL_11]] : memref<1024x1024xf32, strided<[2, 1], offset: ?>> to memref<1024x1024xf32>
// CHECK:           %[[VAL_12:.*]] = bufferization.to_tensor %[[VAL_11]] restrict writable : memref<1024x1024xf32>
// CHECK:           %[[VAL_13:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_12]] : tensor<1024x1024xf32>) outs(%[[VAL_12]] : tensor<1024x1024xf32>) {
// CHECK:           ^bb0(%[[VAL_14:.*]]: f32, %[[VAL_15:.*]]: f32):
// CHECK:             %[[VAL_16:.*]] = math.exp %[[VAL_14]] : f32
// CHECK:             linalg.yield %[[VAL_16]] : f32
// CHECK:           } -> tensor<1024x1024xf32>
// CHECK:           %[[VAL_17:.*]] = arith.muli %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_18:.*]] = arith.index_cast %[[VAL_17]] : i32 to index
// CHECK:           %[[VAL_19:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_18]]], sizes: [1024, 1024], strides: [2, 1] : memref<*xf32> to memref<1024x1024xf32, strided<[2, 1], offset: ?>>
// CHECK:           memref.tensor_store %[[VAL_20:.*]], %[[VAL_19]] : memref<1024x1024xf32, strided<[2, 1], offset: ?>>
// CHECK:           return
// CHECK:         }
