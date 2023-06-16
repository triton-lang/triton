// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %a : !tt.ptr<i32>,
    %b : !tt.ptr<i32>
  ) -> () {
        // offset calculations
        %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
        // a pointer
        %8 = tt.splat %a : (!tt.ptr<i32>) -> tensor<1024x!tt.ptr<i32>>
        %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
        // b pointer
        %18 = tt.splat %b : (!tt.ptr<i32>) -> tensor<1024x!tt.ptr<i32>>
        %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
        %am = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi32>
        %bm = tt.load %19 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi32>
        %5 = arith.addi %am, %bm : tensor<1024xi32>
        tt.store %19, %5 : tensor<1024xi32>
        tt.return
    }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xi32>, %[[VAL_1:.*]]: memref<*xi32>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32) {
// CHECK:           %[[VAL_5:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1024], strides: [1] : memref<*xi32> to memref<1024xi32, strided<[1]>>
// CHECK:           %[[VAL_6:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [1024], strides: [1] : memref<*xi32> to memref<1024xi32, strided<[1]>>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<1024xi32>
// CHECK:           memref.copy %[[VAL_5]], %[[VAL_7]] : memref<1024xi32, strided<[1]>> to memref<1024xi32>
// CHECK:           %[[VAL_8:.*]] = bufferization.to_tensor %[[VAL_7]] restrict writable : memref<1024xi32>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<1024xi32>
// CHECK:           memref.copy %[[VAL_6]], %[[VAL_9]] : memref<1024xi32, strided<[1]>> to memref<1024xi32>
// CHECK:           %[[VAL_10:.*]] = bufferization.to_tensor %[[VAL_9]] restrict writable : memref<1024xi32>
// CHECK:           %[[VAL_11:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_8]], %[[VAL_10]] : tensor<1024xi32>, tensor<1024xi32>) outs(%[[VAL_8]] : tensor<1024xi32>) {
// CHECK:           ^bb0(%[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: i32):
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : i32
// CHECK:             linalg.yield %[[VAL_15]] : i32
// CHECK:           } -> tensor<1024xi32>
// CHECK:           memref.tensor_store %[[VAL_16:.*]], %[[VAL_6]] : memref<1024xi32, strided<[1]>>
// CHECK:           return
// CHECK:         }
