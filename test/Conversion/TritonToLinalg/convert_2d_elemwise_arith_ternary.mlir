// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
    tt.func @kernel(
                        %a : !tt.ptr<i1>,
                        %b : !tt.ptr<f32>,
                        %c : !tt.ptr<f32>,
                        %d : tensor<128x128x!tt.ptr<f32>>
  ) -> () {
        // offset calculations
        %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
        %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<128xi32>) -> tensor<128x1xi32>
        %moff = tt.broadcast %1 : (tensor<128x1xi32>) -> tensor<128x128xi32>
        %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
        %4 = tt.expand_dims %3 {axis = 0 : i32} : (tensor<128xi32>) -> tensor<1x128xi32>
        %koff = tt.broadcast %4 : (tensor<1x128xi32>) -> tensor<128x128xi32>
        %mkoff = arith.addi %moff, %koff : tensor<128x128xi32>
        // a pointer
        %8 = tt.splat %a : (!tt.ptr<i1>) -> tensor<128x128x!tt.ptr<i1>>
        %9 = tt.addptr %8, %mkoff : tensor<128x128x!tt.ptr<i1>>, tensor<128x128xi32>
        // b pointer
        %18 = tt.splat %b : (!tt.ptr<f32>) -> tensor<128x128x!tt.ptr<f32>>
        %19 = tt.addptr %18, %mkoff : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
        // c pointer
        %28 = tt.splat %c : (!tt.ptr<f32>) -> tensor<128x128x!tt.ptr<f32>>
        %29 = tt.addptr %28, %mkoff : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
        %am = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xi1>
        %bm = tt.load %19 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf32>
        %cm = tt.load %29 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf32>
        %100 = arith.select %am, %bm, %cm : tensor<128x128xi1>, tensor<128x128xf32>
        tt.store %d, %100 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf32>
        tt.return
    }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xi1>, %[[VAL_1:.*]]: memref<*xf32>, %[[VAL_2:.*]]: memref<*xf32>, %[[VAL_3:.*]]: memref<128x128xf32>, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK:           %[[VAL_7:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [128, 128], strides: [1, 1] : memref<*xi1> to memref<128x128xi1, strided<[1, 1]>>
// CHECK:           %[[VAL_8:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1]>>
// CHECK:           %[[VAL_9:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: [0], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1]>>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() : memref<128x128xi1>
// CHECK:           memref.copy %[[VAL_7]], %[[VAL_10]] : memref<128x128xi1, strided<[1, 1]>> to memref<128x128xi1>
// CHECK:           %[[VAL_11:.*]] = bufferization.to_tensor %[[VAL_10]] restrict writable : memref<128x128xi1>
// CHECK:           %[[VAL_12:.*]] = memref.alloc() : memref<128x128xf32>
// CHECK:           memref.copy %[[VAL_8]], %[[VAL_12]] : memref<128x128xf32, strided<[1, 1]>> to memref<128x128xf32>
// CHECK:           %[[VAL_13:.*]] = bufferization.to_tensor %[[VAL_12]] restrict writable : memref<128x128xf32>
// CHECK:           %[[VAL_14:.*]] = memref.alloc() : memref<128x128xf32>
// CHECK:           memref.copy %[[VAL_9]], %[[VAL_14]] : memref<128x128xf32, strided<[1, 1]>> to memref<128x128xf32>
// CHECK:           %[[VAL_15:.*]] = bufferization.to_tensor %[[VAL_14]] restrict writable : memref<128x128xf32>
// CHECK:           %[[VAL_16:.*]] = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_11]], %[[VAL_13]], %[[VAL_15]] : tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) outs(%[[VAL_13]] : tensor<128x128xf32>) {
// CHECK:           ^bb0(%[[VAL_17:.*]]: i1, %[[VAL_18:.*]]: f32, %[[VAL_19:.*]]: f32, %[[VAL_20:.*]]: f32):
// CHECK:             %[[VAL_21:.*]] = arith.select %[[VAL_17]], %[[VAL_18]], %[[VAL_19]] : f32
// CHECK:             linalg.yield %[[VAL_21]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           memref.tensor_store %[[VAL_22:.*]], %[[VAL_3]] : memref<128x128xf32>
// CHECK:           return
// CHECK:         }
