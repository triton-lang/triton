// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %a : !tt.ptr<f32>,
    %b : !tt.ptr<f32>,
    %c : tensor<1024x!tt.ptr<f32>>
  ) -> () {
        %cst = arith.constant dense<true> : tensor<1024xi1>
        // offset calculations
        %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
        // a pointer
        %8 = tt.splat %a : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
        %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        // b pointer
        %18 = tt.splat %b : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
        %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        %am = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32>
        %bm = tt.load %19 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32>
        %1 = arith.addf %am, %bm : tensor<1024xf32>
        %2 = arith.subf %1, %bm : tensor<1024xf32>
        %3 = arith.mulf %2, %bm : tensor<1024xf32>
        %4 = arith.divf %3, %bm : tensor<1024xf32>
        %5 = arith.cmpf "oeq", %4, %bm : tensor<1024xf32>
        %6 = arith.select %5, %am, %bm : tensor<1024xi1>, tensor<1024xf32>
        tt.store %c, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
        tt.return
    }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xf32>, %[[VAL_1:.*]]: memref<*xf32>, %[[VAL_2:.*]]: memref<1024xf32>, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32) {
// CHECK:           %[[VAL_6:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
// CHECK:           %[[VAL_7:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
// CHECK:           %[[VAL_8:.*]] = memref.alloc() : memref<1024xf32>
// CHECK:           memref.copy %[[VAL_6]], %[[VAL_8]] : memref<1024xf32, strided<[1]>> to memref<1024xf32>
// CHECK:           %[[VAL_9:.*]] = bufferization.to_tensor %[[VAL_8]] restrict writable : memref<1024xf32>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() : memref<1024xf32>
// CHECK:           memref.copy %[[VAL_7]], %[[VAL_10]] : memref<1024xf32, strided<[1]>> to memref<1024xf32>
// CHECK:           %[[VAL_11:.*]] = bufferization.to_tensor %[[VAL_10]] restrict writable : memref<1024xf32>
// CHECK:           %[[VAL_12:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_9]], %[[VAL_11]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_9]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_13:.*]]: f32, %[[VAL_14:.*]]: f32, %[[VAL_15:.*]]: f32):
// CHECK:             %[[VAL_16:.*]] = arith.addf %[[VAL_13]], %[[VAL_14]] : f32
// CHECK:             linalg.yield %[[VAL_16]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           %[[VAL_17:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_18:.*]], %[[VAL_11]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_18]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_19:.*]]: f32, %[[VAL_20:.*]]: f32, %[[VAL_21:.*]]: f32):
// CHECK:             %[[VAL_22:.*]] = arith.subf %[[VAL_19]], %[[VAL_20]] : f32
// CHECK:             linalg.yield %[[VAL_22]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           %[[VAL_23:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_24:.*]], %[[VAL_11]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_24]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_25:.*]]: f32, %[[VAL_26:.*]]: f32, %[[VAL_27:.*]]: f32):
// CHECK:             %[[VAL_28:.*]] = arith.mulf %[[VAL_25]], %[[VAL_26]] : f32
// CHECK:             linalg.yield %[[VAL_28]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           %[[VAL_29:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_30:.*]], %[[VAL_11]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_30]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_31:.*]]: f32, %[[VAL_32:.*]]: f32, %[[VAL_33:.*]]: f32):
// CHECK:             %[[VAL_34:.*]] = arith.divf %[[VAL_31]], %[[VAL_32]] : f32
// CHECK:             linalg.yield %[[VAL_34]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           %[[VAL_35:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK:           %[[VAL_36:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_37:.*]], %[[VAL_11]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_35]] : tensor<1024xi1>) {
// CHECK:           ^bb0(%[[VAL_38:.*]]: f32, %[[VAL_39:.*]]: f32, %[[VAL_40:.*]]: i1):
// CHECK:             %[[VAL_41:.*]] = arith.cmpf oeq, %[[VAL_38]], %[[VAL_39]] : f32
// CHECK:             linalg.yield %[[VAL_41]] : i1
// CHECK:           } -> tensor<1024xi1>
// CHECK:           %[[VAL_42:.*]] = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_43:.*]], %[[VAL_9]], %[[VAL_11]] : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_9]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_44:.*]]: i1, %[[VAL_45:.*]]: f32, %[[VAL_46:.*]]: f32, %[[VAL_47:.*]]: f32):
// CHECK:             %[[VAL_48:.*]] = arith.select %[[VAL_44]], %[[VAL_45]], %[[VAL_46]] : f32
// CHECK:             linalg.yield %[[VAL_48]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           memref.tensor_store %[[VAL_49:.*]], %[[VAL_2]] : memref<1024xf32>
// CHECK:           return
// CHECK:         }
