// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    // source = %arg1, offset = %1, size = 1, strides = 0
    %cf0 = arith.constant 0.000000e+00 : f32
    %tensor_cf0 = tt.splat %cf0 : (f32) -> tensor<1024xf32>
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %_ptr, %sum_out = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr_iter = %2, %sum_iter = %tensor_cf0) ->  (!tt.ptr<f32>, tensor<1024xf32>) {
      %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
      // offset = 0, size = 1024, strides = 1
      %4 = tt.splat %ptr_iter : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
      // source = %arg1, offset = %1, size = 1024, strides = 0
      %5 = tt.addptr %4, %3 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      // source = %arg1, offset = %1, size = 1024, strides = 1
      %8 = tt.load %5 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32>
      %9 = math.exp %8 : tensor<1024xf32>
      %sum_next = arith.addf %sum_iter, %9 : tensor<1024xf32>
      %cast_i = arith.index_cast %i : index to i32
      %ptr_next = tt.addptr %ptr_iter, %cast_i : !tt.ptr<f32>, i32
      // source = %arg1, offset = %1 + %i, size = 1, strides = 0
      scf.yield %ptr_next, %sum_next : !tt.ptr<f32>, tensor<1024xf32>
    }
    %10 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    %20 = tt.splat %19 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %21 = tt.addptr %20, %10 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %21, %sum_out : tensor<1024xf32>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xf32> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: memref<*xf32> {tt.divisibility = 16 : i32}, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32) {
// CHECK-DAG:           %[[VAL_8:.*]] = arith.constant 3 : index
// CHECK-DAG:           %[[VAL_9:.*]] = arith.constant 12 : index
// CHECK-DAG:           %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_11:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_14:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK:           %[[VAL_15:.*]] = linalg.fill ins(%[[VAL_11]] : f32) outs(%[[VAL_14]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_12:.*]] = arith.muli %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_13:.*]] = arith.index_cast %[[VAL_12]] : i32 to index
// CHECK:           %[[VAL_16:.*]]:2 = scf.for %[[VAL_17:.*]] = %[[VAL_10]] to %[[VAL_9]] step %[[VAL_8]] iter_args(%[[VAL_18:.*]] = %[[VAL_15]], %[[VAL_19:.*]] = %[[VAL_13]]) -> (tensor<1024xf32>, index) {
// CHECK:             %[[VAL_20:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_19]]], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK:             %[[VAL_21:.*]] = memref.alloc() : memref<1024xf32>
// CHECK:             memref.copy %[[VAL_20]], %[[VAL_21]] : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
// CHECK:             %[[VAL_22:.*]] = bufferization.to_tensor %[[VAL_21]] restrict writable : memref<1024xf32>
// CHECK:             %[[VAL_23:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%[[VAL_22]] : tensor<1024xf32>) outs(%[[VAL_22]] : tensor<1024xf32>) {
// CHECK:             ^bb0(%[[VAL_24:.*]]: f32, %[[VAL_25:.*]]: f32):
// CHECK:               %[[VAL_26:.*]] = math.exp %[[VAL_24]] : f32
// CHECK:               linalg.yield %[[VAL_26]] : f32
// CHECK:             } -> tensor<1024xf32>
// CHECK:             %[[VAL_27:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_18]], %[[VAL_28:.*]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_18]] : tensor<1024xf32>) {
// CHECK:             ^bb0(%[[VAL_29:.*]]: f32, %[[VAL_30:.*]]: f32, %[[VAL_31:.*]]: f32):
// CHECK:               %[[VAL_32:.*]] = arith.addf %[[VAL_29]], %[[VAL_30]] : f32
// CHECK:               linalg.yield %[[VAL_32]] : f32
// CHECK:             } -> tensor<1024xf32>
// CHECK:             %[[VAL_33:.*]] = arith.addi %[[VAL_19]], %[[VAL_17]] : index
// CHECK:             scf.yield %[[VAL_34:.*]], %[[VAL_33]] : tensor<1024xf32>, index
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = arith.muli %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_36:.*]] = arith.index_cast %[[VAL_35]] : i32 to index
// CHECK:           %[[VAL_37:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_36]]], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK:           memref.tensor_store %[[VAL_38:.*]]#0, %[[VAL_37]] : memref<1024xf32, strided<[1], offset: ?>>
// CHECK:           return
// CHECK:         }
