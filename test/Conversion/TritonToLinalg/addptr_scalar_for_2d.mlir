// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %cf0 = arith.constant 0.000000e+00 : f32
    %tensor_cf0 = tt.splat %cf0 : (f32) -> tensor<128x128xf32>
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %sum_out, %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%sum_iter = %tensor_cf0,  %ptr_iter = %2) ->  (tensor<128x128xf32>, !tt.ptr<f32> ) {
      %3 = tt.splat %ptr_iter : (!tt.ptr<f32>) -> tensor<128x128x!tt.ptr<f32>>
      // source = %arg1, offset = [%1, 0], size = [128, 128], strides = [0, 0]
      %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
      %5 = tt.expand_dims %4 {axis = 0 : i32} : (tensor<128xi32>) -> tensor<1x128xi32>
      %6 = tt.broadcast %5 : (tensor<1x128xi32>) -> tensor<128x128xi32>
      // offset = [0, 0], size = [128, 128], strides = [0, 1]
      %7 = tt.make_range {end = 384 : i32, start = 128 : i32} : tensor<128xi32>
      %8 = tt.expand_dims %7 {axis = 1 : i32} : (tensor<128xi32>) -> tensor<128x1xi32>
      %9 = tt.broadcast %8 : (tensor<128x1xi32>) -> tensor<128x128xi32>
      // offset = [128, 0], size = [128, 128], strides = [2, 0]
      %10 = arith.addi %6, %9 : tensor<128x128xi32>
      // offset = [128, 0], size = [128, 128], strides = [2, 1]
      %11 = tt.addptr %3, %10 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
      // source = %arg1, offset = [%1 + 128, 0], size = [128, 128], strides = [2, 1]
      %12 = tt.load %11 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf32>
      %17 = math.exp %12 : tensor<128x128xf32>
      %sum_next = arith.addf %sum_iter, %17 : tensor<128x128xf32>
      %cast_i = arith.index_cast %i : index to i32
      %ptr_next = tt.addptr %ptr_iter, %cast_i : !tt.ptr<f32>, i32
      // source = %arg1, offset = %1 + %i, size = 1, strides = 0
      scf.yield %sum_next, %ptr_next : tensor<128x128xf32>, !tt.ptr<f32>
    }
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : (tensor<128xi32>) -> tensor<1x128xi32>
    %6 = tt.broadcast %5 : (tensor<1x128xi32>) -> tensor<128x128xi32>
    // offset = [0, 0], size = [128, 128], strides = [0, 1]
    %7 = tt.make_range {end = 384 : i32, start = 128 : i32} : tensor<128xi32>
    %8 = tt.expand_dims %7 {axis = 1 : i32} : (tensor<128xi32>) -> tensor<128x1xi32>
    %9 = tt.broadcast %8 : (tensor<128x1xi32>) -> tensor<128x128xi32>
    // offset = [128, 0], size = [128, 128], strides = [2, 0]
    %10 = arith.addi %6, %9 : tensor<128x128xi32>
    // offset = [128, 0], size = [128, 128], strides = [2, 1]
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    // source = arg0, offset = %18, size = 1, strides = 0
    %20 = tt.splat %19 : (!tt.ptr<f32>) -> tensor<128x128x!tt.ptr<f32>>
    // source = arg0, offset = [%18, 0], size = [128, 128], strides = [0, 0]
    %21 = tt.addptr %20, %10 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
    // source = %arg0, offset = [%18 + 128, 0], size = [128, 128], strides = [2, 1]
    tt.store %21, %sum_out : tensor<128x128xf32>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xf32> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: memref<*xf32> {tt.divisibility = 16 : i32}, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32) {
// CHECK-DAG:           %[[VAL_8:.*]] = arith.constant 128 : index
// CHECK-DAG:           %[[VAL_9:.*]] = arith.constant 3 : index
// CHECK-DAG:           %[[VAL_10:.*]] = arith.constant 12 : index
// CHECK-DAG:           %[[VAL_11:.*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_12:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_15:.*]] = tensor.empty() : tensor<128x128xf32>
// CHECK:           %[[VAL_16:.*]] = linalg.fill ins(%[[VAL_12]] : f32) outs(%[[VAL_15]] : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK:           %[[VAL_13:.*]] = arith.muli %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_14:.*]] = arith.index_cast %[[VAL_13]] : i32 to index
// CHECK:           %[[VAL_17:.*]]:2 = scf.for %[[VAL_18:.*]] = %[[VAL_11]] to %[[VAL_10]] step %[[VAL_9]] iter_args(%[[VAL_19:.*]] = %[[VAL_16]], %[[VAL_20:.*]] = %[[VAL_14]]) -> (tensor<128x128xf32>, index) {
// CHECK:             %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_8]] : index
// CHECK:             %[[VAL_22:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_21]]], sizes: [128, 128], strides: [2, 1] : memref<*xf32> to memref<128x128xf32, strided<[2, 1], offset: ?>>
// CHECK:             %[[VAL_23:.*]] = memref.alloc() : memref<128x128xf32>
// CHECK:             memref.copy %[[VAL_22]], %[[VAL_23]] : memref<128x128xf32, strided<[2, 1], offset: ?>> to memref<128x128xf32>
// CHECK:             %[[VAL_24:.*]] = bufferization.to_tensor %[[VAL_23]] restrict writable : memref<128x128xf32>
// CHECK:             %[[VAL_25:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_24]] : tensor<128x128xf32>) outs(%[[VAL_24]] : tensor<128x128xf32>) {
// CHECK:             ^bb0(%[[VAL_26:.*]]: f32, %[[VAL_27:.*]]: f32):
// CHECK:               %[[VAL_28:.*]] = math.exp %[[VAL_26]] : f32
// CHECK:               linalg.yield %[[VAL_28]] : f32
// CHECK:             } -> tensor<128x128xf32>
// CHECK:             %[[VAL_29:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_19]], %[[VAL_30:.*]] : tensor<128x128xf32>, tensor<128x128xf32>) outs(%[[VAL_19]] : tensor<128x128xf32>) {
// CHECK:             ^bb0(%[[VAL_31:.*]]: f32, %[[VAL_32:.*]]: f32, %[[VAL_33:.*]]: f32):
// CHECK:               %[[VAL_34:.*]] = arith.addf %[[VAL_31]], %[[VAL_32]] : f32
// CHECK:               linalg.yield %[[VAL_34]] : f32
// CHECK:             } -> tensor<128x128xf32>
// CHECK:             %[[VAL_35:.*]] = arith.addi %[[VAL_20]], %[[VAL_18]] : index
// CHECK:             scf.yield %[[VAL_36:.*]], %[[VAL_35]] : tensor<128x128xf32>, index
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = arith.muli %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_38:.*]] = arith.index_cast %[[VAL_37]] : i32 to index
// CHECK:           %[[VAL_39:.*]] = arith.addi %[[VAL_38]], %[[VAL_8]] : index
// CHECK:           %[[VAL_40:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_39]]], sizes: [128, 128], strides: [2, 1] : memref<*xf32> to memref<128x128xf32, strided<[2, 1], offset: ?>>
// CHECK:           memref.tensor_store %[[VAL_41:.*]]#0, %[[VAL_40]] : memref<128x128xf32, strided<[2, 1], offset: ?>>
// CHECK:           return
// CHECK:         }
