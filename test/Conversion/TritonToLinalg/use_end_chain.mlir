// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>
  )
  {
  %0 = tt.make_range {end = 768 : i32, start = 512 : i32}:tensor<256xi32>
  // offset = [512] size = 256, stride = 1
  %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<256xi32>) -> tensor<256x1xi32>
  // offset = [512,0], size = [256,1], stride = [1,0]
  %2 = tt.broadcast %1 : (tensor<256x1xi32>) -> tensor<256x128xi32>
  // offset = [512,0], size = [256,128], stride = [1,0]
  %5 = tt.make_range {end = 1152 : i32, start = 1024 : i32}:tensor<128xi32>
  // offset = 1024, size = 128, stride = 1
  %6 = tt.expand_dims %5 {axis = 0 : i32} : (tensor<128xi32>) -> tensor<1x128xi32>
  // offset = [0,1024], size = [1,128], stride = [0,1]
  %7 = tt.broadcast %6 : (tensor<1x128xi32>) -> tensor<256x128xi32>
  // offset = [0,1024], size = [256,128], stride = [0,1]
  %c6 = arith.constant 6 : i32
  %splat6 = tt.splat %c6 : (i32) -> tensor<256x128xi32>
  %scale7 = arith.muli %7, %splat6 : tensor<256x128xi32>
  // offset = [0,6144], size = [256,128], stride = [0,6]
  %14 = arith.addi %2, %scale7 : tensor<256x128xi32>
  // offset = [512,6144], size = [256,128], stride = [1,6]
  // mixed use
  %17 = tt.splat %arg1 : (!tt.ptr<bf16>) -> tensor<256x128x!tt.ptr<bf16>>
  %18 = tt.addptr %17, %14 : tensor<256x128x!tt.ptr<bf16>>, tensor<256x128xi32>
  %19 = tt.load %18 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x128xbf16>
  tt.store %18, %19 : tensor<256x128xbf16>
  %20 = arith.sitofp %14 : tensor<256x128xi32> to tensor<256x128xbf16>
  tt.store %18, %20 : tensor<256x128xbf16>
  tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: memref<*xbf16>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32) {
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 6 : index
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 6 : i32
// CHECK:           %[[VAL_30:.*]] = tensor.empty() : tensor<256x128xi32>
// CHECK:           %[[VAL_31:.*]] = linalg.fill ins(%[[VAL_7]] : i32) outs(%[[VAL_30]] : tensor<256x128xi32>) -> tensor<256x128xi32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<256xi32>
// CHECK:           %[[VAL_9:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%[[VAL_8]] : tensor<256xi32>) {
// CHECK:           ^bb0(%[[VAL_10:.*]]: i32):
// CHECK:             %[[VAL_11:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_12:.*]] = arith.index_cast %[[VAL_11]] : index to i32
// CHECK:             linalg.yield %[[VAL_12]] : i32
// CHECK:           } -> tensor<256xi32>
// CHECK:           %[[VAL_13:.*]] = tensor.expand_shape %[[VAL_14:.*]] {{\[\[}}0, 1]] : tensor<256xi32> into tensor<256x1xi32>
// CHECK:           %[[VAL_15:.*]] = tensor.empty() : tensor<256x128xi32>
// CHECK:           %[[VAL_16:.*]] = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_13]] : tensor<256x1xi32>) outs(%[[VAL_15]] : tensor<256x128xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0(%[[VAL_17:.*]]: i32, %[[VAL_18:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_17]] : i32
// CHECK:           } -> tensor<256x128xi32>
// CHECK:           %[[VAL_19:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:           %[[VAL_20:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%[[VAL_19]] : tensor<128xi32>) {
// CHECK:           ^bb0(%[[VAL_21:.*]]: i32):
// CHECK:             %[[VAL_22:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_23:.*]] = arith.index_cast %[[VAL_22]] : index to i32
// CHECK:             linalg.yield %[[VAL_23]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           %[[VAL_24:.*]] = tensor.expand_shape %[[VAL_25:.*]] {{\[\[}}0, 1]] : tensor<128xi32> into tensor<1x128xi32>
// CHECK:           %[[VAL_26:.*]] = tensor.empty() : tensor<256x128xi32>
// CHECK:           %[[VAL_27:.*]] = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_24]] : tensor<1x128xi32>) outs(%[[VAL_26]] : tensor<256x128xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_28:.*]]: i32, %[[VAL_29:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_28]] : i32
// CHECK:           } -> tensor<256x128xi32>
// CHECK:           %[[VAL_32:.*]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_33:.*]], %[[VAL_31]] : tensor<256x128xi32>, tensor<256x128xi32>) outs(%[[VAL_33]] : tensor<256x128xi32>) {
// CHECK:           ^bb0(%[[VAL_34:.*]]: i32, %[[VAL_35:.*]]: i32, %[[VAL_36:.*]]: i32):
// CHECK:             %[[VAL_37:.*]] = arith.muli %[[VAL_34]], %[[VAL_35]] : i32
// CHECK:             linalg.yield %[[VAL_37]] : i32
// CHECK:           } -> tensor<256x128xi32>
// CHECK:           %[[VAL_38:.*]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_39:.*]], %[[VAL_40:.*]] : tensor<256x128xi32>, tensor<256x128xi32>) outs(%[[VAL_39]] : tensor<256x128xi32>) {
// CHECK:           ^bb0(%[[VAL_41:.*]]: i32, %[[VAL_42:.*]]: i32, %[[VAL_43:.*]]: i32):
// CHECK:             %[[VAL_44:.*]] = arith.addi %[[VAL_41]], %[[VAL_42]] : i32
// CHECK:             linalg.yield %[[VAL_44]] : i32
// CHECK:           } -> tensor<256x128xi32>
// CHECK:           %[[VAL_45:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}6656], sizes: [256, 128], strides: [1, %[[VAL_6]]] : memref<*xbf16> to memref<256x128xbf16, strided<[1, ?], offset: 6656>>
// CHECK:           %[[VAL_46:.*]] = memref.alloc() : memref<256x128xbf16>
// CHECK:           memref.copy %[[VAL_45]], %[[VAL_46]] : memref<256x128xbf16, strided<[1, ?], offset: 6656>> to memref<256x128xbf16>
// CHECK:           %[[VAL_47:.*]] = bufferization.to_tensor %[[VAL_46]] restrict writable : memref<256x128xbf16>
// CHECK:           memref.tensor_store %[[VAL_47]], %[[VAL_45]] : memref<256x128xbf16, strided<[1, ?], offset: 6656>>
// CHECK:           %[[VAL_48:.*]] = tensor.empty() : tensor<256x128xbf16>
// CHECK:           %[[VAL_49:.*]] = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_50:.*]] : tensor<256x128xi32>) outs(%[[VAL_48]] : tensor<256x128xbf16>) {
// CHECK:           ^bb0(%[[VAL_51:.*]]: i32, %[[VAL_52:.*]]: bf16):
// CHECK:             %[[VAL_53:.*]] = arith.sitofp %[[VAL_51]] : i32 to bf16
// CHECK:             linalg.yield %[[VAL_53]] : bf16
// CHECK:           } -> tensor<256x128xbf16>
// CHECK:           memref.tensor_store %[[VAL_54:.*]], %[[VAL_45]] : memref<256x128xbf16, strided<[1, ?], offset: 6656>>
// CHECK:           return
// CHECK:         }
