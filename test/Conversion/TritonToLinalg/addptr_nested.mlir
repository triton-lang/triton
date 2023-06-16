// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : i32
  )
  {
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
    // offset = 0, size = 4, stride = 1
    %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<4xi32>) -> tensor<4x1xi32>
    // offset = [0,0], size = [4,1], stride = [1,0]
    %2 = tt.broadcast %1 : (tensor<4x1xi32>) -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [1,0]
    %arg1splat = tt.splat %arg1 : (i32) -> tensor<4x256xi32>
    %offset3 = arith.addi %2, %arg1splat : tensor<4x256xi32>
    // offset = [%arg1,0], size = [4,256], stride = [1,0]
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
    // offset = 0, size = 256, stride = 1
    %4 = tt.expand_dims %3 {axis = 0 : i32} : (tensor<256xi32>) -> tensor<1x256xi32>
    // offset = [0,0], size = [1,256], stride = [0,1]
    %5 = tt.broadcast %4 : (tensor<1x256xi32>) -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [0,1]
    %6 = arith.constant 5 : i32
    %splat6 = tt.splat %6 : (i32) -> tensor<4x256xi32>
    %scale5 = arith.muli %5, %splat6 : tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [0,5]
    %7 = arith.addi %offset3, %scale5: tensor<4x256xi32>
    // offset = [%arg1, 0], size = [4, 256], stride = [1, 5]
    %8 = tt.splat %arg0 : (!tt.ptr<bf16>) -> tensor<4x256x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg0, offset = [%arg1, 0], size = [4, 256], stride = [1, 5]
    %10 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256xbf16>
    %12 = tt.addptr %9, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg0, offset = [%arg1+%arg1, 0], size = [4, 256], stride = [2, 10]
    %13 = tt.load %12 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256xbf16>
    %14 = arith.addf %10, %13 : tensor<4x256xbf16>
    %16 = tt.addptr %12, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg0, offset = [%arg1+%arg1+%arg1, 0], size = [4, 256], stride = [3, 15]
    tt.store %16, %14 : tensor<4x256xbf16>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32) {
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 15 : index
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 5 : index
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_8:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[VAL_9:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_8]]], sizes: [4, 256], strides: [1, %[[VAL_6]]] : memref<*xbf16> to memref<4x256xbf16, strided<[1, ?], offset: ?>>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() : memref<4x256xbf16>
// CHECK:           memref.copy %[[VAL_9]], %[[VAL_10]] : memref<4x256xbf16, strided<[1, ?], offset: ?>> to memref<4x256xbf16>
// CHECK:           %[[VAL_11:.*]] = bufferization.to_tensor %[[VAL_10]] restrict writable : memref<4x256xbf16>
// CHECK:           %[[VAL_12:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[VAL_13:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : index
// CHECK:           %[[VAL_15:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_14]]], sizes: [4, 256], strides: [2, %[[VAL_7]]] : memref<*xbf16> to memref<4x256xbf16, strided<[2, ?], offset: ?>>
// CHECK:           %[[VAL_16:.*]] = memref.alloc() : memref<4x256xbf16>
// CHECK:           memref.copy %[[VAL_15]], %[[VAL_16]] : memref<4x256xbf16, strided<[2, ?], offset: ?>> to memref<4x256xbf16>
// CHECK:           %[[VAL_17:.*]] = bufferization.to_tensor %[[VAL_16]] restrict writable : memref<4x256xbf16>
// CHECK:           %[[VAL_18:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_11]], %[[VAL_17]] : tensor<4x256xbf16>, tensor<4x256xbf16>) outs(%[[VAL_11]] : tensor<4x256xbf16>) {
// CHECK:           ^bb0(%[[VAL_19:.*]]: bf16, %[[VAL_20:.*]]: bf16, %[[VAL_21:.*]]: bf16):
// CHECK:             %[[VAL_22:.*]] = arith.addf %[[VAL_19]], %[[VAL_20]] : bf16
// CHECK:             linalg.yield %[[VAL_22]] : bf16
// CHECK:           } -> tensor<4x256xbf16>
// CHECK:           %[[VAL_23:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[VAL_24:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[VAL_25:.*]] = arith.addi %[[VAL_23]], %[[VAL_24]] : index
// CHECK:           %[[VAL_26:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[VAL_27:.*]] = arith.addi %[[VAL_25]], %[[VAL_26]] : index
// CHECK:           %[[VAL_28:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_27]]], sizes: [4, 256], strides: [3, %[[VAL_5]]] : memref<*xbf16> to memref<4x256xbf16, strided<[3, ?], offset: ?>>
// CHECK:           memref.tensor_store %[[VAL_29:.*]], %[[VAL_28]] : memref<4x256xbf16, strided<[3, ?], offset: ?>>
// CHECK:           return
// CHECK:         }
