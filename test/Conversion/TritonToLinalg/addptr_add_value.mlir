// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32,
  %arg3 : i32
  )
  {
  %0 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
  // offset = 0, size = 4, stride = 1
  %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<4xi32>) -> tensor<4x1xi32>
  // offset = [0,0], size = [4,1], stride = [1,0]
  %2 = tt.broadcast %1 : (tensor<4x1xi32>) -> tensor<4x256xi32>
  // offset = [0,0], size = [4,256], stride = [1,0]
  %arg2splat = tt.splat %arg2 : (i32) -> tensor<4x256xi32>
  %offset2 = arith.addi %2, %arg2splat : tensor<4x256xi32>
  // offset = [%arg2,0], size = [4,256], stride = [1,0]
  %arg3splat = tt.splat %arg3 : (i32) -> tensor<4x256xi32>
  %offset3 = arith.addi %offset2, %arg3splat : tensor<4x256xi32>
  // offset = [%arg2+%arg3,0], size = [4,256], stride = [1,0]
  %c10 = arith.constant 10 : i32
  %c10splat = tt.splat %c10 : (i32) -> tensor<4x256xi32>
  %offset4 = arith.addi %offset3, %c10splat : tensor<4x256xi32>
  // offset = [%arg2+%arg3+10,0], size = [4,256], stride = [1,0]
  %3 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
  // offset = 0, size = 256, stride = 1
  %4 = tt.expand_dims %3 {axis = 0 : i32} : (tensor<256xi32>) -> tensor<1x256xi32>
  // offset = [0,0], size = [1,256], stride = [0,1]
  %5 = tt.broadcast %4 : (tensor<1x256xi32>) -> tensor<4x256xi32>
  // offset = [0,0], size = [4,256], stride = [0,1]
  %c6 = arith.constant 6 : i32
  %splat6 = tt.splat %c6 : (i32) -> tensor<4x256xi32>
  %scale5 = arith.muli %5, %splat6 : tensor<4x256xi32>
  // offset = [0,0], size = [4,256], stride = [0,6]
  %7 = arith.addi %offset4, %scale5: tensor<4x256xi32>
  // offset = [%arg2+%arg3+10, 0], size = [4, 256], stride = [1, 6]
  %8 = tt.splat %arg0 : (!tt.ptr<bf16>) -> tensor<4x256x!tt.ptr<bf16>>
  %9 = tt.addptr %8, %7 : tensor<4x256x!tt.ptr<bf16>>,tensor<4x256xi32>
  // source = %arg0, offset = [%arg2+%arg3+10, 0], size = [4, 256], stride = [1, 6]
  %10 = tt.splat %arg1 : (!tt.ptr<bf16>) -> tensor<4x256x!tt.ptr<bf16>>
  %11 = tt.addptr %10, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
  // source = %arg1, offset = [%arg2+%arg3+10, 0], size = [4, 256], stride = [1, 6]
  %12 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<4x256xbf16>
  tt.store %11, %12 : tensor<4x256xbf16>
  tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: memref<*xbf16>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 6 : index
// CHECK-DAG:           %[[VAL_8:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_2]] : i32 to index
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : index
// CHECK:           %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_8]] : index
// CHECK:           %[[VAL_13:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_12]]], sizes: [4, 256], strides: [1, %[[VAL_7]]] : memref<*xbf16> to memref<4x256xbf16, strided<[1, ?], offset: ?>>
// CHECK:           %[[VAL_14:.*]] = arith.index_cast %[[VAL_2]] : i32 to index
// CHECK:           %[[VAL_15:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_16:.*]] = arith.addi %[[VAL_14]], %[[VAL_15]] : index
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_8]] : index
// CHECK:           %[[VAL_18:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_17]]], sizes: [4, 256], strides: [1, %[[VAL_7]]] : memref<*xbf16> to memref<4x256xbf16, strided<[1, ?], offset: ?>>
// CHECK:           %[[VAL_19:.*]] = memref.alloc() : memref<4x256xbf16>
// CHECK:           memref.copy %[[VAL_13]], %[[VAL_19]] : memref<4x256xbf16, strided<[1, ?], offset: ?>> to memref<4x256xbf16>
// CHECK:           %[[VAL_20:.*]] = bufferization.to_tensor %[[VAL_19]] restrict writable : memref<4x256xbf16>
// CHECK:           memref.tensor_store %[[VAL_20]], %[[VAL_18]] : memref<4x256xbf16, strided<[1, ?], offset: ?>>
// CHECK:           return
// CHECK:         }
