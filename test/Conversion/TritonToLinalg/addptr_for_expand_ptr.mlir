// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>
  )
  {
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %0 = tt.splat %arg0 : (!tt.ptr<bf16>) -> tensor<256x!tt.ptr<bf16>>
    %1 = tt.make_range {end = 2048 : i32, start = 1024 : i32}:tensor<256xi32>
    // source: null, sizes: 256, offsets: 1024, strides: 4
    %2 = tt.addptr %0, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    // source: arg0, sizes: 256, offsets: 1024, strides: 4
    // gep operand is another gep' output, which is passed into the loop as varible, used after update
    %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %2) -> (tensor<256x!tt.ptr<bf16>>) {
      %6 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
      %7 = tt.expand_dims %6 {axis = 1 : i32} : (tensor<256xi32>) -> tensor<256x1xi32>
      %8 = tt.broadcast %7 : (tensor<256x1xi32>) -> tensor<256x256xi32>
      // sizes: [256, 256], offsets: [0, 0], strides: [1, 0]
      %9 = tt.make_range {end = 512 : i32, start = 256 : i32} : tensor<256xi32>
      %10 = tt.expand_dims %9 {axis = 0 : i32} : (tensor<256xi32>) -> tensor<1x256xi32>
      %11 = tt.broadcast %10 : (tensor<1x256xi32>) -> tensor<256x256xi32>
      // sizes: [256, 256], offsets: [0, 256], strides: [0, 1]
      %12 = arith.addi %8, %11 : tensor<256x256xi32>
      // sizes: [256, 256], offsets: [0, 256], strides: [1, 1]
      %13 = tt.expand_dims %ptr {axis = 1 : i32} : (tensor<256x!tt.ptr<bf16>>) -> tensor<256x1x!tt.ptr<bf16>>
      %14 = tt.broadcast %13 : (tensor<256x1x!tt.ptr<bf16>>) -> tensor<256x256x!tt.ptr<bf16>>
      %15 = tt.addptr %14, %12 : tensor<256x256x!tt.ptr<bf16>>, tensor<256x256xi32>
      // source: arg0, sizes: [256, 256], offsets: [1024 + i, 256], strides: [5, 1]
      // perform load
      %16 = tt.load %15 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x256xbf16>
      tt.store %15, %16 : tensor<256x256xbf16>
      // pointer updates
      %17 = tt.splat %i_c3 : (i32) -> tensor<256xi32>
      // sizes: 256, offsets: 3, strides: 0
      %ptr_iter = tt.addptr %ptr, %17 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
      // source: arg0, sizes: 256, offsets: 1024 + i, strides: 4
      scf.yield %ptr_iter : tensor<256x!tt.ptr<bf16>>
    }
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32) {
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 5 : index
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 256 : index
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 1024 : index
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_8:.*]] = arith.constant 12 : index
// CHECK-DAG:           %[[VAL_9:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_7]] to %[[VAL_8]] step %[[VAL_9]] iter_args(%[[VAL_12:.*]] = %[[VAL_6]]) -> (index) {
// CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_5]] : index
// CHECK:             %[[VAL_14:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_13]]], sizes: [256, 256], strides: {{\[}}%[[VAL_4]], 1] : memref<*xbf16> to memref<256x256xbf16, strided<[?, 1], offset: ?>>
// CHECK:             %[[VAL_15:.*]] = memref.alloc() : memref<256x256xbf16>
// CHECK:             memref.copy %[[VAL_14]], %[[VAL_15]] : memref<256x256xbf16, strided<[?, 1], offset: ?>> to memref<256x256xbf16>
// CHECK:             %[[VAL_16:.*]] = bufferization.to_tensor %[[VAL_15]] restrict writable : memref<256x256xbf16>
// CHECK:             memref.tensor_store %[[VAL_16]], %[[VAL_14]] : memref<256x256xbf16, strided<[?, 1], offset: ?>>
// CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_12]], %[[VAL_9]] : index
// CHECK:             scf.yield %[[VAL_17]] : index
// CHECK:           }
// CHECK:           return
// CHECK:         }
