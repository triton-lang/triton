// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>
  )
  {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %0 = tt.splat %arg0 : (!tt.ptr<bf16>) -> tensor<256x!tt.ptr<bf16>>
    %1 = tt.make_range {end = 2048 : i32, start = 1024 : i32}:tensor<256xi32>
    // source: null, sizes: 256, offsets: 1024, strides: 4
    %2 = tt.addptr %0, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    // source: arg0, sizes: 256, offsets: 1024, strides: 4
    %3 = tt.splat %arg1 : (!tt.ptr<bf16>) -> tensor<256x!tt.ptr<bf16>>
    %4 = tt.addptr %3, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    // source: arg1, sizes: 256, offsets: 1024, strides: 4
    %_arg2, %_ptr_ld, %_arg3, %_ptr_st, %_arg4 = scf.for %i = %c0 to %c12 step %c3 iter_args(%arg2 = %c1, %ptr_ld = %2, %arg3 = %c2, %ptr_st = %4, %arg4 = %c3) -> (index, tensor<256x!tt.ptr<bf16>>, index, tensor<256x!tt.ptr<bf16>>, index) {
        // perform load
        %5 = tt.load %ptr_ld {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256xbf16>
        tt.store %ptr_st, %5 : tensor<256xbf16>
        // pointer updates
        %cast3 = arith.index_cast %c3 : index to i32
        %6 = tt.splat %cast3 : (i32) -> tensor<256xi32>
        %ptr_ld_iter = tt.addptr %ptr_ld, %6 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        // source: arg0, sizes: 256, offsets: 1024 + i*3, strides: 4
        %arg2_iter = arith.addi %arg2, %c3 : index
        %arg3_iter = arith.addi %arg3, %c3 : index
        %arg4_iter = arith.addi %arg4, %c3 : index
        %7 = arith.addi %arg2_iter, %arg3_iter : index
        %8 = arith.addi %7, %arg4_iter : index
        %cast8 = arith.index_cast %8 : index to i32
        %9 = tt.splat %cast8 : (i32) -> tensor<256xi32>
        %ptr_st_iter = tt.addptr %ptr_st, %9 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        // source: arg1, sizes: 256, offsets: 1024 + loop-carry variable*i, strides: 4
        scf.yield %arg2_iter, %ptr_ld_iter, %arg3_iter, %ptr_st_iter, %arg4_iter : index, tensor<256x!tt.ptr<bf16>>, index, tensor<256x!tt.ptr<bf16>>, index
    }
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: memref<*xbf16>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32) {
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 4 : index
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 1024 : index
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_9:.*]] = arith.constant 2 : index
// CHECK-DAG:           %[[VAL_10:.*]] = arith.constant 3 : index
// CHECK-DAG:           %[[VAL_11:.*]] = arith.constant 12 : index
// CHECK:           %[[VAL_12:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_6]]], sizes: [256], strides: {{\[}}%[[VAL_5]]] : memref<*xbf16> to memref<256xbf16, strided<[?], offset: ?>>
// CHECK:           %[[VAL_13:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_6]]], sizes: [256], strides: {{\[}}%[[VAL_5]]] : memref<*xbf16> to memref<256xbf16, strided<[?], offset: ?>>
// CHECK:           %[[VAL_14:.*]]:7 = scf.for %[[VAL_15:.*]] = %[[VAL_7]] to %[[VAL_11]] step %[[VAL_10]] iter_args(%[[VAL_16:.*]] = %[[VAL_8]], %[[VAL_17:.*]] = %[[VAL_12]], %[[VAL_18:.*]] = %[[VAL_9]], %[[VAL_19:.*]] = %[[VAL_13]], %[[VAL_20:.*]] = %[[VAL_10]], %[[VAL_21:.*]] = %[[VAL_6]], %[[VAL_22:.*]] = %[[VAL_6]]) -> (index, memref<256xbf16, strided<[?], offset: ?>>, index, memref<256xbf16, strided<[?], offset: ?>>, index, index, index) {
// CHECK:             %[[VAL_23:.*]] = memref.alloc() : memref<256xbf16>
// CHECK:             memref.copy %[[VAL_17]], %[[VAL_23]] : memref<256xbf16, strided<[?], offset: ?>> to memref<256xbf16>
// CHECK:             %[[VAL_24:.*]] = bufferization.to_tensor %[[VAL_23]] restrict writable : memref<256xbf16>
// CHECK:             memref.tensor_store %[[VAL_24]], %[[VAL_19]] : memref<256xbf16, strided<[?], offset: ?>>
// CHECK:             %[[VAL_25:.*]] = arith.addi %[[VAL_21]], %[[VAL_10]] : index
// CHECK:             %[[VAL_26:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_25]]], sizes: [256], strides: {{\[}}%[[VAL_5]]] : memref<*xbf16> to memref<256xbf16, strided<[?], offset: ?>>
// CHECK:             %[[VAL_27:.*]] = arith.addi %[[VAL_16]], %[[VAL_10]] : index
// CHECK:             %[[VAL_28:.*]] = arith.addi %[[VAL_18]], %[[VAL_10]] : index
// CHECK:             %[[VAL_29:.*]] = arith.addi %[[VAL_20]], %[[VAL_10]] : index
// CHECK:             %[[VAL_30:.*]] = arith.addi %[[VAL_27]], %[[VAL_28]] : index
// CHECK:             %[[VAL_31:.*]] = arith.addi %[[VAL_30]], %[[VAL_29]] : index
// CHECK:             %[[VAL_32:.*]] = arith.addi %[[VAL_22]], %[[VAL_31]] : index
// CHECK:             %[[VAL_33:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_32]]], sizes: [256], strides: {{\[}}%[[VAL_5]]] : memref<*xbf16> to memref<256xbf16, strided<[?], offset: ?>>
// CHECK:             scf.yield %[[VAL_27]], %[[VAL_26]], %[[VAL_28]], %[[VAL_33]], %[[VAL_29]], %[[VAL_25]], %[[VAL_32]] : index, memref<256xbf16, strided<[?], offset: ?>>, index, memref<256xbf16, strided<[?], offset: ?>>, index, index, index
// CHECK:           }
// CHECK:           return
// CHECK:         }
