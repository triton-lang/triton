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
    // Example 2, gep operand is another gep's output, which is passed into the loop as varible, used before update
    %_ptr2 = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %2) -> (tensor<256x!tt.ptr<bf16>>) {
        // perform load
        %3 = tt.load %ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256xbf16>
        tt.store %ptr, %3 : tensor<256xbf16>
        // pointer updates
        %4 = tt.splat %i_c3 : (i32) -> tensor<256xi32>
        %ptr_iter = tt.addptr %ptr, %4 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        scf.yield %ptr_iter : tensor<256x!tt.ptr<bf16>>
    }
    // Expected output
    // %offset_dim0 = arith.constant 1024                                                       <- insert instructions to initialize init arg(new)
    // for iter_args (%offset_dim0_iter = %offset_dim0) {                                       <- replace varibles passed in as init arg (new)
    //  %subview = memref.subview %arg0, [%offset_dim0_iter][256][4] : memref<> -> memref<>     <- generate subview on load (new)
    //  ...
    //  %4 = %offset_dim0_iter + %c3                                                            <- replace gep of splat with add (already done)
    //  scf.yield %4                                                                            <- replace yielding an gep output with the corresponding dim variable (new)
    // }
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32) {
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 4 : index
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 1024 : index
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 12 : index
// CHECK-DAG:           %[[VAL_8:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_9:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_5]]], sizes: [256], strides: {{\[}}%[[VAL_4]]] : memref<*xbf16> to memref<256xbf16, strided<[?], offset: ?>>
// CHECK:           %[[VAL_10:.*]]:2 = scf.for %[[VAL_11:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_8]] iter_args(%[[VAL_12:.*]] = %[[VAL_9]], %[[VAL_13:.*]] = %[[VAL_5]]) -> (memref<256xbf16, strided<[?], offset: ?>>, index) {
// CHECK:             %[[VAL_14:.*]] = memref.alloc() : memref<256xbf16>
// CHECK:             memref.copy %[[VAL_12]], %[[VAL_14]] : memref<256xbf16, strided<[?], offset: ?>> to memref<256xbf16>
// CHECK:             %[[VAL_15:.*]] = bufferization.to_tensor %[[VAL_14]] restrict writable : memref<256xbf16>
// CHECK:             memref.tensor_store %[[VAL_15]], %[[VAL_12]] : memref<256xbf16, strided<[?], offset: ?>>
// CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_13]], %[[VAL_8]] : index
// CHECK:             %[[VAL_17:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_16]]], sizes: [256], strides: {{\[}}%[[VAL_4]]] : memref<*xbf16> to memref<256xbf16, strided<[?], offset: ?>>
// CHECK:             scf.yield %[[VAL_17]], %[[VAL_16]] : memref<256xbf16, strided<[?], offset: ?>>, index
// CHECK:           }
// CHECK:           return
// CHECK:         }
