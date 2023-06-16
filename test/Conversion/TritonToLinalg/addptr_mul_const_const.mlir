// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : i32
  )
  {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32}:tensor<1024xi32>
    %2 = tt.splat %0 : (i32) -> tensor<1024xi32>
    %3 = arith.addi %2, %1 : tensor<1024xi32>
    //%3: splat(%0) + range(0, 1024)
    //%3: offset = %0, size = 1024, stride = 1
    // vector and scalar are both constant
    %4 = tt.make_range {end = 4096 : i32, start = 2048 : i32}:tensor<1024xi32>
    %c10 = arith.constant 10 : i32
    %5 = tt.splat %c10 : (i32) -> tensor<1024xi32>
    %6 = arith.muli %5, %4 : tensor<1024xi32>
    //%6: splat(%c10)*range(2048, 4096);
    //%6: offset = %c10*2048, size = 1024, stride = %c10*2
    %7 = arith.addi %3, %6 : tensor<1024xi32>
    //%7: offset = %c10*2048 + %0, size = 1024, stride = %c10*2+1
    %8 = tt.splat %arg0 : (!tt.ptr<bf16>) -> tensor<1024x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    //source=%arg0 offset = %c10*2048 + pid0, size = 1024, stride = %c10*2+1
    %10 = tt.splat %arg1 : (!tt.ptr<bf16>) -> tensor<1024x!tt.ptr<bf16>>
    %11 = tt.addptr %10, %3 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    //source=%arg1, offset = pid0, size = 1024, stride = 1
    %16 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xbf16>
    tt.store %11, %16 : tensor<1024xbf16>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: memref<*xbf16>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32) {
// CHECK:           %[[VAL_7:.*]] = arith.constant 20480 : index
// CHECK:           %[[VAL_8:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_7]] : index
// CHECK:           %[[VAL_10:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_9]]], sizes: [1024], strides: {{\[}}21] : memref<*xbf16> to memref<1024xbf16, strided<[21], offset: ?>>
// CHECK:           %[[VAL_11:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_12:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_11]]], sizes: [1024], strides: [1] : memref<*xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
// CHECK:           %[[VAL_13:.*]] = memref.alloc() : memref<1024xbf16>
// CHECK:           memref.copy %[[VAL_10]], %[[VAL_13]] : memref<1024xbf16, strided<[21], offset: ?>> to memref<1024xbf16>
// CHECK:           %[[VAL_14:.*]] = bufferization.to_tensor %[[VAL_13]] restrict writable : memref<1024xbf16>
// CHECK:           memref.tensor_store %[[VAL_14]], %[[VAL_12]] : memref<1024xbf16, strided<[1], offset: ?>>
// CHECK:           return
// CHECK:         }
