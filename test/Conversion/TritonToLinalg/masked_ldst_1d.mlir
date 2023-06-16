// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32
  )
  {
    %0 = tt.splat %arg0 : (!tt.ptr<bf16>) -> tensor<128x!tt.ptr<bf16>>
    %1 = tt.splat %arg1 : (!tt.ptr<bf16>) -> tensor<128x!tt.ptr<bf16>>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %ldptr = tt.addptr %0, %2 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %stptr = tt.addptr %1, %2 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %nans = arith.constant dense<0xFF80> : tensor<128xbf16>
    %5 = tt.splat %arg2 : (i32) -> tensor<128xi32>
    %mask = arith.cmpi slt, %2, %5 : tensor<128xi32>
    %buff = tt.load %ldptr, %mask, %nans {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128xbf16>
    tt.store %stptr, %buff, %mask : tensor<128xbf16>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: memref<*xbf16>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32) {
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 0xFF80 : bf16
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_8:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [128], strides: [1] : memref<*xbf16> to memref<128xbf16, strided<[1]>>
// CHECK:           %[[VAL_9:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [128], strides: [1] : memref<*xbf16> to memref<128xbf16, strided<[1]>>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() : memref<128xbf16>
// CHECK:           %[[VAL_11:.*]] = arith.index_cast %[[VAL_2]] : i32 to index
// CHECK:           %[[VAL_12:.*]] = arith.minsi %[[VAL_11]], %[[VAL_7]] : index
// CHECK:           %[[VAL_13:.*]] = memref.subview %[[VAL_8]][0] {{\[}}%[[VAL_12]]] [1] : memref<128xbf16, strided<[1]>> to memref<?xbf16, strided<[1]>>
// CHECK:           %[[VAL_14:.*]] = memref.subview %[[VAL_10]][0] {{\[}}%[[VAL_12]]] [1] : memref<128xbf16> to memref<?xbf16, strided<[1]>>
// CHECK:           %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_7]] : index
// CHECK:           scf.if %[[VAL_15]] {
// CHECK:             linalg.fill ins(%[[VAL_6]] : bf16) outs(%[[VAL_10]] : memref<128xbf16>)
// CHECK:           }
// CHECK:           memref.copy %[[VAL_13]], %[[VAL_14]] : memref<?xbf16, strided<[1]>> to memref<?xbf16, strided<[1]>>
// CHECK:           %[[VAL_16:.*]] = bufferization.to_tensor %[[VAL_10]] restrict writable : memref<128xbf16>
// CHECK:           %[[VAL_17:.*]] = arith.index_cast %[[VAL_2]] : i32 to index
// CHECK:           %[[VAL_18:.*]] = arith.minsi %[[VAL_17]], %[[VAL_7]] : index
// CHECK:           %[[VAL_19:.*]] = tensor.extract_slice %[[VAL_16]][0] {{\[}}%[[VAL_18]]] [1] : tensor<128xbf16> to tensor<?xbf16>
// CHECK:           %[[VAL_20:.*]] = memref.subview %[[VAL_9]][0] {{\[}}%[[VAL_18]]] [1] : memref<128xbf16, strided<[1]>> to memref<?xbf16, strided<[1]>>
// CHECK:           memref.tensor_store %[[VAL_19]], %[[VAL_20]] : memref<?xbf16, strided<[1]>>
// CHECK:           return
// CHECK:         }
