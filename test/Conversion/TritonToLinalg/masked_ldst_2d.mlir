// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32,
  %arg3 : i32
  )
  {
    // Mimic a scenario where the raw pointer points to a buffer with dimension (1024, 1024)
    // in row-major, but the actual tensor size is (arg2, arg3).
    // We are trying to load a 128x256 sub-buffer starting at (2, 3).
    // The resulting memref:
    //  offset = 3074
    //  size[1] = 128
    //  size[0] = 256
    //  stride[0] = 1024
    //  stride[1] = 1
    %0 = tt.splat %arg0 : (!tt.ptr<bf16>) -> tensor<128x256x!tt.ptr<bf16>>
    %1 = tt.splat %arg1 : (!tt.ptr<bf16>) -> tensor<128x256x!tt.ptr<bf16>>
    // horizontal index
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %c2 = arith.constant 2 : i32
    %c2tensor = tt.splat %c2 : (i32) -> tensor<128xi32>
    %offset2 = arith.addi %2, %c2tensor : tensor<128xi32>
    %3 = tt.expand_dims %offset2 {axis = 1 : i32} : (tensor<128xi32>) -> tensor<128x1xi32>
    %4 = tt.broadcast %3 : (tensor<128x1xi32>) -> tensor<128x256xi32>
    // vertical index
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %c3 = arith.constant 3 : i32
    %c3tensor = tt.splat %c3 : (i32) -> tensor<256xi32>
    %offset5 = arith.addi %5, %c3tensor : tensor<256xi32>
    %c1024 = arith.constant 1024 : i32
    %c1024tensor = tt.splat %c1024 : (i32) -> tensor<256xi32>
    %scale5 = arith.muli %offset5, %c1024tensor : tensor<256xi32>
    %6 = tt.expand_dims %scale5 {axis = 0 : i32} : (tensor<256xi32>) -> tensor<1x256xi32>
    %7 = tt.broadcast %6 : (tensor<1x256xi32>) -> tensor<128x256xi32>
    // combined index
    %index = arith.addi %4, %7 : tensor<128x256xi32>
    %ldptr = tt.addptr %0, %index : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %stptr = tt.addptr %1, %index : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    // other value for masked load
    %cnan = arith.constant 0xFF80 : bf16
    %nans = tt.splat %cnan : (bf16) -> tensor<128x256xbf16>
    // horizontal mask
    %8 = tt.splat %arg2 : (i32) -> tensor<128xi32>
    %9 = arith.cmpi slt, %offset2, %8 : tensor<128xi32>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : (tensor<128xi1>) -> tensor<128x1xi1>
    %11 = tt.broadcast %10 : (tensor<128x1xi1>) -> tensor<128x256xi1>
    // vertical mask
    %12 = tt.splat %arg3 : (i32) -> tensor<256xi32>
    %13 = arith.cmpi slt, %offset5, %12 : tensor<256xi32>
    %14 = tt.expand_dims %13 {axis = 0 : i32} : (tensor<256xi1>) -> tensor<1x256xi1>
    %15 = tt.broadcast %14 : (tensor<1x256xi1>) -> tensor<128x256xi1>
    // combined mask
    %mask = arith.andi %11, %15 : tensor<128x256xi1>
    // dim0 = min(%arg2, 128), dim1 = min(%arg3, 256)
    // TODO: need reinterpret cast
    %buff = tt.load %ldptr, %mask, %nans {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x256xbf16>
    tt.store %stptr, %buff, %mask : tensor<128x256xbf16>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: memref<*xbf16>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 3074 : index
// CHECK-DAG:           %[[VAL_8:.*]] = arith.constant 1024 : index
// CHECK-DAG:           %[[VAL_9:.*]] = arith.constant 3 : index
// CHECK-DAG:           %[[VAL_10:.*]] = arith.constant 2 : index
// CHECK-DAG:           %[[VAL_11:.*]] = arith.constant 256 : index
// CHECK-DAG:           %[[VAL_12:.*]] = arith.constant 128 : index
// CHECK-DAG:           %[[VAL_13:.*]] = arith.constant 259 : index
// CHECK-DAG:           %[[VAL_14:.*]] = arith.constant 130 : index
// CHECK-DAG:           %[[VAL_15:.*]] = arith.constant 0xFF80 : bf16
// CHECK:           %[[VAL_16:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_7]]], sizes: [128, 256], strides: [1, %[[VAL_8]]] : memref<*xbf16> to memref<128x256xbf16, strided<[1, ?], offset: ?>>
// CHECK:           %[[VAL_17:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_7]]], sizes: [128, 256], strides: [1, %[[VAL_8]]] : memref<*xbf16> to memref<128x256xbf16, strided<[1, ?], offset: ?>>
// CHECK:           %[[VAL_18:.*]] = memref.alloc() : memref<128x256xbf16>
// CHECK:           %[[VAL_19:.*]] = arith.index_cast %[[VAL_2]] : i32 to index
// CHECK:           %[[VAL_20:.*]] = arith.minsi %[[VAL_19]], %[[VAL_14]] : index
// CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_20]], %[[VAL_10]] : index
// CHECK:           %[[VAL_22:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_23:.*]] = arith.minsi %[[VAL_22]], %[[VAL_13]] : index
// CHECK:           %[[VAL_24:.*]] = arith.subi %[[VAL_23]], %[[VAL_9]] : index
// CHECK:           %[[VAL_25:.*]] = arith.minsi %[[VAL_21]], %[[VAL_12]] : index
// CHECK:           %[[VAL_26:.*]] = arith.minsi %[[VAL_24]], %[[VAL_11]] : index
// CHECK:           %[[VAL_27:.*]] = memref.subview %[[VAL_16]][0, 0] {{\[}}%[[VAL_25]], %[[VAL_26]]] [1, 1] : memref<128x256xbf16, strided<[1, ?], offset: ?>> to memref<?x?xbf16, strided<[1, ?], offset: ?>>
// CHECK:           %[[VAL_28:.*]] = memref.subview %[[VAL_18]][0, 0] {{\[}}%[[VAL_25]], %[[VAL_26]]] [1, 1] : memref<128x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
// CHECK:           %[[VAL_29:.*]] = arith.cmpi slt, %[[VAL_25]], %[[VAL_12]] : index
// CHECK:           %[[VAL_30:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_11]] : index
// CHECK:           %[[VAL_31:.*]] = arith.ori %[[VAL_29]], %[[VAL_30]] : i1
// CHECK:           scf.if %[[VAL_31]] {
// CHECK:             linalg.fill ins(%[[VAL_15]] : bf16) outs(%[[VAL_18]] : memref<128x256xbf16>)
// CHECK:           }
// CHECK:           memref.copy %[[VAL_27]], %[[VAL_28]] : memref<?x?xbf16, strided<[1, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
// CHECK:           %[[VAL_32:.*]] = bufferization.to_tensor %[[VAL_18]] restrict writable : memref<128x256xbf16>
// CHECK:           %[[VAL_33:.*]] = arith.index_cast %[[VAL_2]] : i32 to index
// CHECK:           %[[VAL_34:.*]] = arith.minsi %[[VAL_33]], %[[VAL_14]] : index
// CHECK:           %[[VAL_35:.*]] = arith.subi %[[VAL_34]], %[[VAL_10]] : index
// CHECK:           %[[VAL_36:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_37:.*]] = arith.minsi %[[VAL_36]], %[[VAL_13]] : index
// CHECK:           %[[VAL_38:.*]] = arith.subi %[[VAL_37]], %[[VAL_9]] : index
// CHECK:           %[[VAL_39:.*]] = arith.minsi %[[VAL_35]], %[[VAL_12]] : index
// CHECK:           %[[VAL_40:.*]] = arith.minsi %[[VAL_38]], %[[VAL_11]] : index
// CHECK:           %[[VAL_41:.*]] = tensor.extract_slice %[[VAL_32]][0, 0] {{\[}}%[[VAL_39]], %[[VAL_40]]] [1, 1] : tensor<128x256xbf16> to tensor<?x?xbf16>
// CHECK:           %[[VAL_42:.*]] = memref.subview %[[VAL_17]][0, 0] {{\[}}%[[VAL_39]], %[[VAL_40]]] [1, 1] : memref<128x256xbf16, strided<[1, ?], offset: ?>> to memref<?x?xbf16, strided<[1, ?], offset: ?>>
// CHECK:           memref.tensor_store %[[VAL_41]], %[[VAL_42]] : memref<?x?xbf16, strided<[1, ?], offset: ?>>
// CHECK:           return
// CHECK:         }
