// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : !tt.ptr<bf16>
  )
  {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %c64 = arith.constant 128 : i32
    %1 = tt.splat %c64 : (i32) -> tensor<128xi32>
    %2 = arith.muli %0, %1 : tensor<128xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<128xi32>) -> tensor<128x1xi32>
    %4 = tt.broadcast %3 : (tensor<128x1xi32>) -> tensor<128x64xi32>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : (tensor<64xi32>) -> tensor<1x64xi32>
    %7 = tt.broadcast %6 : (tensor<1x64xi32>) -> tensor<128x64xi32>
    %8 = arith.addi %4, %7 : tensor<128x64xi32>
    %10 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : (tensor<256xi32>) -> tensor<1x256xi32>
    %12 = tt.broadcast %11 : (tensor<1x256xi32>) -> tensor<64x256xi32>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %c256 = arith.constant 256 : i32
    %14 = tt.splat %c256 : (i32) -> tensor<64xi32>
    %15 = arith.muli %13, %14 : tensor<64xi32>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : (tensor<64xi32>) -> tensor<64x1xi32>
    %17 = tt.broadcast %16 : (tensor<64x1xi32>) -> tensor<64x256xi32>
    %18 = arith.addi %12, %17 : tensor<64x256xi32>
    %20 = tt.splat %c256 : (i32) -> tensor<128xi32>
    %21 = arith.muli %0, %20 : tensor<128xi32>
    %22 = tt.expand_dims %21 {axis = 1 : i32} : (tensor<128xi32>) -> tensor<128x1xi32>
    %23 = tt.broadcast %22 : (tensor<128x1xi32>) -> tensor<128x256xi32>
    %24 = tt.expand_dims %10 {axis = 0 : i32} : (tensor<256xi32>) -> tensor<1x256xi32>
    %25 = tt.broadcast %24 {axis = 0 : i32} : (tensor<1x256xi32>) -> tensor<128x256xi32>
    %26 = arith.addi %23, %25 : tensor<128x256xi32>
    %30 = tt.splat %arg0 : (!tt.ptr<bf16>) -> tensor<128x64x!tt.ptr<bf16>>
    %31 = tt.addptr %30, %8 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
    %32 = tt.load %31 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<128x64xbf16>
    %40 = tt.splat %arg1 : (!tt.ptr<bf16>) -> tensor<64x256x!tt.ptr<bf16>>
    %41 = tt.addptr %40, %18 : tensor<64x256x!tt.ptr<bf16>>, tensor<64x256xi32>
    %42 = tt.load %41 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<64x256xbf16>
    %50 = tt.splat %arg2 : (!tt.ptr<bf16>) -> tensor<128x256x!tt.ptr<bf16>>
    %51 = tt.addptr %50, %26 : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %cf0 = arith.constant 0.0 : bf16
    %71 = tt.splat %cf0 : (bf16) -> (tensor<128x256xbf16>)
    %60 = tt.dot %32, %42, %71 {allowTF32 = false} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xbf16>
    tt.store %51, %60 : tensor<128x256xbf16>
    tt.store %51, %71 : tensor<128x256xbf16>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: memref<*xbf16>, %[[VAL_2:.*]]: memref<*xbf16>, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32) {
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 256 : index
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 128 : index
// CHECK-DAG:           %[[VAL_8:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK:           %[[VAL_16:.*]] = tensor.empty() : tensor<128x256xbf16>
// CHECK:           %[[VAL_17:.*]] = linalg.fill ins(%[[VAL_8]] : bf16) outs(%[[VAL_16]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
// CHECK:           %[[VAL_9:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [128, 64], strides: {{\[}}%[[VAL_7]], 1] : memref<*xbf16> to memref<128x64xbf16, strided<[?, 1]>>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() : memref<128x64xbf16>
// CHECK:           memref.copy %[[VAL_9]], %[[VAL_10]] : memref<128x64xbf16, strided<[?, 1]>> to memref<128x64xbf16>
// CHECK:           %[[VAL_11:.*]] = bufferization.to_tensor %[[VAL_10]] restrict writable : memref<128x64xbf16>
// CHECK:           %[[VAL_12:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [64, 256], strides: {{\[}}%[[VAL_6]], 1] : memref<*xbf16> to memref<64x256xbf16, strided<[?, 1]>>
// CHECK:           %[[VAL_13:.*]] = memref.alloc() : memref<64x256xbf16>
// CHECK:           memref.copy %[[VAL_12]], %[[VAL_13]] : memref<64x256xbf16, strided<[?, 1]>> to memref<64x256xbf16>
// CHECK:           %[[VAL_14:.*]] = bufferization.to_tensor %[[VAL_13]] restrict writable : memref<64x256xbf16>
// CHECK:           %[[VAL_15:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: [0], sizes: [128, 256], strides: {{\[}}%[[VAL_6]], 1] : memref<*xbf16> to memref<128x256xbf16, strided<[?, 1]>>
// CHECK:           %[[VAL_18:.*]] = tensor.empty() : tensor<128x256xbf16>
// CHECK:           %[[VAL_19:.*]] = linalg.matmul ins(%[[VAL_11]], %[[VAL_14]] : tensor<128x64xbf16>, tensor<64x256xbf16>) outs(%[[VAL_18]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
// CHECK:           memref.tensor_store %[[VAL_19]], %[[VAL_15]] : memref<128x256xbf16, strided<[?, 1]>>
// CHECK:           memref.tensor_store %[[VAL_17]], %[[VAL_15]] : memref<128x256xbf16, strided<[?, 1]>>
// CHECK:           return
// CHECK:         }
