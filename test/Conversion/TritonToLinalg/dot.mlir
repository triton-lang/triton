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
    %11 = tt.expand_dims %10 {axis = 1 : i32} : (tensor<256xi32>) -> tensor<256x1xi32>
    %12 = tt.broadcast %11 : (tensor<256x1xi32>) -> tensor<256x64xi32>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %c256 = arith.constant 256 : i32
    %14 = tt.splat %c256 : (i32) -> tensor<64xi32>
    %15 = arith.muli %13, %14 : tensor<64xi32>
    %16 = tt.expand_dims %15 {axis = 0 : i32} : (tensor<64xi32>) -> tensor<1x64xi32>
    %17 = tt.broadcast %16 : (tensor<1x64xi32>) -> tensor<256x64xi32>
    %18 = arith.addi %12, %17 : tensor<256x64xi32>
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
    %40 = tt.splat %arg1 : (!tt.ptr<bf16>) -> tensor<256x64x!tt.ptr<bf16>>
    %41 = tt.addptr %40, %18 : tensor<256x64x!tt.ptr<bf16>>, tensor<256x64xi32>
    %42 = tt.load %41 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x64xbf16>
    %43 = tt.trans %42 : (tensor<256x64xbf16>) -> tensor<64x256xbf16>
    %50 = tt.splat %arg2 : (!tt.ptr<bf16>) -> tensor<128x256x!tt.ptr<bf16>>
    %51 = tt.addptr %50, %26 : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %52 = tt.load %51 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<128x256xbf16>
    %60 = tt.dot %32, %43, %52 {allowTF32 = false} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xbf16>
    tt.store %51, %60 : tensor<128x256xbf16>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: memref<*xbf16>, %[[VAL_2:.*]]: memref<*xbf16>, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32) {
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 256 : index
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_8:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [128, 64], strides: {{\[}}%[[VAL_7]], 1] : memref<*xbf16> to memref<128x64xbf16, strided<[?, 1]>>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<128x64xbf16>
// CHECK:           memref.copy %[[VAL_8]], %[[VAL_9]] : memref<128x64xbf16, strided<[?, 1]>> to memref<128x64xbf16>
// CHECK:           %[[VAL_10:.*]] = bufferization.to_tensor %[[VAL_9]] restrict writable : memref<128x64xbf16>
// CHECK:           %[[VAL_11:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [256, 64], strides: [1, %[[VAL_6]]] : memref<*xbf16> to memref<256x64xbf16, strided<[1, ?]>>
// CHECK:           %[[VAL_12:.*]] = memref.alloc() : memref<256x64xbf16>
// CHECK:           memref.copy %[[VAL_11]], %[[VAL_12]] : memref<256x64xbf16, strided<[1, ?]>> to memref<256x64xbf16>
// CHECK:           %[[VAL_13:.*]] = bufferization.to_tensor %[[VAL_12]] restrict writable : memref<256x64xbf16>
// CHECK:           %[[VAL_14:.*]] = tensor.empty() : tensor<64x256xbf16>
// CHECK:           %[[VAL_15:.*]] = linalg.transpose ins(%[[VAL_13]] : tensor<256x64xbf16>) outs(%[[VAL_14]] : tensor<64x256xbf16>) permutation = [1, 0]
// CHECK:           %[[VAL_16:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: [0], sizes: [128, 256], strides: {{\[}}%[[VAL_6]], 1] : memref<*xbf16> to memref<128x256xbf16, strided<[?, 1]>>
// CHECK:           %[[VAL_17:.*]] = memref.alloc() : memref<128x256xbf16>
// CHECK:           memref.copy %[[VAL_16]], %[[VAL_17]] : memref<128x256xbf16, strided<[?, 1]>> to memref<128x256xbf16>
// CHECK:           %[[VAL_18:.*]] = bufferization.to_tensor %[[VAL_17]] restrict writable : memref<128x256xbf16>
// CHECK:           %[[VAL_19:.*]] = tensor.empty() : tensor<128x256xbf16>
// CHECK:           %[[VAL_20:.*]] = linalg.matmul ins(%[[VAL_10]], %[[VAL_15]] : tensor<128x64xbf16>, tensor<64x256xbf16>) outs(%[[VAL_19]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
// CHECK:           %[[VAL_21:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_20]], %[[VAL_18]] : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%[[VAL_20]] : tensor<128x256xbf16>) {
// CHECK:           ^bb0(%[[VAL_22:.*]]: bf16, %[[VAL_23:.*]]: bf16, %[[VAL_24:.*]]: bf16):
// CHECK:             %[[VAL_25:.*]] = arith.addf %[[VAL_22]], %[[VAL_23]] : bf16
// CHECK:             linalg.yield %[[VAL_25]] : bf16
// CHECK:           } -> tensor<128x256xbf16>
// CHECK:           memref.tensor_store %[[VAL_26:.*]], %[[VAL_16]] : memref<128x256xbf16, strided<[?, 1]>>
// CHECK:           return
// CHECK:         }
