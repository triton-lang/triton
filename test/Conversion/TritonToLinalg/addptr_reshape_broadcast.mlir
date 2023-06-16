// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
// TODO: expand this example to 3D
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>
  )
  {
  %0 = tt.make_range {end = 1024 : i32, start = 512 : i32}:tensor<256xi32>
  // offset = [512] size = 256, stride = 2
  %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<256xi32>) -> tensor<256x1xi32>
  // offset = [512,0], size = [256,1], stride = [2,0]
  %2 = tt.broadcast %1 : (tensor<256x1xi32>) -> tensor<256x128xi32>
  // offset = [512,0], size = [256,128], stride = [2,0]
  %5 = tt.make_range {end = 1408 : i32, start = 1024 : i32}:tensor<128xi32>
  // offset = 1024, size = 128, stride = 3
  %6 = tt.expand_dims %5 {axis = 0 : i32} : (tensor<128xi32>) -> tensor<1x128xi32>
  // offset = [0,1024], size = [1,128], stride = [0,3]
  %7 = tt.broadcast %6 : (tensor<1x128xi32>) -> tensor<256x128xi32>
  // offset = [0,1024], size = [256,128], stride = [0,3]
  %c6 = arith.constant 6 : i32
  %splat6 = tt.splat %c6 : (i32) -> tensor<256x128xi32>
  %scale7 = arith.muli %7, %splat6 : tensor<256x128xi32>
  // offset = [0,6144], size = [256,128], stride = [0,18]
  %14 = arith.addi %2, %scale7 : tensor<256x128xi32>
  // offset = [512,6144], size = [256,128], stride = [2,18]
  %17 = tt.splat %arg1 : (!tt.ptr<bf16>) -> tensor<256x128x!tt.ptr<bf16>>
  %18 = tt.addptr %17, %14 : tensor<256x128x!tt.ptr<bf16>>, tensor<256x128xi32>
  %19 = tt.load %18 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x128xbf16>
  tt.store %18, %19 : tensor<256x128xbf16>
  tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: memref<*xbf16>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32) {
// CHECK:           %[[VAL_7:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}6656], sizes: [256, 128], strides: [2, 18] : memref<*xbf16> to memref<256x128xbf16, strided<[2, 18], offset: 6656>>
// CHECK:           %[[VAL_8:.*]] = memref.alloc() : memref<256x128xbf16>
// CHECK:           memref.copy %[[VAL_7]], %[[VAL_8]] : memref<256x128xbf16, strided<[2, 18], offset: 6656>> to memref<256x128xbf16>
// CHECK:           %[[VAL_9:.*]] = bufferization.to_tensor %[[VAL_8]] restrict writable : memref<256x128xbf16>
// CHECK:           memref.tensor_store %[[VAL_9]], %[[VAL_7]] : memref<256x128xbf16, strided<[2, 18], offset: 6656>>
// CHECK:           return
// CHECK:         }
