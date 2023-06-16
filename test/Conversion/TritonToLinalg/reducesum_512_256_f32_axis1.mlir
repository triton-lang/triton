// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
    tt.func @kernel(%afloat : !tt.ptr<f32>,
        %res : !tt.ptr<f32>
    ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %c256 = arith.constant 256 : i32
    %ct256 = tt.splat %c256 : (i32) -> tensor<512xi32>
    %ws = arith.muli %ct256, %0 : tensor<512xi32>
    %1 = tt.expand_dims %ws {axis = 1 : i32} : (tensor<512xi32>) -> tensor<512x1xi32>
    %moff = tt.broadcast %1 : (tensor<512x1xi32>) -> tensor<512x256xi32>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : (tensor<256xi32>) -> tensor<1x256xi32>
    %koff = tt.broadcast %4 : (tensor<1x256xi32>) -> tensor<512x256xi32>
    %mkoff = arith.addi %moff, %koff : tensor<512x256xi32>
    // afloat pointer
    %8 = tt.splat %afloat : (!tt.ptr<f32>) -> tensor<512x256x!tt.ptr<f32>>
    %9 = tt.addptr %8, %mkoff : tensor<512x256x!tt.ptr<f32>>, tensor<512x256xi32>
    // res pointer
    %18 = tt.splat %res : (!tt.ptr<f32>) -> tensor<512x!tt.ptr<f32>>
    %19 = tt.addptr %18, %0 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
    %afm = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<512x256xf32>
    %5 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %21 = arith.addf %arg5, %arg6 : f32
      tt.reduce.return %21 : f32
    }) {axis = 1 : i32} : (tensor<512x256xf32>) -> tensor<512xf32>
    tt.store %19, %5 : tensor<512xf32>
    tt.return
    }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xf32>, %[[VAL_1:.*]]: memref<*xf32>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32) {
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 256 : index
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_7:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [512, 256], strides: {{\[}}%[[VAL_5]], 1] : memref<*xf32> to memref<512x256xf32, strided<[?, 1]>>
// CHECK:           %[[VAL_8:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [512], strides: [1] : memref<*xf32> to memref<512xf32, strided<[1]>>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<512x256xf32>
// CHECK:           memref.copy %[[VAL_7]], %[[VAL_9]] : memref<512x256xf32, strided<[?, 1]>> to memref<512x256xf32>
// CHECK:           %[[VAL_10:.*]] = bufferization.to_tensor %[[VAL_9]] restrict writable : memref<512x256xf32>
// CHECK:           %[[VAL_11:.*]] = tensor.empty() : tensor<256x512xf32>
// CHECK:           %[[VAL_12:.*]] = linalg.transpose ins(%[[VAL_10]] : tensor<512x256xf32>) outs(%[[VAL_11]] : tensor<256x512xf32>) permutation = [1, 0]
// CHECK:           %[[VAL_13:.*]] = tensor.empty() : tensor<512xf32>
// CHECK:           %[[VAL_14:.*]] = linalg.fill ins(%[[VAL_6]] : f32) outs(%[[VAL_13]] : tensor<512xf32>) -> tensor<512xf32>
// CHECK:           %[[VAL_15:.*]] = linalg.reduce ins(%[[VAL_12]] : tensor<256x512xf32>) outs(%[[VAL_14]] : tensor<512xf32>) dimensions = [0]
// CHECK:             (%[[VAL_16:.*]]: f32, %[[VAL_17:.*]]: f32) {
// CHECK:               %[[VAL_18:.*]] = arith.addf %[[VAL_16]], %[[VAL_17]] : f32
// CHECK:               linalg.yield %[[VAL_18]] : f32
// CHECK:             }
// CHECK:           memref.tensor_store %[[VAL_15]], %[[VAL_8]] : memref<512xf32, strided<[1]>>
// CHECK:           return
// CHECK:         }
