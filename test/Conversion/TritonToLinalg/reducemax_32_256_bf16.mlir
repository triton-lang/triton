// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
    tt.func @kernel(%afloat : !tt.ptr<bf16>,
        %res : tensor<256x16x!tt.ptr<bf16>>
    ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %c256 = arith.constant 256 : i32
    %ct256 = tt.splat %c256 : (i32) -> tensor<32xi32>
    %ws = arith.muli %ct256, %0 : tensor<32xi32>
    %1 = tt.expand_dims %ws {axis = 1 : i32} : (tensor<32xi32>) -> tensor<32x1xi32>
    %m2 = tt.broadcast %1 : (tensor<32x1xi32>) -> tensor<32x256xi32>
    %100 = tt.expand_dims %m2 {axis = 2 : i32} : (tensor<32x256xi32>) -> tensor<32x256x1xi32>
    %moff = tt.broadcast %100 : (tensor<32x256x1xi32>) -> tensor<32x256x16xi32>
    %33 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %34 = tt.expand_dims %33 {axis = 0 : i32} : (tensor<256xi32>) -> tensor<1x256xi32>
    %k2 = tt.broadcast %34 : (tensor<1x256xi32>) -> tensor<32x256xi32>
    %200 = tt.expand_dims %k2 {axis = 2 : i32} : (tensor<32x256xi32>) -> tensor<32x256x1xi32>
    %koff = tt.broadcast %200 : (tensor<32x256x1xi32>) -> tensor<32x256x16xi32>
    %23 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %24 = tt.expand_dims %23 {axis = 0 : i32} : (tensor<16xi32>) -> tensor<1x16xi32>
    %n2 = tt.broadcast %24 : (tensor<1x16xi32>) -> tensor<256x16xi32>
    %300 = tt.expand_dims %n2 {axis = 0 : i32} : (tensor<256x16xi32>) -> tensor<1x256x16xi32>
    %noff = tt.broadcast %300 : (tensor<1x256x16xi32>) -> tensor<32x256x16xi32>
    %mkoff = arith.addi %moff, %koff : tensor<32x256x16xi32>
    %mknoff = arith.addi %mkoff, %noff : tensor<32x256x16xi32>
    // afloat pointer
    %8 = tt.splat %afloat : (!tt.ptr<bf16>) -> tensor<32x256x16x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %mknoff : tensor<32x256x16x!tt.ptr<bf16>>, tensor<32x256x16xi32>
    %afm = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x256x16xbf16>
    %6 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.cmpf ogt, %arg5, %arg6 : bf16
      %22 = arith.select %21, %arg5, %arg6 : bf16
      tt.reduce.return %22 : bf16
    }) {axis = 0 : i32} : (tensor<32x256x16xbf16>) -> tensor<256x16xbf16>
    tt.store %res, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<256x16xbf16>
    tt.return
    }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: memref<256x16xbf16>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32) {
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 256 : index
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 0xFF80 : bf16
// CHECK:           %[[VAL_7:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [32, 256, 16], strides: {{\[}}%[[VAL_5]], 1, 1] : memref<*xbf16> to memref<32x256x16xbf16, strided<[?, 1, 1]>>
// CHECK:           %[[VAL_8:.*]] = memref.alloc() : memref<32x256x16xbf16>
// CHECK:           memref.copy %[[VAL_7]], %[[VAL_8]] : memref<32x256x16xbf16, strided<[?, 1, 1]>> to memref<32x256x16xbf16>
// CHECK:           %[[VAL_9:.*]] = bufferization.to_tensor %[[VAL_8]] restrict writable : memref<32x256x16xbf16>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<256x16xbf16>
// CHECK:           %[[VAL_11:.*]] = linalg.fill ins(%[[VAL_6]] : bf16) outs(%[[VAL_10]] : tensor<256x16xbf16>) -> tensor<256x16xbf16>
// CHECK:           %[[VAL_12:.*]] = linalg.reduce ins(%[[VAL_9]] : tensor<32x256x16xbf16>) outs(%[[VAL_11]] : tensor<256x16xbf16>) dimensions = [0]
// CHECK:             (%[[VAL_13:.*]]: bf16, %[[VAL_14:.*]]: bf16) {
// CHECK:               %[[VAL_15:.*]] = arith.maxf %[[VAL_13]], %[[VAL_14]] : bf16
// CHECK:               linalg.yield %[[VAL_15]] : bf16
// CHECK:             }
// CHECK:           memref.tensor_store %[[VAL_12]], %[[VAL_1]] : memref<256x16xbf16>
// CHECK:           return
// CHECK:         }
