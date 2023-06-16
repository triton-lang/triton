// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %f32ptr : !tt.ptr<f32>,
    %intptr : !tt.ptr<i32>,
    %f16ptr : !tt.ptr<f16>,
    %save0 : tensor<128x128x!tt.ptr<bf16>>,
    %save1 : tensor<128x128x!tt.ptr<f32>>,
    %save2 : tensor<128x128x!tt.ptr<f32>>,
    %save3 : tensor<128x128x!tt.ptr<f32>>,
    %save4 : tensor<128x128x!tt.ptr<f32>>
  ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<128xi32>) -> tensor<128x1xi32>
    %moff = tt.broadcast %1 : (tensor<128x1xi32>) -> tensor<128x128xi32>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : (tensor<128xi32>) -> tensor<1x128xi32>
    %koff = tt.broadcast %4 : (tensor<1x128xi32>) -> tensor<128x128xi32>
    %mkoff = arith.addi %moff, %koff : tensor<128x128xi32>
    // f32ptr pointer
    %8 = tt.splat %f32ptr : (!tt.ptr<f32>) -> tensor<128x128x!tt.ptr<f32>>
    %9 = tt.addptr %8, %mkoff : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
    // intptr pointer
    %18 = tt.splat %intptr : (!tt.ptr<i32>) -> tensor<128x128x!tt.ptr<i32>>
    %19 = tt.addptr %18, %mkoff : tensor<128x128x!tt.ptr<i32>>, tensor<128x128xi32>
    // f16ptr pointer
    %28 = tt.splat %f16ptr : (!tt.ptr<f16>) -> tensor<128x128x!tt.ptr<f16>>
    %29 = tt.addptr %28, %mkoff : tensor<128x128x!tt.ptr<f16>>, tensor<128x128xi32>
    %afm = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf32>
    %aim = tt.load %19 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xi32>
    %bfm = tt.load %29 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf16>
    %5 = arith.truncf %afm : tensor<128x128xf32> to tensor<128x128xbf16>
    %6 = math.exp %afm : tensor<128x128xf32>
    %7 = arith.sitofp %aim : tensor<128x128xi32> to tensor<128x128xf32>
    %10 = arith.extf %bfm : tensor<128x128xf16> to tensor<128x128xf32>
    %11 = math.sqrt %afm : tensor<128x128xf32>
    tt.store %save0, %5 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xbf16>
    tt.store %save1, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf32>
    tt.store %save2, %7 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf32>
    tt.store %save3, %10 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf32>
    tt.store %save4, %11 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf32>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xf32>, %[[VAL_1:.*]]: memref<*xi32>, %[[VAL_2:.*]]: memref<*xf16>, %[[VAL_3:.*]]: memref<128x128xbf16>, %[[VAL_4:.*]]: memref<128x128xf32>, %[[VAL_5:.*]]: memref<128x128xf32>, %[[VAL_6:.*]]: memref<128x128xf32>, %[[VAL_7:.*]]: memref<128x128xf32>, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32) {
// CHECK:           %[[VAL_11:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1]>>
// CHECK:           %[[VAL_12:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [128, 128], strides: [1, 1] : memref<*xi32> to memref<128x128xi32, strided<[1, 1]>>
// CHECK:           %[[VAL_13:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: [0], sizes: [128, 128], strides: [1, 1] : memref<*xf16> to memref<128x128xf16, strided<[1, 1]>>
// CHECK:           %[[VAL_14:.*]] = memref.alloc() : memref<128x128xf32>
// CHECK:           memref.copy %[[VAL_11]], %[[VAL_14]] : memref<128x128xf32, strided<[1, 1]>> to memref<128x128xf32>
// CHECK:           %[[VAL_15:.*]] = bufferization.to_tensor %[[VAL_14]] restrict writable : memref<128x128xf32>
// CHECK:           %[[VAL_16:.*]] = memref.alloc() : memref<128x128xi32>
// CHECK:           memref.copy %[[VAL_12]], %[[VAL_16]] : memref<128x128xi32, strided<[1, 1]>> to memref<128x128xi32>
// CHECK:           %[[VAL_17:.*]] = bufferization.to_tensor %[[VAL_16]] restrict writable : memref<128x128xi32>
// CHECK:           %[[VAL_18:.*]] = memref.alloc() : memref<128x128xf16>
// CHECK:           memref.copy %[[VAL_13]], %[[VAL_18]] : memref<128x128xf16, strided<[1, 1]>> to memref<128x128xf16>
// CHECK:           %[[VAL_19:.*]] = bufferization.to_tensor %[[VAL_18]] restrict writable : memref<128x128xf16>
// CHECK:           %[[VAL_20:.*]] = tensor.empty() : tensor<128x128xbf16>
// CHECK:           %[[VAL_21:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_15]] : tensor<128x128xf32>) outs(%[[VAL_20]] : tensor<128x128xbf16>) {
// CHECK:           ^bb0(%[[VAL_22:.*]]: f32, %[[VAL_23:.*]]: bf16):
// CHECK:             %[[VAL_24:.*]] = arith.truncf %[[VAL_22]] : f32 to bf16
// CHECK:             linalg.yield %[[VAL_24]] : bf16
// CHECK:           } -> tensor<128x128xbf16>
// CHECK:           %[[VAL_25:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_15]] : tensor<128x128xf32>) outs(%[[VAL_15]] : tensor<128x128xf32>) {
// CHECK:           ^bb0(%[[VAL_26:.*]]: f32, %[[VAL_27:.*]]: f32):
// CHECK:             %[[VAL_28:.*]] = math.exp %[[VAL_26]] : f32
// CHECK:             linalg.yield %[[VAL_28]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           %[[VAL_29:.*]] = tensor.empty() : tensor<128x128xf32>
// CHECK:           %[[VAL_30:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_17]] : tensor<128x128xi32>) outs(%[[VAL_29]] : tensor<128x128xf32>) {
// CHECK:           ^bb0(%[[VAL_31:.*]]: i32, %[[VAL_32:.*]]: f32):
// CHECK:             %[[VAL_33:.*]] = arith.sitofp %[[VAL_31]] : i32 to f32
// CHECK:             linalg.yield %[[VAL_33]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           %[[VAL_34:.*]] = tensor.empty() : tensor<128x128xf32>
// CHECK:           %[[VAL_35:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_19]] : tensor<128x128xf16>) outs(%[[VAL_34]] : tensor<128x128xf32>) {
// CHECK:           ^bb0(%[[VAL_36:.*]]: f16, %[[VAL_37:.*]]: f32):
// CHECK:             %[[VAL_38:.*]] = arith.extf %[[VAL_36]] : f16 to f32
// CHECK:             linalg.yield %[[VAL_38]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           %[[VAL_39:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_15]] : tensor<128x128xf32>) outs(%[[VAL_15]] : tensor<128x128xf32>) {
// CHECK:           ^bb0(%[[VAL_40:.*]]: f32, %[[VAL_41:.*]]: f32):
// CHECK:             %[[VAL_42:.*]] = math.sqrt %[[VAL_40]] : f32
// CHECK:             linalg.yield %[[VAL_42]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           memref.tensor_store %[[VAL_43:.*]], %[[VAL_3]] : memref<128x128xbf16>
// CHECK:           memref.tensor_store %[[VAL_44:.*]], %[[VAL_4]] : memref<128x128xf32>
// CHECK:           memref.tensor_store %[[VAL_45:.*]], %[[VAL_5]] : memref<128x128xf32>
// CHECK:           memref.tensor_store %[[VAL_46:.*]], %[[VAL_6]] : memref<128x128xf32>
// CHECK:           memref.tensor_store %[[VAL_47:.*]], %[[VAL_7]] : memref<128x128xf32>
// CHECK:           return
// CHECK:         }
