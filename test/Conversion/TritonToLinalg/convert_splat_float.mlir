// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
    tt.func @kernel(%fin : f32,
                    %bin : bf16,
                    %save0 : tensor<1024x!tt.ptr<f32>>,
                    %save1 : tensor<128x256x!tt.ptr<bf16>>) -> () {
        %0 = tt.splat %fin : (f32) -> (tensor<1024xf32>)
        %1 = tt.splat %bin : (bf16) -> (tensor<128x256xbf16>)
        tt.store %save0, %0 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
        tt.store %save1, %1 {cache = 1 : i32, evict = 1 : i32} : tensor<128x256xbf16>
        tt.return
    }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: bf16, %[[VAL_2:.*]]: memref<1024xf32>, %[[VAL_3:.*]]: memref<128x256xbf16>, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK:           %[[VAL_8:.*]] = linalg.fill ins(%[[VAL_0]] : f32) outs(%[[VAL_7]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_9:.*]] = tensor.empty() : tensor<128x256xbf16>
// CHECK:           %[[VAL_10:.*]] = linalg.fill ins(%[[VAL_1]] : bf16) outs(%[[VAL_9]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
// CHECK:           memref.tensor_store %[[VAL_8]], %[[VAL_2]] : memref<1024xf32>
// CHECK:           memref.tensor_store %[[VAL_10]], %[[VAL_3]] : memref<128x256xbf16>
// CHECK:           return
// CHECK:         }
