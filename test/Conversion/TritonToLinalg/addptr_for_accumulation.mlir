// RUN: triton-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : !tt.ptr<bf16>,
    %arg3 : i32,
    %arg4 : i32
  )
  {
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
    // offset = 0, size = 4, stride = 1
    %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<4xi32>) -> tensor<4x1xi32>
    // offset = [0,0], size = [4,1], stride = [1,0]
    %2 = tt.broadcast %1 : (tensor<4x1xi32>) -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [1,0]
    %arg3splat = tt.splat %arg3 : (i32) -> tensor<4x256xi32>
    %offset3 = arith.addi %2, %arg3splat : tensor<4x256xi32>
    // offset = [%arg3,0], size = [4,256], stride = [1,0]
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
    // offset = 0, size = 256, stride = 1
    %4 = tt.expand_dims %3 {axis = 0 : i32} : (tensor<256xi32>) -> tensor<1x256xi32>
    // offset = [0,0], size = [1,256], stride = [0,1]
    %5 = tt.broadcast %4 : (tensor<1x256xi32>) -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [0,1]
    %c5 = arith.constant 5 : i32
    %splat6 = tt.splat %c5 : (i32) -> tensor<4x256xi32>
    // scalar = 5
    %scale5 = arith.muli %5, %splat6 : tensor<4x256xi32> // Why we never called the conversion function for the inputs here?
    // offset = [0,0], size = [4,256], stride = [0,5]
    %7 = arith.addi %offset3, %scale5: tensor<4x256xi32> // Why we never called the conversion function for the inputs here?
    // offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %8 = tt.splat %arg0 : (!tt.ptr<bf16>) -> tensor<4x256x!tt.ptr<bf16>> // Why is the input unknown
    %9 = tt.addptr %8, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg0, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %19 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256xbf16> // this will be replaced with a memref.copy
    %11 = tt.splat %arg1 : (!tt.ptr<bf16>) -> tensor<4x256x!tt.ptr<bf16>>
    %12 = tt.addptr %11, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg1, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %sum_out, %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%sum_iter = %19, %ptr_iter = %12) -> (tensor<4x256xbf16>, tensor<4x256x!tt.ptr<bf16>>) {
        %20 = tt.load %ptr_iter {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256xbf16>
        %sum = arith.addf %sum_iter, %20 : tensor<4x256xbf16>
        // pointer updates
        %17 = tt.splat %i_c3 : (i32) -> tensor<4x256xi32>
        // offset: [3, 0], size = [4, 256], stride [0, 0]
        %ptr = tt.addptr %ptr_iter, %17 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
        // source: %arg1, offset = [%arg3+%i, 0], size = [4, 256], stride = [1, 5]
        scf.yield %sum, %ptr : tensor<4x256xbf16>, tensor<4x256x!tt.ptr<bf16>>
    }
    %15 = tt.splat %arg2 : (!tt.ptr<bf16>) -> tensor<4x256x!tt.ptr<bf16>>
    %16 = tt.addptr %15, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg2, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    tt.store %16, %sum_out : tensor<4x256xbf16>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: memref<*xbf16>, %[[VAL_2:.*]]: memref<*xbf16>, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32) {
// CHECK-DAG:           %[[VAL_8:.*]] = arith.constant 5 : index
// CHECK-DAG:           %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_10:.*]] = arith.constant 3 : index
// CHECK-DAG:           %[[VAL_11:.*]] = arith.constant 12 : index
// CHECK-DAG:           %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_13:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_14:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_13]]], sizes: [4, 256], strides: [1, %[[VAL_8]]] : memref<*xbf16> to memref<4x256xbf16, strided<[1, ?], offset: ?>>
// CHECK:           %[[VAL_15:.*]] = memref.alloc() : memref<4x256xbf16>
// CHECK:           memref.copy %[[VAL_14]], %[[VAL_15]] : memref<4x256xbf16, strided<[1, ?], offset: ?>> to memref<4x256xbf16>
// CHECK:           %[[VAL_16:.*]] = bufferization.to_tensor %[[VAL_15]] restrict writable : memref<4x256xbf16>
// CHECK:           %[[VAL_17:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_18:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_17]]], sizes: [4, 256], strides: {{\[}}%[[VAL_9]], %[[VAL_8]]] : memref<*xbf16> to memref<4x256xbf16, strided<[?, ?], offset: ?>>
// CHECK:           %[[VAL_19:.*]]:4 = scf.for %[[VAL_20:.*]] = %[[VAL_12]] to %[[VAL_11]] step %[[VAL_10]] iter_args(%[[VAL_21:.*]] = %[[VAL_16]], %[[VAL_22:.*]] = %[[VAL_18]], %[[VAL_23:.*]] = %[[VAL_17]], %[[VAL_24:.*]] = %[[VAL_12]]) -> (tensor<4x256xbf16>, memref<4x256xbf16, strided<[?, ?], offset: ?>>, index, index) {
// CHECK:             %[[VAL_25:.*]] = memref.alloc() : memref<4x256xbf16>
// CHECK:             memref.copy %[[VAL_22]], %[[VAL_25]] : memref<4x256xbf16, strided<[?, ?], offset: ?>> to memref<4x256xbf16>
// CHECK:             %[[VAL_26:.*]] = bufferization.to_tensor %[[VAL_25]] restrict writable : memref<4x256xbf16>
// CHECK:             %[[VAL_27:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_21]], %[[VAL_26]] : tensor<4x256xbf16>, tensor<4x256xbf16>) outs(%[[VAL_21]] : tensor<4x256xbf16>) {
// CHECK:             ^bb0(%[[VAL_28:.*]]: bf16, %[[VAL_29:.*]]: bf16, %[[VAL_30:.*]]: bf16):
// CHECK:               %[[VAL_31:.*]] = arith.addf %[[VAL_28]], %[[VAL_29]] : bf16
// CHECK:               linalg.yield %[[VAL_31]] : bf16
// CHECK:             } -> tensor<4x256xbf16>
// CHECK:             %[[VAL_32:.*]] = arith.addi %[[VAL_23]], %[[VAL_10]] : index
// CHECK:             %[[VAL_33:.*]] = arith.addi %[[VAL_32]], %[[VAL_24]] : index
// CHECK:             %[[VAL_34:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_33]]], sizes: [4, 256], strides: {{\[}}%[[VAL_9]], %[[VAL_8]]] : memref<*xbf16> to memref<4x256xbf16, strided<[?, ?], offset: ?>>
// CHECK:             scf.yield %[[VAL_35:.*]], %[[VAL_34]], %[[VAL_33]], %[[VAL_12]] : tensor<4x256xbf16>, memref<4x256xbf16, strided<[?, ?], offset: ?>>, index, index
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_37:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: {{\[}}%[[VAL_36]]], sizes: [4, 256], strides: [1, %[[VAL_8]]] : memref<*xbf16> to memref<4x256xbf16, strided<[1, ?], offset: ?>>
// CHECK:           memref.tensor_store %[[VAL_38:.*]]#0, %[[VAL_37]] : memref<4x256xbf16, strided<[1, ?], offset: ?>>
// CHECK:           return
// CHECK:         }
