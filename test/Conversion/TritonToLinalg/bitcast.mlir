// RUN: triton-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func @kernel(%a : !tt.ptr<i32>, %b : !tt.ptr<f32>) -> () {
    // offset calculations
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>

    // a pointer
    %8 = tt.splat %a : (!tt.ptr<i32>) -> tensor<1024x!tt.ptr<i32>>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>

    // b pointer
    %18 = tt.splat %b : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>

    %am = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi32>

    // cast result before doing float add
    %am_bitcast = tt.bitcast %am : tensor<1024xi32> -> tensor<1024xf32>


    tt.store %19, %am_bitcast : tensor<1024xf32>
    tt.return
  }
}

// CHECK: module {
// CHECK:   func.func @kernel(%arg0: memref<*xi32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32) {
// CHECK:   [[RC_:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1]{{.*}} : memref<*xi32> to memref<1024xi32, strided<[1]>>
// CHECK:   [[RC_0_:%.+]] = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1024], strides: [1]{{.*}} : memref<*xf32> to memref<1024xf32, strided<[1]>>
// CHECK:   [[ALLOC_:%.+]] = memref.alloc() : memref<1024xi32>
// CHECK:   memref.copy [[RC_]], [[ALLOC_]] : memref<1024xi32, strided<[1]>> to memref<1024xi32>
// CHECK:   [[VAR_0_:%.+]] = bufferization.to_tensor [[ALLOC_]] restrict writable : memref<1024xi32>
// CHECK:   [[VAR_1_:%.+]] = tensor.empty() : tensor<1024xf32>
// CHECK:   [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_0_]] : tensor<1024xi32>) outs([[VAR_1_]] : tensor<1024xf32>) {
// CHECK:   ^bb0(%in: i32, %out: f32):
// CHECK:     [[VAR_5_:%.+]] = arith.bitcast %in : i32 to f32
// CHECK:     linalg.yield [[VAR_5_]] : f32
// CHECK:   } -> tensor<1024xf32>
// CHECK:   memref.tensor_store [[VAR_2_]], [[RC_0_]] : memref<1024xf32, strided<[1]>>
// CHECK:     return
// CHECK:   }
// CHECK: }
