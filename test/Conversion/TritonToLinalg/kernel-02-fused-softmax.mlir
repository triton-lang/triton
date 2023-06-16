// RUN: triton-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @softmax_kernel_012345(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32) {
    %cst = arith.constant 0xFF800000 : f32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %4 = tt.splat %2 : (!tt.ptr<f32>) -> tensor<128x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %6 = tt.splat %arg4 : (i32) -> tensor<128xi32>
    %7 = arith.cmpi slt, %3, %6 : tensor<128xi32>
    %8 = tt.splat %cst : (f32) -> tensor<128xf32>
    %9 = tt.load %5, %7, %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128xf32>
    %10 = "tt.reduce"(%9) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %21 = arith.cmpf ogt, %arg5, %arg6 : f32
      %22 = arith.select %21, %arg5, %arg6 : f32
      tt.reduce.return %22 : f32
    }) {axis = 0 : i32} : (tensor<128xf32>) -> f32
    %11 = tt.splat %10 : (f32) -> tensor<128xf32>
    %12 = arith.subf %9, %11 : tensor<128xf32>
    %13 = math.exp %12 : tensor<128xf32>
    %14 = "tt.reduce"(%13) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %21 = arith.addf %arg5, %arg6 : f32
      tt.reduce.return %21 : f32
    }) {axis = 0 : i32} : (tensor<128xf32>) -> f32
    %15 = tt.splat %14 : (f32) -> tensor<128xf32>
    %16 = arith.divf %13, %15 : tensor<128xf32>
    %17 = arith.muli %0, %arg3 : i32
    %18 = tt.addptr %arg0, %17 : !tt.ptr<f32>, i32
    %19 = tt.splat %18 : (!tt.ptr<f32>) -> tensor<128x!tt.ptr<f32>>
    %20 = tt.addptr %19, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %20, %16, %7 {cache = 1 : i32, evict = 1 : i32} : tensor<128xf32>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @softmax_kernel_012345
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.muli [[PARAM_5_]], [[PARAM_2_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [128], strides: [1]{{.*}} : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<128xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:           [[VAR_3_:%.+]] = arith.minsi [[VAR_2_]], [[CST_128_]] : index
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_3_]]{{.}} [1]{{.*}} : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_1_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_3_]]{{.}} [1] : memref<128xf32> to memref<?xf32, strided<[1]>>
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_128_]] : index
// CHECK:           scf.if [[VAR_4_]] {
// CHECK:             linalg.fill ins([[CST_0_]] : f32) outs([[RES_]] : memref<128xf32>)
// CHECK:           }
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_]]_1 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK-DAG:       [[VAR_5_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<128xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_0_]] into [[VAR_6_]][] : tensor<f32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_5_]] : tensor<128xf32>) outs([[VAR_inserted_]] : tensor<f32>) dimensions = [0]
// CHECK:             ([[in_1:%.+]]: f32, [[init_1:%.+]]: f32) {
// CHECK:               [[VAR_19_:%.+]] = arith.maxf [[in_1]], [[init_1]] : f32
// CHECK:               linalg.yield [[VAR_19_]] : f32
// CHECK:             }
// CHECK-DAG:       [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]][] : tensor<f32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tensor.empty() : tensor<128xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.fill ins([[VAR_extracted_]] : f32) outs([[VAR_7_]] : tensor<128xf32>) -> tensor<128xf32>
// CHECK:           [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_5_]], [[VAR_8_]] : tensor<128xf32>, tensor<128xf32>) outs([[VAR_5_]] : tensor<128xf32>) {
// CHECK:           ^bb0([[in_1:%.+]]: f32, [[in_2:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:             [[VAR_19_1_:%.+]] = arith.subf [[in_1]], [[in_2]] : f32
// CHECK:             linalg.yield [[VAR_19_1_]] : f32
// CHECK:           } -> tensor<128xf32>
// CHECK:           [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_9_]] : tensor<128xf32>) outs([[VAR_9_]] : tensor<128xf32>) {
// CHECK:           ^bb0([[in_1:%.+]]: f32, [[out_1:%.+]]: f32):
// CHECK:             [[VAR_19_2_:%.+]] = math.exp [[in_1]] : f32
// CHECK:             linalg.yield [[VAR_19_2_]] : f32
// CHECK:           } -> tensor<128xf32>
// CHECK:           [[VAR_11_:%.+]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           [[VAR_inserted_2_:%.+]] = tensor.insert [[CST_0_dot_000000_]] into [[VAR_11_]][] : tensor<f32>
// CHECK:           [[VAR_reduced_3_:%.+]] = linalg.reduce ins([[VAR_10_]] : tensor<128xf32>) outs([[VAR_inserted_2_]] : tensor<f32>) dimensions = [0]
// CHECK:             ([[in_1:%.+]]: f32, [[init_1:%.+]]: f32) {
// CHECK:               [[VAR_19_3_:%.+]] = arith.addf [[in_1]], [[init_1]] : f32
// CHECK:               linalg.yield [[VAR_19_3_]] : f32
// CHECK:             }
// CHECK-DAG:       [[VAR_extracted_4_:%.+]] = tensor.extract [[VAR_reduced_3_]][] : tensor<f32>
// CHECK-DAG:       [[VAR_12_:%.+]] = tensor.empty() : tensor<128xf32>
// CHECK:           [[VAR_13_:%.+]] = linalg.fill ins([[VAR_extracted_4_]] : f32) outs([[VAR_12_]] : tensor<128xf32>) -> tensor<128xf32>
// CHECK:           [[VAR_14_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_10_]], [[VAR_13_]] : tensor<128xf32>, tensor<128xf32>) outs([[VAR_10_]] : tensor<128xf32>) {
// CHECK:           ^bb0([[in_1:%.+]]: f32, [[in_2:%.+]]: f32, [[out_1:%.+]]: f32):
// CHECK:             [[VAR_19_4_:%.+]] = arith.divf [[in_1]], [[in_2]] : f32
// CHECK:             linalg.yield [[VAR_19_4_]] : f32
// CHECK:           } -> tensor<128xf32>
// CHECK:           [[VAR_15_:%.+]] = arith.muli [[PARAM_5_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_16_:%.+]] = arith.index_cast [[VAR_15_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_5_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_16_]]{{.}}, sizes: [128], strides: [1]{{.*}} : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:           [[VAR_18_:%.+]] = arith.minsi [[VAR_17_]], [[CST_128_]] : index
// CHECK-DAG:       [[VAR_extracted_slice_:%.+]] = tensor.extract_slice [[VAR_14_]][0] {{.}}[[VAR_18_]]{{.}} [1] : tensor<128xf32> to tensor<?xf32>
// CHECK-DAG:       [[VAR_subview_6_:%.+]] = memref.subview [[VAR_reinterpret_cast_5_]][0] {{.}}[[VAR_18_]]{{.}} [1]{{.*}} : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:           memref.tensor_store [[VAR_extracted_slice_]], [[VAR_subview_6_]] : memref<?xf32, strided<[1], offset: ?>>
// CHECK:           return
// CHECK:         }
