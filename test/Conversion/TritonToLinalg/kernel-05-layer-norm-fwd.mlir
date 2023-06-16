// RUN: triton-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @_layer_norm_fwd_fused_0123456789(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>, %arg6: i32, %arg7: i32, %arg8: f32) {
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %arg6 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.splat %cst_0 : (f32) -> tensor<256xf32>
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %6 = tt.splat %arg7 : (i32) -> tensor<256xi32>
    %7 = tt.splat %3 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %8 = scf.for %arg9 = %c0_i32 to %arg7 step %c256_i32 iter_args(%arg10 = %4) -> (tensor<256xf32>)  : i32 {
      %32 = tt.splat %arg9 : (i32) -> tensor<256xi32>
      %33 = arith.addi %32, %5 : tensor<256xi32>
      %34 = arith.cmpi slt, %33, %6 : tensor<256xi32>
      %35 = tt.addptr %7, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %36 = tt.load %35, %34, %4 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32>
      %37 = arith.addf %arg10, %36 : tensor<256xf32>
      scf.yield %37 : tensor<256xf32>
    }
    %9 = "tt.reduce"(%8) ({
    ^bb0(%arg9: f32, %arg10: f32):
      %32 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %32 : f32
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %10 = arith.sitofp %arg7 : i32 to f32
    %11 = arith.divf %9, %10 : f32
    %12 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %13 = tt.splat %arg7 : (i32) -> tensor<256xi32>
    %14 = tt.splat %3 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %15 = tt.splat %11 : (f32) -> tensor<256xf32>
    %16 = scf.for %arg9 = %c0_i32 to %arg7 step %c256_i32 iter_args(%arg10 = %4) -> (tensor<256xf32>)  : i32 {
      %32 = tt.splat %arg9 : (i32) -> tensor<256xi32>
      %33 = arith.addi %32, %12 : tensor<256xi32>
      %34 = arith.cmpi slt, %33, %13 : tensor<256xi32>
      %35 = tt.addptr %14, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %36 = tt.load %35, %34, %4 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32>
      %37 = arith.subf %36, %15 : tensor<256xf32>
      %38 = arith.select %34, %37, %4 : tensor<256xi1>, tensor<256xf32>
      %39 = arith.mulf %38, %38 : tensor<256xf32>
      %40 = arith.addf %arg10, %39 : tensor<256xf32>
      scf.yield %40 : tensor<256xf32>
    }
    %17 = "tt.reduce"(%16) ({
    ^bb0(%arg9: f32, %arg10: f32):
      %32 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %32 : f32
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %18 = arith.divf %17, %10 : f32
    %19 = arith.addf %18, %arg8 : f32
    %20 = math.sqrt %19 : f32
    %21 = arith.divf %cst, %20 : f32
    %22 = tt.addptr %arg4, %0 : !tt.ptr<f32>, i32
    tt.store %22, %11 {cache = 1 : i32, evict = 1 : i32} : f32
    %23 = tt.addptr %arg5, %0 : !tt.ptr<f32>, i32
    tt.store %23, %21 {cache = 1 : i32, evict = 1 : i32} : f32
    %24 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %25 = tt.splat %arg7 : (i32) -> tensor<256xi32>
    %26 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %27 = tt.splat %arg3 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %28 = tt.splat %3 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %29 = tt.splat %11 : (f32) -> tensor<256xf32>
    %30 = tt.splat %21 : (f32) -> tensor<256xf32>
    %31 = tt.splat %2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    scf.for %arg9 = %c0_i32 to %arg7 step %c256_i32  : i32 {
      %32 = tt.splat %arg9 : (i32) -> tensor<256xi32>
      %33 = arith.addi %32, %24 : tensor<256xi32>
      %34 = arith.cmpi slt, %33, %25 : tensor<256xi32>
      %35 = tt.addptr %26, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %36 = tt.load %35, %34 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32>
      %37 = tt.addptr %27, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %38 = tt.load %37, %34 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32>
      %39 = tt.addptr %28, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %40 = tt.load %39, %34, %4 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32>
      %41 = arith.subf %40, %29 : tensor<256xf32>
      %42 = arith.mulf %41, %30 : tensor<256xf32>
      %43 = arith.mulf %42, %36 : tensor<256xf32>
      %44 = arith.addf %43, %38 : tensor<256xf32>
      %45 = tt.addptr %31, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      tt.store %45, %44, %34 {cache = 1 : i32, evict = 1 : i32} : tensor<256xf32>
    }
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @_layer_norm_fwd_fused_0123456789
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: memref<*xf32>, [[PARAM_3_:%.+]]: memref<*xf32>, [[PARAM_4_:%.+]]: memref<*xf32>, [[PARAM_5_:%.+]]: memref<*xf32>, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: f32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32) {
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_256_1_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_0_]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[PARAM_9_]], [[PARAM_6_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = scf.for [[VAR_arg12_:%.+]] = [[CST_0_]] to [[PARAM_7_]] step [[CST_256_1_]] iter_args([[VAR_arg13_:%.+]] = [[VAR_1_]]) -> (tensor<256xf32>)  : i32 {
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.index_cast [[VAR_arg12_]] : i32 to index
// CHECK:             [[VAR_27_:%.+]] = arith.addi [[VAR_25_]], [[VAR_26_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_5_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_27_]]{{.}}, sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<256xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.index_cast [[VAR_arg12_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.addi [[VAR_28_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_30_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:             [[VAR_31_:%.+]] = arith.minsi [[VAR_29_]], [[VAR_30_]] : index
// CHECK:             [[VAR_32_:%.+]] = arith.subi [[VAR_31_]], [[VAR_28_]] : index
// CHECK-DAG:         [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_5_]][0] {{.}}[[VAR_32_]]{{.}} [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[VAR_subview_6_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_32_]]{{.}} [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
// CHECK-DAG:         [[VAR_33_:%.+]] = arith.cmpi slt, [[VAR_32_]], [[CST_256_]] : index
// CHECK:             scf.if [[VAR_33_]] {
// CHECK:               linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[RES_]] : memref<256xf32>)
// CHECK:             }
// CHECK:             memref.copy [[VAR_subview_]], [[VAR_subview_]]_6 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:             [[VAR_34_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<256xf32>
// CHECK:             [[VAR_35_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_arg13_]], [[VAR_34_]] : tensor<256xf32>, tensor<256xf32>) outs([[VAR_arg13_]] : tensor<256xf32>) {
// CHECK:             ^bb0([[in_0:.+]]: f32, [[in_1:.+]]: f32, [[out:.+]]: f32):
// CHECK:               [[VAR_36_:%.+]] = arith.addf [[in_0]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_36_]] : f32
// CHECK:             } -> tensor<256xf32>
// CHECK:             scf.yield [[VAR_35_]] : tensor<256xf32>
// CHECK:           }
// CHECK:           [[VAR_4_:%.+]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_0_dot_000000_]] into [[VAR_4_]][] : tensor<f32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_3_]] : tensor<256xf32>) outs([[VAR_inserted_]] : tensor<f32>) dimensions = [0]
// CHECK:             ([[in_0:.+]]: f32, [[in_1:.+]]: f32) {
// CHECK:               [[VAR_25_1_:%.+]] = arith.addf [[in_0]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_25_1_]] : f32
// CHECK:             }
// CHECK-DAG:       [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]][] : tensor<f32>
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.sitofp [[PARAM_7_]] : i32 to f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.divf [[VAR_extracted_]], [[VAR_5_]] : f32
// CHECK-DAG:       [[VAR_7_:%.+]] = tensor.empty() : tensor<256xi32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_7_]] : tensor<256xi32>) {
// CHECK:           ^bb0([[out:.+]]: i32):
// CHECK:             [[VAR_25_2_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_26_1_:%.+]] = arith.index_cast [[VAR_25_2_]] : index to i32
// CHECK:             linalg.yield [[VAR_26_1_]] : i32
// CHECK:           } -> tensor<256xi32>
// CHECK:           [[VAR_9_:%.+]] = tensor.empty() : tensor<256xi32>
// CHECK-DAG:       [[VAR_10_:%.+]] = linalg.fill ins([[PARAM_7_]] : i32) outs([[VAR_9_]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK-DAG:       [[VAR_11_:%.+]] = tensor.empty() : tensor<256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_12_:%.+]] = linalg.fill ins([[VAR_6_]] : f32) outs([[VAR_11_]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = scf.for [[VAR_arg12_1_:%.+]] = [[CST_0_]] to [[PARAM_7_]] step [[CST_256_1_]] iter_args([[VAR_arg13_1_:%.+]] = [[VAR_1_]]) -> (tensor<256xf32>)  : i32 {
// CHECK-DAG:         [[VAR_25_3_:%.+]] = tensor.empty() : tensor<256xi32>
// CHECK:             [[VAR_26_2_:%.+]] = linalg.fill ins([[VAR_arg12_1_]] : i32) outs([[VAR_25_3_]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK:             [[VAR_27_1_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_26_2_]], [[VAR_8_]] : tensor<256xi32>, tensor<256xi32>) outs([[VAR_26_2_]] : tensor<256xi32>) {
// CHECK:             ^bb0([[in_0:.+]]: i32, [[in_1:.+]]: i32, [[out:.+]]: i32):
// CHECK:               [[VAR_44_:%.+]] = arith.addi [[in_0]], [[in_1]] : i32
// CHECK:               linalg.yield [[VAR_44_]] : i32
// CHECK:             } -> tensor<256xi32>
// CHECK:             [[VAR_28_1_:%.+]] = tensor.empty() : tensor<256xi1>
// CHECK:             [[VAR_29_1_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_27_1_]], [[VAR_10_]] : tensor<256xi32>, tensor<256xi32>) outs([[VAR_28_1_]] : tensor<256xi1>) {
// CHECK:             ^bb0([[in_0:.+]]: i32, [[in_1:.+]]: i32, [[out:.+]]: i1):
// CHECK:               [[VAR_44_1_:%.+]] = arith.cmpi slt, [[in_0]], [[in_1]] : i32
// CHECK:               linalg.yield [[VAR_44_1_]] : i1
// CHECK:             } -> tensor<256xi1>
// CHECK-DAG:         [[VAR_30_1_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-DAG:         [[VAR_31_1_:%.+]] = arith.index_cast [[VAR_arg12_1_]] : i32 to index
// CHECK:             [[VAR_32_1_:%.+]] = arith.addi [[VAR_30_1_]], [[VAR_31_1_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_5_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_32_1_]]{{.}}, sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() : memref<256xf32>
// CHECK-DAG:         [[VAR_33_1_:%.+]] = arith.index_cast [[VAR_arg12_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_34_1_:%.+]] = arith.addi [[VAR_33_1_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_35_1_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:             [[VAR_36_1_:%.+]] = arith.minsi [[VAR_34_1_]], [[VAR_35_1_]] : index
// CHECK:             [[VAR_37_:%.+]] = arith.subi [[VAR_36_1_]], [[VAR_33_1_]] : index
// CHECK-DAG:         [[VAR_subview_1_:%.+]] = memref.subview [[VAR_reinterpret_cast_5_1_]][0] {{.}}[[VAR_37_]]{{.}} [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[VAR_subview_6_1_:%.+]] = memref.subview [[RES_1_]][0] {{.}}[[VAR_37_]]{{.}} [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.cmpi slt, [[VAR_37_]], [[CST_256_]] : index
// CHECK:             scf.if [[VAR_38_]] {
// CHECK:               linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[RES_1_]] : memref<256xf32>)
// CHECK:             }
// CHECK:             memref.copy [[VAR_subview_1_]], [[VAR_subview_1_]]_6 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:             [[VAR_39_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<256xf32>
// CHECK:             [[VAR_40_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_39_]], [[VAR_12_]] : tensor<256xf32>, tensor<256xf32>) outs([[VAR_39_]] : tensor<256xf32>) {
// CHECK:             ^bb0([[in_0:.+]]: f32, [[in_1:.+]]: f32, [[out:.+]]: f32):
// CHECK:               [[VAR_44_2_:%.+]] = arith.subf [[in_0]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_44_2_]] : f32
// CHECK:             } -> tensor<256xf32>
// CHECK:             [[VAR_41_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_29_1_]], [[VAR_40_]], [[VAR_1_]] : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs([[VAR_40_]] : tensor<256xf32>) {
// CHECK:             ^bb0([[in_0:.+]]: i1, [[in_1:.+]]: f32, [[in_2:.+]]: f32, [[out:.+]]: f32):
// CHECK:               [[VAR_44_3_:%.+]] = arith.select [[in_0]], [[in_1]], [[in_2]] : f32
// CHECK:               linalg.yield [[VAR_44_3_]] : f32
// CHECK:             } -> tensor<256xf32>
// CHECK:             [[VAR_42_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_41_]], [[VAR_41_]] : tensor<256xf32>, tensor<256xf32>) outs([[VAR_41_]] : tensor<256xf32>) {
// CHECK:             ^bb0([[in_0:.+]]: f32, [[in_1:.+]]: f32, [[out:.+]]: f32):
// CHECK:               [[VAR_44_4_:%.+]] = arith.mulf [[in_0]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_44_4_]] : f32
// CHECK:             } -> tensor<256xf32>
// CHECK:             [[VAR_43_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_arg13_1_]], [[VAR_42_]] : tensor<256xf32>, tensor<256xf32>) outs([[VAR_arg13_1_]] : tensor<256xf32>) {
// CHECK:             ^bb0([[in_0:.+]]: f32, [[in_1:.+]]: f32, [[out:.+]]: f32):
// CHECK:               [[VAR_44_5_:%.+]] = arith.addf [[in_0]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_44_5_]] : f32
// CHECK:             } -> tensor<256xf32>
// CHECK:             scf.yield [[VAR_43_]] : tensor<256xf32>
// CHECK:           }
// CHECK:           [[VAR_14_:%.+]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           [[VAR_inserted_1_:%.+]] = tensor.insert [[CST_0_dot_000000_]] into [[VAR_14_]][] : tensor<f32>
// CHECK:           [[VAR_reduced_2_:%.+]] = linalg.reduce ins([[VAR_13_]] : tensor<256xf32>) outs([[VAR_inserted_1_]] : tensor<f32>) dimensions = [0]
// CHECK:             ([[in_0:.+]]: f32, [[in_1:.+]]: f32) {
// CHECK:               [[VAR_25_4_:%.+]] = arith.addf [[in_0]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_25_4_]] : f32
// CHECK:             }
// CHECK:           [[VAR_extracted_3_:%.+]] = tensor.extract [[VAR_reduced_2_]][] : tensor<f32>
// CHECK:           [[VAR_15_:%.+]] = arith.divf [[VAR_extracted_3_]], [[VAR_5_]] : f32
// CHECK:           [[VAR_16_:%.+]] = arith.addf [[VAR_15_]], [[PARAM_8_]] : f32
// CHECK:           [[VAR_17_:%.+]] = math.sqrt [[VAR_16_]] : f32
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_17_]] : f32
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.index_cast [[PARAM_9_]] : i32 to index
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_4_]] to offset: {{.}}[[VAR_19_]]{{.}}, sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:           affine.store [[VAR_6_]], [[VAR_reinterpret_cast_]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:           [[VAR_20_:%.+]] = arith.index_cast [[PARAM_9_]] : i32 to index
// CHECK:           [[VAR_reinterpret_cast_4_:%.+]] = memref.reinterpret_cast [[PARAM_5_]] to offset: {{.}}[[VAR_20_]]{{.}}, sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:           affine.store [[VAR_18_]], [[VAR_reinterpret_cast_4_]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:           [[VAR_21_:%.+]] = tensor.empty() : tensor<256xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = linalg.fill ins([[VAR_6_]] : f32) outs([[VAR_21_]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = tensor.empty() : tensor<256xf32>
// CHECK:           [[VAR_24_:%.+]] = linalg.fill ins([[VAR_18_]] : f32) outs([[VAR_23_]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK:           scf.for [[VAR_arg12_1_:%.+]] = [[CST_0_]] to [[PARAM_7_]] step [[CST_256_1_]]  : i32 {
// CHECK:             [[VAR_25_5_:%.+]] = arith.index_cast [[VAR_arg12_1_]] : i32 to index
// CHECK-DAG:         [[VAR_reinterpret_cast_5_2_:%.+]] = memref.reinterpret_cast [[PARAM_2_]] to offset: {{.}}[[VAR_25_5_]]{{.}}, sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() : memref<256xf32>
// CHECK-DAG:         [[VAR_26_3_:%.+]] = arith.index_cast [[VAR_arg12_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_27_2_:%.+]] = arith.addi [[VAR_26_3_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_28_2_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:             [[VAR_29_2_:%.+]] = arith.minsi [[VAR_27_2_]], [[VAR_28_2_]] : index
// CHECK:             [[VAR_30_2_:%.+]] = arith.subi [[VAR_29_2_]], [[VAR_26_3_]] : index
// CHECK-DAG:         [[VAR_subview_2_:%.+]] = memref.subview [[VAR_reinterpret_cast_5_2_]][0] {{.}}[[VAR_30_2_]]{{.}} [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[VAR_subview_6_2_:%.+]] = memref.subview [[RES_2_]][0] {{.}}[[VAR_30_2_]]{{.}} [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
// CHECK:             memref.copy [[VAR_subview_2_]], [[VAR_subview_2_]]_6 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK-DAG:         [[VAR_31_2_:%.+]] = bufferization.to_tensor [[RES_2_]] restrict writable : memref<256xf32>
// CHECK-DAG:         [[VAR_32_2_:%.+]] = arith.index_cast [[VAR_arg12_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_reinterpret_cast_7_:%.+]] = memref.reinterpret_cast [[PARAM_3_]] to offset: {{.}}[[VAR_32_2_]]{{.}}, sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() : memref<256xf32>
// CHECK-DAG:         [[VAR_33_2_:%.+]] = arith.index_cast [[VAR_arg12_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_34_2_:%.+]] = arith.addi [[VAR_33_2_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_35_2_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:             [[VAR_36_2_:%.+]] = arith.minsi [[VAR_34_2_]], [[VAR_35_2_]] : index
// CHECK:             [[VAR_37_1_:%.+]] = arith.subi [[VAR_36_2_]], [[VAR_33_2_]] : index
// CHECK-DAG:         [[VAR_subview_9_:%.+]] = memref.subview [[VAR_reinterpret_cast_7_]][0] {{.}}[[VAR_37_1_]]{{.}} [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[VAR_subview_10_:%.+]] = memref.subview [[RES_3_]][0] {{.}}[[VAR_37_1_]]{{.}} [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
// CHECK:             memref.copy [[VAR_subview_9_]], [[VAR_subview_10_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK-DAG:         [[VAR_38_1_:%.+]] = bufferization.to_tensor [[RES_3_]] restrict writable : memref<256xf32>
// CHECK-DAG:         [[VAR_39_1_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-DAG:         [[VAR_40_1_:%.+]] = arith.index_cast [[VAR_arg12_1_]] : i32 to index
// CHECK:             [[VAR_41_1_:%.+]] = arith.addi [[VAR_39_1_]], [[VAR_40_1_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_11_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_41_1_]]{{.}}, sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc() : memref<256xf32>
// CHECK-DAG:         [[VAR_42_1_:%.+]] = arith.index_cast [[VAR_arg12_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_43_1_:%.+]] = arith.addi [[VAR_42_1_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_44_6_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:             [[VAR_45_:%.+]] = arith.minsi [[VAR_43_1_]], [[VAR_44_6_]] : index
// CHECK:             [[VAR_46_:%.+]] = arith.subi [[VAR_45_]], [[VAR_42_1_]] : index
// CHECK-DAG:         [[VAR_subview_13_:%.+]] = memref.subview [[VAR_reinterpret_cast_11_]][0] {{.}}[[VAR_46_]]{{.}} [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[VAR_subview_14_:%.+]] = memref.subview [[RES_4_]][0] {{.}}[[VAR_46_]]{{.}} [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
// CHECK-DAG:         [[VAR_47_:%.+]] = arith.cmpi slt, [[VAR_46_]], [[CST_256_]] : index
// CHECK:             scf.if [[VAR_47_]] {
// CHECK:               linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[RES_4_]] : memref<256xf32>)
// CHECK:             }
// CHECK:             memref.copy [[VAR_subview_13_]], [[VAR_subview_14_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:             [[VAR_48_:%.+]] = bufferization.to_tensor [[RES_4_]] restrict writable : memref<256xf32>
// CHECK:             [[VAR_49_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_48_]], [[VAR_22_]] : tensor<256xf32>, tensor<256xf32>) outs([[VAR_48_]] : tensor<256xf32>) {
// CHECK:             ^bb0([[in_0:.+]]: f32, [[in_1:.+]]: f32, [[out:.+]]: f32):
// CHECK:               [[VAR_61_:%.+]] = arith.subf [[in_0]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_61_]] : f32
// CHECK:             } -> tensor<256xf32>
// CHECK:             [[VAR_50_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_49_]], [[VAR_24_]] : tensor<256xf32>, tensor<256xf32>) outs([[VAR_49_]] : tensor<256xf32>) {
// CHECK:             ^bb0([[in_0:.+]]: f32, [[in_1:.+]]: f32, [[out:.+]]: f32):
// CHECK:               [[VAR_61_1_:%.+]] = arith.mulf [[in_0]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_61_1_]] : f32
// CHECK:             } -> tensor<256xf32>
// CHECK:             [[VAR_51_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_50_]], [[VAR_31_2_]] : tensor<256xf32>, tensor<256xf32>) outs([[VAR_50_]] : tensor<256xf32>) {
// CHECK:             ^bb0([[in_0:.+]]: f32, [[in_1:.+]]: f32, [[out:.+]]: f32):
// CHECK:               [[VAR_61_2_:%.+]] = arith.mulf [[in_0]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_61_2_]] : f32
// CHECK:             } -> tensor<256xf32>
// CHECK:             [[VAR_52_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_51_]], [[VAR_38_1_]] : tensor<256xf32>, tensor<256xf32>) outs([[VAR_51_]] : tensor<256xf32>) {
// CHECK:             ^bb0([[in_0:.+]]: f32, [[in_1:.+]]: f32, [[out:.+]]: f32):
// CHECK:               [[VAR_61_3_:%.+]] = arith.addf [[in_0]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_61_3_]] : f32
// CHECK:             } -> tensor<256xf32>
// CHECK-DAG:         [[VAR_53_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-DAG:         [[VAR_54_:%.+]] = arith.index_cast [[VAR_arg12_1_]] : i32 to index
// CHECK:             [[VAR_55_:%.+]] = arith.addi [[VAR_53_]], [[VAR_54_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_15_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_55_]]{{.}}, sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[VAR_56_:%.+]] = arith.index_cast [[VAR_arg12_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_57_:%.+]] = arith.addi [[VAR_56_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_58_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:             [[VAR_59_:%.+]] = arith.minsi [[VAR_57_]], [[VAR_58_]] : index
// CHECK:             [[VAR_60_:%.+]] = arith.subi [[VAR_59_]], [[VAR_56_]] : index
// CHECK-DAG:         [[VAR_extracted_slice_:%.+]] = tensor.extract_slice [[VAR_52_]][0] {{.}}[[VAR_60_]]{{.}} [1] : tensor<256xf32> to tensor<?xf32>
// CHECK-DAG:         [[VAR_subview_16_:%.+]] = memref.subview [[VAR_reinterpret_cast_15_]][0] {{.}}[[VAR_60_]]{{.}} [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:             memref.tensor_store [[VAR_extracted_slice_]], [[VAR_subview_16_]] : memref<?xf32, strided<[1], offset: ?>>
// CHECK:           }
// CHECK:           return
// CHECK:         }
