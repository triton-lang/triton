// RUN: triton-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @_layer_norm_bwd_dwdb_0123456(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: i32, %arg5: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : (i32) -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %cst : (f32) -> tensor<256x256xf32>
    %6 = tt.splat %arg4 : (i32) -> tensor<256x1xi32>
    %7 = tt.expand_dims %4 {axis = 0 : i32} : (tensor<256xi32>) -> tensor<1x256xi32>
    %8 = tt.splat %arg5 : (i32) -> tensor<1x256xi32>
    %9 = arith.cmpi slt, %7, %8 : tensor<1x256xi32>
    %10 = tt.broadcast %9 : (tensor<1x256xi1>) -> tensor<256x256xi1>
    %11 = tt.splat %arg5 : (i32) -> tensor<256x1xi32>
    %12 = tt.broadcast %7 : (tensor<1x256xi32>) -> tensor<256x256xi32>
    %13 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<256x256x!tt.ptr<f32>>
    %14 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<256x256x!tt.ptr<f32>>
    %15:2 = scf.for %arg6 = %c0_i32 to %arg4 step %c256_i32 iter_args(%arg7 = %5, %arg8 = %5) -> (tensor<256x256xf32>, tensor<256x256xf32>)  : i32 {
      %24 = tt.splat %arg6 : (i32) -> tensor<256xi32>
      %25 = arith.addi %24, %2 : tensor<256xi32>
      %26 = tt.expand_dims %25 {axis = 1 : i32} : (tensor<256xi32>) -> tensor<256x1xi32>
      %27 = arith.cmpi slt, %26, %6 : tensor<256x1xi32>
      %28 = tt.broadcast %27 : (tensor<256x1xi1>) -> tensor<256x256xi1>
      %29 = arith.andi %28, %10 : tensor<256x256xi1>
      %30 = arith.muli %26, %11 : tensor<256x1xi32>
      %31 = tt.broadcast %30 : (tensor<256x1xi32>) -> tensor<256x256xi32>
      %32 = arith.addi %31, %12 : tensor<256x256xi32>
      %33 = tt.addptr %13, %32 : tensor<256x256x!tt.ptr<f32>>, tensor<256x256xi32>
      %34 = tt.load %33, %29, %5 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x256xf32>
      %35 = arith.addf %arg7, %34 : tensor<256x256xf32>
      %36 = tt.addptr %14, %32 : tensor<256x256x!tt.ptr<f32>>, tensor<256x256xi32>
      %37 = tt.load %36, %29, %5 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x256xf32>
      %38 = arith.addf %arg8, %37 : tensor<256x256xf32>
      scf.yield %35, %38 : tensor<256x256xf32>, tensor<256x256xf32>
    }
    %16 = "tt.reduce"(%15#0) ({
    ^bb0(%arg6: f32, %arg7: f32):
      %24 = arith.addf %arg6, %arg7 : f32
      tt.reduce.return %24 : f32
    }) {axis = 0 : i32} : (tensor<256x256xf32>) -> tensor<256xf32>
    %17 = "tt.reduce"(%15#1) ({
    ^bb0(%arg6: f32, %arg7: f32):
      %24 = arith.addf %arg6, %arg7 : f32
      tt.reduce.return %24 : f32
    }) {axis = 0 : i32} : (tensor<256x256xf32>) -> tensor<256xf32>
    %18 = tt.splat %arg5 : (i32) -> tensor<256xi32>
    %19 = arith.cmpi slt, %4, %18 : tensor<256xi32>
    %20 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %21 = tt.addptr %20, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %21, %16, %19 {cache = 1 : i32, evict = 1 : i32} : tensor<256xf32>
    %22 = tt.splat %arg3 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %23 = tt.addptr %22, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %23, %17, %19 {cache = 1 : i32, evict = 1 : i32} : tensor<256xf32>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @_layer_norm_bwd_dwdb_0123456
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: memref<*xf32>, [[PARAM_3_:%.+]]: memref<*xf32>, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[CST_256_1_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<256x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_0_]] : tensor<256x256xf32>) -> tensor<256x256xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[PARAM_6_]], [[CST_256_1_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]]:2 = scf.for [[VAR_arg9_:%.+]] = [[CST_0_]] to [[PARAM_4_]] step [[CST_256_1_]] iter_args([[VAR_arg10_:%.+]] = [[VAR_1_]], [[VAR_arg11_:%.+]] = [[VAR_1_]]) -> (tensor<256x256xf32>, tensor<256x256xf32>)  : i32 {
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.index_cast [[VAR_arg9_]] : i32 to index
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.muli [[VAR_20_]], [[VAR_21_]] : index
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK:             [[VAR_24_:%.+]] = arith.addi [[VAR_22_]], [[VAR_23_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_4_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_24_]]{{.}}, sizes: [256, 256], strides: {{.}}[[VAR_21_]], 1] : memref<*xf32> to memref<256x256xf32, strided<[?, 1], offset: ?>>
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<256x256xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.index_cast [[VAR_arg9_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.addi [[VAR_25_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:             [[VAR_28_:%.+]] = arith.minsi [[VAR_26_]], [[VAR_27_]] : index
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.subi [[VAR_28_]], [[VAR_25_]] : index
// CHECK-DAG:         [[VAR_30_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_31_:%.+]] = arith.addi [[VAR_30_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_32_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:             [[VAR_33_:%.+]] = arith.minsi [[VAR_31_]], [[VAR_32_]] : index
// CHECK-DAG:         [[VAR_34_:%.+]] = arith.subi [[VAR_33_]], [[VAR_30_]] : index
// CHECK-DAG:         [[VAR_35_:%.+]] = arith.minsi [[VAR_29_]], [[CST_256_]] : index
// CHECK:             [[VAR_36_:%.+]] = arith.minsi [[VAR_34_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_subview_5_:%.+]] = memref.subview [[VAR_reinterpret_cast_4_]][0, 0] {{.}}[[VAR_35_]], [[VAR_36_]]{{.}} [1, 1] : memref<256x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK-DAG:         [[VAR_subview_6_:%.+]] = memref.subview [[RES_]][0, 0] {{.}}[[VAR_35_]], [[VAR_36_]]{{.}} [1, 1] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1]>>
// CHECK-DAG:         [[VAR_37_:%.+]] = arith.cmpi slt, [[VAR_35_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.cmpi slt, [[VAR_36_]], [[CST_256_]] : index
// CHECK:             [[VAR_39_:%.+]] = arith.ori [[VAR_37_]], [[VAR_38_]] : i1
// CHECK:             scf.if [[VAR_39_]] {
// CHECK:               linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[RES_]] : memref<256x256xf32>)
// CHECK:             }
// CHECK:             memref.copy [[VAR_subview_5_]], [[VAR_subview_6_]] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1]>>
// CHECK:             [[VAR_40_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<256x256xf32>
// CHECK:             [[VAR_41_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_arg10_]], [[VAR_40_]] : tensor<256x256xf32>, tensor<256x256xf32>) outs([[VAR_arg10_]] : tensor<256x256xf32>) {
// CHECK:             ^bb0([[in_0:.+]]: f32, [[in_1:.+]]: f32, [[out:.+]]: f32):
// CHECK:               [[VAR_64_:%.+]] = arith.addf [[in_0]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_64_]] : f32
// CHECK:             } -> tensor<256x256xf32>
// CHECK-DAG:         [[VAR_42_:%.+]] = arith.index_cast [[VAR_arg9_]] : i32 to index
// CHECK-DAG:         [[VAR_43_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_44_:%.+]] = arith.muli [[VAR_42_]], [[VAR_43_]] : index
// CHECK-DAG:         [[VAR_45_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK:             [[VAR_46_:%.+]] = arith.addi [[VAR_44_]], [[VAR_45_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_7_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_46_]]{{.}}, sizes: [256, 256], strides: {{.}}[[VAR_43_]], 1] : memref<*xf32> to memref<256x256xf32, strided<[?, 1], offset: ?>>
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() : memref<256x256xf32>
// CHECK-DAG:         [[VAR_47_:%.+]] = arith.index_cast [[VAR_arg9_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.addi [[VAR_47_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_49_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:             [[VAR_50_:%.+]] = arith.minsi [[VAR_48_]], [[VAR_49_]] : index
// CHECK-DAG:         [[VAR_51_:%.+]] = arith.subi [[VAR_50_]], [[VAR_47_]] : index
// CHECK-DAG:         [[VAR_52_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_53_:%.+]] = arith.addi [[VAR_52_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_54_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:             [[VAR_55_:%.+]] = arith.minsi [[VAR_53_]], [[VAR_54_]] : index
// CHECK-DAG:         [[VAR_56_:%.+]] = arith.subi [[VAR_55_]], [[VAR_52_]] : index
// CHECK-DAG:         [[VAR_57_:%.+]] = arith.minsi [[VAR_51_]], [[CST_256_]] : index
// CHECK:             [[VAR_58_:%.+]] = arith.minsi [[VAR_56_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_subview_9_:%.+]] = memref.subview [[VAR_reinterpret_cast_7_]][0, 0] {{.}}[[VAR_57_]], [[VAR_58_]]{{.}} [1, 1] : memref<256x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK-DAG:         [[VAR_subview_10_:%.+]] = memref.subview [[RES_1_]][0, 0] {{.}}[[VAR_57_]], [[VAR_58_]]{{.}} [1, 1] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1]>>
// CHECK-DAG:         [[VAR_59_:%.+]] = arith.cmpi slt, [[VAR_57_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_60_:%.+]] = arith.cmpi slt, [[VAR_58_]], [[CST_256_]] : index
// CHECK:             [[VAR_61_:%.+]] = arith.ori [[VAR_59_]], [[VAR_60_]] : i1
// CHECK:             scf.if [[VAR_61_]] {
// CHECK:               linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[RES_1_]] : memref<256x256xf32>)
// CHECK:             }
// CHECK:             memref.copy [[VAR_subview_9_]], [[VAR_subview_10_]] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1]>>
// CHECK:             [[VAR_62_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<256x256xf32>
// CHECK:             [[VAR_63_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_arg11_]], [[VAR_62_]] : tensor<256x256xf32>, tensor<256x256xf32>) outs([[VAR_arg11_]] : tensor<256x256xf32>) {
// CHECK:             ^bb0([[in_0:.+]]: f32, [[in_1:.+]]: f32, [[out:.+]]: f32):
// CHECK:               [[VAR_64_1_:%.+]] = arith.addf [[in_0]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_64_1_]] : f32
// CHECK:             } -> tensor<256x256xf32>
// CHECK:             scf.yield [[VAR_41_]], [[VAR_63_]] : tensor<256x256xf32>, tensor<256x256xf32>
// CHECK:           }
// CHECK:           [[VAR_4_:%.+]] = tensor.empty() : tensor<256xf32>
// CHECK:           [[VAR_5_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_4_]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_3_]]#0 : tensor<256x256xf32>) outs([[VAR_5_]] : tensor<256xf32>) dimensions = [0]
// CHECK:             ([[in_0:.+]]: f32, [[in_1:.+]]: f32) {
// CHECK:               [[VAR_20_1_:%.+]] = arith.addf [[in_0]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_20_1_]] : f32
// CHECK:             }
// CHECK:           [[VAR_6_:%.+]] = tensor.empty() : tensor<256xf32>
// CHECK:           [[VAR_7_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_6_]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK:           [[VAR_reduced_0_:%.+]] = linalg.reduce ins([[VAR_3_]]#1 : tensor<256x256xf32>) outs([[VAR_7_]] : tensor<256xf32>) dimensions = [0]
// CHECK:             ([[in_0:.+]]: f32, [[in_1:.+]]: f32) {
// CHECK:               [[VAR_20_2_:%.+]] = arith.addf [[in_0]], [[in_1]] : f32
// CHECK:               linalg.yield [[VAR_20_2_]] : f32
// CHECK:             }
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_2_]] to offset: {{.}}[[VAR_8_]]{{.}}, sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.addi [[VAR_9_]], [[CST_256_]] : index
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:           [[VAR_12_:%.+]] = arith.minsi [[VAR_10_]], [[VAR_11_]] : index
// CHECK:           [[VAR_13_:%.+]] = arith.subi [[VAR_12_]], [[VAR_9_]] : index
// CHECK-DAG:       [[VAR_extracted_slice_:%.+]] = tensor.extract_slice [[VAR_reduced_]][0] {{.}}[[VAR_13_]]{{.}} [1] : tensor<256xf32> to tensor<?xf32>
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_13_]]{{.}} [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:           memref.tensor_store [[VAR_extracted_slice_]], [[VAR_subview_]] : memref<?xf32, strided<[1], offset: ?>>
// CHECK:           [[VAR_14_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_3_]] to offset: {{.}}[[VAR_14_]]{{.}}, sizes: [256], strides: [1] : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.addi [[VAR_15_]], [[CST_256_]] : index
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:           [[VAR_18_:%.+]] = arith.minsi [[VAR_16_]], [[VAR_17_]] : index
// CHECK:           [[VAR_19_:%.+]] = arith.subi [[VAR_18_]], [[VAR_15_]] : index
// CHECK-DAG:       [[VAR_extracted_slice_2_:%.+]] = tensor.extract_slice [[VAR_reduced_0_]][0] {{.}}[[VAR_19_]]{{.}} [1] : tensor<256xf32> to tensor<?xf32>
// CHECK-DAG:       [[VAR_subview_3_:%.+]] = memref.subview [[VAR_reinterpret_cast_1_]][0] {{.}}[[VAR_19_]]{{.}} [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK:           memref.tensor_store [[VAR_extracted_slice_2_]], [[VAR_subview_3_]] : memref<?xf32, strided<[1], offset: ?>>
// CHECK:           return
// CHECK:         }
