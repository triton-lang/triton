// RUN: triton-opt %s -split-input-file --tritonamdgpu-dot-slicing=slice-k-tile=32 | FileCheck %s

// CHECK: #[[SLICE_V_LAYOUT:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
// CHECK: #blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
// CHECK: #blocked2 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
// CHECK: #[[SLICE_Q_LAYOUT:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
// CHECK: #[[SLICE_K_LAYOUT:.+]] = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>

// CHECK: %[[Q_VIEW_SLICE_1:.+]] = triton_gpu.view_slice %[[Q_PTR:.+]][0, 0] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[SLICE_Q_LAYOUT]]> to tensor<128x32x!tt.ptr<f16, 1>, #[[SLICE_Q_LAYOUT]]>
// CHECK: %[[LOAD_Q_1:.+]] = tt.load %[[Q_VIEW_SLICE_1]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #[[SLICE_Q_LAYOUT]]>
// CHECK: %[[K_VIEW_SLICE_1:.+]] = triton_gpu.view_slice %[[K_PTR:.+]][0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[SLICE_K_LAYOUT]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[SLICE_K_LAYOUT]]>
// CHECK: %[[LOAD_K_1:.+]] = tt.load %[[K_VIEW_SLICE_1]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[SLICE_K_LAYOUT]]>
// CHECK:  %[[QK_DOT_1:.+]] = tt.dot %[[QK_DOT_ARG_1:.+]], %[[QK_DOT_ARG_2:.+]], %[[QK_DOT_ARG_3:.+]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>

// CHECK: %[[Q_VIEW_SLICE_2:.+]] = triton_gpu.view_slice %[[Q_PTR:.+]][0, 32] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[SLICE_Q_LAYOUT]]> to tensor<128x32x!tt.ptr<f16, 1>, #[[SLICE_Q_LAYOUT]]>
// CHECK: %[[LOAD_Q_2:.+]] = tt.load %[[Q_VIEW_SLICE_2]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #[[SLICE_Q_LAYOUT]]>
// CHECK: %[[K_VIEW_SLICE_2:.+]] = triton_gpu.view_slice %[[K_PTR:.+]][32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[SLICE_K_LAYOUT]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[SLICE_K_LAYOUT]]>
// CHECK: %[[LOAD_K_2:.+]] = tt.load %[[K_VIEW_SLICE_2]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[SLICE_K_LAYOUT]]>
// CHECK: %[[QK_DOT_2:.+]] = tt.dot %[[QK_DOT_ARG_2:.+]], %[[QK_DOT_ARG_2:.+]], %[[QK_DOT_1]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>

// CHECK: %[[Q_VIEW_SLICE_3:.+]] = triton_gpu.view_slice %[[Q_PTR:.+]][0, 64] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[SLICE_Q_LAYOUT]]> to tensor<128x32x!tt.ptr<f16, 1>, #[[SLICE_Q_LAYOUT]]>
// CHECK: %[[LOAD_Q_3:.+]] = tt.load %[[Q_VIEW_SLICE_3]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #[[SLICE_Q_LAYOUT]]>
// CHECK: %[[K_VIEW_SLICE_3:.+]] = triton_gpu.view_slice %[[K_PTR:.+]][64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[SLICE_K_LAYOUT]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[SLICE_K_LAYOUT]]>
// CHECK: %[[LOAD_K_3:.+]] = tt.load %[[K_VIEW_SLICE_3]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[SLICE_K_LAYOUT]]>
// CHECK: %[[QK_DOT_3:.+]] = tt.dot %[[QK_DOT_ARG_3:.+]], %[[QK_DOT_ARG_3:.+]], %[[QK_DOT_2]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>

// CHECK: %[[Q_VIEW_SLICE_4:.+]] = triton_gpu.view_slice %[[Q_PTR:.+]][0, 96] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[SLICE_Q_LAYOUT]]> to tensor<128x32x!tt.ptr<f16, 1>, #[[SLICE_Q_LAYOUT]]>
// CHECK: %[[LOAD_Q_4:.+]] = tt.load %[[Q_VIEW_SLICE_4]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #[[SLICE_Q_LAYOUT]]>
// CHECK: %[[K_VIEW_SLICE_4:.+]] = triton_gpu.view_slice %[[K_PTR:.+]][96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[SLICE_K_LAYOUT]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[SLICE_K_LAYOUT]]>
// CHECK: %[[LOAD_K_4:.+]] = tt.load %[[K_VIEW_SLICE_4]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[SLICE_K_LAYOUT]]>
// CHECK: %[[QK_DOT_4:.+]] = tt.dot %[[QK_DOT_ARG_4:.+]], %[[QK_DOT_ARG_4:.+]], %[[QK_DOT_3]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>

// CHECK: %[[QK_VIEW_SLICE_1:.+]] = triton_gpu.view_slice %[[QK_TENSOR:.+]][0, 0] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
// CHECK: %[[QK_DOT_OP_1:.+]] = triton_gpu.convert_layout %[[QK_VIEW_SLICE_1]] : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: %[[V_VIEW_SLICE_1:.+]] = triton_gpu.view_slice %[[V_TENSOR_PTR:.+]][0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[SLICE_V_LAYOUT]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[SLICE_V_LAYOUT]]>
// CHECK: %[[LOAD_V_1:.+]] = tt.load %[[V_VIEW_SLICE_1]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[SLICE_V_LAYOUT]]>
// CHECK: %[[V_DOT_OP_1:.+]] = triton_gpu.convert_layout %[[LOAD_V_1]] : (tensor<32x128xf16, #[[SLICE_V_LAYOUT]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
// CHECK: %[[QKV_DOT_1:.+]] = tt.dot %[[QK_DOT_OP_1]], %[[V_DOT_OP_1]], %[[QKV_DOT_ACC_1:.+]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>

// CHECK: %[[QK_VIEW_SLICE_2:.+]] = triton_gpu.view_slice %[[QK_TENSOR:.+]][0, 32] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
// CHECK: %[[QK_DOT_OP_2:.+]] = triton_gpu.convert_layout %[[QK_VIEW_SLICE_2]] : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: %[[V_VIEW_SLICE_2:.+]] = triton_gpu.view_slice %[[V_TENSOR_PTR:.+]][32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[SLICE_V_LAYOUT]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[SLICE_V_LAYOUT]]>
// CHECK: %[[LOAD_V_2:.+]] = tt.load %[[V_VIEW_SLICE_2]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[SLICE_V_LAYOUT]]>
// CHECK: %[[V_DOT_OP_2:.+]] = triton_gpu.convert_layout %[[LOAD_V_2]] : (tensor<32x128xf16, #[[SLICE_V_LAYOUT]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
// CHECK: %[[QKV_DOT_2:.+]] = tt.dot %[[QK_DOT_OP_2]], %[[V_DOT_OP_2]], %[[QKV_DOT_1]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>

// CHECK: %[[QK_VIEW_SLICE_3:.+]] = triton_gpu.view_slice %[[QK_TENSOR:.+]][0, 64] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
// CHECK: %[[QK_DOT_OP_3:.+]] = triton_gpu.convert_layout %[[QK_VIEW_SLICE_3]] : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: %[[V_VIEW_SLICE_3:.+]] = triton_gpu.view_slice %[[V_TENSOR_PTR:.+]][64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[SLICE_V_LAYOUT]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[SLICE_V_LAYOUT]]>
// CHECK: %[[LOAD_V_3:.+]] = tt.load %[[V_VIEW_SLICE_3]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[SLICE_V_LAYOUT]]>
// CHECK: %[[V_DOT_OP_3:.+]] = triton_gpu.convert_layout %[[LOAD_V_3]] : (tensor<32x128xf16, #[[SLICE_V_LAYOUT]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
// CHECK: %[[QKV_DOT_3:.+]] = tt.dot %[[QK_DOT_OP_3]], %[[V_DOT_OP_3]], %[[QKV_DOT_2]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>

// CHECK: %[[QK_VIEW_SLICE_4:.+]] = triton_gpu.view_slice %[[QK_TENSOR:.+]][0, 96] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
// CHECK: %[[QK_DOT_OP_4:.+]] = triton_gpu.convert_layout %[[QK_VIEW_SLICE_4]] : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: %[[V_VIEW_SLICE_4:.+]] = triton_gpu.view_slice %[[V_TENSOR_PTR:.+]][96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[SLICE_V_LAYOUT]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[SLICE_V_LAYOUT]]>
// CHECK: %[[LOAD_V_4:.+]] = tt.load %[[V_VIEW_SLICE_4]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[SLICE_V_LAYOUT]]>
// CHECK: %[[V_DOT_OP_4:.+]] = triton_gpu.convert_layout %[[LOAD_V_4]] : (tensor<32x128xf16, #[[SLICE_V_LAYOUT]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
// CHECK: %[[QKV_DOT_4:.+]] = tt.dot %[[QK_DOT_OP_4]], %[[V_DOT_OP_4]], %[[QKV_DOT_3]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#mfma = #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [4, 1], isTransposed = true}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @_attn_fwd_0d1d2d34d5d6de7de8de9c10de11de12de13c14de15de16de17c18de19de20de21c2223de24de(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mfma>
    %c0_i64 = arith.constant 0 : i64
    %c128_i64 = arith.constant 128 : i64
    %cst_2 = arith.constant 1.44269502 : f32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %arg7 : i32
    %3 = tt.addptr %arg0, %2 : !tt.ptr<f16, 1>, i32
    %4 = arith.muli %0, %c128_i32 : i32
    %5 = arith.extsi %arg8 : i32 to i64
    %6 = arith.extsi %4 : i32 to i64
    %7 = tt.addptr %arg2, %2 : !tt.ptr<f16, 1>, i32
    %8 = arith.extsi %arg14 : i32 to i64
    %9 = tt.addptr %arg1, %2 : !tt.ptr<f16, 1>, i32
    %10 = arith.extsi %arg11 : i32 to i64
    %11 = tt.addptr %arg5, %2 : !tt.ptr<f16, 1>, i32
    %12 = arith.extsi %arg17 : i32 to i64
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %22 = tt.splat %4 : (i32) -> tensor<128xi32, #blocked2>
    %23 = arith.addi %22, %21 : tensor<128xi32, #blocked2>
    %24 = arith.mulf %arg3, %cst_2 : f32
    %25 = tt.splat %6 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %26 = tt.splat %6 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %27 = arith.extsi %13 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %28 = arith.extsi %14 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %29 = arith.extsi %15 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %30 = arith.extsi %16 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %31 = arith.extsi %17 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %32 = arith.extsi %18 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %33 = arith.extsi %19 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %34 = arith.extsi %20 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %35 = arith.addi %25, %27 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %36 = arith.addi %26, %28 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %37 = tt.expand_dims %35 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
    %38 = tt.expand_dims %36 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
    %39 = tt.splat %5 : (i64) -> tensor<128x1xi64, #blocked>
    %40 = arith.muli %37, %39 : tensor<128x1xi64, #blocked>
    %41 = tt.splat %3 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
    %42 = tt.addptr %41, %40 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
    %43 = tt.broadcast %42 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
    %44 = tt.expand_dims %29 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
    %45 = tt.expand_dims %30 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
    %46 = tt.expand_dims %31 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
    %47 = tt.broadcast %44 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
    %48 = tt.broadcast %45 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
    %49 = tt.broadcast %46 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
    %50 = tt.addptr %43, %47 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
    %51 = tt.load %50 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf16, #blocked>
    %52 = tt.splat %24 : (f32) -> tensor<128x128xf32, #blocked>
    %53 = arith.extf %51 : tensor<128x128xf16, #blocked> to tensor<128x128xf32, #blocked>
    %54 = arith.mulf %53, %52 : tensor<128x128xf32, #blocked>
    %55 = arith.truncf %54 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %56 = tt.expand_dims %32 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
    %57 = tt.splat %9 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
    %58 = tt.addptr %57, %56 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
    %59 = tt.broadcast %58 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
    %60 = tt.splat %10 : (i64) -> tensor<1x128xi64, #blocked1>
    %61 = tt.splat %8 : (i64) -> tensor<128x1xi64, #blocked>
    %62 = tt.splat %7 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
    %63:5 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg22 = %cst_1, %arg23 = %cst, %arg24 = %cst_0, %arg25 = %c0_i64, %arg26 = %c0_i64) -> (tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64)  : i32 {
      %82 = tt.splat %arg26 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %83 = arith.addi %82, %33 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %84 = tt.expand_dims %83 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
      %85 = arith.muli %84, %60 : tensor<1x128xi64, #blocked1>
      %86 = tt.broadcast %85 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
      %87 = tt.addptr %59, %86 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
      %88 = tt.load %87 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf16, #blocked1>
      %89 = triton_gpu.convert_layout %55 : (tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %90 = triton_gpu.convert_layout %88 : (tensor<128x128xf16, #blocked1>) -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %91 = tt.dot %89, %90, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %92 = "tt.reduce"(%91) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32, %arg28: f32):
        %120 = arith.maximumf %arg27, %arg28 : f32
        tt.reduce.return %120 : f32
      }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %93 = arith.maximumf %arg24, %92 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %94 = tt.expand_dims %93 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
      %95 = tt.broadcast %94 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %96 = arith.subf %91, %95 : tensor<128x128xf32, #mfma>
      %97 = tt.extern_elementwise %96 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x128xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %98 = arith.subf %arg24, %93 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %99 = tt.extern_elementwise %98 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %100 = tt.expand_dims %99 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
      %101 = tt.broadcast %100 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %102 = arith.mulf %arg22, %101 : tensor<128x128xf32, #mfma>
      %103 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %104 = arith.addi %103, %34 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %105 = tt.expand_dims %104 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
      %106 = arith.muli %105, %61 : tensor<128x1xi64, #blocked>
      %107 = tt.addptr %62, %106 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
      %108 = tt.broadcast %107 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
      %109 = tt.addptr %108, %48 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
      %110 = tt.load %109 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf16, #blocked>
      %111 = arith.truncf %97 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
      %112 = triton_gpu.convert_layout %111 : (tensor<128x128xf16, #mfma>) -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %113 = triton_gpu.convert_layout %110 : (tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %114 = tt.dot %112, %113, %102 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %115 = "tt.reduce"(%97) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32, %arg28: f32):
        %120 = arith.addf %arg27, %arg28 : f32
        tt.reduce.return %120 : f32
      }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %116 = arith.mulf %arg23, %99 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %117 = arith.addf %116, %115 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %118 = arith.addi %arg25, %c128_i64 : i64
      %119 = arith.addi %arg26, %c128_i64 : i64
      scf.yield %114, %117, %93, %118, %119 : tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64
    }
    %64 = tt.expand_dims %63#1 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
    %65 = tt.broadcast %64 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
    %66 = arith.divf %63#0, %65 : tensor<128x128xf32, #mfma>
    %67 = arith.muli %1, %arg20 : i32
    %68 = tt.addptr %arg4, %67 : !tt.ptr<f32, 1>, i32
    %69 = tt.splat %68 : (!tt.ptr<f32, 1>) -> tensor<128x!tt.ptr<f32, 1>, #blocked2>
    %70 = tt.addptr %69, %23 : tensor<128x!tt.ptr<f32, 1>, #blocked2>, tensor<128xi32, #blocked2>
    %71 = tt.extern_elementwise %63#1 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_log2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %72 = arith.addf %63#2, %71 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %73 = triton_gpu.convert_layout %72 : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #blocked2>
    tt.store %70, %73 {cache = 1 : i32, evict = 1 : i32} : tensor<128xf32, #blocked2>
    %74 = arith.truncf %66 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
    %75 = tt.splat %12 : (i64) -> tensor<128x1xi64, #blocked>
    %76 = arith.muli %38, %75 : tensor<128x1xi64, #blocked>
    %77 = tt.splat %11 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
    %78 = tt.addptr %77, %76 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
    %79 = tt.broadcast %78 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
    %80 = tt.addptr %79, %49 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
    %81 = triton_gpu.convert_layout %80 : (tensor<128x128x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #mfma>
    tt.store %81, %74 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #mfma>
    tt.return
  }
}
