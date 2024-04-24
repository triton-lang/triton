// RUN: triton-opt %s -split-input-file -tritonamdgpu-reorder-instructions | FileCheck %s


// CHECK: #[[BLOCKED:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
// CHECK: #[[BLOCKED2:.+]] = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
// CHECK: #[[MFMA:.+]] = #triton_gpu.mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
// CHECK: #[[SHARED:.+]] = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
// CHECK: #[[SHARED1:.+]] = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>

// CHECK-LABEL: @test_fa_reorder_dot_slicing_4_stages

// CHECK: %[[Q_VIEW_SLICE_0:.+]] = triton_gpu.view_slice %[[Q_PTR:.+]][0, 32] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED]]> to tensor<128x32x!tt.ptr<f16, 1>, #[[BLOCKED]]>
// CHECK-NEXT: %[[Q_VIEW_SLICE_1:.+]] = triton_gpu.view_slice %[[Q_PTR:.+]][0, 64] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED]]> to tensor<128x32x!tt.ptr<f16, 1>, #[[BLOCKED]]>
// CHECK-NEXT: %[[Q_VIEW_SLICE_2:.+]] = triton_gpu.view_slice %[[Q_PTR]][0, 96] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED]]> to tensor<128x32x!tt.ptr<f16, 1>, #[[BLOCKED]]>
// CHECK-NEXT: %[[Q_VIEW_SLICE_3:.+]] = triton_gpu.view_slice %[[Q_PTR]][0, 0] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED]]> to tensor<128x32x!tt.ptr<f16, 1>, #[[BLOCKED]]>

// CHECK-NEXT: %[[LOAD_Q_0:.+]] = tt.load %[[Q_VIEW_SLICE_0]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #[[BLOCKED]]>
// CHECK-NEXT: %[[LOAD_Q_1:.+]] = tt.load %[[Q_VIEW_SLICE_1]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #[[BLOCKED]]>
// CHECK-NEXT: %[[LOAD_Q_2:.+]] = tt.load %[[Q_VIEW_SLICE_2]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #[[BLOCKED]]>
// CHECK-NEXT: %[[LOAD_Q_3:.+]] = tt.load %[[Q_VIEW_SLICE_3]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #[[BLOCKED]]>

// CHECK: %[[Q_SHARED_0:.+]] = triton_gpu.convert_layout %[[TRUNCF_0:.+]] : (tensor<128x32xf16, #[[BLOCKED]]>) -> tensor<128x32xf16, #[[SHARED]]>
// CHECK-NEXT: %[[Q_SHARED_1:.+]] = triton_gpu.convert_layout %[[TRUNCF_1:.+]] : (tensor<128x32xf16, #[[BLOCKED]]>) -> tensor<128x32xf16, #[[SHARED]]>
// CHECK-NEXT: %[[Q_SHARED_2:.+]] = triton_gpu.convert_layout %[[TRUNCF_2:.+]] : (tensor<128x32xf16, #[[BLOCKED]]>) -> tensor<128x32xf16, #[[SHARED]]>
// CHECK-NEXT: %[[Q_SHARED_3:.+]] = triton_gpu.convert_layout %[[TRUNCF_3:.+]] : (tensor<128x32xf16, #[[BLOCKED]]>) -> tensor<128x32xf16, #[[SHARED]]>

// CHECK-NEXT: %[[Q_DOT_0:.+]] = triton_gpu.convert_layout %[[Q_SHARED_0]] : (tensor<128x32xf16, #[[SHARED]]>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[Q_DOT_1:.+]] = triton_gpu.convert_layout %[[Q_SHARED_1]] : (tensor<128x32xf16, #[[SHARED]]>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[Q_DOT_2:.+]] = triton_gpu.convert_layout %[[Q_SHARED_2]] : (tensor<128x32xf16, #[[SHARED]]>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[Q_DOT_3:.+]] = triton_gpu.convert_layout %[[Q_SHARED_3]] : (tensor<128x32xf16, #[[SHARED]]>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>

// CHECK: %[[K_VIEW_SLICE_0:.+]] = triton_gpu.view_slice %[[K_PTR:.+]][32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]>
// CHECK-NEXT: %[[K_VIEW_SLICE_1:.+]] = triton_gpu.view_slice %[[K_PTR]][64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]>
// CHECK-NEXT: %[[K_VIEW_SLICE_2:.+]] = triton_gpu.view_slice %[[K_PTR]][96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]>
// CHECK-NEXT: %[[K_VIEW_SLICE_3:.+]] = triton_gpu.view_slice %[[K_PTR]][0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]>

// CHECK-NEXT: %[[LOAD_K_0:.+]] = tt.load %[[K_VIEW_SLICE_0]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED2]]>
// CHECK-NEXT: %[[LOAD_K_1:.+]] = tt.load %[[K_VIEW_SLICE_1]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED2]]>
// CHECK-NEXT: %[[LOAD_K_2:.+]] = tt.load %[[K_VIEW_SLICE_2]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED2]]>
// CHECK-NEXT: %[[LOAD_K_3:.+]] = tt.load %[[K_VIEW_SLICE_3]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED2]]>

// CHECK-NEXT: %[[K_SHARED_0:.+]] = triton_gpu.convert_layout %[[LOAD_K_0]] : (tensor<32x128xf16, #[[BLOCKED2]]>) -> tensor<32x128xf16, #[[SHARED1]]>
// CHECK-NEXT: %[[K_SHARED_1:.+]] = triton_gpu.convert_layout %[[LOAD_K_1]] : (tensor<32x128xf16, #[[BLOCKED2]]>) -> tensor<32x128xf16, #[[SHARED1]]>
// CHECK-NEXT: %[[K_SHARED_2:.+]] = triton_gpu.convert_layout %[[LOAD_K_2]] : (tensor<32x128xf16, #[[BLOCKED2]]>) -> tensor<32x128xf16, #[[SHARED1]]>
// CHECK-NEXT: %[[K_SHARED_3:.+]] = triton_gpu.convert_layout %[[LOAD_K_3]] : (tensor<32x128xf16, #[[BLOCKED2]]>) -> tensor<32x128xf16, #[[SHARED1]]>

// CHECK-NEXT: %[[K_DOT_0:.+]] = triton_gpu.convert_layout %[[K_SHARED_0]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[K_DOT_1:.+]] = triton_gpu.convert_layout %[[K_SHARED_1]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[K_DOT_2:.+]] = triton_gpu.convert_layout %[[K_SHARED_2]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[K_DOT_3:.+]] = triton_gpu.convert_layout %[[K_SHARED_3]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>

// CHECK-NEXT:  %[[QK_DOT_0:.+]] = tt.dot %[[Q_DOT_3]], %[[K_DOT_3]], %[[acc_0:.+]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>> -> tensor<128x128xf32, #[[MFMA]]>
// CHECK-NEXT:  %[[QK_DOT_1:.+]] = tt.dot %[[Q_DOT_0]], %[[K_DOT_0]], %[[QK_DOT_0]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>> -> tensor<128x128xf32, #[[MFMA]]>
// CHECK-NEXT:  %[[QK_DOT_2:.+]] = tt.dot %[[Q_DOT_1]], %[[K_DOT_1]], %[[QK_DOT_1]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>> -> tensor<128x128xf32, #[[MFMA]]>
// CHECK-NEXT:  %[[QK_DOT_3:.+]] = tt.dot %[[Q_DOT_2]], %[[K_DOT_2]], %[[QK_DOT_2]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>> -> tensor<128x128xf32, #[[MFMA]]>

// CHECK: %[[V_VIEW_SLICE_0:.+]] = triton_gpu.view_slice %[[V_PTR:.+]][32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]>
// CHECK-NEXT: %[[V_VIEW_SLICE_1:.+]] = triton_gpu.view_slice %[[V_PTR]][64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]>
// CHECK-NEXT: %[[V_VIEW_SLICE_2:.+]] = triton_gpu.view_slice %[[V_PTR]][96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]>
// CHECK-NEXT: %[[V_VIEW_SLICE_3:.+]] = triton_gpu.view_slice %[[V_PTR]][0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED2]]>

// CHECK-NEXT: %[[LOAD_V_0:.+]] = tt.load %[[V_VIEW_SLICE_0]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED2]]>
// CHECK-NEXT: %[[LOAD_V_1:.+]] = tt.load %[[V_VIEW_SLICE_1]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED2]]>
// CHECK-NEXT: %[[LOAD_V_2:.+]] = tt.load %[[V_VIEW_SLICE_2]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED2]]>
// CHECK-NEXT: %[[LOAD_V_3:.+]] = tt.load %[[V_VIEW_SLICE_3]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED2]]>

// CHECK-NEXT: %[[V_SHARED_0:.+]] = triton_gpu.convert_layout %[[LOAD_V_0]] : (tensor<32x128xf16, #[[BLOCKED2]]>) -> tensor<32x128xf16, #[[SHARED1]]>
// CHECK-NEXT: %[[V_SHARED_1:.+]] = triton_gpu.convert_layout %[[LOAD_V_1]] : (tensor<32x128xf16, #[[BLOCKED2]]>) -> tensor<32x128xf16, #[[SHARED1]]>
// CHECK-NEXT: %[[V_SHARED_2:.+]] = triton_gpu.convert_layout %[[LOAD_V_2]] : (tensor<32x128xf16, #[[BLOCKED2]]>) -> tensor<32x128xf16, #[[SHARED1]]>
// CHECK-NEXT: %[[V_SHARED_3:.+]] = triton_gpu.convert_layout %[[LOAD_V_3]] : (tensor<32x128xf16, #[[BLOCKED2]]>) -> tensor<32x128xf16, #[[SHARED1]]>

// CHECK-NEXT: %[[V_DOT_0:.+]] = triton_gpu.convert_layout %[[V_SHARED_0]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[V_DOT_1:.+]] = triton_gpu.convert_layout %[[V_SHARED_1]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[V_DOT_2:.+]] = triton_gpu.convert_layout %[[V_SHARED_2]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[V_DOT_3:.+]] = triton_gpu.convert_layout %[[V_SHARED_3]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>

// CHECK-NEXT:  %[[QKV_DOT_0:.+]] = tt.dot %[[QK_DOT_SOFTMAX_3:.+]], %[[V_DOT_3]], %[[acc_1:.+]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>> -> tensor<128x128xf32, #[[MFMA]]>
// CHECK-NEXT:  %[[QKV_DOT_1:.+]] = tt.dot %[[QK_DOT_SOFTMAX_0:.+]], %[[V_DOT_0]], %[[QKV_DOT_0]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>> -> tensor<128x128xf32, #[[MFMA]]>
// CHECK-NEXT:  %[[QKV_DOT_2:.+]] = tt.dot %[[QK_DOT_SOFTMAX_1:.+]], %[[V_DOT_1]], %[[QKV_DOT_1]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>> -> tensor<128x128xf32, #[[MFMA]]>
// CHECK-NEXT:  %[[QKV_DOT_3:.+]] = tt.dot %[[QK_DOT_SOFTMAX_2:.+]], %[[V_DOT_2]], %[[QKV_DOT_2]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>> -> tensor<128x128xf32, #[[MFMA]]>

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#mfma = #triton_gpu.mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @test_fa_reorder_dot_slicing_4_stages(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mfma>
    %c128_i64 = arith.constant 128 : i64
    %c0_i64 = arith.constant 0 : i64
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
    %7 = tt.addptr %arg1, %2 : !tt.ptr<f16, 1>, i32
    %8 = arith.extsi %arg11 : i32 to i64
    %9 = tt.addptr %arg2, %2 : !tt.ptr<f16, 1>, i32
    %10 = arith.extsi %arg14 : i32 to i64
    %11 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked3>
    %20 = tt.splat %4 : (i32) -> tensor<128xi32, #blocked3>
    %21 = arith.addi %20, %19 : tensor<128xi32, #blocked3>
    %22 = arith.mulf %arg3, %cst_2 : f32
    %23 = tt.splat %6 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %24 = tt.splat %6 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %25 = arith.extsi %11 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %26 = arith.extsi %12 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %27 = arith.extsi %13 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %28 = arith.extsi %14 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %29 = arith.extsi %15 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %30 = arith.extsi %16 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %31 = arith.extsi %17 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %32 = arith.extsi %18 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %33 = arith.addi %23, %25 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %34 = arith.addi %24, %26 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %35 = tt.expand_dims %33 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
    %36 = tt.expand_dims %34 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
    %37 = tt.splat %5 : (i64) -> tensor<128x1xi64, #blocked>
    %38 = arith.muli %35, %37 : tensor<128x1xi64, #blocked>
    %39 = tt.splat %3 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
    %40 = tt.addptr %39, %38 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
    %41 = tt.broadcast %40 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
    %42 = tt.expand_dims %27 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
    %43 = tt.expand_dims %28 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
    %44 = tt.expand_dims %29 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi64, #blocked2>
    %45 = tt.broadcast %42 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
    %46 = tt.broadcast %43 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
    %47 = tt.addptr %41, %45 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
    %48 = tt.splat %22 : (f32) -> tensor<128x128xf32, #blocked>
    %49 = tt.splat %22 : (f32) -> tensor<128x128xf32, #blocked>
    %50 = tt.splat %22 : (f32) -> tensor<128x128xf32, #blocked>
    %51 = tt.splat %22 : (f32) -> tensor<128x128xf32, #blocked>
    %52 = tt.expand_dims %30 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi64, #blocked2>
    %53 = tt.splat %7 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked2>
    %54 = tt.addptr %53, %52 : tensor<128x1x!tt.ptr<f16, 1>, #blocked2>, tensor<128x1xi64, #blocked2>
    %55 = tt.broadcast %54 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked2>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked2>
    %56 = tt.splat %8 : (i64) -> tensor<1x128xi64, #blocked2>
    %57 = tt.splat %9 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked2>
    %58 = tt.splat %10 : (i64) -> tensor<1x128xi64, #blocked2>
    %59 = arith.muli %44, %58 : tensor<1x128xi64, #blocked2>
    %60 = tt.broadcast %59 : (tensor<1x128xi64, #blocked2>) -> tensor<128x128xi64, #blocked2>
    %61:5 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg22 = %cst_1, %arg23 = %cst, %arg24 = %cst_0, %arg25 = %c0_i64, %arg26 = %c0_i64) -> (tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64)  : i32 {
      %82 = tt.splat %arg26 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %83 = arith.addi %82, %31 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %84 = tt.expand_dims %83 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi64, #blocked2>
      %85 = arith.muli %84, %56 : tensor<1x128xi64, #blocked2>
      %86 = tt.broadcast %85 : (tensor<1x128xi64, #blocked2>) -> tensor<128x128xi64, #blocked2>
      %87 = tt.addptr %55, %86 : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi64, #blocked2>
      %88 = triton_gpu.view_slice %47[0, 0] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
      %89 = tt.load %88 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
      %90 = arith.extf %89 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
      %91 = triton_gpu.view_slice %48[0, 0] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
      %92 = arith.mulf %90, %91 : tensor<128x32xf32, #blocked>
      %93 = arith.truncf %92 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
      %94 = triton_gpu.convert_layout %93 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
      %95 = triton_gpu.convert_layout %94 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %96 = triton_gpu.view_slice %87[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<32x128x!tt.ptr<f16, 1>, #blocked2>
      %97 = tt.load %96 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked2>
      %98 = triton_gpu.convert_layout %97 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared1>
      %99 = triton_gpu.convert_layout %98 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %100 = tt.dot %95, %99, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %101 = triton_gpu.view_slice %47[0, 32] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
      %102 = tt.load %101 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
      %103 = arith.extf %102 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
      %104 = triton_gpu.view_slice %49[0, 32] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
      %105 = arith.mulf %103, %104 : tensor<128x32xf32, #blocked>
      %106 = arith.truncf %105 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
      %107 = triton_gpu.convert_layout %106 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
      %108 = triton_gpu.convert_layout %107 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %109 = triton_gpu.view_slice %87[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<32x128x!tt.ptr<f16, 1>, #blocked2>
      %110 = tt.load %109 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked2>
      %111 = triton_gpu.convert_layout %110 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared1>
      %112 = triton_gpu.convert_layout %111 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %113 = tt.dot %108, %112, %100 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %114 = triton_gpu.view_slice %47[0, 64] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
      %115 = tt.load %114 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
      %116 = arith.extf %115 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
      %117 = triton_gpu.view_slice %50[0, 64] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
      %118 = arith.mulf %116, %117 : tensor<128x32xf32, #blocked>
      %119 = arith.truncf %118 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
      %120 = triton_gpu.convert_layout %119 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
      %121 = triton_gpu.convert_layout %120 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %122 = triton_gpu.view_slice %87[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<32x128x!tt.ptr<f16, 1>, #blocked2>
      %123 = tt.load %122 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked2>
      %124 = triton_gpu.convert_layout %123 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared1>
      %125 = triton_gpu.convert_layout %124 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %126 = tt.dot %121, %125, %113 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %127 = triton_gpu.view_slice %47[0, 96] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
      %128 = tt.load %127 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
      %129 = arith.extf %128 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
      %130 = triton_gpu.view_slice %51[0, 96] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
      %131 = arith.mulf %129, %130 : tensor<128x32xf32, #blocked>
      %132 = arith.truncf %131 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
      %133 = triton_gpu.convert_layout %132 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
      %134 = triton_gpu.convert_layout %133 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %135 = triton_gpu.view_slice %87[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<32x128x!tt.ptr<f16, 1>, #blocked2>
      %136 = tt.load %135 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked2>
      %137 = triton_gpu.convert_layout %136 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared1>
      %138 = triton_gpu.convert_layout %137 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %139 = tt.dot %134, %138, %126 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %140 = "tt.reduce"(%139) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32, %arg28: f32):
        %191 = arith.maximumf %arg27, %arg28 : f32
        tt.reduce.return %191 : f32
      }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %141 = arith.maximumf %arg24, %140 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %142 = tt.expand_dims %141 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
      %143 = tt.broadcast %142 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %144 = arith.subf %139, %143 : tensor<128x128xf32, #mfma>
      %145 = tt.extern_elementwise %144 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x128xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %146 = arith.subf %arg24, %141 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %147 = tt.extern_elementwise %146 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %148 = tt.expand_dims %147 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
      %149 = tt.broadcast %148 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %150 = arith.mulf %arg22, %149 : tensor<128x128xf32, #mfma>
      %151 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %152 = arith.addi %151, %32 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %153 = tt.expand_dims %152 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi64, #blocked2>
      %154 = tt.addptr %57, %153 : tensor<128x1x!tt.ptr<f16, 1>, #blocked2>, tensor<128x1xi64, #blocked2>
      %155 = tt.broadcast %154 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked2>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked2>
      %156 = tt.addptr %155, %60 : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi64, #blocked2>
      %157 = arith.truncf %145 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
      %158 = triton_gpu.view_slice %157[0, 0] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %159 = triton_gpu.convert_layout %158 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %160 = triton_gpu.view_slice %156[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<32x128x!tt.ptr<f16, 1>, #blocked2>
      %161 = tt.load %160 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked2>
      %162 = triton_gpu.convert_layout %161 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared1>
      %163 = triton_gpu.convert_layout %162 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %164 = tt.dot %159, %163, %150 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %165 = triton_gpu.view_slice %157[0, 32] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %166 = triton_gpu.convert_layout %165 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %167 = triton_gpu.view_slice %156[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<32x128x!tt.ptr<f16, 1>, #blocked2>
      %168 = tt.load %167 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked2>
      %169 = triton_gpu.convert_layout %168 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared1>
      %170 = triton_gpu.convert_layout %169 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %171 = tt.dot %166, %170, %164 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %172 = triton_gpu.view_slice %157[0, 64] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %173 = triton_gpu.convert_layout %172 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %174 = triton_gpu.view_slice %156[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<32x128x!tt.ptr<f16, 1>, #blocked2>
      %175 = tt.load %174 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked2>
      %176 = triton_gpu.convert_layout %175 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared1>
      %177 = triton_gpu.convert_layout %176 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %178 = tt.dot %173, %177, %171 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %179 = triton_gpu.view_slice %157[0, 96] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %180 = triton_gpu.convert_layout %179 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %181 = triton_gpu.view_slice %156[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked2> to tensor<32x128x!tt.ptr<f16, 1>, #blocked2>
      %182 = tt.load %181 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked2>
      %183 = triton_gpu.convert_layout %182 : (tensor<32x128xf16, #blocked2>) -> tensor<32x128xf16, #shared1>
      %184 = triton_gpu.convert_layout %183 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %185 = tt.dot %180, %184, %178 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %186 = "tt.reduce"(%145) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32, %arg28: f32):
        %191 = arith.addf %arg27, %arg28 : f32
        tt.reduce.return %191 : f32
      }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %187 = arith.mulf %arg23, %147 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %188 = arith.addf %187, %186 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %189 = arith.addi %arg25, %c128_i64 : i64
      %190 = arith.addi %arg26, %c128_i64 : i64
      scf.yield %185, %188, %141, %189, %190 : tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64
    }
    %62 = tt.expand_dims %61#1 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
    %63 = tt.broadcast %62 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
    %64 = arith.divf %61#0, %63 : tensor<128x128xf32, #mfma>
    %65 = arith.muli %1, %arg20 : i32
    %66 = tt.addptr %arg4, %65 : !tt.ptr<f32, 1>, i32
    %67 = tt.splat %66 : (!tt.ptr<f32, 1>) -> tensor<128x!tt.ptr<f32, 1>, #blocked3>
    %68 = tt.addptr %67, %21 : tensor<128x!tt.ptr<f32, 1>, #blocked3>, tensor<128xi32, #blocked3>
    %69 = tt.extern_elementwise %61#1 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_log2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %70 = arith.addf %61#2, %69 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %71 = triton_gpu.convert_layout %70 : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #blocked3>
    tt.store %68, %71 {cache = 1 : i32, evict = 1 : i32} : tensor<128xf32, #blocked3>
    %72 = tt.addptr %arg5, %2 : !tt.ptr<f16, 1>, i32
    %73 = arith.extsi %arg17 : i32 to i64
    %74 = arith.truncf %64 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
    %75 = tt.splat %73 : (i64) -> tensor<128x1xi64, #blocked1>
    %76 = arith.muli %36, %75 : tensor<128x1xi64, #blocked1>
    %77 = tt.splat %72 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
    %78 = tt.addptr %77, %76 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
    %79 = tt.broadcast %78 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
    %80 = tt.addptr %79, %46 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
    %81 = triton_gpu.convert_layout %74 : (tensor<128x128xf16, #mfma>) -> tensor<128x128xf16, #blocked1>
    tt.store %80, %81 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #blocked1>
    tt.return
  }
}


// -----


// CHECK: #[[BLOCKED:.+]] = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
// CHECK: #[[BLOCKED1:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
// CHECK: #[[MFMA:.+]] = #triton_gpu.mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
// CHCEK: #[[SHARED:.+]] = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
// CHECK: #[[SHARED1:.+]] = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>

// CHECK-LABEL: @test_reordering_with_boundary_check

// CHECK: %[[Q_PTR:.+]] = tt.addptr %[[PTR_ARG_0:.+]], %[[PTR_ARG_1:.+]] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED1]]>, tensor<128x128xi64, #[[BLOCKED1]]>
// CHECK: %[[Q_VIEW_SLICE_0:.+]] = triton_gpu.view_slice %[[Q_PTR]][0, 32] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED1]]> to tensor<128x32x!tt.ptr<f16, 1>, #[[BLOCKED1]]>
// CHECK-NEXT: %[[Q_VIEW_SLICE_1:.+]] = triton_gpu.view_slice %[[Q_PTR]][0, 64] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED1]]> to tensor<128x32x!tt.ptr<f16, 1>, #[[BLOCKED1]]>
// CHECK-NEXT: %[[Q_VIEW_SLICE_2:.+]] = triton_gpu.view_slice %[[Q_PTR]][0, 96] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED1]]> to tensor<128x32x!tt.ptr<f16, 1>, #[[BLOCKED1]]>
// CHECK-NEXT: %[[Q_VIEW_SLICE_3:.+]] = triton_gpu.view_slice %[[Q_PTR]][0, 0] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED1]]> to tensor<128x32x!tt.ptr<f16, 1>, #[[BLOCKED1]]>

// CHECK: %[[Q_BOUNDARY_CHECK_0:.+]] = triton_gpu.view_slice %[[Q_BC_OPERAND:.+]][0, 32] [128, 32] [1, 1] : tensor<128x128xi1, #[[BLOCKED1]]> to tensor<128x32xi1, #[[BLOCKED1]]>
// CHECK-NEXT: %[[Q_BOUNDARY_CHECK_1:.+]] = triton_gpu.view_slice %[[Q_BC_OPERAND]][0, 64] [128, 32] [1, 1] : tensor<128x128xi1, #[[BLOCKED1]]> to tensor<128x32xi1, #[[BLOCKED1]]>
// CHECK-NEXT: %[[Q_BOUNDARY_CHECK_2:.+]] = triton_gpu.view_slice %[[Q_BC_OPERAND]][0, 96] [128, 32] [1, 1] : tensor<128x128xi1, #[[BLOCKED1]]> to tensor<128x32xi1, #[[BLOCKED1]]>
// CHECK-NEXT: %[[Q_BOUNDARY_CHECK_3:.+]] = triton_gpu.view_slice %[[Q_BC_OPERAND]][0, 0] [128, 32] [1, 1] : tensor<128x128xi1, #[[BLOCKED1]]> to tensor<128x32xi1, #[[BLOCKED1]]>

// CHECK-NEXT: %[[LOAD_Q_0:.+]] = tt.load %[[Q_VIEW_SLICE_0]], %[[Q_BOUNDARY_CHECK_0]], %[[Q_PADDING_0:.+]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #[[BLOCKED1]]>
// CHECK-NEXT: %[[LOAD_Q_1:.+]] = tt.load %[[Q_VIEW_SLICE_1]], %[[Q_BOUNDARY_CHECK_1]], %[[Q_PADDING_1:.+]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #[[BLOCKED1]]>
// CHECK-NEXT: %[[LOAD_Q_2:.+]] = tt.load %[[Q_VIEW_SLICE_2]], %[[Q_BOUNDARY_CHECK_2]], %[[Q_PADDING_2:.+]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #[[BLOCKED1]]>
// CHECK-NEXT: %[[LOAD_Q_3:.+]] = tt.load %[[Q_VIEW_SLICE_3]], %[[Q_BOUNDARY_CHECK_3]], %[[Q_PADDING_3:.+]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #[[BLOCKED1]]>

// CHECK: %[[Q_SHARED_0:.+]] = triton_gpu.convert_layout %[[TRUNCF_0:.+]] : (tensor<128x32xf16, #[[BLOCKED1]]>) -> tensor<128x32xf16, #[[SHARED]]>
// CHECK-NEXT: %[[Q_SHARED_1:.+]] = triton_gpu.convert_layout %[[TRUNCF_1:.+]] : (tensor<128x32xf16, #[[BLOCKED1]]>) -> tensor<128x32xf16, #[[SHARED]]>
// CHECK-NEXT: %[[Q_SHARED_2:.+]] = triton_gpu.convert_layout %[[TRUNCF_2:.+]] : (tensor<128x32xf16, #[[BLOCKED1]]>) -> tensor<128x32xf16, #[[SHARED]]>
// CHECK-NEXT: %[[Q_SHARED_3:.+]] = triton_gpu.convert_layout %[[TRUNCF_3:.+]] : (tensor<128x32xf16, #[[BLOCKED1]]>) -> tensor<128x32xf16, #[[SHARED]]>

// CHECK-NEXT: %[[Q_DOT_0:.+]] = triton_gpu.convert_layout %[[Q_SHARED_0]] : (tensor<128x32xf16, #[[SHARED]]>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[Q_DOT_1:.+]] = triton_gpu.convert_layout %[[Q_SHARED_1]] : (tensor<128x32xf16, #[[SHARED]]>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[Q_DOT_2:.+]] = triton_gpu.convert_layout %[[Q_SHARED_2]] : (tensor<128x32xf16, #[[SHARED]]>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[Q_DOT_3:.+]] = triton_gpu.convert_layout %[[Q_SHARED_3]] : (tensor<128x32xf16, #[[SHARED]]>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>

// CHECK: %[[K_VIEW_SLICE_0:.+]] = triton_gpu.view_slice %[[K_PTR:.+]][32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED]]>
// CHECK-NEXT: %[[K_VIEW_SLICE_1:.+]] = triton_gpu.view_slice %[[K_PTR]][64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED]]>
// CHECK-NEXT: %[[K_VIEW_SLICE_2:.+]] = triton_gpu.view_slice %[[K_PTR]][96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED]]>
// CHECK-NEXT: %[[K_VIEW_SLICE_3:.+]] = triton_gpu.view_slice %[[K_PTR]][0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED]]>

// CHECK: %[[K_BOUNDARY_CHECK_0:.+]] = triton_gpu.view_slice %[[K_BC_OPERAND:.+]][32, 0] [32, 128] [1, 1] : tensor<128x128xi1, #[[BLOCKED]]> to tensor<32x128xi1, #[[BLOCKED]]>
// CHECK-NEXT: %[[K_BOUNDARY_CHECK_1:.+]] = triton_gpu.view_slice %[[K_BC_OPERAND]][64, 0] [32, 128] [1, 1] : tensor<128x128xi1, #[[BLOCKED]]> to tensor<32x128xi1, #[[BLOCKED]]>
// CHECK-NEXT: %[[K_BOUNDARY_CHECK_2:.+]] = triton_gpu.view_slice %[[K_BC_OPERAND]][96, 0] [32, 128] [1, 1] : tensor<128x128xi1, #[[BLOCKED]]> to tensor<32x128xi1, #[[BLOCKED]]>
// CHECK-NEXT: %[[K_BOUNDARY_CHECK_3:.+]] = triton_gpu.view_slice %[[K_BC_OPERAND]][0, 0] [32, 128] [1, 1] : tensor<128x128xi1, #[[BLOCKED]]> to tensor<32x128xi1, #[[BLOCKED]]>

// CHECK: %[[LOAD_K_0:.+]] = tt.load %[[K_VIEW_SLICE_0]], %[[K_BOUNDARY_CHECK_0]], %[[K_PADDING_0:.+]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED]]>
// CHECK-NEXT: %[[LOAD_K_1:.+]] = tt.load %[[K_VIEW_SLICE_1]], %[[K_BOUNDARY_CHECK_1]], %[[K_PADDING_1:.+]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED]]>
// CHECK-NEXT: %[[LOAD_K_2:.+]] = tt.load %[[K_VIEW_SLICE_2]], %[[K_BOUNDARY_CHECK_2]], %[[K_PADDING_2:.+]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED]]>
// CHECK-NEXT: %[[LOAD_K_3:.+]] = tt.load %[[K_VIEW_SLICE_3]], %[[K_BOUNDARY_CHECK_3]], %[[K_PADDING_3:.+]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED]]>

// CHECK-NEXT: %[[K_SHARED_0:.+]] = triton_gpu.convert_layout %[[LOAD_K_0]] : (tensor<32x128xf16, #[[BLOCKED]]>) -> tensor<32x128xf16, #[[SHARED1]]>
// CHECK-NEXT: %[[K_SHARED_1:.+]] = triton_gpu.convert_layout %[[LOAD_K_1]] : (tensor<32x128xf16, #[[BLOCKED]]>) -> tensor<32x128xf16, #[[SHARED1]]>
// CHECK-NEXT: %[[K_SHARED_2:.+]] = triton_gpu.convert_layout %[[LOAD_K_2]] : (tensor<32x128xf16, #[[BLOCKED]]>) -> tensor<32x128xf16, #[[SHARED1]]>
// CHECK-NEXT: %[[K_SHARED_3:.+]] = triton_gpu.convert_layout %[[LOAD_K_3]] : (tensor<32x128xf16, #[[BLOCKED]]>) -> tensor<32x128xf16, #[[SHARED1]]>

// CHECK-NEXT: %[[K_DOT_0:.+]] = triton_gpu.convert_layout %[[K_SHARED_0]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[K_DOT_1:.+]] = triton_gpu.convert_layout %[[K_SHARED_1]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[K_DOT_2:.+]] = triton_gpu.convert_layout %[[K_SHARED_2]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[K_DOT_3:.+]] = triton_gpu.convert_layout %[[K_SHARED_3]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>

// CHECK: %[[V_VIEW_SLICE_0:.+]] = triton_gpu.view_slice %[[V_PTR:.+]][32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED]]>
// CHECK-NEXT: %[[V_VIEW_SLICE_1:.+]] = triton_gpu.view_slice %[[V_PTR]][64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED]]>
// CHECK-NEXT: %[[V_VIEW_SLICE_2:.+]] = triton_gpu.view_slice %[[V_PTR]][96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED]]>
// CHECK-NEXT: %[[V_VIEW_SLICE_3:.+]] = triton_gpu.view_slice %[[V_PTR]][0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #[[BLOCKED]]> to tensor<32x128x!tt.ptr<f16, 1>, #[[BLOCKED]]>

// CHECK: %[[V_BOUNDARY_CHECK_0:.+]] = triton_gpu.view_slice %[[K_BC_OPERAND:.+]][32, 0] [32, 128] [1, 1] : tensor<128x128xi1, #[[BLOCKED]]> to tensor<32x128xi1, #[[BLOCKED]]>
// CHECK-NEXT: %[[V_BOUNDARY_CHECK_1:.+]] = triton_gpu.view_slice %[[K_BC_OPERAND]][64, 0] [32, 128] [1, 1] : tensor<128x128xi1, #[[BLOCKED]]> to tensor<32x128xi1, #[[BLOCKED]]>
// CHECK-NEXT: %[[V_BOUNDARY_CHECK_2:.+]] = triton_gpu.view_slice %[[K_BC_OPERAND]][96, 0] [32, 128] [1, 1] : tensor<128x128xi1, #[[BLOCKED]]> to tensor<32x128xi1, #[[BLOCKED]]>
// CHECK-NEXT: %[[V_BOUNDARY_CHECK_3:.+]] = triton_gpu.view_slice %[[K_BC_OPERAND]][0, 0] [32, 128] [1, 1] : tensor<128x128xi1, #[[BLOCKED]]> to tensor<32x128xi1, #[[BLOCKED]]>

// CHECK: %[[LOAD_V_0:.+]] = tt.load %[[V_VIEW_SLICE_0]], %[[V_BOUNDARY_CHECK_0]], %[[V_PADDING_0:.+]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED]]>
// CHECK-NEXT: %[[LOAD_V_1:.+]] = tt.load %[[V_VIEW_SLICE_1]], %[[V_BOUNDARY_CHECK_1]], %[[V_PADDING_1:.+]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED]]>
// CHECK-NEXT: %[[LOAD_V_2:.+]] = tt.load %[[V_VIEW_SLICE_2]], %[[V_BOUNDARY_CHECK_2]], %[[V_PADDING_2:.+]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED]]>
// CHECK-NEXT: %[[LOAD_V_3:.+]] = tt.load %[[V_VIEW_SLICE_3]], %[[V_BOUNDARY_CHECK_3]], %[[V_PADDING_3:.+]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #[[BLOCKED]]>

// CHECK-NEXT: %[[V_SHARED_0:.+]] = triton_gpu.convert_layout %[[LOAD_V_0]] : (tensor<32x128xf16, #[[BLOCKED]]>) -> tensor<32x128xf16, #[[SHARED1]]>
// CHECK-NEXT: %[[V_SHARED_1:.+]] = triton_gpu.convert_layout %[[LOAD_V_1]] : (tensor<32x128xf16, #[[BLOCKED]]>) -> tensor<32x128xf16, #[[SHARED1]]>
// CHECK-NEXT: %[[V_SHARED_2:.+]] = triton_gpu.convert_layout %[[LOAD_V_2]] : (tensor<32x128xf16, #[[BLOCKED]]>) -> tensor<32x128xf16, #[[SHARED1]]>
// CHECK-NEXT: %[[V_SHARED_3:.+]] = triton_gpu.convert_layout %[[LOAD_V_3]] : (tensor<32x128xf16, #[[BLOCKED]]>) -> tensor<32x128xf16, #[[SHARED1]]>

// CHECK-NEXT: %[[V_DOT_0:.+]] = triton_gpu.convert_layout %[[V_SHARED_0]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[V_DOT_1:.+]] = triton_gpu.convert_layout %[[V_SHARED_1]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[V_DOT_2:.+]] = triton_gpu.convert_layout %[[V_SHARED_2]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>
// CHECK-NEXT: %[[V_DOT_3:.+]] = triton_gpu.convert_layout %[[V_SHARED_3]] : (tensor<32x128xf16, #[[SHARED1]]>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>

// CHECK-NEXT:  %[[QKV_DOT_0:.+]] = tt.dot %[[QK_DOT_SOFTMAX_3:.+]], %[[V_DOT_3]], %[[acc_1:.+]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>> -> tensor<128x128xf32, #[[MFMA]]>
// CHECK-NEXT:  %[[QKV_DOT_1:.+]] = tt.dot %[[QK_DOT_SOFTMAX_0:.+]], %[[V_DOT_0]], %[[QKV_DOT_0]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>> -> tensor<128x128xf32, #[[MFMA]]>
// CHECK-NEXT:  %[[QKV_DOT_2:.+]] = tt.dot %[[QK_DOT_SOFTMAX_1:.+]], %[[V_DOT_1]], %[[QKV_DOT_1]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>> -> tensor<128x128xf32, #[[MFMA]]>
// CHECK-NEXT:  %[[QKV_DOT_3:.+]] = tt.dot %[[QK_DOT_SOFTMAX_2:.+]], %[[V_DOT_2]], %[[QKV_DOT_2]] {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MFMA]], kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MFMA]], kWidth = 4}>> -> tensor<128x128xf32, #[[MFMA]]>

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#mfma = #triton_gpu.mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @test_reordering_with_boundary_check(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<128> : tensor<128x1xi64, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<1x128xi64, #blocked1>
    %cst_1 = arith.constant dense<128> : tensor<1x128xi64, #blocked>
    %cst_2 = arith.constant dense<0> : tensor<1x128xi64, #blocked1>
    %cst_3 = arith.constant dense<0> : tensor<1x128xi64, #blocked>
    %cst_4 = arith.constant dense<0> : tensor<128x1xi64, #blocked1>
    %cst_5 = arith.constant dense<0> : tensor<128x1xi64, #blocked>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked1>
    %cst_8 = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %cst_9 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mfma>
    %c128_i64 = arith.constant 128 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst_11 = arith.constant 1.44269502 : f32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %arg7 : i32
    %3 = tt.addptr %arg0, %2 : !tt.ptr<f16, 1>, i32
    %4 = arith.muli %0, %c128_i32 : i32
    %5 = arith.extsi %arg20 : i32 to i64
    %6 = arith.extsi %arg8 : i32 to i64
    %7 = arith.extsi %4 : i32 to i64
    %8 = tt.addptr %arg1, %2 : !tt.ptr<f16, 1>, i32
    %9 = arith.extsi %arg11 : i32 to i64
    %10 = tt.addptr %arg2, %2 : !tt.ptr<f16, 1>, i32
    %11 = arith.extsi %arg14 : i32 to i64
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %22 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %23 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %24 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %25 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %26 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked3>
    %27 = tt.splat %4 : (i32) -> tensor<128xi32, #blocked3>
    %28 = arith.addi %27, %26 : tensor<128xi32, #blocked3>
    %29 = arith.mulf %arg3, %cst_11 : f32
    %30 = tt.splat %7 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %31 = tt.splat %7 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %32 = tt.splat %7 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %33 = arith.extsi %12 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %34 = arith.extsi %13 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %35 = arith.extsi %14 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %36 = arith.extsi %15 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %37 = arith.extsi %16 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %38 = arith.extsi %17 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %39 = arith.extsi %18 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %40 = arith.extsi %19 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %41 = arith.extsi %20 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %42 = arith.extsi %21 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %43 = arith.extsi %22 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %44 = arith.extsi %23 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %45 = arith.extsi %24 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %46 = arith.extsi %25 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %47 = arith.addi %30, %33 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %48 = arith.addi %31, %34 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %49 = arith.addi %32, %35 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %50 = tt.expand_dims %47 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
    %51 = tt.expand_dims %48 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
    %52 = tt.expand_dims %49 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi64, #blocked2>
    %53 = tt.splat %6 : (i64) -> tensor<128x1xi64, #blocked1>
    %54 = arith.muli %50, %53 : tensor<128x1xi64, #blocked1>
    %55 = tt.splat %3 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
    %56 = tt.addptr %55, %54 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
    %57 = tt.broadcast %56 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
    %58 = tt.expand_dims %36 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
    %59 = tt.expand_dims %37 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi64, #blocked2>
    %60 = tt.expand_dims %38 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
    %61 = tt.expand_dims %39 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
    %62 = tt.expand_dims %40 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
    %63 = tt.broadcast %58 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
    %64 = tt.broadcast %59 : (tensor<1x128xi64, #blocked2>) -> tensor<128x128xi64, #blocked2>
    %65 = tt.addptr %57, %63 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
    %66 = arith.cmpi sge, %51, %cst_4 : tensor<128x1xi64, #blocked1>
    %67 = tt.splat %5 : (i64) -> tensor<128x1xi64, #blocked1>
    %68 = tt.splat %5 : (i64) -> tensor<128x1xi64, #blocked>
    %69 = arith.cmpi slt, %51, %67 : tensor<128x1xi64, #blocked1>
    %70 = arith.andi %66, %69 : tensor<128x1xi1, #blocked1>
    %71 = tt.broadcast %70 : (tensor<128x1xi1, #blocked1>) -> tensor<128x128xi1, #blocked1>
    %72 = arith.cmpi sge, %60, %cst_2 : tensor<1x128xi64, #blocked1>
    %73 = arith.cmpi sge, %61, %cst_3 : tensor<1x128xi64, #blocked>
    %74 = arith.cmpi slt, %60, %cst_0 : tensor<1x128xi64, #blocked1>
    %75 = arith.cmpi slt, %61, %cst_1 : tensor<1x128xi64, #blocked>
    %76 = arith.andi %72, %74 : tensor<1x128xi1, #blocked1>
    %77 = arith.andi %73, %75 : tensor<1x128xi1, #blocked>
    %78 = tt.broadcast %76 : (tensor<1x128xi1, #blocked1>) -> tensor<128x128xi1, #blocked1>
    %79 = tt.broadcast %77 : (tensor<1x128xi1, #blocked>) -> tensor<128x128xi1, #blocked>
    %80 = arith.andi %71, %78 : tensor<128x128xi1, #blocked1>
    %81 = tt.splat %29 : (f32) -> tensor<128x128xf32, #blocked1>
    %82 = tt.splat %29 : (f32) -> tensor<128x128xf32, #blocked1>
    %83 = tt.splat %29 : (f32) -> tensor<128x128xf32, #blocked1>
    %84 = tt.splat %29 : (f32) -> tensor<128x128xf32, #blocked1>
    %85 = tt.expand_dims %41 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
    %86 = tt.expand_dims %42 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
    %87 = tt.splat %8 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
    %88 = tt.addptr %87, %85 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
    %89 = tt.broadcast %88 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
    %90 = tt.splat %9 : (i64) -> tensor<1x128xi64, #blocked>
    %91 = arith.cmpi sge, %86, %cst_5 : tensor<128x1xi64, #blocked>
    %92 = arith.cmpi slt, %86, %cst : tensor<128x1xi64, #blocked>
    %93 = arith.andi %91, %92 : tensor<128x1xi1, #blocked>
    %94 = tt.broadcast %93 : (tensor<128x1xi1, #blocked>) -> tensor<128x128xi1, #blocked>
    %95 = tt.splat %5 : (i64) -> tensor<1x128xi64, #blocked>
    %96 = tt.splat %10 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
    %97 = tt.splat %11 : (i64) -> tensor<1x128xi64, #blocked>
    %98 = arith.muli %62, %97 : tensor<1x128xi64, #blocked>
    %99 = tt.broadcast %98 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
    %100:5 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg22 = %cst_10, %arg23 = %cst_8, %arg24 = %cst_9, %arg25 = %c0_i64, %arg26 = %c0_i64) -> (tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64)  : i32 {
      %121 = tt.splat %arg26 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %122 = tt.splat %arg26 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %123 = arith.addi %121, %43 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %124 = arith.addi %122, %44 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %125 = tt.expand_dims %123 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
      %126 = tt.expand_dims %124 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
      %127 = arith.muli %125, %90 : tensor<1x128xi64, #blocked>
      %128 = tt.broadcast %127 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
      %129 = tt.addptr %89, %128 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
      %130 = arith.cmpi sge, %126, %cst_3 : tensor<1x128xi64, #blocked>
      %131 = arith.cmpi slt, %126, %95 : tensor<1x128xi64, #blocked>
      %132 = arith.andi %130, %131 : tensor<1x128xi1, #blocked>
      %133 = tt.broadcast %132 : (tensor<1x128xi1, #blocked>) -> tensor<128x128xi1, #blocked>
      %134 = arith.andi %94, %133 : tensor<128x128xi1, #blocked>
      %135 = triton_gpu.view_slice %65[0, 0] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<128x32x!tt.ptr<f16, 1>, #blocked1>
      %136 = triton_gpu.view_slice %80[0, 0] [128, 32] [1, 1] : tensor<128x128xi1, #blocked1> to tensor<128x32xi1, #blocked1>
      %137 = triton_gpu.view_slice %cst_7[0, 0] [128, 32] [1, 1] : tensor<128x128xf16, #blocked1> to tensor<128x32xf16, #blocked1>
      %138 = tt.load %135, %136, %137 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked1>
      %139 = arith.extf %138 : tensor<128x32xf16, #blocked1> to tensor<128x32xf32, #blocked1>
      %140 = triton_gpu.view_slice %81[0, 0] [128, 32] [1, 1] : tensor<128x128xf32, #blocked1> to tensor<128x32xf32, #blocked1>
      %141 = arith.mulf %139, %140 : tensor<128x32xf32, #blocked1>
      %142 = arith.truncf %141 : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %143 = triton_gpu.convert_layout %142 : (tensor<128x32xf16, #blocked1>) -> tensor<128x32xf16, #shared>
      %144 = triton_gpu.convert_layout %143 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %145 = triton_gpu.view_slice %129[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<32x128x!tt.ptr<f16, 1>, #blocked>
      %146 = triton_gpu.view_slice %134[0, 0] [32, 128] [1, 1] : tensor<128x128xi1, #blocked> to tensor<32x128xi1, #blocked>
      %147 = triton_gpu.view_slice %cst_6[0, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked> to tensor<32x128xf16, #blocked>
      %148 = tt.load %145, %146, %147 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked>
      %149 = triton_gpu.convert_layout %148 : (tensor<32x128xf16, #blocked>) -> tensor<32x128xf16, #shared1>
      %150 = triton_gpu.convert_layout %149 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %151 = tt.dot %144, %150, %cst_10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %152 = triton_gpu.view_slice %65[0, 32] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<128x32x!tt.ptr<f16, 1>, #blocked1>
      %153 = triton_gpu.view_slice %80[0, 32] [128, 32] [1, 1] : tensor<128x128xi1, #blocked1> to tensor<128x32xi1, #blocked1>
      %154 = triton_gpu.view_slice %cst_7[0, 32] [128, 32] [1, 1] : tensor<128x128xf16, #blocked1> to tensor<128x32xf16, #blocked1>
      %155 = tt.load %152, %153, %154 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked1>
      %156 = arith.extf %155 : tensor<128x32xf16, #blocked1> to tensor<128x32xf32, #blocked1>
      %157 = triton_gpu.view_slice %82[0, 32] [128, 32] [1, 1] : tensor<128x128xf32, #blocked1> to tensor<128x32xf32, #blocked1>
      %158 = arith.mulf %156, %157 : tensor<128x32xf32, #blocked1>
      %159 = arith.truncf %158 : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %160 = triton_gpu.convert_layout %159 : (tensor<128x32xf16, #blocked1>) -> tensor<128x32xf16, #shared>
      %161 = triton_gpu.convert_layout %160 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %162 = triton_gpu.view_slice %129[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<32x128x!tt.ptr<f16, 1>, #blocked>
      %163 = triton_gpu.view_slice %134[32, 0] [32, 128] [1, 1] : tensor<128x128xi1, #blocked> to tensor<32x128xi1, #blocked>
      %164 = triton_gpu.view_slice %cst_6[32, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked> to tensor<32x128xf16, #blocked>
      %165 = tt.load %162, %163, %164 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked>
      %166 = triton_gpu.convert_layout %165 : (tensor<32x128xf16, #blocked>) -> tensor<32x128xf16, #shared1>
      %167 = triton_gpu.convert_layout %166 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %168 = tt.dot %161, %167, %151 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %169 = triton_gpu.view_slice %65[0, 64] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<128x32x!tt.ptr<f16, 1>, #blocked1>
      %170 = triton_gpu.view_slice %80[0, 64] [128, 32] [1, 1] : tensor<128x128xi1, #blocked1> to tensor<128x32xi1, #blocked1>
      %171 = triton_gpu.view_slice %cst_7[0, 64] [128, 32] [1, 1] : tensor<128x128xf16, #blocked1> to tensor<128x32xf16, #blocked1>
      %172 = tt.load %169, %170, %171 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked1>
      %173 = arith.extf %172 : tensor<128x32xf16, #blocked1> to tensor<128x32xf32, #blocked1>
      %174 = triton_gpu.view_slice %83[0, 64] [128, 32] [1, 1] : tensor<128x128xf32, #blocked1> to tensor<128x32xf32, #blocked1>
      %175 = arith.mulf %173, %174 : tensor<128x32xf32, #blocked1>
      %176 = arith.truncf %175 : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %177 = triton_gpu.convert_layout %176 : (tensor<128x32xf16, #blocked1>) -> tensor<128x32xf16, #shared>
      %178 = triton_gpu.convert_layout %177 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %179 = triton_gpu.view_slice %129[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<32x128x!tt.ptr<f16, 1>, #blocked>
      %180 = triton_gpu.view_slice %134[64, 0] [32, 128] [1, 1] : tensor<128x128xi1, #blocked> to tensor<32x128xi1, #blocked>
      %181 = triton_gpu.view_slice %cst_6[64, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked> to tensor<32x128xf16, #blocked>
      %182 = tt.load %179, %180, %181 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked>
      %183 = triton_gpu.convert_layout %182 : (tensor<32x128xf16, #blocked>) -> tensor<32x128xf16, #shared1>
      %184 = triton_gpu.convert_layout %183 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %185 = tt.dot %178, %184, %168 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %186 = triton_gpu.view_slice %65[0, 96] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<128x32x!tt.ptr<f16, 1>, #blocked1>
      %187 = triton_gpu.view_slice %80[0, 96] [128, 32] [1, 1] : tensor<128x128xi1, #blocked1> to tensor<128x32xi1, #blocked1>
      %188 = triton_gpu.view_slice %cst_7[0, 96] [128, 32] [1, 1] : tensor<128x128xf16, #blocked1> to tensor<128x32xf16, #blocked1>
      %189 = tt.load %186, %187, %188 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked1>
      %190 = arith.extf %189 : tensor<128x32xf16, #blocked1> to tensor<128x32xf32, #blocked1>
      %191 = triton_gpu.view_slice %84[0, 96] [128, 32] [1, 1] : tensor<128x128xf32, #blocked1> to tensor<128x32xf32, #blocked1>
      %192 = arith.mulf %190, %191 : tensor<128x32xf32, #blocked1>
      %193 = arith.truncf %192 : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %194 = triton_gpu.convert_layout %193 : (tensor<128x32xf16, #blocked1>) -> tensor<128x32xf16, #shared>
      %195 = triton_gpu.convert_layout %194 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %196 = triton_gpu.view_slice %129[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<32x128x!tt.ptr<f16, 1>, #blocked>
      %197 = triton_gpu.view_slice %134[96, 0] [32, 128] [1, 1] : tensor<128x128xi1, #blocked> to tensor<32x128xi1, #blocked>
      %198 = triton_gpu.view_slice %cst_6[96, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked> to tensor<32x128xf16, #blocked>
      %199 = tt.load %196, %197, %198 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked>
      %200 = triton_gpu.convert_layout %199 : (tensor<32x128xf16, #blocked>) -> tensor<32x128xf16, #shared1>
      %201 = triton_gpu.convert_layout %200 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %202 = tt.dot %195, %201, %185 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %203 = "tt.reduce"(%202) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32, %arg28: f32):
        %270 = arith.maximumf %arg27, %arg28 : f32
        tt.reduce.return %270 : f32
      }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %204 = arith.maximumf %arg24, %203 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %205 = tt.expand_dims %204 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
      %206 = tt.broadcast %205 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %207 = arith.subf %202, %206 : tensor<128x128xf32, #mfma>
      %208 = tt.extern_elementwise %207 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x128xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %209 = arith.subf %arg24, %204 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %210 = tt.extern_elementwise %209 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %211 = tt.expand_dims %210 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
      %212 = tt.broadcast %211 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %213 = arith.mulf %arg22, %212 : tensor<128x128xf32, #mfma>
      %214 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %215 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %216 = arith.addi %214, %45 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %217 = arith.addi %215, %46 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %218 = tt.expand_dims %216 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
      %219 = tt.expand_dims %217 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
      %220 = tt.addptr %96, %218 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
      %221 = tt.broadcast %220 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
      %222 = tt.addptr %221, %99 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
      %223 = arith.cmpi sge, %219, %cst_5 : tensor<128x1xi64, #blocked>
      %224 = arith.cmpi slt, %219, %68 : tensor<128x1xi64, #blocked>
      %225 = arith.andi %223, %224 : tensor<128x1xi1, #blocked>
      %226 = tt.broadcast %225 : (tensor<128x1xi1, #blocked>) -> tensor<128x128xi1, #blocked>
      %227 = arith.andi %226, %79 : tensor<128x128xi1, #blocked>
      %228 = arith.truncf %208 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
      %229 = triton_gpu.view_slice %228[0, 0] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %230 = triton_gpu.convert_layout %229 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %231 = triton_gpu.view_slice %222[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<32x128x!tt.ptr<f16, 1>, #blocked>
      %232 = triton_gpu.view_slice %227[0, 0] [32, 128] [1, 1] : tensor<128x128xi1, #blocked> to tensor<32x128xi1, #blocked>
      %233 = triton_gpu.view_slice %cst_6[0, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked> to tensor<32x128xf16, #blocked>
      %234 = tt.load %231, %232, %233 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked>
      %235 = triton_gpu.convert_layout %234 : (tensor<32x128xf16, #blocked>) -> tensor<32x128xf16, #shared1>
      %236 = triton_gpu.convert_layout %235 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %237 = tt.dot %230, %236, %213 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %238 = triton_gpu.view_slice %228[0, 32] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %239 = triton_gpu.convert_layout %238 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %240 = triton_gpu.view_slice %222[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<32x128x!tt.ptr<f16, 1>, #blocked>
      %241 = triton_gpu.view_slice %227[32, 0] [32, 128] [1, 1] : tensor<128x128xi1, #blocked> to tensor<32x128xi1, #blocked>
      %242 = triton_gpu.view_slice %cst_6[32, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked> to tensor<32x128xf16, #blocked>
      %243 = tt.load %240, %241, %242 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked>
      %244 = triton_gpu.convert_layout %243 : (tensor<32x128xf16, #blocked>) -> tensor<32x128xf16, #shared1>
      %245 = triton_gpu.convert_layout %244 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %246 = tt.dot %239, %245, %237 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %247 = triton_gpu.view_slice %228[0, 64] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %248 = triton_gpu.convert_layout %247 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %249 = triton_gpu.view_slice %222[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<32x128x!tt.ptr<f16, 1>, #blocked>
      %250 = triton_gpu.view_slice %227[64, 0] [32, 128] [1, 1] : tensor<128x128xi1, #blocked> to tensor<32x128xi1, #blocked>
      %251 = triton_gpu.view_slice %cst_6[64, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked> to tensor<32x128xf16, #blocked>
      %252 = tt.load %249, %250, %251 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked>
      %253 = triton_gpu.convert_layout %252 : (tensor<32x128xf16, #blocked>) -> tensor<32x128xf16, #shared1>
      %254 = triton_gpu.convert_layout %253 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %255 = tt.dot %248, %254, %246 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %256 = triton_gpu.view_slice %228[0, 96] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
      %257 = triton_gpu.convert_layout %256 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %258 = triton_gpu.view_slice %222[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<32x128x!tt.ptr<f16, 1>, #blocked>
      %259 = triton_gpu.view_slice %227[96, 0] [32, 128] [1, 1] : tensor<128x128xi1, #blocked> to tensor<32x128xi1, #blocked>
      %260 = triton_gpu.view_slice %cst_6[96, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked> to tensor<32x128xf16, #blocked>
      %261 = tt.load %258, %259, %260 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked>
      %262 = triton_gpu.convert_layout %261 : (tensor<32x128xf16, #blocked>) -> tensor<32x128xf16, #shared1>
      %263 = triton_gpu.convert_layout %262 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %264 = tt.dot %257, %263, %255 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %265 = "tt.reduce"(%208) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32, %arg28: f32):
        %270 = arith.addf %arg27, %arg28 : f32
        tt.reduce.return %270 : f32
      }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %266 = arith.mulf %arg23, %210 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %267 = arith.addf %266, %265 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %268 = arith.addi %arg25, %c128_i64 : i64
      %269 = arith.addi %arg26, %c128_i64 : i64
      scf.yield %264, %267, %204, %268, %269 : tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64
    }
    %101 = tt.expand_dims %100#1 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
    %102 = tt.broadcast %101 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
    %103 = arith.divf %100#0, %102 : tensor<128x128xf32, #mfma>
    %104 = arith.muli %1, %arg20 : i32
    %105 = tt.addptr %arg4, %104 : !tt.ptr<f32, 1>, i32
    %106 = tt.splat %105 : (!tt.ptr<f32, 1>) -> tensor<128x!tt.ptr<f32, 1>, #blocked3>
    %107 = tt.addptr %106, %28 : tensor<128x!tt.ptr<f32, 1>, #blocked3>, tensor<128xi32, #blocked3>
    %108 = tt.extern_elementwise %100#1 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_log2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %109 = arith.addf %100#2, %108 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %110 = triton_gpu.convert_layout %109 : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #blocked3>
    tt.store %107, %110 {cache = 1 : i32, evict = 1 : i32} : tensor<128xf32, #blocked3>
    %111 = tt.addptr %arg5, %2 : !tt.ptr<f16, 1>, i32
    %112 = arith.extsi %arg17 : i32 to i64
    %113 = arith.truncf %103 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
    %114 = tt.splat %112 : (i64) -> tensor<128x1xi64, #blocked2>
    %115 = arith.muli %52, %114 : tensor<128x1xi64, #blocked2>
    %116 = tt.splat %111 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked2>
    %117 = tt.addptr %116, %115 : tensor<128x1x!tt.ptr<f16, 1>, #blocked2>, tensor<128x1xi64, #blocked2>
    %118 = tt.broadcast %117 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked2>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked2>
    %119 = tt.addptr %118, %64 : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi64, #blocked2>
    %120 = triton_gpu.convert_layout %113 : (tensor<128x128xf16, #mfma>) -> tensor<128x128xf16, #blocked2>
    tt.store %119, %120 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #blocked2>
    tt.return
  }
}