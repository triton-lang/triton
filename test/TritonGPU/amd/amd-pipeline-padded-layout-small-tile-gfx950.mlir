// RUN: triton-opt %s -split-input-file -tritonamdgpu-pipeline="use_async_copy=1" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: pipeline_padded_layout_gfx950
  tt.func @pipeline_padded_layout_gfx950(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK: ttg.async_wait %{{.*}}
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %2 = tt.broadcast %1 : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #blocked>
    %4 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #blocked>
    %5 = tt.addptr %3, %2 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi32, #blocked>
    %6 = tt.addptr %4, %2 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi32, #blocked>
    
    %7 = scf.for %arg3 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<16x16xf32, #mma>)  : i32 {
      %9 = tt.load %5 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked>
      %10 = tt.load %6 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked>
      %11 = ttg.convert_layout %9 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %12 = ttg.convert_layout %10 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %13 = tt.dot %11, %12, %arg4 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
      scf.yield %13 : tensor<16x16xf32, #mma>
    } {tt.scheduled_max_stage = 1 : i32}
    
    tt.return
  }
}
