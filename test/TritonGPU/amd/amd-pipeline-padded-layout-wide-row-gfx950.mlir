// RUN: triton-opt %s -split-input-file -tritonamdgpu-pipeline="use_async_copy=1" | FileCheck %s

// Each row of A holds 1024 f16 elements, 128 vectors of 8 against a warp of 64
// lanes, so one warp covers only part of a row. The padded layout must not be
// staggered over more rows than the 4 the tile has.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: dot_operand_row_wider_than_warp
  tt.func @dot_operand_row_wider_than_warp(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK: ttg.async_copy_global_to_local
    // CHECK: ttg.async_wait
    %cst = arith.constant dense<0.000000e+00> : tensor<4x16xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x1024xi32, #blocked>
    %2 = tt.broadcast %1 : tensor<1x1024xi32, #blocked> -> tensor<4x1024xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<4x1024x!tt.ptr<f16>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<4x1024x!tt.ptr<f16>, #blocked>, tensor<4x1024xi32, #blocked>
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x16xi32, #blocked1>
    %7 = tt.broadcast %6 : tensor<1x16xi32, #blocked1> -> tensor<1024x16xi32, #blocked1>
    %8 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x16x!tt.ptr<f16>, #blocked1>
    %9 = tt.addptr %8, %7 : tensor<1024x16x!tt.ptr<f16>, #blocked1>, tensor<1024x16xi32, #blocked1>

    %10 = scf.for %arg2 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg3 = %cst) -> (tensor<4x16xf32, #mma>)  : i32 {
      %11 = tt.load %4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<4x1024x!tt.ptr<f16>, #blocked>
      %12 = tt.load %9 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<1024x16x!tt.ptr<f16>, #blocked1>
      %13 = ttg.convert_layout %11 : tensor<4x1024xf16, #blocked> -> tensor<4x1024xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %14 = ttg.convert_layout %12 : tensor<1024x16xf16, #blocked1> -> tensor<1024x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %15 = tt.dot %13, %14, %arg3 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<4x1024xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<1024x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<4x16xf32, #mma>
      scf.yield %15 : tensor<4x16xf32, #mma>
    } {tt.scheduled_max_stage = 1 : i32}

    tt.return
  }
}
