// RUN: triton-opt %s -split-input-file -tritonamdgpu-pipeline="use_async_copy=1" | FileCheck %s

// The A operand is a single row of 2048 f16 elements, i.e. 256 vectors of 8
// against a warp size of 64, so warpSize / contigLanes truncates to zero.
// The padded layout must not be staggered over 16 rows when the tile has one,
// which would add bases for rows that do not exist and leave the layout
// broadcasting in the offset dimension.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: dot_operand_row_wider_than_warp
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_wait
  tt.func @dot_operand_row_wider_than_warp(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<3072> : tensor<2048x1xi32, #blocked>
    %cst_0 = arith.constant dense<768> : tensor<2048x1xi32, #blocked>
    %cst_1 = arith.constant dense<3072> : tensor<1x2048xi32, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<2048x16xf16, #blocked>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x2048xf16, #blocked1>
    %c2048_i32 = arith.constant 2048 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<1x16xf32, #mma>
    %0 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2048xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<2048x1xi32, #blocked>
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %4 = tt.splat %c0_i32 : i32 -> tensor<1x16xi32, #blocked>
    %5 = arith.addi %4, %3 : tensor<1x16xi32, #blocked>
    %6 = tt.broadcast %5 : tensor<1x16xi32, #blocked> -> tensor<2048x16xi32, #blocked>
    %7 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<2048x16x!tt.ptr<f16>, #blocked>
    %8 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<2048xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x2048xi32, #blocked1>
    %10 = tt.splat %c0_i32 : i32 -> tensor<1x2048xi32, #blocked1>
    %11 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x2048x!tt.ptr<f16>, #blocked1>
    %12 = scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg4 = %cst_4) -> (tensor<1x16xf32, #mma>)  : i32 {
      %15 = arith.muli %arg3, %c2048_i32 overflow<nsw> : i32
      %16 = tt.splat %15 : i32 -> tensor<2048x1xi32, #blocked>
      %17 = arith.addi %16, %1 : tensor<2048x1xi32, #blocked>
      %18 = arith.cmpi ult, %17, %cst : tensor<2048x1xi32, #blocked>
      %19 = arith.muli %17, %cst_0 overflow<nsw> : tensor<2048x1xi32, #blocked>
      %20 = tt.broadcast %19 : tensor<2048x1xi32, #blocked> -> tensor<2048x16xi32, #blocked>
      %21 = arith.addi %20, %6 : tensor<2048x16xi32, #blocked>
      %22 = tt.broadcast %18 : tensor<2048x1xi1, #blocked> -> tensor<2048x16xi1, #blocked>
      %23 = tt.addptr %7, %21 : tensor<2048x16x!tt.ptr<f16>, #blocked>, tensor<2048x16xi32, #blocked>
      %24 = tt.load %23, %22, %cst_2 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<2048x16x!tt.ptr<f16>, #blocked>
      %25 = tt.splat %15 : i32 -> tensor<1x2048xi32, #blocked1>
      %26 = arith.addi %25, %9 : tensor<1x2048xi32, #blocked1>
      %27 = arith.cmpi ult, %26, %cst_1 : tensor<1x2048xi32, #blocked1>
      %28 = arith.addi %10, %26 : tensor<1x2048xi32, #blocked1>
      %29 = tt.addptr %11, %28 : tensor<1x2048x!tt.ptr<f16>, #blocked1>, tensor<1x2048xi32, #blocked1>
      %30 = tt.load %29, %27, %cst_3 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<1x2048x!tt.ptr<f16>, #blocked1>
      %31 = ttg.convert_layout %30 : tensor<1x2048xf16, #blocked1> -> tensor<1x2048xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %32 = ttg.convert_layout %24 : tensor<2048x16xf16, #blocked> -> tensor<2048x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %33 = tt.dot %31, %32, %arg4 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<1x2048xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<2048x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<1x16xf32, #mma>
      scf.yield %33 : tensor<1x16xf32, #mma>
    } {tt.scheduled_max_stage = 1 : i32}
    %13 = arith.truncf %12 : tensor<1x16xf32, #mma> to tensor<1x16xf16, #mma>
    %14 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #mma>
    tt.store %14, %13 : tensor<1x16x!tt.ptr<f16>, #mma>
    tt.return
  }
}
