// RUN: triton-opt %s -split-input-file -tritonamdgpu-stream-pipeline="num_stages=3 use_async_copy=1 use_pingpong=1" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @simple_pipelined_load
  // We expect one ttg.async_wait in the epilogue, one in the loop and one in the prologue
  // CHECK: ttg.async_wait
  // CHECK-NOT: ttg.async_wait
  // CHECK: scf.for
  // CHECK: ttg.async_wait
  // CHECK-NOT: ttg.async_wait
  // CHECK: scf.yield
  // CHECK: ttg.async_wait
  // CHECK-NOT: ttg.async_wait
  tt.func @simple_pipelined_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %arg3: i32, %arg4: i32) -> tensor<128x16xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %3 = tt.broadcast %0 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %4 = tt.broadcast %2 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %5 = tt.addptr %3, %4 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    %6 = scf.for %arg6 = %c0_i32 to %arg3 step %arg4 iter_args(%arg5 = %cst) -> (tensor<128x16xf32, #mma>)  : i32 {
      %7 = tt.load %5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %8 = ttg.convert_layout %7 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %9 = tt.dot %arg2, %8, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      scf.yield %9 : tensor<128x16xf32, #mma>
    }
    tt.return %6 : tensor<128x16xf32, #mma>
  }
}
