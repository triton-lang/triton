// RUN: triton-opt %s -split-input-file --tritongpu-reduce-data-duplication 2>&1 | FileCheck %s
// CHECK: #triton_gpu.shared
// CHECK-NOT: maxPhase = 0

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 4, 1], order = [2, 0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
module attributes {"triton_gpu.compute-capability" = 89 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @thin_matmul_kernel(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x32xf32, #mma>
    %c32_i32 = arith.constant 32 : i32
    %c512_i32 = arith.constant 512 : i32
    %cst_0 = arith.constant dense<32> : tensor<1x16x1xi32, #blocked>
    %cst_1 = arith.constant dense<32> : tensor<2x1xi32, #blocked1>
    %cst_2 = arith.constant dense<2> : tensor<16x1xi32, #blocked2>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<16x1xi32, #blocked2>
    %4 = arith.muli %3, %cst_2 : tensor<16x1xi32, #blocked2>
    %5 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x2xi32, #blocked2>
    %7 = tt.broadcast %6 : tensor<1x2xi32, #blocked2> -> tensor<16x2xi32, #blocked2>
    %8 = tt.addptr %arg0, %1 : !tt.ptr<f32, 1>, i32
    %9 = tt.splat %8 : !tt.ptr<f32, 1> -> tensor<16x1x!tt.ptr<f32, 1>, #blocked2>
    %10 = tt.addptr %9, %4 : tensor<16x1x!tt.ptr<f32, 1>, #blocked2>, tensor<16x1xi32, #blocked2>
    %11 = tt.broadcast %10 : tensor<16x1x!tt.ptr<f32, 1>, #blocked2> -> tensor<16x2x!tt.ptr<f32, 1>, #blocked2>
    %12 = tt.addptr %11, %7 : tensor<16x2x!tt.ptr<f32, 1>, #blocked2>, tensor<16x2xi32, #blocked2>
    %13 = tt.load %12 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x2xf32, #blocked2>
    %27 = triton_gpu.convert_layout %13 : tensor<16x2xf32, #blocked2> -> tensor<16x2xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    tt.return
  }
}
