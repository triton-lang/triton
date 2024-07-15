// RUN: triton-opt %s -tritongpu-remove-layout-conversions | FileCheck %s
// Minimized reproducer for https://github.com/pytorch/pytorch/issues/130101

// CHECK: tt.return
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 2], order = [2, 1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, triton_gpu.target = "cuda:86", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @triton_() -> tensor<1x256xi32, #blocked> attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<1x256xi32, #blocked1>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked2}>}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked2}>}>> -> tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked2}>>
    %2 = tt.expand_dims %1 {axis = 2 : i32} : tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked2}>> -> tensor<1x2x1xi32, #blocked2>
    %3 = tt.broadcast %2 : tensor<1x2x1xi32, #blocked2> -> tensor<1x2x128xi32, #blocked2>
    %4 = tt.reshape %3 {allow_reorder = false} : tensor<1x2x128xi32, #blocked2> -> tensor<1x256xi32, #blocked1>
    %5 = tt.broadcast %2 : tensor<1x2x1xi32, #blocked2> -> tensor<2x2x64xi32, #blocked2>
    %6 = tt.reshape %5 {allow_reorder = false} : tensor<2x2x64xi32, #blocked2> -> tensor<1x256xi32, #blocked1>
    %7 = arith.cmpi ne, %4, %cst : tensor<1x256xi32, #blocked1>
    %8 = arith.select %7, %6, %cst : tensor<1x256xi1, #blocked1>, tensor<1x256xi32, #blocked1>
    %9 = triton_gpu.convert_layout %8 : tensor<1x256xi32, #blocked1> -> tensor<1x256xi32, #blocked>
    tt.return %9 : tensor<1x256xi32, #blocked>
  }
}
