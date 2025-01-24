// RUN: triton-opt %s -split-input-file -tritonamdgpu-stream-pipeline='num_stages=2' -cse -canonicalize -triton-amdgpu-refine-ops='arch=gfx942' -canonicalize | FileCheck %s

#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [32, 32], isTransposed = true}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel(
      %arg0: tensor<256x64x!tt.ptr<f16>, #blocked> {tt.contiguity=16 : i32, tt.divisibility=16: i32, tt.constancy=16: i32},
      %arg1: tensor<64x128x!tt.ptr<f16>, #blocked> {tt.contiguity=16 : i32, tt.divisibility=16: i32, tt.constancy=16: i32})  -> tensor<256x128xf32, #mma> attributes {noinline = false} {

    %output = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32

    %shift0 = arith.constant dense<64> : tensor<256x64xi32, #blocked>
    %shift1 = arith.constant dense<64> : tensor<64x128xi32, #blocked>

    %0:3 = scf.for %arg2 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(
      %loop_arg0 = %output,
      %loop_arg1 = %arg0,
      %loop_arg2 = %arg1) -> (
        tensor<256x128xf32, #mma>,
        tensor<256x64x!tt.ptr<f16>, #blocked>,
        tensor<64x128x!tt.ptr<f16>, #blocked>) : i32 {
      %1 = tt.load %loop_arg1 : tensor<256x64x!tt.ptr<f16>, #blocked>
      %2 = tt.load %loop_arg2 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %3 = ttg.convert_layout %1 : tensor<256x64xf16, #blocked> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %4 = ttg.convert_layout %2 : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %5 = tt.dot %3, %4, %loop_arg0 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
      %6 = tt.addptr %loop_arg1, %shift0 : tensor<256x64x!tt.ptr<f16>, #blocked>, tensor<256x64xi32, #blocked>
      %7 = tt.addptr %loop_arg2, %shift1 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      scf.yield %5, %6, %7 : tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked>
    }

    tt.return %0#0 : tensor<256x128xf32, #mma>
  }
}


// TODO: add TT GEMM case to the test
