// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 -dump-swp-failure | FileCheck %s

// CHECK-LABEL: @dont_pipeline_128x1
// CHECK-NOT: local_load{{.*}}128x1
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @dont_pipeline_128x1(%arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false, swp = true} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_4 = arith.constant dense<-1.000000e+30> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>

    %99:1 = scf.for %arg25 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg31 = %cst_4) -> (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) : i32 {
      %94 = tt.splat %arg6 : !tt.ptr<i32> -> tensor<128x1x!tt.ptr<i32>, #blocked>
      %151 = tt.load %94 : tensor<128x1x!tt.ptr<i32>, #blocked>
      %161 = triton_gpu.convert_layout %151 : tensor<128x1xi32, #blocked> -> tensor<128x1xi32, #mma>
      %162 = tt.broadcast %161 : tensor<128x1xi32, #mma> -> tensor<128x64xi32, #mma>
      %170 = arith.sitofp %162 : tensor<128x64xi32, #mma> to tensor<128x64xf32, #mma>

      %173 = "tt.reduce"(%170) <{axis = 1 : i32}> ({
      ^bb0(%arg33: f32, %arg34: f32):
        %207 = arith.maxnumf %arg33, %arg34 : f32
        tt.reduce.return %207 : f32
      }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %175 = arith.maxnumf %arg31, %173 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>

      %201 = arith.truncf %170 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
      %202 = triton_gpu.convert_layout %201 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>

      %192 = arith.constant dense<0.> : tensor<128x64xf32, #mma>
      %203 = arith.constant dense<0.> : tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %204 = tt.dot %202, %203, %192 : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>

      scf.yield %175 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    }
    tt.return
  }
}
