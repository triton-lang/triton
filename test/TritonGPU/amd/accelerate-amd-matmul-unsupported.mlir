// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx942 matrix-instruction-size=0" --verify-diagnostics

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_fp8e4m3fn(
      %arg0: tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x256xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x256x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    // expected-error @+1 {{no matching matrix core intrinsic due to unsupported element type}}
    %1 = tt.dot %arg0, %arg1, %cst : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
    tt.store %arg2, %1 : tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
