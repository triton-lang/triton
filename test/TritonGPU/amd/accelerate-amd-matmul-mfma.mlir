// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul='arch-generation-name=gfx940 matrix-instruction-size=0' | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_fp8e5m2
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_fp8e5m2(
      %arg0: tensor<128x64xf8E5M2, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x256xf8E5M2, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x256x!tt.ptr<f32>, #blocked> ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
// CHECK: %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
// CHECK: %0 = triton_gpu.convert_layout %cst : {{.*}} -> tensor<128x256xf32, #mma>
// CHECK: %1 = triton_gpu.convert_layout %arg0 : {{.*}} -> tensor<128x64xf8E5M2, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
// CHECK: %2 = tt.fp_to_fp %1 : {{.*}} -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
// CHECK: %3 = triton_gpu.convert_layout %arg1 : {{.*}} -> tensor<64x256xf8E5M2, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
// CHECK: %4 = tt.fp_to_fp %3 : tensor<64x256xf8E5M2, {{.*}} -> tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
// CHECK: %5 = tt.dot %2, %4, %0 : {{.*}} -> tensor<128x256xf32, #mma>
// CHECK: %6 = triton_gpu.convert_layout %5 : {{.*}} -> tensor<128x256xf32, #blocked>
// CHECK tt.store %arg2, %6
    %1 = tt.dot %arg0, %arg1, %cst : tensor<128x64xf8E5M2, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xf8E5M2, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
    tt.store %arg2, %1 : tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
