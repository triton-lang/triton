// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul='arch-generation-name=gfx940 matrix-instruction-size=0' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_fp8e5m2
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_fp8e5m2(
      %arg0: tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x256xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x256x!tt.ptr<f32>, #blocked> ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    // CHECK: %[[A0:.+]] = ttg.convert_layout %arg0 : {{.*}} -> tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    // CHECK: %[[A1:.+]] = tt.fp_to_fp %[[A0]] : {{.*}} -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    // CHECK: %[[B0:.+]] = ttg.convert_layout %arg1 : {{.*}} -> tensor<64x256xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    // CHECK: %[[B1:.+]] = tt.fp_to_fp %[[B0]] : tensor<64x256xf8E5M2, {{.*}} -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    // CHECK: tt.dot %[[A1]], %[[B1]]
    %1 = tt.dot %arg0, %arg1, %cst : tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
    tt.store %arg2, %1 : tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Verify that we use FMA when the N dimension is too small for any mma.
// CHECK-NOT: #triton_gpu.amd_mfma
// CHECK-LABEL: small_n_size
#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.target" = "hip:gfx942", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @small_n_size(
    %a: tensor<4x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
    %b: tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>)
    -> tensor<4x128xf32, #blocked> {
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<4x128xf32, #blocked>
    %result = tt.dot %a, %b, %zero_f32 : tensor<4x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<4x128xf32, #blocked>
    tt.return %result : tensor<4x128xf32, #blocked>
  }
}
