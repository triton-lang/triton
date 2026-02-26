// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="arch=gfx950" | FileCheck %s

// Test bf8_fp8 with non-transposed layout.
// A=bf8, B=fp8 -> intrinsic mfma.bf8.fp8, operands passed as (A, B)

// CHECK-LABEL: mfma_16x16x32_bf8_fp8_non_transposed
#mma_nt = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = false}>
#dotOp0_nt = #ttg.dot_op<{opIdx = 0, parent = #mma_nt, kWidth = 8}>
#dotOp1_nt = #ttg.dot_op<{opIdx = 1, parent = #mma_nt, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_bf8_fp8_non_transposed_layout(
      %arg0: tensor<16x32xf8E5M2, #dotOp0_nt>,
      %arg1: tensor<32x16xf8E4M3FN, #dotOp1_nt>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma_nt>
    // CHECK: rocdl.mfma.f32.16x16x32.bf8.fp8
    %dot = tt.dot %arg0, %arg1, %cst : tensor<16x32xf8E5M2, #dotOp0_nt> * tensor<32x16xf8E4M3FN, #dotOp1_nt> -> tensor<16x16xf32, #mma_nt>
    tt.return
  }
}

// -----

// Test bf8_fp8 with transposed layout.
// A=bf8, B=fp8, but operands get swapped internally to (B, A)
// Check that we swap intrinsic type selection

// CHECK-LABEL: mfma_16x16x32_bf8_fp8_transposed
#mma_t = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dotOp0_t = #ttg.dot_op<{opIdx = 0, parent = #mma_t, kWidth = 8}>
#dotOp1_t = #ttg.dot_op<{opIdx = 1, parent = #mma_t, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_bf8_fp8_transposed_layout(
      %arg0: tensor<128x128xf8E5M2, #dotOp0_t>,
      %arg1: tensor<128x128xf8E4M3FN, #dotOp1_t>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma_t>
    // CHECK: rocdl.mfma.f32.16x16x32.fp8.bf8
    %dot = tt.dot %arg0, %arg1, %cst : tensor<128x128xf8E5M2, #dotOp0_t> * tensor<128x128xf8E4M3FN, #dotOp1_t> -> tensor<128x128xf32, #mma_t>
    tt.return
  }
}