// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="arch=gfx950" | FileCheck %s

// Test bf8_fp8 with non-transposed layout.
// A=bf8, B=fp8 -> intrinsic mfma.bf8.fp8, operands passed as (A, B)
// Hardware sees: (bf8_data, fp8_data) interpreted as (bf8, fp8)

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

// Test fp8_bf8 with non-transposed layout.
// A=fp8, B=bf8 -> intrinsic mfma.fp8.bf8, operands passed as (A, B)
// Hardware sees: (fp8_data, bf8_data) interpreted as (fp8, bf8)

// CHECK-LABEL: mfma_16x16x32_fp8_bf8_non_transposed
#mma_nt = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = false}>
#dotOp0_nt = #ttg.dot_op<{opIdx = 0, parent = #mma_nt, kWidth = 8}>
#dotOp1_nt = #ttg.dot_op<{opIdx = 1, parent = #mma_nt, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_fp8_bf8_non_transposed_layout(
      %arg0: tensor<16x32xf8E4M3FN, #dotOp0_nt>,
      %arg1: tensor<32x16xf8E5M2, #dotOp1_nt>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma_nt>
    // CHECK: rocdl.mfma.f32.16x16x32.fp8.bf8
    %dot = tt.dot %arg0, %arg1, %cst : tensor<16x32xf8E4M3FN, #dotOp0_nt> * tensor<32x16xf8E5M2, #dotOp1_nt> -> tensor<16x16xf32, #mma_nt>
    tt.return
  }
}

// -----

// Test bf8_fp8 with transposed layout.
// A=bf8, B=fp8, but operands get swapped internally to (B, A)
// Check that we swap intrinsic type selection: we expect fp8.bf8
// Hardware sees: (fp8_data, bf8_data) interpreted as (fp8, bf8)

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

// -----

// Test fp8_bf8 with transposed layout.
// A=fp8, B=bf8, but operands get swapped internally to (B, A)
// Check that we swap intrinsic type selection: we expect bf8.fp8
// Hardware sees: (bf8_data, fp8_data) interpreted as (bf8, fp8)

// CHECK-LABEL: mfma_16x16x32_fp8_bf8_transposed
#mma_t = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dotOp0_t = #ttg.dot_op<{opIdx = 0, parent = #mma_t, kWidth = 8}>
#dotOp1_t = #ttg.dot_op<{opIdx = 1, parent = #mma_t, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_fp8_bf8_transposed_layout(
      %arg0: tensor<128x128xf8E4M3FN, #dotOp0_t>,
      %arg1: tensor<128x128xf8E5M2, #dotOp1_t>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma_t>
    // CHECK: rocdl.mfma.f32.16x16x32.bf8.fp8
    %dot = tt.dot %arg0, %arg1, %cst : tensor<128x128xf8E4M3FN, #dotOp0_t> * tensor<128x128xf8E5M2, #dotOp1_t> -> tensor<128x128xf32, #mma_t>
    tt.return
  }
}

// -----

// Test 32x32 MFMA with mixed bf8_fp8 with transposed layout.

// CHECK-LABEL: mfma_32x32x16_bf8_fp8_transposed
#mma_32_t = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#dotOp0_32_t = #ttg.dot_op<{opIdx = 0, parent = #mma_32_t, kWidth = 8}>
#dotOp1_32_t = #ttg.dot_op<{opIdx = 1, parent = #mma_32_t, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_32x32x16_bf8_fp8_transposed_layout(
      %arg0: tensor<128x128xf8E5M2, #dotOp0_32_t>,
      %arg1: tensor<128x128xf8E4M3FN, #dotOp1_32_t>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma_32_t>
    // CHECK: rocdl.mfma.f32.32x32x16.fp8.bf8
    %dot = tt.dot %arg0, %arg1, %cst : tensor<128x128xf8E5M2, #dotOp0_32_t> * tensor<128x128xf8E4M3FN, #dotOp1_32_t> -> tensor<128x128xf32, #mma_32_t>
    tt.return
  }
}

// -----

// Check symmetric types (bf8_bf8) with transposed layout.
// For symmetric types, swapping doesn't change the intrinsic.

// CHECK-LABEL: mfma_16x16x32_bf8_bf8_transposed
#mma_t = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dotOp0_t = #ttg.dot_op<{opIdx = 0, parent = #mma_t, kWidth = 8}>
#dotOp1_t = #ttg.dot_op<{opIdx = 1, parent = #mma_t, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_bf8_bf8_transposed_layout(
      %arg0: tensor<128x128xf8E5M2, #dotOp0_t>,
      %arg1: tensor<128x128xf8E5M2, #dotOp1_t>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma_t>
    // CHECK: rocdl.mfma.f32.16x16x32.bf8.bf8
    %dot = tt.dot %arg0, %arg1, %cst : tensor<128x128xf8E5M2, #dotOp0_t> * tensor<128x128xf8E5M2, #dotOp1_t> -> tensor<128x128xf32, #mma_t>
    tt.return
  }
}

// -----

// Check symmetric types (fp8_fp8) with transposed layout.
// For symmetric types, swapping doesn't change the intrinsic.

// CHECK-LABEL: mfma_16x16x32_fp8_fp8_transposed
#mma_t = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dotOp0_t = #ttg.dot_op<{opIdx = 0, parent = #mma_t, kWidth = 8}>
#dotOp1_t = #ttg.dot_op<{opIdx = 1, parent = #mma_t, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_16x16x32_fp8_fp8_transposed_layout(
      %arg0: tensor<128x128xf8E4M3FN, #dotOp0_t>,
      %arg1: tensor<128x128xf8E4M3FN, #dotOp1_t>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma_t>
    // CHECK: rocdl.mfma.f32.16x16x32.fp8.fp8
    %dot = tt.dot %arg0, %arg1, %cst : tensor<128x128xf8E4M3FN, #dotOp0_t> * tensor<128x128xf8E4M3FN, #dotOp1_t> -> tensor<128x128xf32, #mma_t>
    tt.return
  }
}

// -----

// Test 32x32 MFMA with symmetric bf8_bf8 and non-transposed layout.

// CHECK-LABEL: mfma_32x32x16_bf8_bf8_non_transposed
#mma_32_nt = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = false}>
#dotOp0_32_nt = #ttg.dot_op<{opIdx = 0, parent = #mma_32_nt, kWidth = 8}>
#dotOp1_32_nt = #ttg.dot_op<{opIdx = 1, parent = #mma_32_nt, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_32x32x16_bf8_bf8_non_transposed_layout(
      %arg0: tensor<32x16xf8E5M2, #dotOp0_32_nt>,
      %arg1: tensor<16x32xf8E5M2, #dotOp1_32_nt>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma_32_nt>
    // CHECK: rocdl.mfma.f32.32x32x16.bf8.bf8
    %dot = tt.dot %arg0, %arg1, %cst : tensor<32x16xf8E5M2, #dotOp0_32_nt> * tensor<16x32xf8E5M2, #dotOp1_32_nt> -> tensor<32x32xf32, #mma_32_nt>
    tt.return
  }
}
