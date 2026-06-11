// RUN: triton-opt %s  -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942" | FileCheck %s

// CHECK-LABEL:mfma_xf32

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 8], isTransposed = true}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_xf32(
    %arg0: tensor<64x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>,
    %arg1: tensor<128x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    // Check that we generate xf32 instructions
    // CHECK: rocdl.mfma.f32.16x16x8.xf32
    %dot = tt.dot %arg0, %arg1, %cst_0, inputPrecision = tf32 :
      tensor<64x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x64xf32, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL:mfma_not_xf32

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 4], isTransposed = true}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_not_xf32(
    %arg0: tensor<64x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>,
    %arg1: tensor<128x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    // Check that we don't generate xf32 instructions if the input precision is "ieee"
    // CHECK: rocdl.mfma.f32.16x16x4f32
    %dot = tt.dot %arg0, %arg1, %cst_0, inputPrecision = ieee :
      tensor<64x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x64xf32, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL:mfma_f64_ignore_xf32

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 4], isTransposed = true, elementBitWidth = 64}>
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @mfma_f64_ignore_xf32(
    %a: tensor<32x256xf64, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>,
    %b: tensor<256x32xf64, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>) {
    %zero_f64 = arith.constant dense<0.000000e+00> : tensor<32x32xf64, #mma>
    // CHECK: rocdl.mfma.f64.16x16x4f64
    %dot = tt.dot %a, %b, %zero_f64, inputPrecision = tf32 : tensor<32x256xf64, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<256x32xf64, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf64, #mma>
    tt.return
  }
}
