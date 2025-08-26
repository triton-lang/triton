// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul | FileCheck %s

// Verify that for SM_120 with FP8 inputs, tt.dot_scaled is preserved and
// scales are converted to linear layout for hardware acceleration.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked_k = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.target" = "cuda:120", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sm120_dot_scaled_basic
  tt.func public @sm120_dot_scaled_basic(
    %a: tensor<128x32xi8, #blocked_k>,
    %scale_a: tensor<128x2xi8, #blocked>,
    %b: tensor<32x128xi8, #blocked>,
    %scale_b: tensor<128x2xi8, #blocked>
  ) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK-DAG: tt.dot_scaled
    // CHECK-DAG: #linear
    // CHECK-DAG: #linear1
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e4m3 rhs = e4m3 {fastMath = false}
      : tensor<128x32xi8, #blocked_k>, tensor<128x2xi8, #blocked>
        * tensor<32x128xi8, #blocked>, tensor<128x2xi8, #blocked>
        -> tensor<128x128xf32, #blocked>
    tt.return %d : tensor<128x128xf32, #blocked>
  }
}

// -----

// Verify that for SM_120 with FP4 inputs, tt.dot_scaled is decomposed into:
// 1. ttg.fp4_to_fp for unpacking FP4 values
// 2. Scale application with arith.mulf
// 3. Regular tt.dot operation with MMA encoding

#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2_k = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.target" = "cuda:120", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sm120_dot_scaled_fp4_fallback
  // CHECK-NOT: tt.dot_scaled
  // CHECK: ttg.fp4_to_fp
  // CHECK: tt.dot
  // CHECK: #mma
  tt.func public @sm120_dot_scaled_fp4_fallback(
    %a: tensor<128x32xi8, #blocked2_k>,
    %scale_a: tensor<128x2xi8, #blocked2>,
    %b: tensor<32x128xi8, #blocked2>,
    %scale_b: tensor<128x2xi8, #blocked2>
  ) -> tensor<128x128xf32, #blocked2> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked2>
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e2m1 rhs = e2m1 {fastMath = false}
      : tensor<128x32xi8, #blocked2_k>, tensor<128x2xi8, #blocked2>
        * tensor<32x128xi8, #blocked2>, tensor<128x2xi8, #blocked2>
        -> tensor<128x128xf32, #blocked2>
    tt.return %d : tensor<128x128xf32, #blocked2>
  }
}

// -----

// Verify that for SM_100 (Blackwell), tt.dot_scaled uses the specialized
// MMAv5 path with tensor memory and tc_gen5_mma_scaled instruction.

#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3_1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3_2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: sm100_dot_scaled_mma_v5
  // CHECK: ttng.tc_gen5_mma_scaled
  tt.func public @sm100_dot_scaled_mma_v5(%a: tensor<128x64xi8, #blocked3_2>, %scale_a: tensor<128x2xi8, #blocked3_1>, %b: tensor<64x128xi8, #blocked3>, %scale_b: tensor<128x2xi8, #blocked3_1>) -> tensor<128x128xf32, #blocked3> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked3>
    %d = tt.dot_scaled %a scale %scale_a, %b scale %scale_b, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x64xi8, #blocked3_2>, tensor<128x2xi8, #blocked3_1> * tensor<64x128xi8, #blocked3>, tensor<128x2xi8, #blocked3_1> -> tensor<128x128xf32, #blocked3>
    tt.return %d : tensor<128x128xf32, #blocked3>
  }
}
