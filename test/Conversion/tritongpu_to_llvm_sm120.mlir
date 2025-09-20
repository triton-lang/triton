// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul --allocate-shared-memory-nv='compute-capability=120' --convert-triton-gpu-to-llvm='compute-capability=120' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked_k = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>

module attributes {"ttg.target" = "cuda:120", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sm120_mmav2_dot_scaled
  // CHECK: mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X
  tt.func public @sm120_mmav2_dot_scaled(
    %a: tensor<128x32xf8E5M2, #blocked_k>,
    %sa: tensor<128x2xi8, #blocked>,
    %b: tensor<32x128xf8E5M2, #blocked>,
    %sb: tensor<128x2xi8, #blocked>,
    %out: !tt.ptr<f32>
  ){
    %c = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %a_d = ttg.convert_layout %a : tensor<128x32xf8E5M2, #blocked_k> -> tensor<128x32xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %b_d = ttg.convert_layout %b : tensor<32x128xf8E5M2, #blocked> -> tensor<32x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %d = tt.dot_scaled %a_d scale %sa, %b_d scale %sb, %c lhs = e5m2 rhs = e5m2 {fastMath = false}
      : tensor<128x32xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<128x2xi8, #blocked>
        * tensor<32x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, tensor<128x2xi8, #blocked>
        -> tensor<128x128xf32, #blocked>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %out_ptrs = tt.broadcast %out_splat : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x128x!tt.ptr<f32>, #blocked>
    %zero = arith.constant dense<0> : tensor<128x128xi1, #blocked>
    tt.store %out_ptrs, %d, %zero : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
