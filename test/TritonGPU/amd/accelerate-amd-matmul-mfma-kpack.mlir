// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx942 kPack=2" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHECK{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [16, 16, 16], isTransposed = true}>
// CHECK-LABEL: kpack_kwidth_small_k_f16
// CHECK: tt.dot {{.*}} : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<16x256xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kpack_kwidth_small_k_f16(
      %a: tensor<16x16xf16, #dotOp0>,
      %b: tensor<16x256xf16, #dotOp1>,
      %o_ptr: tensor<16x256x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x256xf32, #blocked>
    %ab = tt.dot %a, %b, %cst : tensor<16x16xf16, #dotOp0> * tensor<16x256xf16, #dotOp1> -> tensor<16x256xf32, #blocked>
    tt.store %o_ptr, %ab : tensor<16x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHECK{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [16, 16, 16], isTransposed = true}>
// CHECK-LABEL: kpack_kwidth_large_k_f16
// CHECK: tt.dot {{.*}} : tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x256xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kpack_kwidth_large_k_f16(
      %a: tensor<16x32xf16, #dotOp0>,
      %b: tensor<32x256xf16, #dotOp1>,
      %o_ptr: tensor<16x256x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x256xf32, #blocked>
    %ab = tt.dot %a, %b, %cst : tensor<16x32xf16, #dotOp0> * tensor<32x256xf16, #dotOp1> -> tensor<16x256xf32, #blocked>
    tt.store %o_ptr, %ab : tensor<16x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHECK{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [64, 4, 16], isTransposed = true}>
// CHECK-LABEL: asymmetric_mfma_64x4_f32
// CHECK: tt.dot {{.*}} : tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x4xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x4xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @asymmetric_mfma_64x4_f32(
      %a: tensor<64x32xf32, #dotOp0>,
      %b: tensor<32x4xf32, #dotOp1>,
      %o_ptr: tensor<64x4x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x4xf32, #blocked>
    %ab = tt.dot %a, %b, %cst : tensor<64x32xf32, #dotOp0> * tensor<32x4xf32, #dotOp1> -> tensor<64x4xf32, #blocked>
    tt.store %o_ptr, %ab : tensor<64x4x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHECK{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [4, 64, 16], isTransposed = false}>
// CHECK-LABEL: asymmetric_mfma_4x64_f32
// CHECK: tt.dot {{.*}} : tensor<4x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<4x64xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @asymmetric_mfma_4x64_f32(
      %a: tensor<4x32xf32, #dotOp0>,
      %b: tensor<32x64xf32, #dotOp1>,
      %o_ptr: tensor<4x64x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<4x64xf32, #blocked>
    %ab = tt.dot %a, %b, %cst : tensor<4x32xf32, #dotOp0> * tensor<32x64xf32, #dotOp1> -> tensor<4x64xf32, #blocked>
    tt.store %o_ptr, %ab : tensor<4x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
