// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx1250-strict" | FileCheck %s
//
// gfx1250-strict uses native WMMA, fp8/bf8 dots never select the
// K=128 intrinsics and fall back to K=64 instead.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK{LITERAL}: #mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
// CHECK-LABEL: wmma_dot_bf16
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250-strict", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_bf16(
      %a: tensor<32x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %b: tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %out: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: tt.dot {{.*}} : tensor<32x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %d = tt.dot %a, %b, %cst : tensor<32x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf32, #blocked>
    tt.store %out, %d : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK{LITERAL}: #mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 64]}>
// CHECK-LABEL: wmma_dot_fp8_k128_uses_k64
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250-strict", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_fp8_k128_uses_k64(
      %a: tensor<32x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %b: tensor<128x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %out: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: tt.dot {{.*}} : tensor<32x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %d = tt.dot %a, %b, %cst : tensor<32x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf32, #blocked>
    tt.store %out, %d : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK{LITERAL}: #mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 128]}>
// CHECK-LABEL: wmma_dot_scaled_mxfp8
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250-strict", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_scaled_mxfp8(
      %arg0: tensor<32x128xf8E4M3FN, #blocked>,
      %arg1: tensor<128x32xf8E4M3FN, #blocked1>,
      %scale_a_ptr: tensor<32x4x!tt.ptr<i8>, #blocked2>,
      %scale_b_ptr: tensor<32x4x!tt.ptr<i8>, #blocked2>,
      %arg4: tensor<32x32x!tt.ptr<f32>, #blocked3>
      ) {
    // CHECK: tt.dot_scaled {{.*}} -> tensor<32x32xf32, #mma>
    %arg2 = tt.load %scale_a_ptr : tensor<32x4x!tt.ptr<i8>, #blocked2>
    %arg3 = tt.load %scale_b_ptr : tensor<32x4x!tt.ptr<i8>, #blocked2>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked3>
    %1 = tt.dot_scaled %arg0 scale %arg2, %arg1 scale %arg3, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x128xf8E4M3FN, #blocked>, tensor<32x4xi8, #blocked2> * tensor<128x32xf8E4M3FN, #blocked1>, tensor<32x4xi8, #blocked2> -> tensor<32x32xf32, #blocked3>
    tt.store %arg4, %1 : tensor<32x32x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}
