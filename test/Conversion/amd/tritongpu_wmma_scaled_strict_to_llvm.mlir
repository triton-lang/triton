// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250-strict --convert-builtin-func-to-llvm --verify-diagnostics | FileCheck %s
//
// gfx1250-strict lowers 16x16x128 scaled WMMA, but rejects the 32x16x128 fp4
// scaled WMMA layouts.

#scale_a = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
#scale_b = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = true, instrShape = [16, 16, 128]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250-strict", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_strict_scaled_fp8
  tt.func @wmma_strict_scaled_fp8(
      %a: tensor<32x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>,
      %a_scale: tensor<32x4xi8, #scale_a>,
      %b: tensor<128x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>,
      %b_scale: tensor<32x4xi8, #scale_b>,
      %c: tensor<32x32xf32, #mma>) {
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4"
    %d = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %c
        lhs = e4m3 rhs = e4m3 {fastMath = false} :
        tensor<32x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>,
        tensor<32x4xi8, #scale_a> *
        tensor<128x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>,
        tensor<32x4xi8, #scale_b> -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = false, instrShape = [32, 16, 128]}>
#mma1 = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = false, instrShape = [32, 16, 64]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250-strict", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wmma_strict_reject_scaled_fp4_32x16(
      %a: tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>,
      %a_scale: tensor<64x4xi8, #linear>,
      %b: tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>,
      %b_scale: tensor<32x4xi8, #linear1>,
      %c: tensor<64x32xf32, #mma>) {
    // expected-error @+2 {{wmma scale intrinsic llvm.amdgcn.wmma.scale.f32.32x16x128.f4 is not supported on gfx1250-strict}}
    // expected-error @+1 {{failed to legalize operation}}
    %d = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %c
        lhs = e2m1 rhs = e2m1 {fastMath = false} :
        tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>,
        tensor<64x4xi8, #linear> *
        tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>,
        tensor<32x4xi8, #linear1> -> tensor<64x32xf32, #mma>
    tt.return
  }
}
