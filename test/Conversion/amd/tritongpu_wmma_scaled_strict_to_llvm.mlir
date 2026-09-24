// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250-strict --convert-builtin-func-to-llvm | FileCheck %s
//
// gfx1250-strict has no scaled N16 WMMA. 

#scale_a = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
#scale_b = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
#result = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = true, instrShape = [16, 16, 128]}>
#operand = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = true, instrShape = [16, 16, 64]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250-strict", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @wmma_n16_emulate_scaled_fp4
  // CHECK-NOT: wmma.scale
  // CHECK: llvm.fmul
  // CHECK: wmma.f32.16x16x4.f32
  // CHECK-NOT: wmma.scale
  tt.func @wmma_n16_emulate_scaled_fp4(
      %a: tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #operand, kWidth = 16}>>,
      %a_scale: tensor<32x4xi8, #scale_a>,
      %b: tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #operand, kWidth = 16}>>,
      %b_scale: tensor<32x4xi8, #scale_b>,
      %c: tensor<32x32xf32, #result>) {
    %d = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %c
        lhs = e2m1 rhs = e2m1 {fastMath = false} :
        tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #operand, kWidth = 16}>>,
        tensor<32x4xi8, #scale_a> *
        tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #operand, kWidth = 16}>>,
        tensor<32x4xi8, #scale_b> -> tensor<32x32xf32, #result>
    tt.return
  }
}

// -----

#scale_a = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
#scale_b = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = true, instrShape = [16, 16, 128]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250-strict", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @wmma_n16_emulate_scaled_fp8
  // CHECK-NOT: wmma.scale
  // CHECK: llvm.fmul
  // CHECK: wmma.f32.16x16x4.f32
  // CHECK-NOT: wmma.scale
  tt.func @wmma_n16_emulate_scaled_fp8(
      %a: tensor<32x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>,
      %a_scale: tensor<32x4xi8, #scale_a>,
      %b: tensor<128x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>,
      %b_scale: tensor<32x4xi8, #scale_b>,
      %c: tensor<32x32xf32, #mma>) {
    %d = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %c
        lhs = e4m3 rhs = e4m3 {fastMath = false} :
        tensor<32x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>,
        tensor<32x4xi8, #scale_a> *
        tensor<128x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>,
        tensor<32x4xi8, #scale_b> -> tensor<32x32xf32, #mma>
    tt.return
  }
}
