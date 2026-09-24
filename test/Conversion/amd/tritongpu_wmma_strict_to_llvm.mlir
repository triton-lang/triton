// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250-strict --convert-builtin-func-to-llvm | FileCheck %s
//
// gfx1250-strict has no N16 WMMA, fallback to fp32

#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250-strict", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @wmma_n16_emulate_bf16
  // CHECK-NOT: wmma.f32.16x16x32.bf16
  // CHECK: wmma.f32.16x16x4.f32
  // CHECK-NOT: wmma.f32.16x16x32.bf16
  tt.func @wmma_n16_emulate_bf16(
      %a: tensor<16x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
      %b: tensor<32x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
      %c: tensor<16x16xf32, #mma>,
      %out: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %d = tt.dot %a, %b, %c, inputPrecision = ieee :
        tensor<16x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> *
        tensor<32x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> ->
        tensor<16x16xf32, #mma>
    %ptr = tt.splat %out : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #mma>
    tt.store %ptr, %d : tensor<16x16x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 64]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250-strict", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @wmma_n16_emulate_fp8
  // CHECK-NOT: wmma.f32.16x16x64.fp8.fp8
  // CHECK: wmma.f32.16x16x4.f32
  // CHECK-NOT: wmma.f32.16x16x64.fp8.fp8
  tt.func @wmma_n16_emulate_fp8(
      %a: tensor<16x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
      %b: tensor<64x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
      %c: tensor<16x16xf32, #mma>,
      %out: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %d = tt.dot %a, %b, %c, inputPrecision = ieee :
        tensor<16x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> *
        tensor<64x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> ->
        tensor<16x16xf32, #mma>
    %ptr = tt.splat %out : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #mma>
    tt.store %ptr, %d : tensor<16x16x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 64]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250-strict", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @wmma_n16_emulate_i8
  // CHECK-NOT: wmma.i32.16x16x64.iu8
  // CHECK: llvm.sitofp
  // CHECK: wmma.f32.16x16x4.f32
  // CHECK: llvm.fptosi
  // CHECK-NOT: wmma.i32.16x16x64.iu8
  tt.func @wmma_n16_emulate_i8(
      %a: tensor<16x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
      %b: tensor<64x16xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
      %c: tensor<16x16xi32, #mma>,
      %out: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %d = tt.dot %a, %b, %c :
        tensor<16x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> *
        tensor<64x16xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> ->
        tensor<16x16xi32, #mma>
    %ptr = tt.splat %out : !tt.ptr<i32> -> tensor<16x16x!tt.ptr<i32>, #mma>
    tt.store %ptr, %d : tensor<16x16x!tt.ptr<i32>, #mma>
    tt.return
  }
}
