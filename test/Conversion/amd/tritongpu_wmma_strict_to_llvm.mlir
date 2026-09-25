// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250-strict --convert-builtin-func-to-llvm --verify-diagnostics | FileCheck %s
//
// gfx1250-strict lowers native WMMA layouts, but rejects layouts backed by the
// fp8/bf8 K=128 intrinsics 

#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250-strict", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_strict_bf16
  tt.func @wmma_strict_bf16(
      %a: tensor<16x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
      %b: tensor<32x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
      %c: tensor<16x16xf32, #mma>,
      %out: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.wmma.f32.16x16x32.bf16"
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
  // CHECK-LABEL: wmma_strict_fp8_k64
  tt.func @wmma_strict_fp8_k64(
      %a: tensor<16x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
      %b: tensor<64x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
      %c: tensor<16x16xf32, #mma>,
      %out: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8"
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
  // CHECK-LABEL: wmma_strict_i8
  tt.func @wmma_strict_i8(
      %a: tensor<16x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
      %b: tensor<64x16xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
      %c: tensor<16x16xi32, #mma>,
      %out: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.wmma.i32.16x16x64.iu8"
    %d = tt.dot %a, %b, %c :
        tensor<16x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> *
        tensor<64x16xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> ->
        tensor<16x16xi32, #mma>
    %ptr = tt.splat %out : !tt.ptr<i32> -> tensor<16x16x!tt.ptr<i32>, #mma>
    tt.store %ptr, %d : tensor<16x16x!tt.ptr<i32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 128]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250-strict", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wmma_strict_reject_fp8_k128(
      %a: tensor<16x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>,
      %b: tensor<128x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>,
      %c: tensor<16x16xf32, #mma>,
      %out: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // expected-error @+2 {{wmma intrinsic llvm.amdgcn.wmma.f32.16x16x128.fp8.fp8 is not supported on gfx1250-strict}}
    // expected-error @+1 {{failed to legalize operation}}
    %d = tt.dot %a, %b, %c, inputPrecision = ieee :
        tensor<16x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> *
        tensor<128x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> ->
        tensor<16x16xf32, #mma>
    %ptr = tt.splat %out : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #mma>
    tt.store %ptr, %d : tensor<16x16x!tt.ptr<f32>, #mma>
    tt.return
  }
}
