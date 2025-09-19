// RUN: triton-opt %s --tritongpu-reduce-data-duplication --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch="gfx942" -split-input-file | FileCheck %s --check-prefix=GFX942
// RUN: triton-opt %s --tritongpu-reduce-data-duplication --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch="gfx950" -split-input-file | FileCheck %s --check-prefix=GFX950

#mfma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#dotop = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GFX942-LABEL: shortcut_mfma16
  tt.func public @shortcut_mfma16(%arg0: tensor<16x16xf16, #mfma>) {
    // GFX942-NOT: store
    // GFX942-NOT: load
    // GFX942: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #mfma> -> tensor<16x16xf16, #dotop>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GFX942-LABEL: mfma_dot_cvt_bf8_mfma32_v3
  tt.func public @mfma_dot_cvt_bf8_mfma32_v3(%arg0: tensor<128x32xf8E5M2, #mfma>) {
    // GFX942-NOT: store
    // GFX942-NOT: load
    // GFX942: rocdl.ds_bpermute
    // GFX942: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf8E5M2, #mfma> -> tensor<128x32xf8E5M2, #dotop0>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
// GFX950-LABEL: mfma_dot_cvt_bf8_mfma32_v4
  tt.func public @mfma_dot_cvt_bf8_mfma32_v4(%arg0: tensor<128x32xf8E5M2, #mfma>) {
    // GFX950-NOT: rocdl.ds_bpermute
    // GFX950-COUNT-2: llvm.call_intrinsic "llvm.amdgcn.permlane32.swap"
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf8E5M2, #mfma> -> tensor<128x32xf8E5M2, #dotop0>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GFX942-LABEL: mfma_dot_cvt_bf8_mfma16_v3
  tt.func public @mfma_dot_cvt_bf8_mfma16_v3(%arg0: tensor<128x32xf8E5M2, #mfma>) {
    // GFX942-NOT: store
    // GFX942-NOT: load
    // GFX942: rocdl.ds_bpermute
    // GFX942: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf8E5M2, #mfma> -> tensor<128x32xf8E5M2, #dotop0>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GFX950-LABEL: mfma_dot_cvt_bf8_mfma16_v4
  tt.func public @mfma_dot_cvt_bf8_mfma16_v4(%arg0: tensor<128x32xf8E5M2, #mfma>) {
    // GFX950-NOT: rocdl.ds_bpermute
    // GFX950: llvm.call_intrinsic "llvm.amdgcn.permlane32.swap"
    // GFX950: llvm.call_intrinsic "llvm.amdgcn.permlane16.swap"
    // GFX950: llvm.call_intrinsic "llvm.amdgcn.permlane32.swap"
    // GFX950: llvm.call_intrinsic "llvm.amdgcn.permlane16.swap"
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf8E5M2, #mfma> -> tensor<128x32xf8E5M2, #dotop0>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [64, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GFX950-LABEL: mfma_linear_permlane_swap
  tt.func public @mfma_linear_permlane_swap(%arg0: tensor<128x128xf16, #mma>) {
  // GFX950-COUNT-16: llvm.call_intrinsic "llvm.amdgcn.permlane32.swap"
    %1 = ttg.convert_layout %arg0: tensor<128x128xf16, #mma> -> tensor<128x128xf16, #linear>
    tt.return
  }
}
