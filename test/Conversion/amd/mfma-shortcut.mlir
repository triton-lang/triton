// RUN: triton-opt %s --decompose-unsupported-amd-conversions --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch="gfx90a" -split-input-file | FileCheck %s

#mfma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#dotop = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: shortcut_mfma16
  tt.func public @shortcut_mfma16(%arg0: tensor<16x16xf16, #mfma>) {
    // CHECK-NOT: store
    // CHECK-NOT: load
    // CHECK: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #mfma> -> tensor<16x16xf16, #dotop>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#dotop = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: no_shortcut_mfma16
  tt.func public @no_shortcut_mfma16(%arg0: tensor<16x16xf16, #mfma>) {
    // CHECK: store
    // CHECK: load
    // CHECK: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #mfma> -> tensor<16x16xf16, #dotop>
    tt.return
  }
}
