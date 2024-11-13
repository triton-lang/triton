// RUN: triton-opt %s --decompose-unsupported-amd-conversions --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch="gfx942" -split-input-file | FileCheck %s

#mfma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#dotop = #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth=4}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: shortcut_mfma16
  tt.func public @shortcut_mfma16(%arg0: tensor<16x16xf16, #mfma>) {
    // CHECK-NOT: store
    // CHECK-NOT: load
    // CHECK: llvm.return
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xf16, #mfma> -> tensor<16x16xf16, #dotop>
    tt.return
  }
}

// -----

#mfma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#dotop = #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: no_shortcut_mfma16
  tt.func public @no_shortcut_mfma16(%arg0: tensor<16x16xf16, #mfma>) {
    // CHECK: store
    // CHECK: load
    // CHECK: llvm.return
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xf16, #mfma> -> tensor<16x16xf16, #dotop>
    tt.return
  }
}

// -----

#mfma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#dotop0 = #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: mfma_dot_cvt_f8
  tt.func public @mfma_dot_cvt_f8(%arg0: tensor<128x32xf8E4M3FNUZ, #mfma>) {
    // CHECK-NOT: store
    // CHECK-NOT: load
    // CHECK: rocdl.ds_bpermute
    %0 = triton_gpu.convert_layout %arg0 : tensor<128x32xf8E4M3FNUZ, #mfma> -> tensor<128x32xf8E4M3FNUZ, #dotop0>
    tt.return
  }
}
