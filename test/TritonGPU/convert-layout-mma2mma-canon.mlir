// RUN: triton-opt -canonicalize %s | FileCheck %s

module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mma_to_mma
  tt.func @mma_to_mma(%arg0: tensor<128x128xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8] }>>) -> tensor<128x128xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>> {
    // The canonicalization should decompose this MMA->MMA convert into two converts
    // with a blocked layout in the middle.
    // CHECK: ttg.convert_layout %
    // CHECK-SAME: -> tensor<128x128xf16, #ttg.blocked<
    // CHECK: ttg.convert_layout %
    // CHECK-SAME: tensor<128x128xf16, #ttg.blocked<
    // CHECK-SAME: -> tensor<128x128xf16, #ttg.nvidia_mma<
    %0 = ttg.convert_layout %arg0 : tensor<128x128xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8] }>> -> tensor<128x128xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>>
    tt.return %0 : tensor<128x128xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>>
  }
}


