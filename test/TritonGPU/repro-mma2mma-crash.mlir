// Minimal reproducer for a direct MMA→MMA convert.
// This is intended to be run on an environment with triton-opt built from main.
// If the underlying issue is present, the pass pipeline below may plan an
// unsupported conversion path and assert (e.g., "Unexpected mma -> mma layout conversion").
//
// RUN: triton-opt -split-input-file %s -pass-pipeline='builtin.module(canonicalize,tritongpu-remove-layout-conversions,canonicalize)' --verify-each

module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Direct convert_layout from one Nvidia MMA encoding to another.
  // Adjust versions/warps/instrShape if needed to match the failure signature you see locally.
  tt.func @mma_to_mma(%arg0: tensor<128x128xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8] }>>) -> tensor<128x128xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>> {
    %0 = ttg.convert_layout %arg0
         : tensor<128x128xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8] }>>
        -> tensor<128x128xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>>
    tt.return %0 : tensor<128x128xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>>
  }
}


