// RUN: triton-opt %s -split-input-file -tritongpu-reduce-data-duplication | FileCheck %s

//       CHECK:   #[[$SHARED:.*]] = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [0, 1]}
//       CHECK-LABEL: apply_swizzle
//       CHECK:   %{{.*}} = ttg.local_alloc %{{.*}} : (tensor<16x256xf16, #{{.*}}>) -> !ttg.memdesc<16x256xf16, #[[$SHARED]], #smem>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @apply_swizzle(%arg0: tensor<16x256xf16, #blocked>) {
    %0 = ttg.convert_layout %arg0 : tensor<16x256xf16, #blocked> -> tensor<16x256xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
}

// -----

//       CHECK-LABEL:   conversion_shortcut_blocked_dotop_warp32
//       CHECK-NOT:  ttg.local_alloc
//       CHECK: ttg.convert_layout
//       CHECK-NOT:  ttg.local_alloc
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [0, 1]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @conversion_shortcut_blocked_dotop_warp32(%arg0: tensor<64x64xf16, #blocked>) {
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

//       CHECK-LABEL:   conversion_shortcut_blocked_dotop_warp64
//       CHECK-NOT:  ttg.local_alloc
//       CHECK: ttg.convert_layout
//       CHECK-NOT:  ttg.local_alloc
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 2], warpsPerCTA = [2, 2], order = [0, 1]}>
module attributes {"ttg.target" = "hip:gfx940", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @conversion_shortcut_blocked_dotop_warp64(%arg0: tensor<64x64xf16, #blocked>) {
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}
