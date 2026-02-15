// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm --debug| FileCheck %s

// CHECK-LABEL: convert_identity
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @convert_identity(%arg0: tensor<128x128xf16, #blocked>) {
    %1 = ttg.convert_layout %arg0 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked>
    tt.return
  }
}
