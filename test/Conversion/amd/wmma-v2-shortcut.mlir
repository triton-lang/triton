// RUN: triton-opt %s --tritongpu-reduce-data-duplication --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch="gfx1200" -reconcile-unrealized-casts -split-input-file | FileCheck %s

#wmmaTv2 = #ttg.amd_wmma<{version = 2, warpsPerCTA = [1, 1], isTranspose = true}>
#dotop0v2 = #ttg.dot_op<{opIdx = 0, parent = #wmmaTv2, kWidth=8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_dot_cvt_bf16_wmma_v2
  tt.func @wmma_dot_cvt_bf16_wmma_v2(%arg0: tensor<16x16xbf16, #wmmaTv2>) {
    // CHECK-NOT: %0
    %0 = ttg.convert_layout %arg0 : tensor<16x16xbf16, #wmmaTv2> -> tensor<16x16xbf16, #dotop0v2>
    tt.return
  }
}
