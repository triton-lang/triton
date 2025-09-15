// RUN: triton-opt %s --tritongpu-reduce-data-duplication --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch="gfx1100" -split-input-file | FileCheck %s

#wmmaT = #ttg.amd_wmma<{version = 1, warpsPerCTA = [1, 1], isTranspose = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #wmmaT, kWidth=16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_dot_cvt_bf16_wmma
  tt.func public @wmma_dot_cvt_bf16_wmma(%arg0: tensor<16x16xbf16, #wmmaT>) {
    // CHECK-NOT: store
    // CHECK-NOT: load
    // CHECK-COUNT-4: rocdl.permlanex16
    // CHECK: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<16x16xbf16, #wmmaT> -> tensor<16x16xbf16, #dotop0>
    tt.return
  }
}
