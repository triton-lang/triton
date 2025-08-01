// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch="gfx1200" | FileCheck %s

#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [8, 0], [16, 0]], lane = [[1, 0], [2, 0], [4, 0], [0, 8], [0, 16]], warp = [[0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 2, isTranspose = false, warpsPerCTA = [1, 2]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_dot0_transpose
  tt.func @wmma_dot0_transpose(%arg0: !tt.ptr<bf16>) {
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>, #linear>
    // CHECK-COUNT-2: llvm.amdgcn.global.load.tr.b128
    %1 = amdgpu.global_load_transpose %0 : tensor<32x32x!tt.ptr<bf16>, #linear> -> tensor<32x32xbf16, #linear1>
    %2 = ttg.convert_layout %1 : tensor<32x32xbf16, #linear1> -> tensor<32x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    tt.return
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 8], [0, 16]], lane = [[0, 1], [0, 2], [0, 4], [8, 0], [16, 0]], warp = [[0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 2, isTranspose = false, warpsPerCTA = [1, 2]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_dot1_transpose
  tt.func @wmma_dot1_transpose(%arg0: !tt.ptr<bf16>) {
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>, #linear>
    // CHECK-COUNT-2: llvm.amdgcn.global.load.tr.b128
    %1 = amdgpu.global_load_transpose %0 : tensor<32x32x!tt.ptr<bf16>, #linear> -> tensor<32x32xbf16, #linear1>
    %2 = ttg.convert_layout %1 : tensor<32x32xbf16, #linear1> -> tensor<32x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    tt.return
  }
}
