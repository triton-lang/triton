// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

// CHECK-LABEL: amd_in_thread_transpose
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose(%arg0: tensor<32x32xf16, #blocked>) {
    // CHECK-COUNT-16: llvm.extractvalue
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<4xf16>
    %0 = amdgpu.in_thread_transpose %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #linear>
    ttg.local_alloc %0 : (tensor<32x32xf16, #linear>) -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    tt.return
  }
}
