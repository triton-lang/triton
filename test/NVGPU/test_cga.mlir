// RUN: triton-opt %s -split-input-file --convert-nv-gpu-to-llvm | FileCheck %s
#SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 2 : i32} {
  tt.func @test_mbarrier() {
    %ptr = llvm.mlir.zero : !llvm.ptr<3>

    // CHECK: llvm.inline_asm
    %v = nvgpu.cluster_id
    llvm.store %v, %ptr : i32, !llvm.ptr<3>

    tt.return
  }
} // end module
