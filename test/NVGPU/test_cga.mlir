// RUN: triton-opt %s -split-input-file --convert-nv-gpu-to-llvm | FileCheck %s
#SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 2 : i32} {
  tt.func @test_mbarrier() {
    %addr = arith.constant 32 : i32
    %data = arith.constant 123 : i32
    %pred = arith.constant 1 : i1
    %id0 = arith.constant 0 : i32
    %id1 = arith.constant 1 : i32
    // CHECK: llvm.inline_asm
    // CHECK: llvm.inline_asm
    // CHECK: llvm.inline_asm
    nvgpu.cga_barrier_sync
    nvgpu.cga_barrier_arrive
    nvgpu.cga_barrier_wait

    %ptr = llvm.mlir.zero : !llvm.ptr<i32, 3>

    // CHECK: llvm.inline_asm
    %v = nvgpu.cluster_id
    llvm.store %v, %ptr : !llvm.ptr<i32, 3>

    tt.return
  }
} // end module
