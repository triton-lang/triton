// RUN: triton-opt %s -split-input-file --convert-nv-gpu-to-llvm | FileCheck %s
#SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32,  "triton_gpu.num-ctas" = 2 : i32} {
  tt.func @test_mbarrier() {
    %mbarrier = llvm.mlir.zero : !llvm.ptr<i64, 3>
    %pred = arith.constant 1 : i1
    // CHECK: llvm.inline_asm
    nvgpu.mbarrier_init %mbarrier, %pred { count = 32 : i32 } : !llvm.ptr<i64, 3>
    // CHECK: llvm.inline_asm
    nvgpu.mbarrier_arrive %mbarrier, %pred {arriveType = 1 : i32}: !llvm.ptr<i64, 3>
    // CHECK: llvm.inline_asm
    nvgpu.mbarrier_arrive %mbarrier, %pred {arriveType = 0 : i32}: !llvm.ptr<i64, 3>
    // CHECK: llvm.inline_asm
    nvgpu.mbarrier_arrive %mbarrier, %pred {arriveType = 2 : i32, txCount = 128 : i32}: !llvm.ptr<i64, 3>
    // CHECK: llvm.inline_asm
    nvgpu.mbarrier_wait %mbarrier, %pred : !llvm.ptr<i64, 3>, i1
    tt.return
  }
} // end module
