// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm='arch=gfx942' | FileCheck %s
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @conditional_barrier() {
    // CHECK: llvm.func @conditional_barrier
    // CHECK: llvm.cond_br
    // CHECK: bb1
    // CHECK: llvm.br
    // CHECK: bb2
    // CHECK: llvm.add
    // CHECK: llvm.cond_br
    // CHECK: bb3
    // CHECK: llvm.br
    // CHECK: bb4
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = rocdl.workitem.id.x : i32
    %1 = arith.divsi %0, %c256_i32 : i32
    %2 = arith.cmpi ne, %1, %c0_i32 : i32
    %3 = arith.cmpi eq, %1, %c0_i32 : i32
    amdgpu.cond_barrier %2
    %4 = arith.addi %0, %c256_i32 : i32
    amdgpu.cond_barrier %3
    tt.return
  }
}
