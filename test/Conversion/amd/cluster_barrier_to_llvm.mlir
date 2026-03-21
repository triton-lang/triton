// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx1250 | FileCheck %s

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: cluster_barrier_arrive
  tt.func @cluster_barrier_arrive() {
    // CHECK: rocdl.s.barrier.signal id = -3
    amdg.cluster_barrier_arrive
    tt.return
  }
}
// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: cluster_barrier_wait
  tt.func @cluster_barrier_wait() {
    // CHECK: rocdl.s.barrier.wait id = -3
    amdg.cluster_barrier_wait
    tt.return
  }
}
