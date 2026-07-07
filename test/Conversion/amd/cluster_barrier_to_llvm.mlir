// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s

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

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: atomic_cluster_barrier
  tt.func @atomic_cluster_barrier(%ptr: !tt.ptr<i32>, %cmp: i32, %val: i32) {
    // CHECK-NOT: rocdl.s.barrier
    // CHECK: llvm.cmpxchg {{.*}} syncscope("agent") acquire monotonic
    // CHECK: rocdl.s.barrier.signal id = -3
    // CHECK: rocdl.s.barrier.wait id = -3
    // CHECK-NOT: rocdl.s.barrier
    %old = tt.atomic_cas acquire, gpu, %ptr, %cmp, %val : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
  }
}
