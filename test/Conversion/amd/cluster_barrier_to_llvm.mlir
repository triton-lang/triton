// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --verify-diagnostics | FileCheck %s

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
  tt.func @unsupported_scalar_atomic_cas(%ptr: !tt.ptr<i32>, %cmp: i32, %val: i32) {
    // expected-error@+2 {{scalar atomic CAS is not supported in multi-CTA kernels on AMD}}
    // expected-error@+1 {{failed to legalize operation 'tt.atomic_cas' that was explicitly marked illegal}}
    %old = tt.atomic_cas acquire, gpu, %ptr, %cmp, %val : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @unsupported_scalar_atomic_rmw(%ptr: !tt.ptr<i32>, %val: i32) {
    // expected-error@+2 {{scalar atomic RMW is not supported in multi-CTA kernels on AMD}}
    // expected-error@+1 {{failed to legalize operation 'tt.atomic_rmw' that was explicitly marked illegal}}
    %old = tt.atomic_rmw add, acquire, gpu, %ptr, %val : (!tt.ptr<i32>, i32) -> i32
    tt.return
  }
}
