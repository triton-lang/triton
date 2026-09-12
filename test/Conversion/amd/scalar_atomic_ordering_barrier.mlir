// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 --convert-builtin-func-to-llvm | FileCheck %s
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx906 --convert-builtin-func-to-llvm | FileCheck %s

// One elected thread runs the atomic, so the barrier broadcasting its result
// must carry the acquire for the rest of the CTA. A "local" annotation drops
// buffer_gl0_inv, and on RDNA the other CU's L0 then serves stale data.

// CHECK-DAG: [[$LOCAL_MMRA_TAG:#[A-Za-z0-9_]+]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"local">

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @scalar_atomic_acquire
  tt.func public @scalar_atomic_acquire(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    // CHECK: llvm.atomicrmw add {{.*}} acquire
    // CHECK: llvm.fence syncscope("workgroup") release{{$}}
    // CHECK-NEXT: rocdl.s.barrier
    // CHECK-NEXT: llvm.fence syncscope("workgroup") acquire{{$}}
    %0 = tt.atomic_rmw add, acquire, gpu, %arg0, %c0, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %arg1, %0 : !tt.ptr<i32>
    tt.return
  }

  // Relaxed orders nothing, so the broadcast barrier is LDS staging only.
  // CHECK-LABEL: llvm.func @scalar_atomic_relaxed
  tt.func public @scalar_atomic_relaxed(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    // CHECK: llvm.atomicrmw add {{.*}} monotonic
    // CHECK: llvm.fence syncscope("workgroup") release {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // CHECK-NEXT: rocdl.s.barrier
    // CHECK-NEXT: llvm.fence syncscope("workgroup") acquire {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    %0 = tt.atomic_rmw add, relaxed, gpu, %arg0, %c0, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %arg1, %0 : !tt.ptr<i32>
    tt.return
  }

  // Release-only is ordered by the barrier *before* the atomic, so the
  // broadcast barrier after it stays LDS-only.
  // CHECK-LABEL: llvm.func @scalar_atomic_release
  tt.func public @scalar_atomic_release(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
    %true = arith.constant true
    %c1 = arith.constant 1 : i32
    // CHECK: llvm.fence syncscope("workgroup") release{{$}}
    // CHECK-NEXT: rocdl.s.barrier
    // CHECK-NEXT: llvm.fence syncscope("workgroup") acquire{{$}}
    // CHECK: llvm.atomicrmw xchg {{.*}} release
    // CHECK: llvm.fence syncscope("workgroup") release {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // CHECK-NEXT: rocdl.s.barrier
    // CHECK-NEXT: llvm.fence syncscope("workgroup") acquire {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    %0 = tt.atomic_rmw exch, release, gpu, %arg0, %c1, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %arg1, %0 : !tt.ptr<i32>
    tt.return
  }
}
