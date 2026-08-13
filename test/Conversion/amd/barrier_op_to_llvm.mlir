// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx906 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=TAGGED,GENERIC
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=TAGGED,GENERIC
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=TAGGED,GENERIC
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=TAGGED,GENERIC
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1200 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=TAGGED,GENERIC
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=TAGGED,GENERIC

// TAGGED-DAG: [[$LOCAL_MMRA_TAG:#[A-Za-z0-9_]+]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"local">

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // TAGGED-LABEL: llvm.func @lower_barrier
  tt.func @lower_barrier() {
    // TAGGED: llvm.fence syncscope("workgroup") release {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // TAGGED-NEXT: rocdl.s.barrier
    // TAGGED-NEXT: llvm.fence syncscope("workgroup") acquire {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    ttg.barrier local
    tt.return
  }

  // Global ordering keeps the fence untagged so LLVM emits wait + invalidate.
  // GENERIC-LABEL: llvm.func @lower_barrier_local_and_global
  tt.func @lower_barrier_local_and_global() {
    // GENERIC: llvm.fence syncscope("workgroup") release{{$}}
    // GENERIC-NEXT: rocdl.s.barrier
    // GENERIC-NEXT: llvm.fence syncscope("workgroup") acquire{{$}}
    ttg.barrier local|global_read
    tt.return
  }
}
