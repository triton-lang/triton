// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx906 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=GENERIC
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=TAGGED
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=TAGGED
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=GENERIC
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1200 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=RDNA4
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=TAGGED

// RDNA & GCN5_1 targets fall back to the generic barrier lowering without MMRA tags.
// TAGGED-DAG: [[$LOCAL_MMRA_TAG:#[A-Za-z0-9_]+]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"local">

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // TAGGED-LABEL: llvm.func @lower_barrier
  // GENERIC-LABEL: llvm.func @lower_barrier
  // RDNA4-LABEL: llvm.func @lower_barrier
  tt.func @lower_barrier() {
    // TAGGED: llvm.fence syncscope("workgroup") release {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // TAGGED-NEXT: rocdl.s.barrier
    // TAGGED-NEXT: llvm.fence syncscope("workgroup") acquire {llvm.mmra = [[$LOCAL_MMRA_TAG]]}

    // GENERIC: llvm.fence syncscope("workgroup") release{{$}}
    // GENERIC-NEXT: rocdl.s.barrier
    // GENERIC-NEXT: llvm.fence syncscope("workgroup") acquire{{$}}

    // RDNA4: llvm.fence syncscope("workgroup") release{{$}}
    // RDNA4-NEXT: rocdl.s.barrier.signal id = -1
    // RDNA4-NEXT: rocdl.s.barrier.wait id = -1
    // RDNA4-NEXT: llvm.fence syncscope("workgroup") acquire{{$}}
    ttg.barrier local
    tt.return
  }
}
