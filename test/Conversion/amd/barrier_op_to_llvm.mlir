// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=TAGGED
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=TAGGED
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=GENERIC
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1200 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=RDNA4
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=TAGGED
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1100 cu-mode=True" --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=TAGGED
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1200 cu-mode=True" --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=TAGGED

// In WGP mode an RDNA work-group spans two CUs with a vector L0 each, so a
// work-group barrier cannot be tagged as synchronizing only one address space:
// an `"amdgpu-synchronize-as":"local"` acquire fence emits no cache invalidate,
// leaving the sibling CU's L0 stale. Those targets fall back to the generic
// lowering. In CU mode the work-group shares one L0, as on CDNA, so the tagged
// lowering applies.
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
