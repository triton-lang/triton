// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s --enable-var-scope --check-prefix=CDNA5

// CDNA5: [[$MMRA_TAG:#[A-Za-z0-9_]+]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"local">

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx1250", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CDNA5-LABEL: init_barrier
  tt.func @init_barrier(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // CDNA5:   %[[INIT_VAL1:.+]] = llvm.mlir.constant(4294967297 : i64) : i64
    // CDNA5:   %[[ALLOC_PTR:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<3>, i32)>
    // CDNA5:   llvm.cond_br %{{.+}}, ^[[BB0:.+]], ^[[BB1:.+]]
    // CDNA5-NEXT: ^[[BB0]]:
    // CDNA5-NEXT:   llvm.store %[[INIT_VAL1]], %[[ALLOC_PTR]] : i64, !llvm.ptr<3>
    // CDNA5-NEXT:   llvm.br ^[[BB1]]
    // CDNA5-NEXT: ^[[BB1]]:
    // CDNA5-NEXT:   llvm.fence syncscope("workgroup") release {llvm.mmra = [[$MMRA_TAG]]}
    // CDNA5-NEXT:   rocdl.s.barrier
    // CDNA5-NEXT:   llvm.fence syncscope("workgroup") acquire {llvm.mmra = [[$MMRA_TAG]]}
    // CDNA5-NEXT:   llvm.return
    amdg.init_barrier %alloc, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // CDNA5-LABEL: wait_barrier
  tt.func @wait_barrier(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %phase: i32) {
    // CDNA5: rocdl.s.sleep {{.*}}
    // CDNA5: llvm.load {{.*}} : !llvm.ptr<3> -> i64
    // CDNA5: llvm.icmp "ne" {{%arg1, %.*|%.*, %arg1}} : i32
    amdg.wait_barrier %alloc, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // CDNA5-LABEL: arrive_barrier
  tt.func @arrive_barrier(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // CDNA5: %[[UPDATE_VAL1:.+]] = llvm.mlir.constant(1 : i64) : i64
    // CDNA5: %[[ALLOC_PTR:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<3>, i32)>
    // CDNA5: rocdl.ds.atomic.barrier.arrive.rtn.b64 %[[ALLOC_PTR]], %[[UPDATE_VAL1]] : !llvm.ptr<3>, i64 -> i64
    %prior_phase = amdg.arrive_barrier %alloc, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> i32
    tt.return
  }

  // CDNA5-LABEL: async_copy_mbarrier_arrive
  tt.func @async_copy_mbarrier_arrive(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // CDNA5: %[[ALLOC_PTR:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<3>, i32)>
    // CDNA5: rocdl.ds.atomic.async.barrier.arrive.b64 %[[ALLOC_PTR]] : !llvm.ptr<3>
    amdg.async_copy_mbarrier_arrive %alloc : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}
