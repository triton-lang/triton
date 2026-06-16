// RUN: triton-opt %s -split-input-file -convert-warp-pipeline="gfx-arch=gfx950" | FileCheck %s --check-prefix=TAIL
// RUN: triton-opt %s -split-input-file -convert-warp-pipeline="gfx-arch=gfx950 backedge-barrier-to-head=true" | FileCheck %s --check-prefix=HEAD

// Verify the opt-in schedule hint moves the cluster-0 backedge barrier to the loop head.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @generic_stage0_backedge_barrier(%n: index, %ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %v0  = arith.constant 0.0 : f32
    %v1  = arith.constant 1.0 : f32

    scf.for %i = %c0 to %n step %c1 {
      scf.execute_region {
        tt.store %ptr0, %v0 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "stage0"}

      scf.execute_region {
        tt.store %ptr1, %v1 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "stage1"}

      scf.yield
    } {triton.warp_pipeline.pipelined_for}

    tt.return
  }
}

// TAIL-LABEL: tt.func @generic_stage0_backedge_barrier
// TAIL: scf.for
// TAIL-NOT: rocdl.s.barrier
// TAIL: tt.store
// TAIL: rocdl.sched.barrier
// TAIL-NEXT: rocdl.s.barrier
// TAIL-NEXT: rocdl.sched.barrier
// TAIL: tt.store
// TAIL: rocdl.sched.barrier
// TAIL-NEXT: rocdl.s.barrier
// TAIL-NEXT: rocdl.sched.barrier
// TAIL: amdg.cond_barrier

// HEAD-LABEL: tt.func @generic_stage0_backedge_barrier
// HEAD: scf.for
// HEAD: rocdl.sched.barrier
// HEAD-NEXT: rocdl.s.barrier
// HEAD-NEXT: rocdl.sched.barrier
// HEAD: tt.store
// HEAD: rocdl.sched.barrier
// HEAD-NEXT: rocdl.s.barrier
// HEAD-NEXT: rocdl.sched.barrier
// HEAD: tt.store
// HEAD-NOT: rocdl.s.barrier
// HEAD: amdg.cond_barrier
