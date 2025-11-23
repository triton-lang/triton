// RUN: triton-opt %s -split-input-file -convert-warp-pipeline | FileCheck %s

// ---- 2-stage pipeline (basic) ----
//
// Use tt.store inside the regions so they have memory effects and are
// preserved through the convert-warp-pipeline pass.

tt.func @two_stage_backend(%n: index, %ptr: !tt.ptr<f32>) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %v0  = arith.constant 0.0 : f32
  %v1  = arith.constant 1.0 : f32

  // Frontend has already annotated total stages.
  scf.for %i = %c0 to %n step %c1 {

    // Stage 0 cluster
    scf.execute_region {
      // Memory-effecting Triton op: cannot be DCE'd.
      tt.store %ptr, %v0 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage0"}

    // Stage 1 cluster
    scf.execute_region {
      tt.store %ptr, %v1 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage1"}

    scf.yield
  } {triton.warp_pipeline.pipelined_for}

  tt.return
}

// CHECK-LABEL: tt.func @two_stage_backend(
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %c1 = arith.constant 1 : index
// CHECK-NOT: no_inline

// === Pre-loop sync + role setup ===
// CHECK: gpu.barrier
// CHECK: arith.divsi
// CHECK: %[[WARPLOW:.+]] = arith.cmpi eq
// CHECK: %[[WARPHIGH:.+]] = arith.cmpi ne
// CHECK: amdg.cond_barrier %[[WARPHIGH]]

// After conversion, the for body is flattened and cluster barriers inserted.
// CHECK: scf.for
// CHECK-NOT:   scf.execute_region
// CHECK: rocdl.sched.barrier
// CHECK: rocdl.s.barrier
// CHECK: rocdl.sched.barrier
// CHECK-NOT:   scf.execute_region

// CHECK: amdg.cond_barrier %[[WARPLOW]]
// CHECK: tt.return


// ---- 3-stage pipeline (ensures multiple clusters handled) ----

tt.func @three_stage_backend(%n: index, %ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %v0  = arith.constant 0.0 : f32
  %v1  = arith.constant 1.0 : f32
  %v2  = arith.constant 2.0 : f32

  scf.for %i = %c0 to %n step %c1 {

    // Stage 0
    scf.execute_region {
      tt.store %ptr0, %v0 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage0"}

    // Stage 1
    scf.execute_region {
      tt.store %ptr0, %v1 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage1"}

    // Stage 2
    scf.execute_region {
      tt.store %ptr1, %v2 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage2"}

    scf.yield
  } {triton.warp_pipeline.pipelined_for}

  tt.return
}

// CHECK-LABEL: tt.func @three_stage_backend(
// CHECK-NOT: no_inline
// CHECK: gpu.barrier
// CHECK: amdg.cond_barrier
// CHECK: scf.for
// CHECK-NOT:   scf.execute_region
// CHECK: rocdl.sched.barrier
// CHECK: rocdl.s.barrier
// CHECK: rocdl.sched.barrier
// CHECK: amdg.cond_barrier
// CHECK: tt.return


// ---- Negative: no total_stages → pass should not touch the loop ----
//
// Still uses tt.store so the execute_region has side effects, but there is
// no pipelined_for annotation, so the pass must leave it as-is.

tt.func @no_total_stages(%n: index, %ptr: !tt.ptr<f32>) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %v0  = arith.constant 3.0 : f32

  scf.for %i = %c0 to %n step %c1 {
    scf.execute_region {
      tt.store %ptr, %v0 : !tt.ptr<f32>
      scf.yield
    }
    scf.yield
  }

  tt.return
}

// CHECK-LABEL: tt.func @no_total_stages(
// CHECK-NOT: gpu.barrier
// CHECK-NOT: amdg.cond_barrier
// CHECK: scf.for
// CHECK:   scf.execute_region
// CHECK: tt.return
