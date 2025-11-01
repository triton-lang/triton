// RUN: triton-opt %s -convert-warp-pipeline | FileCheck %s

// ---- 2-stage pipeline (basic) ----

tt.func @two_stage_backend(%n: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index

  // Frontend has already annotated total stages.
  scf.for %i = %c0 to %n step %c1 {

    // Stage 0 cluster
    scf.execute_region {
      %a0 = arith.addi %i, %c1 : index
      %x0 = arith.addi %a0, %c1 : index
      scf.yield
    } {triton.warp_pipeline.stage}

    // Stage 1 cluster
    scf.execute_region {
      %a1 = arith.addi %i, %c1 : index
      %x1 = arith.muli %a1, %c1 : index
      scf.yield
    } {triton.warp_pipeline.stage}

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
// CHECK: amdgpu.cond_barrier %[[WARPHIGH]]

// CHECK: scf.for
// CHECK-NOT:   scf.execute_region
// CHECK: rocdl.sched.barrier
// CHECK: rocdl.s.barrier
// CHECK: rocdl.sched.barrier
// CHECK-NOT:   scf.execute_region

// CHECK: amdgpu.cond_barrier %[[WARPLOW]]
// CHECK: tt.return


// ---- 3-stage pipeline (ensures multiple clusters handled) ----

tt.func @three_stage_backend(%n: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index

  scf.for %i = %c0 to %n step %c1 {

    // Stage 0
    scf.execute_region {
      %x0 = arith.addi %i, %c1 : index
      scf.yield
    } {triton.warp_pipeline.stage}
    // Stage 1
    scf.execute_region {
      %x1 = arith.muli %i, %c1 : index
      scf.yield
    } {triton.warp_pipeline.stage}
    // Stage 2
    scf.execute_region {
      %x2 = arith.addi %i, %c1 : index
      scf.yield
    } {triton.warp_pipeline.stage}

    scf.yield
  } {triton.warp_pipeline.pipelined_for}

  tt.return
}

// CHECK-LABEL: tt.func @three_stage_backend(
// CHECK-NOT: no_inline
// CHECK: gpu.barrier
// CHECK: amdgpu.cond_barrier
// CHECK: scf.for
// CHECK-NOT:   scf.execute_region
// CHECK: rocdl.sched.barrier
// CHECK: rocdl.s.barrier
// CHECK: rocdl.sched.barrier
// CHECK: amdgpu.cond_barrier
// CHECK: tt.return


// ---- Negative: no total_stages → pass should not touch the loop ----

tt.func @no_total_stages(%n: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    scf.execute_region {
      %x = arith.addi %i, %c1 : index
      scf.yield
    }
    scf.yield
  }
  tt.return
}

// CHECK-LABEL: tt.func @no_total_stages(
// CHECK-NOT: gpu.barrier
// CHECK-NOT: amdgpu.cond_barrier
// CHECK: scf.for
// CHECK:   scf.execute_region
// CHECK: tt.return
