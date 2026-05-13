// RUN: triton-opt %s -split-input-file -tritonamdgpu-warp-pipeline -verify-diagnostics

// Loops are not allowed inside a warp_pipeline_stage region; see isLoopOp
// in WarpPipeliner.cpp for the rationale (no scheduling benefit, opaque to
// MemoryEffectOpInterface, also covers the "no nested warp pipelines"
// rule).  Both the loop-form (createPipeline) and flat-form
// (createFlatPipeline) must reject loops between borders.

// ---- Loop-form: scf.for inside a stage ----

tt.func @loop_form_for_in_cluster(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %n step %c1 {
    %a = arith.addi %i, %c1 : index
    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage"}

    // expected-error @+1 {{loop op cannot appear inside a warp_pipeline_stage region}}
    scf.for %j = %c0 to %n step %c1 {
      scf.yield
    }

    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage"}
    %b = arith.addi %a, %i : index

    scf.yield
  }

  tt.return
}

// -----

// ---- Loop-form: scf.while inside a stage ----

tt.func @loop_form_while_in_cluster(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %n step %c1 {
    %a = arith.addi %i, %c1 : index
    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage"}

    // expected-error @+1 {{loop op cannot appear inside a warp_pipeline_stage region}}
    scf.while (%w = %c0) : (index) -> index {
      %cond = arith.cmpi slt, %w, %n : index
      scf.condition(%cond) %w : index
    } do {
    ^bb0(%w: index):
      %wn = arith.addi %w, %c1 : index
      scf.yield %wn : index
    }

    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage"}
    %b = arith.addi %a, %i : index

    scf.yield
  }

  tt.return
}

// -----

// ---- Loop-form: nested warp-pipelined scf.for is still a loop ----
//
// Even an already-pipelined inner loop is rejected: nesting warp pipelines
// is a hard constraint, and the loop-op check enforces it for free.

tt.func @loop_form_nested_pipelined_for(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %n step %c1 {
    %a = arith.addi %i, %c1 : index
    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage"}

    // expected-error @+1 {{loop op cannot appear inside a warp_pipeline_stage region}}
    scf.for %j = %c0 to %n step %c1 {
      scf.execute_region {
        scf.yield
      } {triton.warp_pipeline.stage = "inner"}
      scf.yield
    } {triton.warp_pipeline.pipelined_for}

    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage"}
    %b = arith.addi %a, %i : index

    scf.yield
  }

  tt.return
}

// -----

// ---- Flat-form: scf.for between flat borders ----

tt.func @flat_form_for_in_cluster(%n: index, %ptr: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %v0 = arith.constant 0.0 : f32

  tt.store %ptr, %v0 : !tt.ptr<f32>
  rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage0"}

  // expected-error @+1 {{loop op cannot appear inside a warp_pipeline_stage region}}
  scf.for %j = %c0 to %n step %c1 {
    scf.yield
  }

  rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage1"}
  tt.store %ptr, %v0 : !tt.ptr<f32>

  tt.return
}
