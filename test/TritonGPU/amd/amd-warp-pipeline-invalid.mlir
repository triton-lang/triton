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
    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage"}

    // expected-error @+1 {{loop op cannot appear inside a warp_pipeline_stage region}}
    scf.for %j = %c0 to %n step %c1 {
      scf.yield
    }

    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage"}
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
    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage"}

    // expected-error @+1 {{loop op cannot appear inside a warp_pipeline_stage region}}
    scf.while (%w = %c0) : (index) -> index {
      %cond = arith.cmpi slt, %w, %n : index
      scf.condition(%cond) %w : index
    } do {
    ^bb0(%w: index):
      %wn = arith.addi %w, %c1 : index
      scf.yield %wn : index
    }

    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage"}
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
    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage"}

    // expected-error @+1 {{loop op cannot appear inside a warp_pipeline_stage region}}
    scf.for %j = %c0 to %n step %c1 {
      scf.execute_region {
        scf.yield
      } {triton.warp_pipeline.stage = "inner"}
      scf.yield
    } {triton.warp_pipeline.pipelined_for}

    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage"}
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
  rocdl.sched.barrier none {triton.warp_pipeline.border = "stage0"}

  // expected-error @+1 {{loop op cannot appear inside a warp_pipeline_stage region}}
  scf.for %j = %c0 to %n step %c1 {
    scf.yield
  }

  rocdl.sched.barrier none {triton.warp_pipeline.border = "stage1"}
  tt.store %ptr, %v0 : !tt.ptr<f32>

  tt.return
}

// -----

// ---- Loop-form: phase_gap on a later stage is rejected ----

tt.func @phase_gap_on_second_stage(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %n step %c1 {
    %a = arith.addi %i, %c1 : index
    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage0"}

    %b = arith.addi %a, %i : index
    // expected-error @+1 {{warp-pipeline phase_gap may only be specified on the first stage}}
    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage1", triton.warp_pipeline.phase_gap = 2 : i32}

    scf.yield
  }

  tt.return
}

// -----

// ---- Loop-form: phase_gap is currently limited to two stages ----

tt.func @phase_gap_too_wide(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %n step %c1 {
    %a = arith.addi %i, %c1 : index
    // expected-error @+1 {{warp-pipeline phase_gap must be 1 or 2}}
    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage0", triton.warp_pipeline.phase_gap = 3 : i32}

    %b = arith.addi %a, %i : index
    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage1"}

    scf.yield
  }

  tt.return
}

// -----

// ---- Loop-form: phase_gap must be an i32 integer ----

tt.func @phase_gap_wrong_type(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %n step %c1 {
    %a = arith.addi %i, %c1 : index
    // expected-error @+1 {{warp-pipeline phase_gap must be an i32 integer}}
    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage0", triton.warp_pipeline.phase_gap = "2"}

    %b = arith.addi %a, %i : index
    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage1"}

    scf.yield
  }

  tt.return
}

// -----

// ---- Flat-form: phase_gap is not silently ignored ----

tt.func @flat_phase_gap(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %a = arith.addi %c0, %c1 : index
  // expected-error @+1 {{flat warp pipelines only support phase_gap=1}}
  rocdl.sched.barrier none {triton.warp_pipeline.border = "stage0", triton.warp_pipeline.phase_gap = 2 : i32}
  %b = arith.addi %a, %n : index
  rocdl.sched.barrier none {triton.warp_pipeline.border = "stage1"}

  tt.return
}

// -----

// ---- Loop-form: one stage is not a pipeline ----

tt.func @single_stage_pipeline(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // expected-error @+1 {{warp_pipeline_stage borders did not produce at least two stages}}
  scf.for %i = %c0 to %n step %c1 {
    %a = arith.addi %i, %c1 : index
    rocdl.sched.barrier none {triton.warp_pipeline.border = "stage0"}

    scf.yield
  }

  tt.return
}
