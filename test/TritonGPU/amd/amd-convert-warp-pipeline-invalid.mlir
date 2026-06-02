// RUN: triton-opt %s -split-input-file -convert-warp-pipeline="gfx-arch=gfx950" -verify-diagnostics

// validatePipelinedForBody runs upfront, before any IR mutation, so a
// malformed `pipelined_for` body fails the pass with no partial conversion.

// ==== Non-warp-pipeline scf.execute_region inside a pipelined_for body ====

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @bad_unmarked_execute_region(%n: index, %ptr: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v0 = arith.constant 0.0 : f32
    %v1 = arith.constant 1.0 : f32

    scf.for %i = %c0 to %n step %c1 {
      scf.execute_region {
        tt.store %ptr, %v0 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "stage0"}

      // expected-error @+1 {{non-warp-pipeline scf.execute_region inside pipelined_for body}}
      scf.execute_region {
        tt.store %ptr, %v1 : !tt.ptr<f32>
        scf.yield
      }

      scf.yield
    } {triton.warp_pipeline.pipelined_for}

    tt.return
  }
}

// -----

// ==== Multiple pre-existing barriers between two stages ====

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @bad_double_barrier_between_stages(%n: index, %ptr: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v0 = arith.constant 0.0 : f32
    %v1 = arith.constant 1.0 : f32

    scf.for %i = %c0 to %n step %c1 {
      scf.execute_region {
        tt.store %ptr, %v0 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "stage0"}

      amdg.async_wait {num_inst = 0 : i32}
      // expected-error @+1 {{multiple pre-existing barriers between pipeline stages}}
      amdg.async_wait {num_inst = 0 : i32}

      scf.execute_region {
        tt.store %ptr, %v1 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "stage1"}

      scf.yield
    } {triton.warp_pipeline.pipelined_for}

    tt.return
  }
}

// -----

// ==== Both top-of-loop and bottom-of-loop pre-existing barriers ====

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @bad_top_and_bottom_barriers(%n: index, %ptr: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v0 = arith.constant 0.0 : f32
    %v1 = arith.constant 1.0 : f32

    // expected-error @+1 {{both top-of-loop and bottom-of-loop pre-existing barriers}}
    scf.for %i = %c0 to %n step %c1 {
      amdg.async_wait {num_inst = 0 : i32}

      scf.execute_region {
        tt.store %ptr, %v0 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "stage0"}

      scf.execute_region {
        tt.store %ptr, %v1 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "stage1"}

      amdg.async_wait {num_inst = 0 : i32}

      scf.yield
    } {triton.warp_pipeline.pipelined_for}

    tt.return
  }
}

// -----

// ==== Unexpected op inside a pipelined_for body ====
//
// Anything that is not a warp-pipeline stage, an ignorable barrier/wait,
// or scf.yield must be rejected upfront.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @bad_unexpected_op_in_body(%n: index, %ptr: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v0 = arith.constant 0.0 : f32
    %v1 = arith.constant 1.0 : f32

    scf.for %i = %c0 to %n step %c1 {
      scf.execute_region {
        tt.store %ptr, %v0 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "stage0"}

      // expected-error @+1 {{unexpected op inside pipelined_for body}}
      %x = arith.addi %i, %c1 : index

      scf.execute_region {
        tt.store %ptr, %v1 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "stage1"}

      scf.yield
    } {triton.warp_pipeline.pipelined_for}

    tt.return
  }
}
