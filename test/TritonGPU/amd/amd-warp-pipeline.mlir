// RUN: triton-opt %s -tritonamdgpu-warp-pipeline | FileCheck %s

// ---- 3-stage example (two borders) ----

tt.func @three_stage_example(%n: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index

  scf.for %i = %c0 to %n step %c1 {
    // Stage 0 (before first border)
    %a  = arith.addi %i, %c1 : index
    %a2 = arith.muli %a, %c1 : index

    // explicit split point → next stage begins
    rocdl.sched.barrier 0 {triton.warp_pipeline.border} 

    // Stage 1
    %b  = arith.addi %a2, %i : index

    // explicit split point → next stage begins
    rocdl.sched.barrier 0 {triton.warp_pipeline.border} 

    // Stage 2
    %c  = arith.addi %b, %a : index
    %d  = arith.muli %c, %c1 : index

    scf.yield
  }

  tt.return
}

// CHECK-LABEL: tt.func @three_stage_example(
// CHECK: scf.for
//
// Inside the loop we expect exactly three execute_region clusters:
// CHECK:   scf.execute_region
// CHECK:   scf.execute_region
// CHECK:   scf.execute_region
// CHECK: triton.warp_pipeline.lead_stages = 1 : i32, triton.warp_pipeline.total_stages = 3 : i32
//
// And the split markers must be gone:
// CHECK-NOT: rocdl.sched.barrier
// CHECK: tt.return


// ---- 2-stage example (one border) ----

tt.func @two_stage_example(%n: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index

  scf.for %i = %c0 to %n step %c1 {
    // Stage 0
    %x = arith.addi %i, %c1 : index

    // split to Stage 1
    rocdl.sched.barrier 0 {triton.warp_pipeline.border} 

    // Stage 1
    %y = arith.muli %x, %c1 : index

    scf.yield
  }

  tt.return
}

// CHECK-LABEL: tt.func @two_stage_example(
// CHECK: scf.for
// CHECK:   scf.execute_region
// CHECK:   scf.execute_region
// CHECK: triton.warp_pipeline.lead_stages = 1 : i32, triton.warp_pipeline.total_stages = 2 : i32
// CHECK-NOT: rocdl.sched.barrier
// CHECK: tt.return


// ---- Negative: no border → no structuring ----

tt.func @no_split_example(%n: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index

  scf.for %i = %c0 to %n step %c1 {
    %x = arith.addi %i, %c1 : index
    %y = arith.muli %x, %c1 : index
    scf.yield
  }

  tt.return
}

// CHECK-LABEL: tt.func @no_split_example(
// CHECK: scf.for
// CHECK-NOT: scf.execute_region
// CHECK-NOT: total_stages
// CHECK: tt.return
