// RUN: triton-opt %s -split-input-file -convert-warp-pipeline="gfx-arch=gfx950" | FileCheck %s
// RUN: triton-opt %s -split-input-file -convert-warp-pipeline="gfx-arch=gfx950" -verify-diagnostics=only-expected

// warp_pipeline_stage(schedule_hint="mfma-valu") marks a register-only dot
// stage with a mask-0, size-0 sched.group.barrier at its head for the
// MFMA <-> VALU co-execution pass; delay=N becomes s_nops at the stage head.

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-LABEL: tt.func @hinted_dot_stage
tt.func @hinted_dot_stage(%a: tensor<64x32xf16, #dot_a>, %b: tensor<32x64xf16, #dot_b>, %n: index, %ptr: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
  %half = arith.constant dense<5.000000e-01> : tensor<64x64xf32, #mma>
  %v0 = arith.constant 0.0 : f32
  // CHECK: scf.for
  %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero, %q = %zero) -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) {
    // The dot stage is cluster 0, whose wrap-around barrier sits at the loop
    // tail: the marker is the first op of the body.
    // CHECK-NEXT: rocdl.sched.group.barrier none, 0, 1
    // CHECK: tt.dot
    // CHECK: math.exp2
    %d, %p = scf.execute_region -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) {
      %dot = tt.dot %a, %b, %acc : tensor<64x32xf16, #dot_a> * tensor<32x64xf16, #dot_b> -> tensor<64x64xf32, #mma>
      %s = arith.mulf %q, %half : tensor<64x64xf32, #mma>
      %e = math.exp2 %s : tensor<64x64xf32, #mma>
      scf.yield %dot, %e : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
    } {triton.warp_pipeline.stage = "dot", triton.warp_pipeline.schedule_hint = "mfma-valu"}
    // The memory stage: two s_nop 7 for delay = 16 inside its cluster barrier,
    // between the s_barrier and the sched_barrier that closes it, so nothing
    // of the stage can be scheduled above them.
    // CHECK: rocdl.sched.barrier none
    // CHECK-NEXT: rocdl.s.barrier
    // CHECK-NEXT: rocdl.s.nop 7
    // CHECK-NEXT: rocdl.s.nop 7
    // CHECK-NEXT: rocdl.sched.barrier none
    // CHECK-NEXT: tt.store
    scf.execute_region {
      tt.store %ptr, %v0 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "mem", triton.warp_pipeline.delay = 16 : i32}
    scf.yield %d, %p : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
  } {triton.warp_pipeline.pipelined_for}
  tt.return
}

}

// -----

// A stage whose VALU work consumes its own dot result does not qualify: the
// hint is dropped with a warning and no marker is emitted.

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-LABEL: tt.func @dependent_dot_stage
// CHECK-NOT: rocdl.sched.group.barrier
tt.func @dependent_dot_stage(%a: tensor<64x32xf16, #dot_a>, %b: tensor<32x64xf16, #dot_b>, %n: index, %ptr: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
  %v0 = arith.constant 0.0 : f32
  %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero, %q = %zero) -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) {
    %d, %p = scf.execute_region -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) {
      %dot = tt.dot %a, %b, %acc : tensor<64x32xf16, #dot_a> * tensor<32x64xf16, #dot_b> -> tensor<64x64xf32, #mma>
      // expected-warning @below {{schedule_hint="mfma-valu" dropped: this op consumes the stage's tt.dot result}}
      %e = math.exp2 %dot : tensor<64x64xf32, #mma>
      scf.yield %dot, %e : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
    } {triton.warp_pipeline.stage = "dot", triton.warp_pipeline.schedule_hint = "mfma-valu"}
    scf.execute_region {
      tt.store %ptr, %v0 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "mem"}
    scf.yield %d, %p : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
  } {triton.warp_pipeline.pipelined_for}
  tt.return
}

}

// -----

// A stage that touches memory does not qualify either.

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-LABEL: tt.func @memory_in_dot_stage
// CHECK-NOT: rocdl.sched.group.barrier
tt.func @memory_in_dot_stage(%a: tensor<64x32xf16, #dot_a>, %b: tensor<32x64xf16, #dot_b>, %n: index, %ptr: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
  %v0 = arith.constant 0.0 : f32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> (tensor<64x64xf32, #mma>) {
    %d = scf.execute_region -> (tensor<64x64xf32, #mma>) {
      %dot = tt.dot %a, %b, %acc : tensor<64x32xf16, #dot_a> * tensor<32x64xf16, #dot_b> -> tensor<64x64xf32, #mma>
      // expected-warning @below {{schedule_hint="mfma-valu" dropped: the stage accesses memory}}
      tt.store %ptr, %v0 : !tt.ptr<f32>
      scf.yield %dot : tensor<64x64xf32, #mma>
    } {triton.warp_pipeline.stage = "dot", triton.warp_pipeline.schedule_hint = "mfma-valu"}
    scf.execute_region {
      tt.store %ptr, %v0 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "mem"}
    scf.yield %d : tensor<64x64xf32, #mma>
  } {triton.warp_pipeline.pipelined_for}
  tt.return
}

}
