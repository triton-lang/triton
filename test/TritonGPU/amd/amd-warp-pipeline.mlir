// RUN: triton-opt %s -split-input-file -tritonamdgpu-warp-pipeline | FileCheck %s

#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 4]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 16]], warp = [[0, 1], [0, 2], [0, 8]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [4, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0]], warp = [[1, 0], [2, 0], [8, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.padded_shared<[512:+16] {offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 16], [0, 1], [0, 2], [0, 8], [0, 4]], block = []}>
#shared1 = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0], [1, 0], [2, 0], [8, 0], [4, 0]], block = []}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {

// -- 3-stage example (two borders) ----
tt.func @three_stage_example(%n: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index

  scf.for %i = %c0 to %n step %c1 {
    // Stage 0 (before first border)
    %a  = arith.addi %i, %c1 : index
    %a2 = arith.muli %a, %c1 : index

    // explicit split point
    rocdl.sched.barrier 0 {triton.warp_pipeline.border="stage"}

    // Stage 1
    %b  = arith.addi %a2, %i : index

    // explicit split point
    rocdl.sched.barrier 0 {triton.warp_pipeline.border="stage"}

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
// CHECK:     arith.addi
// CHECK:     arith.muli
// CHECK:     scf.yield
// CHECK:   scf.execute_region
// CHECK:     arith.addi
// CHECK:     scf.yield
// CHECK:   scf.execute_region
// CHECK:     arith.addi
// CHECK:     arith.muli
// CHECK:     scf.yield
// CHECK: triton.warp_pipeline.pipelined_for
//
// And the split markers must be gone:
// CHECK-NOT: rocdl.sched.barrier
// CHECK: tt.return


// -- 2-stage example (one border) ----

tt.func @two_stage_example(%n: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index

  scf.for %i = %c0 to %n step %c1 {
    // Stage 0
    %x = arith.addi %i, %c1 : index

    // split to Stage 1
    rocdl.sched.barrier 0 {triton.warp_pipeline.border="stage"}

    // Stage 1
    %y = arith.muli %x, %c1 : index

    scf.yield
  }

  tt.return
}

// CHECK-LABEL: tt.func @two_stage_example(
// CHECK: scf.for
// CHECK:   scf.execute_region
// CHECK:     arith.addi
// CHECK:     scf.yield
// CHECK:   scf.execute_region
// CHECK:     arith.muli
// CHECK:     scf.yield
// CHECK: triton.warp_pipeline.pipelined_for
// CHECK-NOT: rocdl.sched.barrier
// CHECK: tt.return

// -- pipelining with pre-existing barrier (ignorable ops) ----

// CHECK-LABEL: tt.func public @triple_buf_two_stages
// CHECK: scf.for
// CHECK:   scf.execute_region
// CHECK:     local_load
// CHECK:     local_load
// CHECK:     async_copy_global_to_local
// CHECK:     async_commit_group
// CHECK:     scf.yield
// CHECK:   triton.warp_pipeline.stage
// CHECK:   ttg.async_wait
// CHECK:   scf.execute_region
// CHECK:     async_copy_global_to_local
// CHECK:     async_commit_group
// CHECK:     tt.dot
// CHECK:     scf.yield
// CHECK:   triton.warp_pipeline.stage
// CHECK: triton.warp_pipeline.pipelined_for
// CHECK-NOT: rocdl.sched.barrier
// CHECK: tt.return

tt.func public @triple_buf_two_stages(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: tensor<256x256xf32, #mma>, %arg5: i32, %arg6: i32, %arg7: tensor<256x32xi32, #linear>, %arg8: tensor<32x256xi32, #linear1>, %arg9: !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, %arg10: !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, %arg11: !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, %arg12: !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, %arg13: !ttg.async.token, %arg14: !ttg.async.token, %arg15: !ttg.async.token, %arg16: tensor<256x32x!tt.ptr<bf16>, #linear>, %arg17: tensor<32x256x!tt.ptr<bf16>, #linear1>, %arg18: tensor<256xi64, #ttg.slice<{dim = 1, parent = #mma}>>, %arg19: tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma}>>, %arg20: i64, %arg21: i64, %arg22: !tt.ptr<bf16>, %arg23: i32) attributes {noinline = false} {
  %0 = ttg.local_alloc : () -> !ttg.memdesc<3x256x32xbf16, #shared, #smem, mutable>
  %1 = ttg.local_alloc : () -> !ttg.memdesc<3x32x256xbf16, #shared1, #smem, mutable>
  %2:11 = scf.for %arg24 = %arg0 to %arg6 step %arg1 iter_args(%arg25 = %arg4, %arg26 = %arg1, %arg27 = %arg9, %arg28 = %arg11, %arg29 = %arg13, %arg30 = %arg10, %arg31 = %arg12, %arg32 = %arg14, %arg33 = %arg15, %arg34 = %arg16, %arg35 = %arg17) -> (tensor<256x256xf32, #mma>, i32, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.async.token, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.async.token, !ttg.async.token, tensor<256x32x!tt.ptr<bf16>, #linear>, tensor<32x256x!tt.ptr<bf16>, #linear1>)  : i32 {
    %32 = tt.addptr %arg34, %arg7 : tensor<256x32x!tt.ptr<bf16>, #linear>, tensor<256x32xi32, #linear>
    %33 = tt.addptr %arg35, %arg8 : tensor<32x256x!tt.ptr<bf16>, #linear1>, tensor<32x256xi32, #linear1>
    %34 = arith.addi %arg26, %arg1 : i32
    %35 = arith.cmpi slt, %34, %arg3 : i32
    %36 = arith.select %35, %34, %arg0 : i32
    %37 = ttg.memdesc_index %0[%36] : !ttg.memdesc<3x256x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>
    %38 = ttg.memdesc_index %1[%36] : !ttg.memdesc<3x32x256xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>
    %39 = ttg.local_load %arg27 token %arg29 : !ttg.memdesc<256x32xbf16, #shared, #smem, mutable> -> tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %40 = ttg.local_load %arg30 token %arg29 : !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable> -> tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %41 = ttg.async_copy_global_to_local %32, %37 : tensor<256x32x!tt.ptr<bf16>, #linear> -> <256x32xbf16, #shared, #smem, mutable>
    %42 = ttg.async_commit_group tokens %41
    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage"}
    %43 = ttg.async_wait %arg32, %arg33 {num = 0 : i32}
    %44 = ttg.async_copy_global_to_local %33, %38 : tensor<32x256x!tt.ptr<bf16>, #linear1> -> <32x256xbf16, #shared1, #smem, mutable>
    %45 = ttg.async_commit_group tokens %44
    %46 = tt.dot %39, %40, %arg25 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x256xf32, #mma>
    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage"}
    scf.yield %46, %36, %arg28, %37, %43, %arg31, %38, %42, %45, %32, %33 : tensor<256x256xf32, #mma>, i32, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.async.token, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.async.token, !ttg.async.token, tensor<256x32x!tt.ptr<bf16>, #linear>, tensor<32x256x!tt.ptr<bf16>, #linear1>
  }
  ttg.local_dealloc %1 : !ttg.memdesc<3x32x256xbf16, #shared1, #smem, mutable>
  ttg.local_dealloc %0 : !ttg.memdesc<3x256x32xbf16, #shared, #smem, mutable>
  tt.return
}

// -- Flat (unrolled) pipeline: borders outside scf.for ----
//
// Simulates a static_range epilogue that was unrolled at the Python level
// following a regular pipelined main loop.  The flat backward walk must stop
// at the prior scf.for (loops are disallowed inside a stage) so the main
// loop is not absorbed into stage 0.

tt.func @flat_pipeline_example(%n: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index

  // Pipelined main loop: gets the pipelined_for attribute and acts as a
  // hard boundary for the flat epilogue's backward walk.
  scf.for %i = %c0 to %n step %c1 {
    %x = arith.addi %i, %c1 : index
    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "load"}
    %y = arith.muli %x, %c1 : index
    scf.yield
  }

  // Stage 0 (ops before the first epilogue border)
  %a  = arith.addi %c0, %c1 : index
  %a2 = arith.muli %a, %c1 : index

  rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage0_epi", triton.warp_pipeline.priority = 1 : i32}

  // Stage 1
  %b  = arith.addi %a2, %c0 : index
  %b2 = arith.muli %b, %c1 : index

  rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage1_epi", triton.warp_pipeline.priority = 0 : i32}

  tt.return
}

// CHECK-LABEL: tt.func @flat_pipeline_example(
// Pipelined main loop forms its own warp pipeline (one execute_region per
// stage, then the pipelined_for attribute on the loop).
// CHECK: scf.for
// CHECK:   scf.execute_region
// CHECK:   scf.execute_region
// CHECK: triton.warp_pipeline.pipelined_for
// Flat epilogue execute_regions created from the borders.  Crucially, they
// must NOT absorb the pipelined main loop above.
// CHECK: scf.execute_region
// CHECK:   arith.addi
// CHECK:   arith.muli
// CHECK:   scf.yield
// CHECK: triton.warp_pipeline.priority = 1
// CHECK-SAME: triton.warp_pipeline.stage = "stage0_epi"
// CHECK: scf.execute_region
// CHECK:   arith.addi
// CHECK:   arith.muli
// CHECK:   scf.yield
// CHECK: triton.warp_pipeline.priority = 0
// CHECK-SAME: triton.warp_pipeline.stage = "stage1_epi"
// Border markers must be erased:
// CHECK-NOT: rocdl.sched.barrier
// CHECK: tt.return

// -- Post-unroll IV remap is sunk past ignorable ops (FA-kernel pattern) ----
// The FA kernel body begins with async_wait.  After MLIR loop unrolling, IV
// remap ops (arith.addi/muli) land between the last border of iter N and the
// async_wait at the start of iter N+1, which would otherwise poison cluster
// building.  The sink pre-pass moves scalar ops past adjacent ignorable ops so
// they join the next cluster naturally.
tt.func @unroll_iv_remap_sunk_past_async_wait(%n: index, %ptr: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %v0 = arith.constant 0.0 : f32

  scf.for %i = %c0 to %n step %c2 {
    // iter 0: async_wait FIRST, then stage1 / stage2 bodies.
    ttg.async_wait {num = 0 : i32}
    tt.store %ptr, %v0 : !tt.ptr<f32>
    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage1"}
    tt.store %ptr, %v0 : !tt.ptr<f32>
    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage2"}

    // IV remap injected by unroller; sits between iter-0 last border and
    // iter-1 async_wait -- the poisonous spot.
    %i_1 = arith.addi %i, %c1 : index

    // iter 1: async_wait FIRST, then stage1 (uses %i_1) / stage2.
    ttg.async_wait {num = 0 : i32}
    %off = arith.muli %i_1, %c1 : index
    tt.store %ptr, %v0 : !tt.ptr<f32>
    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage1"}
    tt.store %ptr, %v0 : !tt.ptr<f32>
    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage2"}

    scf.yield
  }
  tt.return
}

// CHECK-LABEL: tt.func @unroll_iv_remap_sunk_past_async_wait(
// CHECK: scf.for
// iter 0: async_wait, stage1 region, stage2 region.
// CHECK:   ttg.async_wait
// CHECK:   scf.execute_region
// CHECK:     tt.store
// CHECK:   scf.execute_region
// CHECK:     tt.store
// iter 1 starts with async_wait; IV remap was sunk past it into iter-1 stage1.
// CHECK:   ttg.async_wait
// CHECK:   scf.execute_region {{.*}} {
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.muli
// CHECK:     tt.store
// CHECK:   scf.execute_region
// CHECK:     tt.store
// CHECK: triton.warp_pipeline.pipelined_for
// No free arith ops or leftover sched.barrier markers in the loop body.
// CHECK-NOT: rocdl.sched.barrier
// CHECK: tt.return

// -- Negative: no border → no structuring ----
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
}
// CHECK-LABEL: tt.func @no_split_example(
// CHECK: scf.for
// CHECK-NOT: scf.execute_region
// CHECK-NOT: pipelined_for
// CHECK: tt.return
