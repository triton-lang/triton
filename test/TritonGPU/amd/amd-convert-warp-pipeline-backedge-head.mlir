// RUN: triton-opt %s -split-input-file -convert-warp-pipeline="gfx-arch=gfx950" | FileCheck %s --check-prefix=HEURISTIC
// RUN: triton-opt %s -split-input-file -convert-warp-pipeline="gfx-arch=gfx942" | FileCheck %s --check-prefix=NOHEURISTIC

// The internal heuristic should match the rotated 4-stage dot/memory shape:
// two low-priority dot stages, two high-priority memory stages, and a local
// wraparound dependency.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @heuristic_dot_low_priority(%n: index, %ptr: tensor<16x16x!tt.ptr<f16>, #blocked>, %acc: tensor<16x16xf32, #mma>, %rhs: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %smem = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>

    scf.for %i = %c0 to %n step %c1 iter_args(%iter_acc = %acc) -> tensor<16x16xf32, #mma> {
      %stage0 = scf.execute_region -> tensor<16x16xf32, #mma> no_inline {
        %lhs = ttg.local_load %smem : !ttg.memdesc<16x16xf16, #shared, #smem, mutable> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
        %dot = tt.dot %lhs, %rhs, %iter_acc : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<16x16xf32, #mma>
        scf.yield %dot : tensor<16x16xf32, #mma>
      } {triton.warp_pipeline.stage = "dot0", triton.warp_pipeline.priority = 0 : i32}

      scf.execute_region no_inline {
        %loaded = tt.load %ptr : tensor<16x16x!tt.ptr<f16>, #blocked>
        ttg.local_store %loaded, %smem : tensor<16x16xf16, #blocked> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
        scf.yield
      } {triton.warp_pipeline.stage = "mem0", triton.warp_pipeline.priority = 1 : i32}

      %stage2 = scf.execute_region -> tensor<16x16xf32, #mma> no_inline {
        %lhs = ttg.local_load %smem : !ttg.memdesc<16x16xf16, #shared, #smem, mutable> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
        %dot = tt.dot %lhs, %rhs, %stage0 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<16x16xf32, #mma>
        scf.yield %dot : tensor<16x16xf32, #mma>
      } {triton.warp_pipeline.stage = "dot1", triton.warp_pipeline.priority = 0 : i32}

      scf.execute_region no_inline {
        %loaded = tt.load %ptr : tensor<16x16x!tt.ptr<f16>, #blocked>
        ttg.local_store %loaded, %smem : tensor<16x16xf16, #blocked> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
        scf.yield
      } {triton.warp_pipeline.stage = "mem1", triton.warp_pipeline.priority = 1 : i32}

      scf.yield %stage2 : tensor<16x16xf32, #mma>
    } {triton.warp_pipeline.pipelined_for}

    ttg.local_dealloc %smem : !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// Matched gfx950 pipeline: the heuristic moves the cluster-0 backedge barrier
// to the loop head while keeping the cluster-0 priority reset at the loop tail.
// HEURISTIC-LABEL: tt.func @heuristic_dot_low_priority
// HEURISTIC: rocdl.s.setprio 0
// HEURISTIC-NEXT: scf.for
// Key difference: the backedge barrier has moved to the loop head.
// HEURISTIC-NEXT: rocdl.sched.barrier
// HEURISTIC-NEXT: ttg.barrier local
// HEURISTIC-NEXT: rocdl.sched.barrier
// HEURISTIC: ttg.local_load
// HEURISTIC: tt.dot
// HEURISTIC: rocdl.s.setprio 1
// HEURISTIC: ttg.local_store
// HEURISTIC: rocdl.s.setprio 0
// HEURISTIC: ttg.local_load
// HEURISTIC: tt.dot
// HEURISTIC: rocdl.s.setprio 1
// HEURISTIC: ttg.local_store
// Key difference: priority still resets for the next cluster-0 iteration
// at the loop tail, separate from the moved backedge barrier.
// HEURISTIC-NEXT: rocdl.s.setprio 0
// HEURISTIC-NEXT: scf.yield
// HEURISTIC: rocdl.sched.barrier
// HEURISTIC-NEXT: ttg.barrier local
// HEURISTIC-NEXT: rocdl.sched.barrier
// HEURISTIC: rocdl.s.setprio 0
// HEURISTIC: amdg.cond_barrier

// Same matched input on gfx942: the arch gate disables the heuristic, so the
// cluster-0 priority reset and backedge barrier remain together at the tail.
// NOHEURISTIC-LABEL: tt.func @heuristic_dot_low_priority
// NOHEURISTIC: rocdl.s.setprio 0
// NOHEURISTIC-NEXT: scf.for
// Key difference: without the heuristic, there is no head barrier.
// NOHEURISTIC-NOT: rocdl.sched.barrier
// NOHEURISTIC-NOT: ttg.barrier local
// NOHEURISTIC: ttg.local_load
// NOHEURISTIC: tt.dot
// NOHEURISTIC: rocdl.s.setprio 1
// NOHEURISTIC: ttg.local_store
// NOHEURISTIC: rocdl.s.setprio 0
// NOHEURISTIC: ttg.local_load
// NOHEURISTIC: tt.dot
// NOHEURISTIC: rocdl.s.setprio 1
// NOHEURISTIC: ttg.local_store
// Key difference: priority reset and the backedge barrier stay together
// at the loop tail.
// NOHEURISTIC-NEXT: rocdl.s.setprio 0
// NOHEURISTIC: rocdl.sched.barrier
// NOHEURISTIC-NEXT: ttg.barrier local
// NOHEURISTIC-NEXT: rocdl.sched.barrier
// NOHEURISTIC-NEXT: scf.yield
// NOHEURISTIC: rocdl.s.setprio 0
// NOHEURISTIC: amdg.cond_barrier
