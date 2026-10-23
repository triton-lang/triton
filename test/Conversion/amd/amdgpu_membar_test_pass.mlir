// RUN: triton-opt %s --convert-scf-to-cf --allocate-shared-memory --test-tritonamdgpu-membar | FileCheck %s

// The test pass runs AMDMembarAnalysis without any ROCDL op in its input, so
// it must load the ROCDL dialect itself to emit the scheduling fence.
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// The pipeliner rotates the slot views of the stages in flight through iter_args:
// the view read in an iteration was written two iterations earlier. With three
// slots and a counter advancing by one slot per iteration, the read slot
// (c - 1) mod 3 differs from the refilled slot (c + 1) mod 3, so the refill
// needs no barrier after the synced local_load. A scheduling-only fence keeps
// the point the dropped barrier gave the instruction scheduler.
// CHECK-LABEL: test_pass_rotated_slot_views
tt.func @test_pass_rotated_slot_views(%A: !tt.ptr<f16>, %ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  %offset = arith.constant dense<0> : tensor<128x32xi32, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable>
  %s0 = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  %ld0 = amdg.buffer_load_to_local %A[%offset] into %s0 : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
  %cg0 = ttg.async_commit_group tokens %ld0
  %s1 = ttg.memdesc_index %alloc[%c1_i32] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  %ld1 = amdg.buffer_load_to_local %A[%offset] into %s1 : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
  %cg1 = ttg.async_commit_group tokens %ld1
  %w0 = ttg.async_wait %cg0 {num = 1 : i32}
  %res:5 = scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%c = %c1_i32, %r = %s0, %n = %s1, %tok = %w0, %cgp = %cg1) -> (i32, !ttg.memdesc<128x32xf16, #shared, #smem, mutable>, !ttg.memdesc<128x32xf16, #shared, #smem, mutable>, !ttg.async.token, !ttg.async.token) : i32 {
    %cp1 = arith.addi %c, %c1_i32 : i32
    %lt = arith.cmpi slt, %cp1, %c3_i32 : i32
    %cn = arith.select %lt, %cp1, %c0_i32 : i32
    %sw = ttg.memdesc_index %alloc[%cn] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    // CHECK: ttg.local_load
    // CHECK-NEXT: rocdl.sched.barrier none
    // CHECK-NEXT: amdg.buffer_load_to_local
    %v = ttg.local_load %r token %tok : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #AL>
    %ld = amdg.buffer_load_to_local %A[%offset] into %sw : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
    %cg = ttg.async_commit_group tokens %ld
    %w = ttg.async_wait %cgp {num = 1 : i32}
    scf.yield %cn, %n, %sw, %w, %cg : i32, !ttg.memdesc<128x32xf16, #shared, #smem, mutable>, !ttg.memdesc<128x32xf16, #shared, #smem, mutable>, !ttg.async.token, !ttg.async.token
  }
  tt.return
}
}
