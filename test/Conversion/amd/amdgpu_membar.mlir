// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --allocate-shared-memory -test-tritonamdgpu-membar | FileCheck %s

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#A_SHARED = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// Check that we only get a single barrier when using AsyncWait
// CHECK-LABEL: pipelined_async_copy_local_to_global
tt.func @pipelined_async_copy_local_to_global(%A: !tt.ptr<f16>) {
  %index_0 = arith.constant 0 : i32
  %index_1 = arith.constant 1 : i32
  %a_ptr = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_a = ttg.memdesc_index %alloc[%index_0] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_b = ttg.memdesc_index %alloc[%index_1] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Load TileA
  %1 = ttg.async_copy_global_to_local %a_ptr, %tile_a: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Wait for TileA
  %2 = ttg.async_wait %1 {num = 4 : i32}
  // Read TileA
  %4 = ttg.local_load %tile_a token %2 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  // Load into TileB
  %3 = ttg.async_copy_global_to_local %a_ptr, %tile_b : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // There should be a single barrier after async_wait
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NOT: ttg.barrier local
  // CHECK: tt.return
  tt.return
}
// Same as above but different order of ops
// CHECK-LABEL: pipelined_async_copy_local_to_global_2
tt.func @pipelined_async_copy_local_to_global_2(%A: !tt.ptr<f16>) {
  %index_0 = arith.constant 0 : i32
  %index_1 = arith.constant 1 : i32
  %a_ptr = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_a = ttg.memdesc_index %alloc[%index_0] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_b = ttg.memdesc_index %alloc[%index_1] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Load Tile
  %1 = ttg.async_copy_global_to_local %a_ptr, %tile_a: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Wait for TileA
  %2 = ttg.async_wait %1 {num = 4 : i32}
  // Load into TileB
  %3 = ttg.async_copy_global_to_local %a_ptr, %tile_b : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Read TileA
  %4 = ttg.local_load %tile_a token %2 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  // There should be a single barrier after async_wait
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NOT: ttg.barrier local
  // CHECK: tt.return
  tt.return
}
// Check that multiple LocalLoads waiting on the same AsyncWait produce one barrier
// CHECK-LABEL: pipelined_async_copy_local_to_global_3
tt.func @pipelined_async_copy_local_to_global_3(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>) {
  %index_0 = arith.constant 0 : i32
  %index_1 = arith.constant 1 : i32
  %a_ptr = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #AL>
  %b_ptr = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #AL>

  %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_a_1 = ttg.memdesc_index %alloc_a[%index_0] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_a_2 = ttg.memdesc_index %alloc_a[%index_1] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_b_1 = ttg.memdesc_index %alloc_b[%index_0] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_b_2 = ttg.memdesc_index %alloc_b[%index_1] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  // Load TileA_1
  %1 = ttg.async_copy_global_to_local %a_ptr, %tile_a_1: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Load TileB_1
  %2 = ttg.async_copy_global_to_local %b_ptr, %tile_b_1: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Wait for TileA
  %3 = ttg.async_wait %1, %2 {num = 4 : i32}
  // Read TileA_1
  %4 = ttg.local_load %tile_a_1 token %3 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  // Read TileB_1
  %5 = ttg.local_load %tile_b_1 token %3 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  // Load into TileA_2
  %6 = ttg.async_copy_global_to_local %a_ptr, %tile_a_2 : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Load into TileB_2
  %7 = ttg.async_copy_global_to_local %b_ptr, %tile_b_2 : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  // There should be a single barrier after async_wait
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NOT: ttg.barrier local
  // CHECK: tt.return
  tt.return
}

// Check that we do not get a barrier for LocalLoad if the token comes from a previous loop iteration
// CHECK-LABEL: async_wait_in_previous_loop_iteration
tt.func @async_wait_in_previous_loop_iteration(%a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %2 = ttg.async_wait %1 {num = 4 : i32}

  // CHECK: cf.br
  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %2) -> (!ttg.async.token)  : i32 {
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

    // CHECK-NOT: ttg.barrier local
    // CHECK: ttg.async_wait
    %8 = ttg.async_wait %7 {num = 4 : i32}
    // CHECK: ttg.barrier local
    // CHECK-NOT: ttg.barrier local
    scf.yield %8: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}

// Check we do get a barrier for LocalLoad if the initial loop token does not come from AsyncWait
// CHECK-LABEL: intial_loop_token_is_not_from_async_wait
tt.func @intial_loop_token_is_not_from_async_wait(%a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %1) -> (!ttg.async.token)  : i32 {
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
    // CHECK: ttg.local_load
    // CHECK: ttg.barrier local
    // CHECK: ttg.async_copy_global_to_local
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    %8 = ttg.async_wait %7 {num = 4 : i32}
    scf.yield %8: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}

// Same as above but the loop carried token does not come from AsyncWait
// CHECK-LABEL: loop_carried_token_not_from_async_wait
tt.func @loop_carried_token_not_from_async_wait(%a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %2 = ttg.async_wait %1 {num = 4 : i32}
  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %2) -> (!ttg.async.token)  : i32 {
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
    // CHECK: ttg.local_load
    // CHECK: ttg.barrier local
    // CHECK: ttg.async_copy_global_to_local
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    scf.yield %7: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}


// Check that we do not get a barrier for an if where both branches yield an AsyncToken from AsyncWait
// CHECK-LABEL: async_wait_inside_if
tt.func @async_wait_inside_if(%cond: i1, %a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %2 = ttg.async_wait %1 {num = 4 : i32}

  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %2) -> (!ttg.async.token)  : i32 {
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
    // CHECK: ttg.local_load
    // CHECK-NOT: ttg.barrier local
    // CHECK: ttg.async_copy_global_to_local
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    %103 = scf.if %cond -> (!ttg.async.token) {
      %8 = ttg.async_wait %7 {num = 4 : i32}
      scf.yield %8 : !ttg.async.token
    } else {
      %9 = ttg.async_wait %7 {num = 4 : i32}
      scf.yield %9 : !ttg.async.token
    }
    scf.yield %103: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}

// Check that we do get a barrier for an if where one branch does not yield an token from AsyncWait
// CHECK-LABEL: non_async_wait_token_from_then
tt.func @non_async_wait_token_from_then(%cond: i1, %a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %2 = ttg.async_wait %1 {num = 4 : i32}

  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %2) -> (!ttg.async.token)  : i32 {
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
    // We should get a barrier because the then branch does not yield an token from AsyncWait
    // CHECK: ttg.local_load
    // CHECK: ttg.barrier local
    // CHECK: ttg.async_copy_global_to_local
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    %103 = scf.if %cond -> (!ttg.async.token) {
      scf.yield %7 : !ttg.async.token
    } else {
      %8 = ttg.async_wait %7 {num = 4 : i32}
      scf.yield %8 : !ttg.async.token
    }
    scf.yield %103: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}

// See above
// CHECK-LABEL: non_async_wait_token_from_else
tt.func @non_async_wait_token_from_else(%cond: i1, %a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %2 = ttg.async_wait %1 {num = 4 : i32}

  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %2) -> (!ttg.async.token)  : i32 {
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
    // We should get a barrier because the else branch does not yield an token from AsyncWait
    // CHECK: ttg.local_load
    // CHECK: ttg.barrier local
    // CHECK: ttg.async_copy_global_to_local
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    %103 = scf.if %cond -> (!ttg.async.token) {
      %8 = ttg.async_wait %7 {num = 4 : i32}
      scf.yield %8 : !ttg.async.token
    } else {
      %9 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
      scf.yield %9 : !ttg.async.token
    }
    scf.yield %103: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}

// CHECK-LABEL: missing_barrier_reused_allocation
tt.func @missing_barrier_reused_allocation(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %alloc1 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %offset = arith.constant dense<0> : tensor<128x32xi32, #AL>

  %slice1_0 = ttg.memdesc_index %alloc1[%c0_i32] : !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %async1 = amdg.buffer_load_to_local %A[%offset] into %slice1_0 : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %token1 = ttg.async_commit_group tokens %async1
  %wait1 = amdg.async_wait %token1 {num_inst = 0 : i32}
  // CHECK: ttg.barrier local
  // CHECK: ttg.local_load
  %local_load = ttg.local_load %slice1_0 token %wait1 {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<128x32xf16, #AL>
  ttg.local_dealloc %alloc1 : !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %alloc2 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %slice2_0 = ttg.memdesc_index %alloc2[%c0_i32] : !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // op2: Async load into alloc2 (overlapping with the dealloc'd alloc1 that is still being local_load'd from)
  // CHECK: ttg.barrier local
  // CHECK-NEXT: amdg.buffer_load_to_local
  %async2 = amdg.buffer_load_to_local %B[%offset] into %slice2_0 : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %token2 = ttg.async_commit_group tokens %async2
  %wait2 = amdg.async_wait %token2 {num_inst = 0 : i32}
  // CHECK: ttg.barrier local
  tt.return
}

}

// -----
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: must_barrier_tdm_copy_select_cmpi_unbounded_base
// The select/cmpi modular-wrap idiom only equals (base + 1) % N when
// -1 <= base < N. Here the base is a plain function argument with no
// provable bound, so the matcher must reject the modular form and a barrier
// must be emitted between the TDM copy and the local_load.
// Positive coverage for the iter_arg path lives in
// test/Analysis/test-membar.mlir.
tt.func @must_barrier_tdm_copy_select_cmpi_unbounded_base(%desc: !tt.tensordesc<128x128xf16, #shared>, %phase: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  %c_pred = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>

  %write_sum = arith.addi %phase, %c1_i32 : i32
  %write_cmp = arith.cmpi slt, %write_sum, %c3_i32 : i32
  %write_idx = arith.select %write_cmp, %write_sum, %c0_i32 : i32
  %write_view = ttg.memdesc_index %alloc[%write_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

  %read_idx = arith.remsi %phase, %c3_i32 : i32
  %read_view = ttg.memdesc_index %alloc[%read_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

  // CHECK: amdg.async_tdm_copy_global_to_local
  %copy_desc = amdg.update_tensor_descriptor %desc add_offsets = [%c0_i32, %c0_i32] pred = %c_pred {clamp_bounds} : !tt.tensordesc<128x128xf16, #shared>
  %token = amdg.async_tdm_copy_global_to_local %copy_desc into %write_view : !tt.tensordesc<128x128xf16, #shared> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  %load = ttg.local_load %read_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #AL>
  tt.return
}
}

// -----
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: disjoint_tdm_copy_remsi
// Test that TDM copy and local_load with provably disjoint dynamic indices
// using arith.remsi (gluon/canonical modular wrap) do not require a barrier.
tt.func @disjoint_tdm_copy_remsi(%desc: !tt.tensordesc<128x128xf16, #shared>, %phase: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %c_pred = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>

  // Write index: (phase + 2) % 3
  %write_phase = arith.addi %phase, %c2_i32 : i32
  %write_idx = arith.remsi %write_phase, %c3_i32 : i32
  %write_view = ttg.memdesc_index %alloc[%write_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

  // Read index: phase % 3
  %read_idx = arith.remsi %phase, %c3_i32 : i32
  %read_view = ttg.memdesc_index %alloc[%read_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

  // CHECK: amdg.async_tdm_copy_global_to_local
  %copy_desc = amdg.update_tensor_descriptor %desc add_offsets = [%c0_i32, %c0_i32] pred = %c_pred {clamp_bounds} : !tt.tensordesc<128x128xf16, #shared>
  %token = amdg.async_tdm_copy_global_to_local %copy_desc into %write_view : !tt.tensordesc<128x128xf16, #shared> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.local_load
  %load = ttg.local_load %read_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #AL>
  tt.return
}

// CHECK-LABEL: disjoint_select_cmpi_iter_arg_cf
// After scf-to-cf, the loop-carried phase is a cf block argument. The analysis
// proves the select/cmpi wrap is bounded from the incoming init and backedge
// operands.
tt.func @disjoint_select_cmpi_iter_arg_cf(%cst: tensor<128x128xf16, #AL>, %ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  %init = arith.constant -1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>

  // CHECK: cf.br
  %res = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%phase = %init) -> (i32) : i32 {
    %w_sum = arith.addi %phase, %c1_i32 : i32
    %w_cmp = arith.cmpi sge, %w_sum, %c3_i32 : i32
    %w_idx = arith.select %w_cmp, %c0_i32, %w_sum : i32
    %w_view = ttg.memdesc_index %alloc[%w_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    %r_idx = arith.remsi %phase, %c3_i32 : i32
    %r_view = ttg.memdesc_index %alloc[%r_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    // CHECK: ttg.barrier local
    // CHECK: ttg.local_store
    ttg.local_store %cst, %w_view : tensor<128x128xf16, #AL> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK-NOT: ttg.barrier local
    // CHECK: ttg.local_load
    %load = ttg.local_load %r_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #AL>

    scf.yield %w_idx : i32
  }
  tt.return
}

// CHECK-LABEL: disjoint_remsi_loop_carried_cf
// Next-iteration hazard after scf-to-cf lowering:
//   read_i  = phase_i % 3
//   write_i = (phase_i + 1) % 3
//   phase_{i+1} = phase_i + 1
// Dynamic index comparisons are disabled across the cf backedge, so the
// carried write cannot be proven disjoint from read_{i+1}.
tt.func @disjoint_remsi_loop_carried_cf(%cst: tensor<128x128xf16, #AL>, %ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>

  // CHECK: cf.br
  %res = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%phase = %c0_i32) -> (i32) : i32 {
    %r_idx = arith.remsi %phase, %c3_i32 : i32
    %r_view = ttg.memdesc_index %alloc[%r_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK: ttg.barrier local
    // CHECK-NEXT: ttg.local_load
    %load = ttg.local_load %r_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #AL>

    %w_sum = arith.addi %phase, %c1_i32 : i32
    %w_idx = arith.remsi %w_sum, %c3_i32 : i32
    %w_view = ttg.memdesc_index %alloc[%w_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK-NOT: ttg.barrier local
    // CHECK: ttg.local_store
    ttg.local_store %cst, %w_view : tensor<128x128xf16, #AL> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    %next_phase = arith.addi %phase, %c1_i32 : i32
    scf.yield %next_phase : i32
  }
  tt.return
}

// CHECK-LABEL: must_barrier_remsi_loop_carried_future_disjoint_cf
// Future-precision case after scf-to-cf lowering. The current analysis
// invalidates dynamic indices across the cf backedge, so it still requires a
// conservative barrier before the read.
tt.func @must_barrier_remsi_loop_carried_future_disjoint_cf(%cst: tensor<128x128xf16, #AL>, %ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable>

  // CHECK: cf.br
  %res = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%phase = %c0_i32) -> (i32) : i32 {
    %r_idx = arith.remsi %phase, %c4_i32 : i32
    %r_view = ttg.memdesc_index %alloc[%r_idx] : !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK: ttg.barrier local
    // CHECK-NEXT: ttg.local_load
    %load = ttg.local_load %r_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #AL>

    %w_sum = arith.addi %phase, %c1_i32 : i32
    %w_idx = arith.remsi %w_sum, %c4_i32 : i32
    %w_view = ttg.memdesc_index %alloc[%w_idx] : !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK-NOT: ttg.barrier local
    // CHECK: ttg.local_store
    ttg.local_store %cst, %w_view : tensor<128x128xf16, #AL> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    %next_phase = arith.addi %phase, %c2_i32 : i32
    scf.yield %next_phase : i32
  }
  tt.return
}
}

// -----

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// Two async loads need no barrier between them. Dynamic slice indices, so
// membar cannot prove the tiles disjoint and only the filter applies.
// CHECK-LABEL: no_barrier_between_async_loads
tt.func @no_barrier_between_async_loads(%A: !tt.ptr<f16>, %i: i32, %j: i32) {
  %offset = arith.constant dense<0> : tensor<128x32xi32, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
  %tile_a = ttg.memdesc_index %alloc[%i] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  %tile_b = ttg.memdesc_index %alloc[%j] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  // CHECK: amdg.buffer_load_to_local
  // CHECK-NOT: ttg.barrier local
  // CHECK: amdg.buffer_load_to_local
  %async_a = amdg.buffer_load_to_local %A[%offset] into %tile_a : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
  %async_b = amdg.buffer_load_to_local %A[%offset] into %tile_b : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
  // CHECK: tt.return
  tt.return
}
}

// -----

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// Same for ttg.async_copy_global_to_local.
// CHECK-LABEL: no_barrier_between_async_copies
tt.func @no_barrier_between_async_copies(%A: !tt.ptr<f16>, %i: i32, %j: i32) {
  %a_ptr = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
  %tile_a = ttg.memdesc_index %alloc[%i] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  %tile_b = ttg.memdesc_index %alloc[%j] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  // CHECK: ttg.async_copy_global_to_local
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.async_copy_global_to_local
  %async_a = ttg.async_copy_global_to_local %a_ptr, %tile_a : tensor<128x32x!tt.ptr<f16>, #AL> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  %async_b = ttg.async_copy_global_to_local %a_ptr, %tile_b : tensor<128x32x!tt.ptr<f16>, #AL> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  // CHECK: tt.return
  tt.return
}
}

// -----

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// Three-buffer pipelined loop without async tokens:
//   prologue: load buf[0]; load buf[1]
//   loop i:   load buf[i]; wait(2); j = (i+1)%3; read buf[j]; load buf[j]
// None of the load/load pairs can be proven disjoint (the prologue indices have
// a different SSA base than the loop index, and (i+0) vs (i+1)%3 have
// mismatched moduli), so all of them rely on the filter. What remains is the
// barrier owned by the async_wait and the WAR barrier before the refill of
// buf[j], which stays because the LocalLoad is not synced via an async wait.
// CHECK-LABEL: no_barrier_between_async_loads_in_pipelined_loop
tt.func @no_barrier_between_async_loads_in_pipelined_loop(%A: !tt.ptr<f16>, %ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %offset = arith.constant dense<0> : tensor<128x32xi32, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable>

  %s0 = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  // CHECK: amdg.buffer_load_to_local
  // CHECK-NOT: ttg.barrier local
  %ld0 = amdg.buffer_load_to_local %A[%offset] into %s0 : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
  ttg.async_commit_group

  %s1 = ttg.memdesc_index %alloc[%c1_i32] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  // CHECK: amdg.buffer_load_to_local
  // CHECK-NOT: ttg.barrier local
  %ld1 = amdg.buffer_load_to_local %A[%offset] into %s1 : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
  ttg.async_commit_group

  // CHECK: cf.br
  %res = scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%i = %c2_i32) -> (i32) : i32 {
    %si = ttg.memdesc_index %alloc[%i] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    // CHECK-NOT: ttg.barrier local
    // CHECK: amdg.buffer_load_to_local
    %ldi = amdg.buffer_load_to_local %A[%offset] into %si : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
    ttg.async_commit_group

    // CHECK: ttg.async_wait
    // CHECK-NEXT: ttg.barrier local
    ttg.async_wait {num = 2 : i32}

    %ip1 = arith.addi %i, %c1_i32 : i32
    %j = arith.remsi %ip1, %c3_i32 : i32
    %sj = ttg.memdesc_index %alloc[%j] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    // CHECK-NOT: ttg.barrier local
    // CHECK: ttg.local_load
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: amdg.buffer_load_to_local
    %v = ttg.local_load %sj : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #AL>
    %ldj = amdg.buffer_load_to_local %A[%offset] into %sj : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
    ttg.async_commit_group

    scf.yield %j : i32
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// A TDM copy is an async load too, so the LocalLoad synced via async_tdm_wait
// does not need a barrier against the following copy into the same buffer.
// Only the barrier belonging to the wait remains.
// CHECK-LABEL: tdm_copy_after_synced_local_load
tt.func @tdm_copy_after_synced_local_load(%desc: !tt.tensordesc<64x64xf16, #shared>) {
  %buf = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %tok_a = amdg.async_tdm_copy_global_to_local %desc into %buf : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  // CHECK-NOT: ttg.barrier local
  // CHECK: amdg.async_tdm_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NOT: ttg.barrier local
  %wait = amdg.async_tdm_wait %tok_a {num = 0 : i32}
  %val = ttg.local_load %buf token %wait : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #blocked>
  %tok_b = amdg.async_tdm_copy_global_to_local %desc into %buf : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  // CHECK: tt.return
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// Same for a fused TDM copy. It writes to several destinations and the
// LocalLoad reads the second one, so all of them have to be considered.
// CHECK-LABEL: tdm_fused_copy_after_synced_local_load
tt.func @tdm_fused_copy_after_synced_local_load(%desc0: !tt.tensordesc<64x64xf16, #shared>, %desc1: !tt.tensordesc<64x64xf16, #shared>) {
  %dst0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %dst1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %tok_a = amdg.async_tdm_fused_copy_global_to_local %desc0, %desc1 into %dst0, %dst1 {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<64x64xf16, #shared>, !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  // CHECK-NOT: ttg.barrier local
  // CHECK: amdg.async_tdm_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NOT: ttg.barrier local
  %wait = amdg.async_tdm_wait %tok_a {num = 0 : i32}
  %val = ttg.local_load %dst1 token %wait : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #blocked>
  %tok_b = amdg.async_tdm_fused_copy_global_to_local %desc0, %desc1 into %dst0, %dst1 {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<64x64xf16, #shared>, !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  // CHECK: tt.return
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#idx_parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// Same for a TDM gather.
// CHECK-LABEL: tdm_gather_after_synced_local_load
tt.func @tdm_gather_after_synced_local_load(%desc: !tt.tensordesc<64x128xf16>, %rows: tensor<64xi32, #ttg.slice<{dim = 0, parent = #idx_parent}>>) {
  %buf = ttg.local_alloc : () -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
  %tok_a = amdg.async_tdm_gather %desc[%rows] to %buf : tensor<64xi32, #ttg.slice<{dim = 0, parent = #idx_parent}>>, !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
  // CHECK-NOT: ttg.barrier local
  // CHECK: amdg.async_tdm_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NOT: ttg.barrier local
  %wait = amdg.async_tdm_wait %tok_a {num = 0 : i32}
  %val = ttg.local_load %buf token %wait : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> tensor<256x128xf16, #blocked>
  %tok_b = amdg.async_tdm_gather %desc[%rows] to %buf : tensor<64xi32, #ttg.slice<{dim = 0, parent = #idx_parent}>>, !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
  // CHECK: tt.return
  tt.return
}
}
