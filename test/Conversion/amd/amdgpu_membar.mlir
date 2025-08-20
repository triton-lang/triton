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
  // CHECK-NOT: gpu.barrier
  // CHECK: ttg.async_wait
  // CHECK-NEXT: gpu.barrier
  // CHECK-NOT: gpu.barrier
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
  // CHECK-NOT: gpu.barrier
  // CHECK: ttg.async_wait
  // CHECK-NEXT: gpu.barrier
  // CHECK-NOT: gpu.barrier
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
  // CHECK-NOT: gpu.barrier
  // CHECK: ttg.async_wait
  // CHECK-NEXT: gpu.barrier
  // CHECK-NOT: gpu.barrier
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

    // CHECK-NOT: gpu.barrier
    // CHECK: ttg.async_wait
    %8 = ttg.async_wait %7 {num = 4 : i32}
    // CHECK: gpu.barrier
    // CHECK-NOT: gpu.barrier
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
    // CHECK: gpu.barrier
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
    // CHECK: gpu.barrier
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
    // CHECK-NOT: gpu.barrier
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
    // CHECK: gpu.barrier
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
    // CHECK: gpu.barrier
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

}
