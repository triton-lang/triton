// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --allocate-shared-memory -test-print-membar | FileCheck %s --check-prefixes=CHECK,CF
// RUN: triton-opt %s -split-input-file                     --allocate-shared-memory -test-print-membar | FileCheck %s --check-prefixes=CHECK,SCF
// RUN: triton-opt %s -split-input-file --allocate-shared-memory -test-print-membar -test-print-membar | FileCheck %s --check-prefixes=CHECK,SCF

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#A_SHARED = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @async_store_wait
tt.func @async_store_wait(%arg: tensor<32x16xf16, #AL>) {
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // CHECK: async_tma_store_wait
  ttng.async_tma_store_wait {pendings = 0 : i32}
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttg.local_store
  ttg.local_store %arg, %alloc : tensor<32x16xf16, #AL> -> !ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @warpgroup_wait_followed_by_cta_barrier
tt.func @warpgroup_wait_followed_by_cta_barrier(%acc: tensor<256xf32, #blocked>, %value: i32) {
  %c1 = arith.constant 1 : i32
  // CHECK: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
  // CHECK-NEXT: arith.addi
  // CHECK-NEXT: ttg.barrier local
  %wait = ttng.warp_group_dot_wait %acc {pendings = 0 : i32} : tensor<256xf32, #blocked>
  %next = arith.addi %value, %c1 : i32
  ttg.barrier local
  tt.return
}

// CHECK-LABEL: @warpgroup_wait_followed_by_barrier_expect
tt.func @warpgroup_wait_followed_by_barrier_expect(%acc: tensor<256xf32, #blocked>, %value: i32) {
  %true = arith.constant true
  %c1 = arith.constant 1 : i32
  %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
  // CHECK: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
  // CHECK-NEXT: arith.addi
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.barrier_expect
  %wait = ttng.warp_group_dot_wait %acc {pendings = 0 : i32} : tensor<256xf32, #blocked>
  %next = arith.addi %value, %c1 : i32
  ttng.barrier_expect %barrier, 8, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
  tt.return
}

// CHECK-LABEL: @warpgroup_wait_before_memory_effect
tt.func @warpgroup_wait_before_memory_effect(%acc: tensor<256xf32, #blocked>) {
  %allocation = ttg.local_alloc : () -> !ttg.memdesc<256xf32, #shared, #smem, mutable>
  // CHECK: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttg.local_store
  %wait = ttng.warp_group_dot_wait %acc {pendings = 0 : i32} : tensor<256xf32, #blocked>
  ttg.local_store %acc, %allocation : tensor<256xf32, #blocked> -> !ttg.memdesc<256xf32, #shared, #smem, mutable>
  ttg.barrier local
  tt.return
}

// CHECK-LABEL: @warpgroup_wait_before_barrier_invalidation
tt.func @warpgroup_wait_before_barrier_invalidation(%acc: tensor<256xf32, #blocked>) {
  %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
  // CHECK: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.inval_barrier
  %wait = ttng.warp_group_dot_wait %acc {pendings = 0 : i32} : tensor<256xf32, #blocked>
  ttng.inval_barrier %barrier : !ttg.memdesc<1xi64, #shared, #smem, mutable>
  tt.return
}
}

// -----

#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @wait_then_arrive_barrier
tt.func @wait_then_arrive_barrier(%phase: i32) {
  %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  // CHECK: ttng.wait_barrier
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.arrive_barrier
  ttng.wait_barrier %barrier, %phase : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.arrive_barrier %barrier, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}

// CHECK-LABEL: @arrive_then_wait_barrier
tt.func @arrive_then_wait_barrier(%phase: i32) {
  %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  // CHECK: ttng.arrive_barrier
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.wait_barrier
  ttng.arrive_barrier %barrier, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.wait_barrier %barrier, %phase : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}

// Batched expects use thread zero; invalidation can use another elected lane.
// The second expect must preserve the first barrier's pending write.
// CHECK-LABEL: @batched_expect_preserves_earlier_barrier
tt.func @batched_expect_preserves_earlier_barrier() {
  %true = arith.constant true
  %a = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %b = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  // CHECK: ttng.init_barrier [[A:%.*]], 1
  // CHECK-NEXT: ttng.init_barrier [[B:%.*]], 1
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.barrier_expect [[A]], 0,
  // CHECK-NEXT: ttng.barrier_expect [[B]], 0,
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.inval_barrier [[A]]
  // CHECK-NEXT: ttng.inval_barrier [[B]]
  ttng.init_barrier %a, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.init_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.barrier_expect %a, 0, %true : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.barrier_expect %b, 0, %true : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %a : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.inval_barrier %b : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 18944 : i32} {
// An async-proxy read of one stage does not conflict with a generic-proxy write
// to a disjoint stage. The store wait still synchronizes subsequent accesses.
// CHECK-LABEL: @tma_store_disjoint_footprints
tt.func @tma_store_disjoint_footprints(%desc: !tt.tensordesc<64x64xf16, #shared>, %input: tensor<64x64xf16, #blocked>) -> tensor<64x64xf16, #blocked> {
  %c0 = arith.constant 0 : i32
  %allocation = ttg.local_alloc : () -> !ttg.memdesc<3x64x64xf16, #shared, #ttg.shared_memory, mutable>
  %low = ttg.memdesc_index %allocation[%c0] : !ttg.memdesc<3x64x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory, mutable>
  %prefix = ttg.memdesc_subslice %allocation [2, 0, 0] : !ttg.memdesc<3x64x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<1x64x64xf16, #shared, #ttg.shared_memory, mutable, 3x64x64>
  %high = ttg.memdesc_index %prefix[%c0] : !ttg.memdesc<1x64x64xf16, #shared, #ttg.shared_memory, mutable, 3x64x64> -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory, mutable>
  ttg.local_store %input, %low : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory, mutable>
  ttg.barrier local
  ttng.fence_async_shared {bCluster = false}
  // CHECK: ttng.async_tma_copy_local_to_global
  // CHECK-NEXT: ttg.local_store
  // CHECK-NEXT: ttng.async_tma_store_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: {{.*}} = ttg.local_load
  ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %low : !tt.tensordesc<64x64xf16, #shared>, !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory, mutable>
  ttg.local_store %input, %high : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory, mutable>
  ttng.async_tma_store_wait {pendings = 0 : i32}
  %output = ttg.local_load %high : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<64x64xf16, #blocked>
  tt.return %output : tensor<64x64xf16, #blocked>
}

// CHECK-LABEL: tma_special_cases
tt.func @tma_special_cases(%arg1: !tt.tensordesc<256x64xf16, #shared>, %arg2: !tt.tensordesc<1x64xf16, #shared>) -> (tensor<256x64xf16, #blocked>){
  %true = arith.constant 1 : i1
  %cx = arith.constant dense<1> : tensor<32xi32>
  %c0 = arith.constant 0 : i32
  %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  %gather_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #ttg.shared_memory, mutable>
  //      CHECK: ttng.init_barrier
  // CHECK-NEXT: ttng.init_barrier
  ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.barrier_expect
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  // CHECK-NEXT: ttng.wait_barrier
  ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.async_tma_copy_global_to_local %arg1[%c0, %c0] %alloc, %barrier, %true : !tt.tensordesc<256x64xf16, #shared>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.barrier_expect
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.wait_barrier
  ttng.async_tma_copy_global_to_local %arg1[%c0, %c0] %alloc, %barrier, %true : !tt.tensordesc<256x64xf16, #shared>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: ttg.local_load
  %t = ttg.local_load %alloc : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #blocked>

  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.barrier_expect
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  // CHECK-NEXT: ttng.wait_barrier
  ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.async_tma_copy_global_to_local %arg1[%c0, %c0] %alloc, %barrier, %true : !tt.tensordesc<256x64xf16, #shared>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: memdesc_subslice
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.barrier_expect
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.async_tma_gather
  // CHECK-NEXT: ttng.wait_barrier
  %view = ttg.memdesc_subslice %gather_alloc [0, 0]  : !ttg.memdesc<32x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x64xf16, #shared, #ttg.shared_memory, mutable>
  ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.async_tma_gather %arg2[%cx, %c0] %view, %barrier, %true : !tt.tensordesc<1x64xf16, #shared>, tensor<32xi32>, i32, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<32x64xf16, #shared, #ttg.shared_memory, mutable>, i1
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.inval_barrier
  // CHECK-NEXT: ttng.inval_barrier
  ttng.inval_barrier %barrier : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.inval_barrier %barrier : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  tt.return %t : tensor<256x64xf16, #blocked>
}
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 18944 : i32} {
// CHECK-LABEL: tma_special_cases_cf
tt.func @tma_special_cases_cf(%arg1: !tt.tensordesc<256x64xf16, #shared>, %i1 : i1, %arg2: tensor<256x64xf16, #blocked>) -> (tensor<256x64xf16, #blocked>){
  %true = arith.constant 1 : i1
  %c0 = arith.constant 0 : i32
  %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  // CF: cf.cond_br
  // SCF: scf.if
  scf.if %i1 {
    //  CHECK-NOT: ttg.barrier local
    //      CHECK: ttng.async_tma_copy_global_to_local
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttng.barrier_expect
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttng.wait_barrier
    // CF-NEXT: cf.br
    // SCF-NEXT: } else {
    ttng.async_tma_copy_global_to_local %arg1[%c0, %c0] %alloc, %barrier, %true : !tt.tensordesc<256x64xf16, #shared>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
    ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  } else {
    //  CHECK-NOT: ttg.barrier local
    //      CHECK: ttg.local_store
    // CF-NEXT: cf.br
    // SCF-NEXT: }
    ttg.local_store %arg2, %alloc : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  }
  //      CHECK: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  %t = ttg.local_load %alloc : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #blocked>
  tt.return %t : tensor<256x64xf16, #blocked>
}
}

// -----

// CHECK-LABEL: tmem_copy_after_alloc
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>

//#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @tmem_copy_after_alloc(%arg0: tensor<128x16xf8E4M3FN, #blocked>) {
    // CHECK: local_alloc
    %0 = ttg.local_alloc %arg0 {allocation.offset = 53248 : i32} : (tensor<128x16xf8E4M3FN, #blocked>) -> !ttg.memdesc<128x16xf8E4M3FN, #shared, #smem>
    // CHECK: tmem_alloc
    %1 = ttng.tmem_alloc  {tensor_memory_col_offset = 256 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory, mutable>
    // ttg.barrier local
    // CHECK: tmem_copy
    ttng.tmem_copy %0, %1 : !ttg.memdesc<128x16xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @unused_integer_atomic_adds
  tt.func @unused_integer_atomic_adds(
      %indices: tensor<128xi32, #blocked>,
      %values: tensor<128xi32, #blocked>,
      %mask: tensor<128xi1, #blocked>) -> tensor<128xi32, #blocked> {
    %zero = arith.constant dense<0> : tensor<128xi32, #blocked>
    %shared = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    // CHECK: ttg.local_store
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw add
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw add
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw add
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: {{.*}}ttg.local_load
    ttg.local_store %zero, %shared : tensor<128xi32, #blocked> -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    %first = ttg.local_atomic_scatter_rmw add, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %second = ttg.local_atomic_scatter_rmw add, %shared[%indices], %values, %mask {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>, tensor<128xi1, #blocked>) -> tensor<128xi32, #blocked>
    %third = ttg.local_atomic_scatter_rmw add, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %final = ttg.local_load %shared : !ttg.memdesc<128xi32, #shared, #smem, mutable> -> tensor<128xi32, #blocked>
    tt.return %final : tensor<128xi32, #blocked>
  }

  // CHECK-LABEL: @first_atomic_result_used
  tt.func @first_atomic_result_used(
      %indices: tensor<128xi32, #blocked>,
      %values: tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked> {
    %shared = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    // CHECK: {{.*}}ttg.local_atomic_scatter_rmw add
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw add
    %first = ttg.local_atomic_scatter_rmw add, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %second = ttg.local_atomic_scatter_rmw add, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    tt.return %first : tensor<128xi32, #blocked>
  }

  // CHECK-LABEL: @second_atomic_result_used
  tt.func @second_atomic_result_used(
      %indices: tensor<128xi32, #blocked>,
      %values: tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked> {
    %shared = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    // CHECK: {{.*}}ttg.local_atomic_scatter_rmw add
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw add
    %first = ttg.local_atomic_scatter_rmw add, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %second = ttg.local_atomic_scatter_rmw add, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    tt.return %second : tensor<128xi32, #blocked>
  }

  // CHECK-LABEL: @unused_commutative_integer_atomics
  tt.func @unused_commutative_integer_atomics(
      %indices: tensor<128xi32, #blocked>,
      %values: tensor<128xi32, #blocked>) {
    %shared = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    // CHECK: {{.*}}ttg.local_atomic_scatter_rmw and
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw and
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw or
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw or
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw xor
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw xor
    // CHECK-NEXT: ttg.barrier local
    // CHECK: {{.*}}ttg.local_atomic_scatter_rmw max
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw max
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw min
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw min
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw umax
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw umax
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw umin
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw umin
    %and0 = ttg.local_atomic_scatter_rmw and, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %and1 = ttg.local_atomic_scatter_rmw and, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %or0 = ttg.local_atomic_scatter_rmw or, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %or1 = ttg.local_atomic_scatter_rmw or, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %xor0 = ttg.local_atomic_scatter_rmw xor, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %xor1 = ttg.local_atomic_scatter_rmw xor, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %max0 = ttg.local_atomic_scatter_rmw max, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %max1 = ttg.local_atomic_scatter_rmw max, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %min0 = ttg.local_atomic_scatter_rmw min, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %min1 = ttg.local_atomic_scatter_rmw min, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %umax0 = ttg.local_atomic_scatter_rmw umax, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %umax1 = ttg.local_atomic_scatter_rmw umax, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %umin0 = ttg.local_atomic_scatter_rmw umin, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %umin1 = ttg.local_atomic_scatter_rmw umin, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    tt.return
  }

  // CHECK-LABEL: @unused_mixed_integer_atomics
  tt.func @unused_mixed_integer_atomics(
      %indices: tensor<128xi32, #blocked>,
      %values: tensor<128xi32, #blocked>) {
    %shared = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    // CHECK: {{.*}}ttg.local_atomic_scatter_rmw add
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw max
    %add = ttg.local_atomic_scatter_rmw add, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %max = ttg.local_atomic_scatter_rmw max, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    tt.return
  }

  // CHECK-LABEL: @unused_atomic_exchanges
  tt.func @unused_atomic_exchanges(
      %indices: tensor<128xi32, #blocked>,
      %values: tensor<128xi32, #blocked>) {
    %shared = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    // CHECK: {{.*}}ttg.local_atomic_scatter_rmw exch
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw exch
    %first = ttg.local_atomic_scatter_rmw exch, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %second = ttg.local_atomic_scatter_rmw exch, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    tt.return
  }

  // CHECK-LABEL: @unused_mixed_width_integer_atomic_adds
  tt.func @unused_mixed_width_integer_atomic_adds(
      %indices32: tensor<128xi32, #blocked>,
      %values32: tensor<128xi32, #blocked>,
      %indices64: tensor<64xi32, #blocked>,
      %values64: tensor<64xi64, #blocked>) {
    %shared32 = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    %shared64 = ttg.memdesc_reinterpret %shared32 : !ttg.memdesc<128xi32, #shared, #smem, mutable> -> !ttg.memdesc<64xi64, #shared, #smem, mutable>
    // CHECK: {{.*}}ttg.local_atomic_scatter_rmw add
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw add
    %narrow = ttg.local_atomic_scatter_rmw add, %shared32[%indices32], %values32 {axis = 0 : i32} : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %wide = ttg.local_atomic_scatter_rmw add, %shared64[%indices64], %values64 {axis = 0 : i32} : (!ttg.memdesc<64xi64, #shared, #smem, mutable>, tensor<64xi64, #blocked>, tensor<64xi32, #blocked>) -> tensor<64xi64, #blocked>
    tt.return
  }

  // CHECK-LABEL: @unused_floating_atomic_adds
  tt.func @unused_floating_atomic_adds(
      %indices: tensor<128xi32, #blocked>,
      %values: tensor<128xf32, #blocked>) {
    %shared = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    // CHECK: {{.*}}ttg.local_atomic_scatter_rmw fadd
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: {{.*}}ttg.local_atomic_scatter_rmw fadd
    %first = ttg.local_atomic_scatter_rmw fadd, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xf32, #shared, #smem, mutable>, tensor<128xf32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xf32, #blocked>
    %second = ttg.local_atomic_scatter_rmw fadd, %shared[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<128xf32, #shared, #smem, mutable>, tensor<128xf32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

// Private functions model callable Gluon helpers. Their barrier arguments are
// initialized by callers; waits may complete groups issued by those callers.
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#rows = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#cols = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
!barrier = !ttg.memdesc<1xi64, #bar, #ttg.shared_memory, mutable>
!ptrs = tensor<128x!tt.ptr<i32>, #blocked>
!values = tensor<128xi32, #blocked>
!row_tile = tensor<16x16xf32, #rows>
!col_tile = tensor<16x16xf32, #cols>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Same-thread publications share one rendezvous even with pure work and
  // runtime predicates between them. Each barrier starts with count one.
  // CHECK-LABEL: @same_thread_publications
  tt.func private @same_thread_publications(%ptr: !tt.ptr<i32>, %first: !barrier, %second: !barrier, %pred: i1) -> i32 {
    // CHECK: tt.load
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttng.arrive_barrier
    // CHECK-NEXT: arith.addi
    // CHECK-NEXT: ttng.barrier_expect
    %value = tt.load %ptr : !tt.ptr<i32>
    ttng.arrive_barrier %first, 1, %pred : !barrier
    %twice = arith.addi %value, %value : i32
    ttng.barrier_expect %second, 0, %pred : !barrier
    tt.return %twice : i32
  }

  // Reads need completion before handing storage back, just as writes need
  // publication before handing their results to another partition.
  // CHECK-LABEL: @global_reads_and_writes
  tt.func private @global_reads_and_writes(%src: !ptrs, %dst: !ptrs, %read_done: !barrier, %write_done: !barrier) {
    // CHECK: tt.load
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttng.arrive_barrier
    // CHECK-NEXT: tt.store
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttng.arrive_barrier
    %value = tt.load %src : !ptrs
    ttng.arrive_barrier %read_done, 1 : !barrier
    tt.store %dst, %value : !ptrs
    ttng.arrive_barrier %write_done, 1 : !barrier
    tt.return
  }

  // Completion-only operations can publish together, but return cannot leave
  // their completion visible to only the issuing threads.
  // CHECK-LABEL: @completion_batch_at_return
  tt.func private @completion_batch_at_return(%desc: !tt.ptr<i8>) {
    // CHECK: ttg.async_wait
    // CHECK-NEXT: ttng.tensormap_fenceproxy_acquire
    // CHECK-NEXT: ttng.async_tma_store_wait
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: tt.return
    ttg.async_wait {num = 0 : i32}
    ttng.tensormap_fenceproxy_acquire %desc : !tt.ptr<i8>
    ttng.async_tma_store_wait {pendings = 0 : i32, read_only}
    tt.return
  }

  // The conversion's interior barrier follows its initial scratch writes.
  // Its final scratch reads also need ordering before the subsequent arrive.
  // CHECK-LABEL: @completion_before_scratch
  tt.func private @completion_before_scratch(%input: !row_tile, %done: !barrier) -> !col_tile {
    // CHECK: ttg.async_wait
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttg.convert_layout
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttng.arrive_barrier
    ttg.async_wait {num = 0 : i32}
    %converted = ttg.convert_layout %input : !row_tile -> !col_tile
    ttng.arrive_barrier %done, 1 : !barrier
    tt.return %converted : !col_tile
  }

  // A completion on just one incoming path is still a demand at the join.
  // CHECK-LABEL: @completion_through_cfg_join
  tt.func private @completion_through_cfg_join(%pred: i1, %dst: !ptrs, %value: !values) {
    cf.cond_br %pred, ^wait, ^skip
  ^wait:
    // CHECK: ttg.async_wait
    // CHECK-NEXT: cf.br
    ttg.async_wait {num = 0 : i32}
    cf.br ^join
  ^skip:
    cf.br ^join
  ^join:
    // CHECK: ttg.barrier local
    // CHECK-NEXT: tt.store
    tt.store %dst, %value : !ptrs
    tt.return
  }

  // The preheader is synchronized, but iteration two follows a global store.
  // The caller initializes the barrier with count two for this two-trip loop.
  // CHECK-LABEL: @release_after_backedge
  tt.func private @release_after_backedge(%done: !barrier, %dst: !ptrs, %value: !values) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    ttg.barrier local
    // CHECK: cf.br ^[[LOOP:bb[0-9]+]]
    cf.br ^loop(%zero : i32)
    // CHECK: ^[[LOOP]](
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttng.arrive_barrier
  ^loop(%iteration: i32):
    ttng.arrive_barrier %done, 1 : !barrier
    tt.store %dst, %value : !ptrs
    %next = arith.addi %iteration, %one : i32
    %again = arith.cmpi slt, %next, %two : i32
    cf.cond_br %again, ^loop(%next : i32), ^exit
  ^exit:
    tt.return
  }

  // Capture writes must follow completion, and each partition has its own
  // completion/publication state. The two destination pointers are disjoint.
  // CHECK-LABEL: @partition_scopes
  tt.func @partition_scopes(%default_ptr: !tt.ptr<i32>, %worker_ptr: !tt.ptr<i32>) {
    %one = arith.constant 1 : i32
    // CHECK: ttg.async_wait
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttg.warp_specialize
    ttg.async_wait {num = 0 : i32}
    ttg.warp_specialize(%worker_ptr)
    default {
      // CHECK: default {
      // CHECK-NEXT: ttg.async_wait
      // CHECK-NEXT: ttg.async_wait
      // CHECK-NEXT: ttg.barrier local
      // CHECK-NEXT: tt.store
      ttg.async_wait {num = 0 : i32}
      ttg.async_wait {num = 0 : i32}
      tt.store %default_ptr, %one : !tt.ptr<i32>
      ttg.warp_yield
    }
    partition0(%ptr: !tt.ptr<i32>) num_warps(4) {
      %two = arith.constant 2 : i32
      // CHECK: partition0
      // CHECK: ttg.async_wait
      // CHECK-NEXT: ttg.barrier local
      // CHECK-NEXT: tt.store
      ttg.async_wait {num = 0 : i32}
      tt.store %ptr, %two : !tt.ptr<i32>
      ttg.warp_return
    } : (!tt.ptr<i32>) -> ()
    tt.return
  }
}

// -----

#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
!barrier = !ttg.memdesc<2xi64, #bar, #ttg.shared_memory, mutable>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A relaxed cluster rendezvous preserves pending work without adding thread
  // effects or demands.
  // CHECK-LABEL: @relaxed_cluster_preserves_thread_state
  tt.func private @relaxed_cluster_preserves_thread_state(%src: !tt.ptr<i32>, %read_done: !barrier, %next_done: !barrier) -> i32 {
    // CHECK: ttg.async_wait
    // CHECK-NEXT: ttng.cluster_barrier {relaxed = true}
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: tt.load
    // CHECK-NEXT: ttng.cluster_barrier {relaxed = true}
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttng.arrive_barrier
    // CHECK-NEXT: ttng.cluster_barrier {relaxed = true}
    // CHECK-NEXT: ttng.arrive_barrier
    // CHECK-NEXT: tt.return
    ttg.async_wait {num = 0 : i32}
    ttng.cluster_barrier {relaxed = true}
    %value = tt.load %src : !tt.ptr<i32>
    ttng.cluster_barrier {relaxed = true}
    ttng.arrive_barrier %read_done, 1 : !barrier
    ttng.cluster_barrier {relaxed = true}
    ttng.arrive_barrier %next_done, 1 : !barrier
    tt.return %value : i32
  }

  // Nonidentity routing has an unknown issuer set and cannot share the
  // surrounding fixed-thread publications' rendezvous.
  // CHECK-LABEL: @different_issuers
  tt.func private @different_issuers(%barrier: !barrier) attributes {noinline = true} {
    // CHECK: ttng.arrive_barrier
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttng.arrive_barrier {{.*}} {fromCTA = 0 : i32}
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttng.arrive_barrier
    ttng.arrive_barrier %barrier, 1 : !barrier
    ttng.arrive_barrier %barrier, 1 {fromCTA = 0 : i32} : !barrier
    ttng.arrive_barrier %barrier, 1 : !barrier
    tt.return
  }

  // A callee's entry publication demand orders the caller's prior effects.
  // Each local barrier receives three arrivals, including the routed arrival.
  // CHECK-LABEL: @call_publication
  tt.func @call_publication() {
    %phase = arith.constant 0 : i32
    %barrier = ttg.local_alloc : () -> !barrier
    ttng.init_barrier %barrier, 3 : !barrier
    // CHECK: ttg.async_wait
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: tt.call @different_issuers
    ttg.async_wait {num = 0 : i32}
    tt.call @different_issuers(%barrier) : (!barrier) -> ()
    ttng.wait_barrier %barrier, %phase : !barrier
    ttng.inval_barrier %barrier : !barrier
    tt.return
  }
}
