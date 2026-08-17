// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --allocate-shared-memory -test-print-membar | FileCheck %s --check-prefixes=CHECK,CF
// RUN: triton-opt %s -split-input-file                     --allocate-shared-memory -test-print-membar | FileCheck %s --check-prefixes=CHECK,SCF

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
  // CHECK: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32, warpGroupLocal}
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
  // CHECK: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32, warpGroupLocal}
  // CHECK-NEXT: arith.addi
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
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 18944 : i32} {
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

  // CHECK-NEXT: ttng.barrier_expect
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  // CHECK-NEXT: ttng.wait_barrier
  ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.async_tma_copy_global_to_local %arg1[%c0, %c0] %alloc, %barrier, %true : !tt.tensordesc<256x64xf16, #shared>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  // CHECK-NEXT: ttng.barrier_expect
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.wait_barrier
  ttng.async_tma_copy_global_to_local %arg1[%c0, %c0] %alloc, %barrier, %true : !tt.tensordesc<256x64xf16, #shared>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: ttg.local_load
  %t = ttg.local_load %alloc : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #blocked>

  // CHECK-NEXT: ttng.barrier_expect
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  // CHECK-NEXT: ttng.wait_barrier
  ttng.barrier_expect %barrier, 49152, %true : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>
  ttng.async_tma_copy_global_to_local %arg1[%c0, %c0] %alloc, %barrier, %true : !tt.tensordesc<256x64xf16, #shared>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
  ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>

  // CHECK-NEXT: memdesc_subslice
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
