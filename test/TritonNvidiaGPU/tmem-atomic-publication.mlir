// RUN: triton-opt %s -split-input-file --tritongpu-optimize-thread-regions --convert-scf-to-cf --allocate-shared-memory-nv=compute-capability=100 --triton-nvidia-gpu-membar='compute-capability=100 ptx-version=87' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
!data = tensor<128x2xi32, #blocked>
!tile = !ttg.memdesc<128x2xi32, #tmem, #ttng.tensor_memory, mutable>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100"} {
  // Private stores can wait at the join, immediately before the TMEM load.
  // CHECK-LABEL: @private_tile
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: {{.*}}ttg.thread_predicate
  // CHECK-NEXT: cf.cond_br {{.*}}, {{.*}}, ^[[RELAXED_JOIN:bb[0-9]+]]
  // CHECK-NOT: ttng.tmem_wait store
  // CHECK-NOT: ttg.barrier
  // CHECK: tt.atomic_rmw add, relaxed
  // CHECK-NOT: ttng.tmem_wait store
  // CHECK-NOT: ttg.barrier
  // CHECK: ^[[RELAXED_JOIN]]:
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: {{.*}}ttng.tmem_load
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttg.barrier all
  // CHECK-NEXT: {{.*}}ttg.thread_predicate
  // CHECK-NEXT: cf.cond_br {{.*}}, {{.*}}, ^[[RELEASE_JOIN:bb[0-9]+]]
  // CHECK-NOT: ttng.tmem_wait store
  // CHECK-NOT: ttg.barrier
  // CHECK: tt.atomic_rmw add, release
  // CHECK-NOT: ttng.tmem_wait store
  // CHECK-NOT: ttg.barrier
  // CHECK: ^[[RELEASE_JOIN]]:
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: {{.*}}ttng.tmem_load
  // One wait after the relaxed atomic's join completes both private loads.
  // CHECK-NOT: ttng.tmem_wait load
  // CHECK: ttg.thread_predicate
  // CHECK-NEXT: cf.cond_br {{.*}}, {{.*}}, ^[[LOAD_JOIN:bb[0-9]+]]
  // CHECK-NOT: {{ttng\.tmem_wait|ttg\.barrier}}
  // CHECK: tt.atomic_rmw add, relaxed
  // CHECK-NOT: {{ttng\.tmem_wait|ttg\.barrier}}
  // CHECK: ^[[LOAD_JOIN]]:
  // CHECK-NOT: ttng.tmem_wait load
  // CHECK: ttng.tmem_load
  // CHECK: ttng.tmem_wait load
  // CHECK-NOT: ttng.tmem_wait load
  // CHECK: tt.return
  tt.func public @private_tile(%flag: !tt.ptr<i32>) {
    %one = arith.constant 1 : i32
    %true = arith.constant true
    %data = arith.constant dense<1> : !data
    %tile = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !tile
    %other = ttng.tmem_alloc %data {tensor_memory_col_offset = 4 : i32, tensor_memory_row_offset = 0 : i32} : (!data) -> !tile
    ttng.tmem_store %data, %tile, %true : !data -> !tile
    %a = tt.atomic_rmw add, relaxed, cta, %flag, %one : (!tt.ptr<i32>, i32) -> i32
    %v = ttng.tmem_load %tile : !tile -> !data
    ttng.tmem_store %v, %tile, %true : !data -> !tile
    %b = tt.atomic_rmw add, release, cta, %flag, %one : (!tt.ptr<i32>, i32) -> i32
    %w = ttng.tmem_load %tile : !tile -> !data
    %c = tt.atomic_rmw add, relaxed, cta, %flag, %one : (!tt.ptr<i32>, i32) -> i32
    %last = ttng.tmem_load %other : !tile -> !data
    %sum = arith.addi %w, %last : !data
    tt.print "tile" {hex = false, isSigned = array<i32: 1>} : %sum : !data
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#narrow = #ttng.tensor_memory_encoding<blockM = 128, blockN = 2, colStride = 1>
!data = tensor<128x2xi32, #blocked>
!pool = !ttg.memdesc<128x4xi32, #tmem, #ttng.tensor_memory, mutable>
!view = !ttg.memdesc<128x2xi32, #tmem, #ttng.tensor_memory, mutable, 128x4>
!peer = !ttg.memdesc<128x2xi32, #narrow, #ttng.tensor_memory>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100"} {
  tt.func private @publish(%flag: !tt.ptr<i32>) attributes {noinline = true} {
    %one = arith.constant 1 : i32
    %old = tt.atomic_rmw add, release, cta, %flag, %one : (!tt.ptr<i32>, i32) -> i32
    tt.return
  }

  // A distinct worker-visible allocation reuses the pool's first two columns.
  // Only the overlapping store must complete before the release atomic.
  // The private load still needs a wait and rendezvous before the call.
  // CHECK-LABEL: @reused_storage
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttg.barrier all
  // CHECK-NEXT: {{.*}}ttg.thread_predicate
  // CHECK-NOT: ttg.barrier
  // CHECK: tt.atomic_rmw add, release
  // CHECK-NOT: ttg.barrier
  // CHECK: ttng.tmem_load
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttg.barrier all
  // CHECK-NEXT: {{.*}}ttg.thread_predicate
  // CHECK-NEXT: cf.cond_br {{.*}}, {{.*}}, ^[[DISJOINT_JOIN:bb[0-9]+]]
  // CHECK-NOT: ttng.tmem_wait store
  // CHECK-NOT: ttg.barrier
  // CHECK: tt.atomic_rmw add, release
  // CHECK-NOT: ttng.tmem_wait store
  // CHECK-NOT: ttg.barrier
  // CHECK: ^[[DISJOINT_JOIN]]:
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: {{.*}}ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: tt.call @publish
  // CHECK-NEXT: {{.*}}ttng.tmem_load
  tt.func public @reused_storage(%flag: !tt.ptr<i32>, %overlap: i1) {
    %one = arith.constant 1 : i32
    %true = arith.constant true
    %data = arith.constant dense<1> : !data
    %pool = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !pool
    %values = scf.if %overlap -> (!data) {
      %a = ttng.tmem_subslice %pool {offset = 0 : i32} : !pool -> !view
      ttng.tmem_store %data, %a, %true : !data -> !view
      %old = tt.atomic_rmw add, release, cta, %flag, %one : (!tt.ptr<i32>, i32) -> i32
      %v = ttng.tmem_load %a : !view -> !data
      scf.yield %v : !data
    } else {
      %b = ttng.tmem_subslice %pool {offset = 2 : i32} : !pool -> !view
      ttng.tmem_store %data, %b, %true : !data -> !view
      %old = tt.atomic_rmw add, release, cta, %flag, %one : (!tt.ptr<i32>, i32) -> i32
      %v = ttng.tmem_load %b : !view -> !data
      tt.call @publish(%flag) : (!tt.ptr<i32>) -> ()
      %after = ttng.tmem_load %b : !view -> !data
      %sum = arith.addi %v, %after : !data
      scf.yield %sum : !data
    }
    %peer = ttng.tmem_alloc %values {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (!data) -> !peer
    ttg.warp_specialize(%peer)
    default {
      ttg.warp_yield
    }
    partition0(%tile: !peer) num_warps(8) {
      %v = ttng.tmem_load %tile : !peer -> !data
      tt.print "tile" {hex = false, isSigned = array<i32: 1>} : %v : !data
      ttg.warp_return
    } : (!peer) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
!data = tensor<128x1xi32, #blocked>
!tile = !ttg.memdesc<128x1xi32, #tmem, #ttng.tensor_memory, mutable>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // The first release lets the worker read. The second lets the producer overwrite.
  // CHECK-LABEL: @peer_handoff
  // CHECK: default {
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttg.barrier all
  // CHECK-NEXT: {{.*}}ttg.thread_predicate
  // CHECK: tt.atomic_rmw exch, release
  // CHECK: partition0
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttg.barrier all
  // CHECK-NEXT: {{.*}}ttg.thread_predicate
  // CHECK-NEXT: cf.cond_br {{.*}}, {{.*}}, ^[[PEER_JOIN:bb[0-9]+]]
  // CHECK-NOT: {{ttng\.tmem_wait|ttg\.barrier}}
  // CHECK: tt.atomic_rmw add, release
  // CHECK-NOT: {{ttng\.tmem_wait|ttg\.barrier}}
  // CHECK: ^[[PEER_JOIN]]:
  // CHECK-NOT: ttng.tmem_wait load
  // CHECK: ttg.warp_return
  tt.func public @peer_handoff(%flag: !tt.ptr<i32>) {
    %tile = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !tile
    ttg.warp_specialize(%tile, %flag)
    default {
      %one = arith.constant 1 : i32
      %two = arith.constant 2 : i32
      %true = arith.constant true
      %a = arith.constant dense<1> : !data
      %b = arith.constant dense<2> : !data
      ttng.tmem_store %a, %tile, %true : !data -> !tile
      %old = tt.atomic_rmw exch, release, cta, %flag, %one : (!tt.ptr<i32>, i32) -> i32
      %done = tt.atomic_poll acquire, cta, %flag, %two : !tt.ptr<i32>, i32 -> i1
      ttng.tmem_store %b, %tile, %true : !data -> !tile
      ttg.warp_yield
    }
    partition0(%captured: !tile, %ready: !tt.ptr<i32>) num_warps(4) {
      %one = arith.constant 1 : i32
      %done = tt.atomic_poll acquire, cta, %ready, %one : !tt.ptr<i32>, i32 -> i1
      %v = ttng.tmem_load %captured : !tile -> !data
      %old = tt.atomic_rmw add, release, cta, %ready, %one : (!tt.ptr<i32>, i32) -> i32
      tt.print "tile" {hex = false, isSigned = array<i32: 1>} : %v : !data
      ttg.warp_return
    } : (!tile, !tt.ptr<i32>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#row = #ttg.slice<{dim = 1, parent = #blocked}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
!data = tensor<128x1xi32, #blocked>
!ptrs = tensor<128x1x!tt.ptr<i32>, #blocked>
!tile = !ttg.memdesc<128x1xi32, #tmem, #ttng.tensor_memory, mutable>
!bar = !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Passing the counter to a noinline helper keeps its arrival undistributed.
  tt.func private @wait_ready(%bar: !bar) attributes {noinline = true} {
    %phase = arith.constant 0 : i32
    ttng.wait_barrier %bar, %phase : !bar
    tt.return
  }

  // The release and unrelated mbarrier arrival leave the private store pending.
  // One wait completes both stores before the loads.
  // CHECK-LABEL: @distributed_release_batches_stores
  // CHECK: ttng.tmem_store
  // CHECK-NOT: ttng.tmem_wait store
  // CHECK: tt.atomic_rmw add, release
  // CHECK-NOT: ttng.tmem_wait store
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: ttng.tmem_wait store
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: {{.*}}ttng.tmem_load
  // CHECK-NEXT: {{.*}}ttng.tmem_load
  // CHECK-NOT: ttng.tmem_wait store
  // CHECK: tt.return
  tt.func public @distributed_release_batches_stores(%flags: !tt.ptr<i32>) {
    %true = arith.constant true
    %ones = arith.constant dense<1> : !data
    %rows = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #row>
    %offsets = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32, #row> -> !data
    %base = tt.splat %flags : !tt.ptr<i32> -> !ptrs
    %addresses = tt.addptr %base, %offsets : !ptrs, !data
    %bar = ttg.local_alloc : () -> !bar
    ttng.init_barrier %bar, 1 : !bar
    %a = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !tile
    %b = ttng.tmem_alloc {tensor_memory_col_offset = 4 : i32, tensor_memory_row_offset = 0 : i32} : () -> !tile
    ttng.tmem_store %ones, %a, %true : !data -> !tile
    %old = tt.atomic_rmw add, release, cta, %addresses, %ones : (!ptrs, !data) -> !data
    ttng.arrive_barrier %bar, 1 : !bar
    ttng.tmem_store %old, %b, %true : !data -> !tile
    %left = ttng.tmem_load %a : !tile -> !data
    %right = ttng.tmem_load %b : !tile -> !data
    %sum = arith.addi %left, %right : !data
    tt.print "tiles" {hex = false, isSigned = array<i32: 1>} : %sum : !data
    tt.call @wait_ready(%bar) : (!bar) -> ()
    tt.return
  }
}
