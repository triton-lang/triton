// RUN: triton-opt %s -split-input-file --triton-nvidia-preferred-cluster-fallback=compute-capability=100 | FileCheck %s

#blockedM = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0], [2, 0]]}>
#blockedN = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[0, 1], [0, 2]]}>

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_cross_cta_convert
  tt.func @reject_cross_cta_convert(%arg0: tensor<256x128xf16, #blockedM>) -> tensor<256x128xf16, #blockedN> {
    %cvt = ttg.convert_layout %arg0 : tensor<256x128xf16, #blockedM> -> tensor<256x128xf16, #blockedN>
    tt.return %cvt : tensor<256x128xf16, #blockedN>
  }
}

// -----

#blockedReduce = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0], [2, 0]]}>
#sliceReduce = #ttg.slice<{dim = 0, parent = #blockedReduce}>

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_cross_cta_reduce
  tt.func @reject_cross_cta_reduce(%arg0: tensor<256x128xf16, #blockedReduce>) -> tensor<128xf16, #sliceReduce> {
    %red = "tt.reduce"(%arg0) ({
    ^bb0(%lhs: f16, %rhs: f16):
      %add = arith.addf %lhs, %rhs : f16
      tt.reduce.return %add : f16
    }) {axis = 0 : i32} : (tensor<256x128xf16, #blockedReduce>) -> tensor<128xf16, #sliceReduce>
    tt.return %red : tensor<128xf16, #sliceReduce>
  }
}

// -----

#blockedInline = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1], [2]]}>

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_any_inline_asm
  tt.func @reject_any_inline_asm(%arg0: tensor<128xi32, #blockedInline>) -> tensor<128xi32, #blockedInline> {
    %asm = tt.elementwise_inline_asm "add.u32 $0, $1, 1;" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %arg0 : tensor<128xi32, #blockedInline> -> tensor<128xi32, #blockedInline>
    tt.return %asm : tensor<128xi32, #blockedInline>
  }
}

// -----

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_atomic_poll
  tt.func @reject_atomic_poll(%ptr: !tt.ptr<i32>, %expected: i32) {
    %matched = tt.atomic_poll acquire, gpu, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    tt.return
  }
}

// -----

#blockedStore = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[0], [0]]}>
#sharedStore = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0], [0]]}>
#barrierStore = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1], [2]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_async_shared_store
  tt.func @reject_async_shared_store(%src: tensor<128xi32, #blockedStore>, %dst: !ttg.memdesc<128xi32, #sharedStore, #smem, mutable>, %barrier: !ttg.memdesc<4xi64, #barrierStore, #smem, mutable>) {
    ttng.async_shared_store %src, %dst, %barrier : tensor<128xi32, #blockedStore> -> !ttg.memdesc<128xi32, #sharedStore, #smem, mutable>, !ttg.memdesc<4xi64, #barrierStore, #smem, mutable>
    tt.return
  }
}

// -----

#blockedShared = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0], [2, 0]]}>
#sharedCrossCTA = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1], [0, 2]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_cross_cta_local_store
  tt.func @reject_cross_cta_local_store(%src: tensor<256x128xf16, #blockedShared>, %dst: !ttg.memdesc<256x128xf16, #sharedCrossCTA, #smem, mutable>) {
    ttg.local_store %src, %dst : tensor<256x128xf16, #blockedShared> -> !ttg.memdesc<256x128xf16, #sharedCrossCTA, #smem, mutable>
    tt.return
  }
}

// -----

#barrierLocal = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1], [2]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_count_without_fallback
  tt.func @reject_count_without_fallback(%barrier: !ttg.memdesc<4xi64, #barrierLocal, #smem, mutable>) {
    ttng.init_barrier %barrier, 2 : !ttg.memdesc<4xi64, #barrierLocal, #smem, mutable>
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1], [2]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: "ttng.preferred-cluster-fallback-ctas" = 2 : i32
  // CHECK-LABEL: tt.func @safe_explicit_fallback_count
  tt.func @safe_explicit_fallback_count(%barrier: !ttg.memdesc<4xi64, #barrier, #smem, mutable>) {
    ttng.init_barrier %barrier, 4 {fallback_count = 2 : i32} : !ttg.memdesc<4xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_clc
  tt.func @reject_clc(%clc: i128) -> i32 {
    %pid = ttng.clc_get_program_id %clc, x : i128 -> i32
    tt.return %pid : i32
  }
}

// -----

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.instrumentation_mode" = "consan"} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_consan_marked_module
  tt.func @reject_consan_marked_module() {
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.instrumentation_mode" = "gsan"} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_gsan_marked_module
  tt.func @reject_gsan_marked_module() {
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_two_ctas
  tt.func @reject_two_ctas() {
    tt.return
  }
}

// -----

#barrierAll = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0], [0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_mbarrier_group_larger_than_pair
  tt.func @reject_mbarrier_group_larger_than_pair(%barrier: !ttg.memdesc<1xi64, #barrierAll, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #barrierAll, #smem, mutable>
    tt.return
  }
}

// -----

#split = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1], [2]]}>
#broadcast = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[0], [0]]}>

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_cross_cta_histogram
  tt.func @reject_cross_cta_histogram(%src: tensor<512xi32, #split>) -> tensor<128xi32, #broadcast> {
    %hist = tt.histogram %src : tensor<512xi32, #split> -> tensor<128xi32, #broadcast>
    tt.return %hist : tensor<128xi32, #broadcast>
  }
}

// -----

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_atomic_load
  tt.func @reject_atomic_load(%ptr: !tt.ptr<i32>) {
    %unused = tt.atomic_load acquire, gpu, %ptr : (!tt.ptr<i32>) -> i32
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1], [2]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_expect_from_cta
  tt.func @reject_expect_from_cta(%barrier: !ttg.memdesc<4xi64, #barrier, #smem, mutable>, %pred: i1) {
    ttng.barrier_expect %barrier, 128 {fromCTA = 0 : i32}, %pred : !ttg.memdesc<4xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1], [2]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_arrive_from_cta
  tt.func @reject_arrive_from_cta(%barrier: !ttg.memdesc<4xi64, #barrier, #smem, mutable>) {
    ttng.arrive_barrier %barrier, 1 {fromCTA = 1 : i32} : !ttg.memdesc<4xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1], [2]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_arrive_multicast
  tt.func @reject_arrive_multicast(%barrier: !ttg.memdesc<4xi64, #barrier, #smem, mutable>) {
    ttng.arrive_barrier %barrier, 1 {multicastCTA = 2 : i32} : !ttg.memdesc<4xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 32, CGALayout = [[1, 0], [0, 1]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_remote_shared_subview
  tt.func @reject_remote_shared_subview(%alloc: !ttg.memdesc<256x256xf32, #shared, #smem, mutable>) -> !ttg.memdesc<128x256xf32, #shared, #smem, mutable, 256x256> {
    %tile = ttg.memdesc_subslice %alloc [128, 0] : !ttg.memdesc<256x256xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x256xf32, #shared, #smem, mutable, 256x256>
    tt.return %tile : !ttg.memdesc<128x256xf32, #shared, #smem, mutable, 256x256>
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 32, CGALayout = [[0, 1], [0, 2]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // The row offset changes only the address within each CTA's shared memory.
  // CHECK: "ttng.preferred-cluster-fallback-ctas" = 2 : i32
  // CHECK-LABEL: tt.func @safe_local_shared_subview
  tt.func @safe_local_shared_subview(%alloc: !ttg.memdesc<256x256xf32, #shared, #smem, mutable>) -> !ttg.memdesc<128x256xf32, #shared, #smem, mutable, 256x256> {
    %tile = ttg.memdesc_subslice %alloc [128, 0] : !ttg.memdesc<256x256xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x256xf32, #shared, #smem, mutable, 256x256>
    tt.return %tile : !ttg.memdesc<128x256xf32, #shared, #smem, mutable, 256x256>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 1], [0, 2]]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1], [0, 2]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // Indexing the rows preserves CTA ownership of the columns.
  // CHECK: "ttng.preferred-cluster-fallback-ctas" = 2 : i32
  // CHECK-LABEL: tt.func @safe_local_gather_scatter_atomic
  tt.func @safe_local_gather_scatter_atomic(%src: !ttg.memdesc<2x512xi32, #shared, #smem, mutable>, %indices: tensor<2x512xi32, #blocked>, %values: tensor<2x512xi32, #blocked>) -> tensor<2x512xi32, #blocked> {
    %gathered = ttg.local_gather %src[%indices] {axis = 0 : i32} : !ttg.memdesc<2x512xi32, #shared, #smem, mutable>, tensor<2x512xi32, #blocked> -> tensor<2x512xi32, #blocked>
    ttg.local_scatter %src[%indices], %values {axis = 0 : i32} : !ttg.memdesc<2x512xi32, #shared, #smem, mutable>, tensor<2x512xi32, #blocked>, tensor<2x512xi32, #blocked>
    %old = ttg.local_atomic_scatter_rmw add, %src[%indices], %gathered {axis = 0 : i32} : (!ttg.memdesc<2x512xi32, #shared, #smem, mutable>, tensor<2x512xi32, #blocked>, tensor<2x512xi32, #blocked>) -> tensor<2x512xi32, #blocked>
    tt.return %old : tensor<2x512xi32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 1], [0, 2]]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1], [0, 2]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_cross_cta_gather
  tt.func @reject_cross_cta_gather(%src: !ttg.memdesc<2x512xi32, #shared, #smem, mutable>, %indices: tensor<2x512xi32, #blocked>) -> tensor<2x512xi32, #blocked> {
    %gathered = ttg.local_gather %src[%indices] {axis = 1 : i32} : !ttg.memdesc<2x512xi32, #shared, #smem, mutable>, tensor<2x512xi32, #blocked> -> tensor<2x512xi32, #blocked>
    tt.return %gathered : tensor<2x512xi32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 1], [0, 2]]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1], [0, 2]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_cross_cta_scatter
  tt.func @reject_cross_cta_scatter(%dst: !ttg.memdesc<2x512xi32, #shared, #smem, mutable>, %indices: tensor<2x512xi32, #blocked>, %values: tensor<2x512xi32, #blocked>) {
    ttg.local_scatter %dst[%indices], %values {axis = 1 : i32} : !ttg.memdesc<2x512xi32, #shared, #smem, mutable>, tensor<2x512xi32, #blocked>, tensor<2x512xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 1], [0, 2]]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1], [0, 2]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_cross_cta_atomic_scatter
  tt.func @reject_cross_cta_atomic_scatter(%dst: !ttg.memdesc<2x512xi32, #shared, #smem, mutable>, %indices: tensor<2x512xi32, #blocked>, %values: tensor<2x512xi32, #blocked>) -> tensor<2x512xi32, #blocked> {
    %old = ttg.local_atomic_scatter_rmw add, %dst[%indices], %values {axis = 1 : i32} : (!ttg.memdesc<2x512xi32, #shared, #smem, mutable>, tensor<2x512xi32, #blocked>, tensor<2x512xi32, #blocked>) -> tensor<2x512xi32, #blocked>
    tt.return %old : tensor<2x512xi32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 0], [0, 0]]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 0], [0, 0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // The destination is local, but broadcasting the result needs other CTAs.
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_atomic_result_broadcast
  tt.func @reject_atomic_result_broadcast(%dst: !ttg.memdesc<2x128xi32, #shared, #smem, mutable>, %indices: tensor<2x128xi32, #blocked>, %values: tensor<2x128xi32, #blocked>) -> tensor<2x128xi32, #blocked> {
    %old = ttg.local_atomic_scatter_rmw add, %dst[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<2x128xi32, #shared, #smem, mutable>, tensor<2x128xi32, #blocked>, tensor<2x128xi32, #blocked>) -> tensor<2x128xi32, #blocked>
    tt.return %old : tensor<2x128xi32, #blocked>
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1], [2]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // Routing preserves the upper CTA bit, so every destination stays in its pair.
  // CHECK: "ttng.preferred-cluster-fallback-ctas" = 2 : i32
  // CHECK-LABEL: tt.func @safe_barrier_routing_within_pair
  tt.func @safe_barrier_routing_within_pair(%barrier: !ttg.memdesc<4xi64, #barrier, #smem, mutable>, %pred: i1) {
    ttng.barrier_expect %barrier, 128 {fromCTA = 2 : i32}, %pred : !ttg.memdesc<4xi64, #barrier, #smem, mutable>
    ttng.arrive_barrier %barrier, 1 {fromCTA = 2 : i32} : !ttg.memdesc<4xi64, #barrier, #smem, mutable>
    ttng.arrive_barrier %barrier, 1 {multicastCTA = 1 : i32} : !ttg.memdesc<4xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_gpu_inline_asm
  tt.func @reject_gpu_inline_asm() {
    ttg.inline_asm "bar.sync 0;" {constraints = "", pure = false} : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: "ttng.preferred-cluster-fallback-ctas" = 2 : i32
  // CHECK-LABEL: tt.func @safe_pure_extern
  tt.func @safe_pure_extern(%arg: f32) -> f32 {
    %result = tt.extern_elementwise %arg {libname = "", libpath = "", pure = true, symbol = "__nv_log1pf"} : (f32) -> f32
    tt.return %result : f32
  }
}

// -----

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_impure_extern
  tt.func @reject_impure_extern(%arg: i32) -> i32 {
    %result = tt.extern_elementwise %arg {libname = "", libpath = "", pure = false, symbol = "opaque"} : (i32) -> i32
    tt.return %result : i32
  }
}
