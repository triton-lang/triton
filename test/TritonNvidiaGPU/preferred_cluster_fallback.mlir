// RUN: triton-opt %s -split-input-file --triton-nvidia-preferred-cluster-fallback=compute-capability=100 | FileCheck %s

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: "ttng.preferred-cluster-fallback-ctas" = 2 : i32
  // CHECK-LABEL: tt.func @safe_no_cross_cta
  tt.func @safe_no_cross_cta() {
    tt.return
  }
}

// -----

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
  // CHECK-LABEL: tt.func @reject_atomic_rmw
  tt.func @reject_atomic_rmw(%arg0: !tt.ptr<f32>) {
    %true = arith.constant true
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %cst, %true : (!tt.ptr<f32>, f32, i1) -> f32
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_atomic_cas
  tt.func @reject_atomic_cas(%arg0: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %0 = tt.atomic_cas relaxed, gpu, %arg0, %c0, %c1 : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
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
#mcast = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0], [0, 0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttng.preferred-cluster-fallback-ctas
  // CHECK-LABEL: tt.func @reject_static_mma_completion_count
  tt.func @reject_static_mma_completion_count(%barrier: !ttg.memdesc<4xi64, #barrierLocal, #smem, mutable>, %desc: !ttg.memdesc<128x128xf16, #mcast, #smem>) {
    ttng.init_barrier %barrier, 2 : !ttg.memdesc<4xi64, #barrierLocal, #smem, mutable>
    ttng.tc_gen5_commit %barrier descs %desc : !ttg.memdesc<4xi64, #barrierLocal, #smem, mutable>, !ttg.memdesc<128x128xf16, #mcast, #smem>
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
