// RUN: triton-opt %s -split-input-file --triton-nvidia-tma-lowering --allocate-shared-memory -test-print-membar | FileCheck --dump-input=fail --dump-input-context=30 %s

// -----

#blockedSplitM = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#blockedSplitN = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[0, 1]]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @convert_layout_cluster_barrier
  // CHECK: ttg.convert_layout
  // CHECK-NEXT: nvg.cluster_barrier {relaxed = false}
  // CHECK-NEXT: ttg.local_alloc
  tt.func @convert_layout_cluster_barrier() -> tensor<256x128xf16, #blockedSplitM> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf16, #blockedSplitM>
    %cvt = ttg.convert_layout %cst : tensor<256x128xf16, #blockedSplitM> -> tensor<256x128xf16, #blockedSplitN>
    %buf = ttg.local_alloc %cvt : (tensor<256x128xf16, #blockedSplitN>) -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    ttg.local_store %cvt, %buf : tensor<256x128xf16, #blockedSplitN> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    %ld = ttg.local_load %buf : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> tensor<256x128xf16, #blockedSplitM>
    tt.return %ld : tensor<256x128xf16, #blockedSplitM>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[0, 1]]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // If there is a cross-CTA read dependency at kernel exit, we must end with a cluster barrier.
  // CHECK-LABEL: @end_cluster_barrier_after_cross_reduce
  // CHECK: "tt.reduce"{{.*}}axis = 1
  // CHECK: nvg.cluster_barrier {relaxed = false}
  // CHECK-NEXT: tt.return
  tt.func @end_cluster_barrier_after_cross_reduce(%arg0: tensor<256x128xf16, #blocked>) -> tensor<256xf16, #slice1> {
    %red = "tt.reduce"(%arg0) ({
    ^bb0(%lhs: f16, %rhs: f16):
      %add = arith.addf %lhs, %rhs : f16
      tt.reduce.return %add : f16
    }) {axis = 1 : i32} : (tensor<256x128xf16, #blocked>) -> tensor<256xf16, #slice1>
    tt.return %red : tensor<256xf16, #slice1>
  }
}

// -----

#sharedA = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#sharedB = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, CTASplitM = 2, twoCTAs = true>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 8 : i32, "ttng.two-ctas" = true, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Negative test: in 2CTA kernels with non-zero tensor memory size, TMEM
  // teardown sync at kernel exit means we should not add an extra cluster barrier.
  // CHECK-LABEL: @no_end_cluster_barrier_for_mma_with_tmem_teardown
  // CHECK: ttng.tmem_alloc
  // CHECK: ttng.tc_gen5_mma
  // CHECK-NOT: nvg.cluster_barrier {relaxed = false}
  // CHECK: tt.return
  tt.func @no_end_cluster_barrier_for_mma_with_tmem_teardown() {
    %true = arith.constant true
    %a = ttg.local_alloc : () -> !ttg.memdesc<256x32xf16, #sharedA, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<32x128xf16, #sharedB, #smem, mutable>
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a, %b, %acc, %true, %true {two_ctas} :
       !ttg.memdesc<256x32xf16, #sharedA, #smem, mutable>,
       !ttg.memdesc<32x128xf16, #sharedB, #smem, mutable>,
       !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[0, 1]]}>
#slice0 = #ttg.slice<{dim = 0, parent = #blocked}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // First reduction does not cross CTAs; second one does.
  // There should be no cluster barrier between them as tt.reduce always syncs internally before touching distributed shared memory.
  // but there should be a cluster barrier after the reduction that touches distributed shared memory
  // CHECK-LABEL: @reduce_nocross_then_cross
  // CHECK: "tt.reduce"{{.*}}axis = 0
  // CHECK-NOT: nvg.cluster_barrier
  // CHECK: ttg.barrier local
  // CHECK: "tt.reduce"{{.*}}axis = 1
  // CHECK: nvg.cluster_barrier {relaxed = false}
  // CHECK: "tt.reduce"{{.*}}axis = 0
  tt.func @reduce_nocross_then_cross(%t1: tensor<256x128xf16, #blocked>, %t2: tensor<256x128xf16, #blocked>) -> (tensor<128xf16, #slice0>, tensor<256xf16, #slice1>, tensor<128xf16, #slice0>) {
    %red_nc = "tt.reduce"(%t1) ({
    ^bb0(%lhs: f16, %rhs: f16):
      %add = arith.addf %lhs, %rhs : f16
      tt.reduce.return %add : f16
    }) {axis = 0 : i32} : (tensor<256x128xf16, #blocked>) -> tensor<128xf16, #slice0>

    %red_c = "tt.reduce"(%t1) ({
    ^bb0(%lhs: f16, %rhs: f16):
      %add = arith.addf %lhs, %rhs : f16
      tt.reduce.return %add : f16
    }) {axis = 1 : i32} : (tensor<256x128xf16, #blocked>) -> tensor<256xf16, #slice1>

    %red_nc2 = "tt.reduce"(%t2) ({
    ^bb0(%lhs: f16, %rhs: f16):
      %add = arith.addf %lhs, %rhs : f16
      tt.reduce.return %add : f16
    }) {axis = 0 : i32} : (tensor<256x128xf16, #blocked>) -> tensor<128xf16, #slice0>

    tt.return %red_nc, %red_c, %red_nc2 : tensor<128xf16, #slice0>, tensor<256xf16, #slice1>, tensor<128xf16, #slice0>
  }
}


// -----

#blockedSplitM = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#slice0 = #ttg.slice<{dim = 0, parent = #blockedSplitM}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0]]}>
#shared1d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reduce_cluster_barrier
  // CHECK: "tt.reduce"
  // CHECK: nvg.cluster_barrier {relaxed = false}
  // CHECK-NEXT: ttg.local_alloc
  tt.func @reduce_cluster_barrier() -> tensor<128xf16, #slice0> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf16, #blockedSplitM>
    %red = "tt.reduce"(%cst) ({
    ^bb0(%lhs: f16, %rhs: f16):
      %add = arith.addf %lhs, %rhs : f16
      tt.reduce.return %add : f16
    }) {axis = 0 : i32} : (tensor<256x128xf16, #blockedSplitM>) -> tensor<128xf16, #slice0>
    %buf = ttg.local_alloc %red : (tensor<128xf16, #slice0>) -> !ttg.memdesc<128xf16, #shared1d, #smem, mutable>
    ttg.local_store %red, %buf : tensor<128xf16, #slice0> -> !ttg.memdesc<128xf16, #shared1d, #smem, mutable>
    %ld = ttg.local_load %buf : !ttg.memdesc<128xf16, #shared1d, #smem, mutable> -> tensor<128xf16, #slice0>
    tt.return %ld : tensor<128xf16, #slice0>
  }
}

// -----

#sharedA = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#sharedB = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 2], order = [0, 1], CGALayout = [[1, 0]]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, CTASplitM = 2, twoCTAs = true>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 8 : i32, "ttng.two-ctas" = true, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // TODO: Improve the heuristics so that we don't emit a cluster_arrive/wait when num-ctas == 2!
  // A wait with explicit deps proves the inputs are no longer in use whenever num-ctas == 2 but
  // we currently emit a cluster barrier
  // CHECK-LABEL: @mma_v5_two_ctas_wait_barrier_no_cluster
  // CHECK: ttng.init_barrier
  // CHECK-NEXT: ttng.fence_mbarrier_init_release_cluster
  // CHECK-NEXT: ttng.cluster_arrive {relaxed = true}
  // CHECK-NEXT: ttng.cluster_wait
  // CHECK: ttng.wait_barrier
  // CHECK: nvg.cluster_barrier {relaxed = false}
  // CHECK: ttg.local_store
  // CHECK: tt.return
  tt.func @mma_v5_two_ctas_wait_barrier_no_cluster() -> tensor<256x32xf16, #blocked> {
    %a = ttg.local_alloc : () -> !ttg.memdesc<256x32xf16, #sharedA, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<32x128xf16, #sharedB, #smem, mutable>
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<256x32xf16, #blocked>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.fence_mbarrier_init_release_cluster
    ttng.cluster_arrive {relaxed = true}
    ttng.cluster_wait
    ttng.tc_gen5_mma %a, %b, %acc, %true, %true, %barrier[%true] {is_async, two_ctas} :
       !ttg.memdesc<256x32xf16, #sharedA, #smem, mutable>,
       !ttg.memdesc<32x128xf16, #sharedB, #smem, mutable>,
       !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.wait_barrier %barrier, %c0 deps %a, %b :
      !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>,
      !ttg.memdesc<256x32xf16, #sharedA, #smem, mutable>,
      !ttg.memdesc<32x128xf16, #sharedB, #smem, mutable>
    ttg.local_dealloc %a : !ttg.memdesc<256x32xf16, #sharedA, #smem, mutable>
    ttg.local_dealloc %b : !ttg.memdesc<32x128xf16, #sharedB, #smem, mutable>
    ttg.local_dealloc %barrier : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256x32xf16, #sharedA, #smem, mutable>
    ttg.local_store %cst, %buf : tensor<256x32xf16, #blocked> -> !ttg.memdesc<256x32xf16, #sharedA, #smem, mutable>
    %ld = ttg.local_load %buf : !ttg.memdesc<256x32xf16, #sharedA, #smem, mutable> -> tensor<256x32xf16, #blocked>
    tt.return %ld : tensor<256x32xf16, #blocked>
  }
}

// -----

#blockedSplitM = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#slice0 = #ttg.slice<{dim = 0, parent = #blockedSplitM}>
#shared1d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // We make sure we generate cluster barriers when we touch distributed shared memory in a loop.
  // TODO: A better codegen would be:
  // for {
  //   reduce
  //   cluster_arrive
  //   cluster_wait
  // }
  // local_alloc
  // but not even the membar allocation pass generates this pattern
  // An even better codegen would be the above + predicate on the last iteration
  // and come out of the for loop analysis with a read dependency on the last reduce

  // CHECK-LABEL: @scf_for_reduce_cluster_barrier
  // CHECK: scf.for
  // CHECK: nvg.cluster_barrier {relaxed = false}
  // CHECK: tt.reduce
  // CHECK: scf.yield
  // CHECK: nvg.cluster_barrier {relaxed = false}
  // CHECK-NEXT: ttg.local_alloc
  tt.func @scf_for_reduce_cluster_barrier() -> tensor<128xf16, #slice0> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf16, #blockedSplitM>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant dense<0.000000e+00> : tensor<128xf16, #slice0>
    %loop_res = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (tensor<128xf16, #slice0>) {
      %red = "tt.reduce"(%cst) ({
      ^bb0(%lhs: f16, %rhs: f16):
        %add = arith.addf %lhs, %rhs : f16
        tt.reduce.return %add : f16
      }) {axis = 0 : i32} : (tensor<256x128xf16, #blockedSplitM>) -> tensor<128xf16, #slice0>
      scf.yield %red : tensor<128xf16, #slice0>
    }
    %buf = ttg.local_alloc %loop_res : (tensor<128xf16, #slice0>) -> !ttg.memdesc<128xf16, #shared1d, #smem, mutable>
    ttg.local_store %loop_res, %buf : tensor<128xf16, #slice0> -> !ttg.memdesc<128xf16, #shared1d, #smem, mutable>
    %ld = ttg.local_load %buf : !ttg.memdesc<128xf16, #shared1d, #smem, mutable> -> tensor<128xf16, #slice0>
    tt.return %ld : tensor<128xf16, #slice0>
  }
}

// -----

#blockedSrc = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#blockedDst = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Negative test: no cluster barrier should be inserted for multiCTA when the layouts don't cross CTAs
  // CHECK-LABEL: @no_cluster_convert_block_trivial
  // CHECK-NOT: nvg.cluster_barrier
  // CHECK: tt.return
  tt.func @no_cluster_convert_block_trivial() -> tensor<256x128xf16, #blockedSrc> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf16, #blockedSrc>
    %cvt = ttg.convert_layout %cst : tensor<256x128xf16, #blockedSrc> -> tensor<256x128xf16, #blockedDst>
    %buf = ttg.local_alloc %cvt : (tensor<256x128xf16, #blockedDst>) -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    ttg.local_store %cvt, %buf : tensor<256x128xf16, #blockedDst> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    %ld = ttg.local_load %buf : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> tensor<256x128xf16, #blockedSrc>
    tt.return %ld : tensor<256x128xf16, #blockedSrc>
  }
}

// -----

#blockedTmaSrc = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#blockedTmaDst = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#nvmmaTma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#barrierEncTma = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Non-distributed convert_layout followed by multicast TMA should still
  // insert a cluster barrier before the TMA.
  // TODO: There is a world where we can do better:
  // 1. We emit the relaxed = false cluster_arrive/wait pair automatically
  // 2. We realise that we can avoid emitting it there as there is a cluster barrier after the convert_layout
  // CHECK-LABEL: @convert_layout_trivial_then_tma_multicast_cluster_barrier
  // CHECK: ttng.init_barrier
  // CHECK-NEXT: ttng.fence_mbarrier_init_release_cluster
  // CHECK-NEXT: ttng.cluster_arrive {relaxed = true}
  // CHECK-NEXT: ttng.cluster_wait
  // CHECK-NOT: ttng.cluster_wait
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.convert_layout
  // CHECK: nvg.cluster_barrier {relaxed = false}
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  tt.func @convert_layout_trivial_then_tma_multicast_cluster_barrier(%input: tensor<64x128xf16, #blockedTmaSrc>, %desc: !tt.tensordesc<tensor<64x128xf16, #nvmmaTma>>) -> tensor<64x128xf16, #blockedTmaDst> {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrierEncTma, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #barrierEncTma, #smem, mutable>
    ttng.fence_mbarrier_init_release_cluster
    ttng.cluster_arrive {relaxed = true}
    ttng.cluster_wait
    %cvt = ttg.convert_layout %input : tensor<64x128xf16, #blockedTmaSrc> -> tensor<64x128xf16, #blockedTmaDst>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %barrier, %true {multicast} :
        !tt.tensordesc<tensor<64x128xf16, #nvmmaTma>>, !ttg.memdesc<1xi64, #barrierEncTma, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttng.wait_barrier %barrier, %c0 deps %dst :
        !ttg.memdesc<1xi64, #barrierEncTma, #smem, mutable>,
        !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_dealloc %dst : !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_dealloc %barrier : !ttg.memdesc<1xi64, #barrierEncTma, #smem, mutable>
    tt.return %cvt : tensor<64x128xf16, #blockedTmaDst>
  }
}

// -----

#blockedSplitM = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blockedSplitM}>
#shared1d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Negative test: no cluster barrier should be inserted for multiCTA reduce when the axis is not split
  // CHECK-LABEL: @no_cluster_reduce_unsplit_axis
  // CHECK-NOT: nvg.cluster_barrier
  // CHECK: tt.return
  tt.func @no_cluster_reduce_unsplit_axis() -> tensor<256xf16, #slice1> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf16, #blockedSplitM>
    %red = "tt.reduce"(%cst) ({
    ^bb0(%lhs: f16, %rhs: f16):
      %add = arith.addf %lhs, %rhs : f16
      tt.reduce.return %add : f16
    }) {axis = 1 : i32} : (tensor<256x128xf16, #blockedSplitM>) -> tensor<256xf16, #slice1>
    %buf = ttg.local_alloc %red : (tensor<256xf16, #slice1>) -> !ttg.memdesc<256xf16, #shared1d, #smem, mutable>
    ttg.local_store %red, %buf : tensor<256xf16, #slice1> -> !ttg.memdesc<256xf16, #shared1d, #smem, mutable>
    %ld = ttg.local_load %buf : !ttg.memdesc<256xf16, #shared1d, #smem, mutable> -> tensor<256xf16, #slice1>
    tt.return %ld : tensor<256xf16, #slice1>
  }
}

// -----

#sharedA = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#sharedB = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 2], order = [0, 1], CGALayout = [[0, 0]]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1, CTASplitN = 2>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Negative test: no cluster barrier should be inserted for multiCTA MMA when the twoCTAs is not set
  // CHECK-LABEL: @no_cluster_mma_without_two_ctas
  // CHECK: ttng.init_barrier
  // CHECK-NEXT: ttng.fence_mbarrier_init_release_cluster
  // CHECK-NEXT: ttng.cluster_arrive {relaxed = true}
  // CHECK-NEXT: ttng.cluster_wait
  // CHECK: tt.return
  tt.func @no_cluster_mma_without_two_ctas() -> tensor<128x16xf16, #blocked> {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #blocked>
    %a = ttg.local_alloc : () -> !ttg.memdesc<128x16xf16, #sharedA, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<16x128xf16, #sharedB, #smem, mutable>
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.fence_mbarrier_init_release_cluster
    ttng.cluster_arrive {relaxed = true}
    ttng.cluster_wait
    ttng.tc_gen5_mma %a, %b, %acc, %true, %true, %barrier[%true] {is_async} :
       !ttg.memdesc<128x16xf16, #sharedA, #smem, mutable>,
       !ttg.memdesc<16x128xf16, #sharedB, #smem, mutable>,
       !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.wait_barrier %barrier, %c0 deps %a, %b :
      !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>,
      !ttg.memdesc<128x16xf16, #sharedA, #smem, mutable>,
      !ttg.memdesc<16x128xf16, #sharedB, #smem, mutable>
    ttg.local_dealloc %a : !ttg.memdesc<128x16xf16, #sharedA, #smem, mutable>
    ttg.local_dealloc %b : !ttg.memdesc<16x128xf16, #sharedB, #smem, mutable>
    ttg.local_dealloc %barrier : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128x16xf16, #sharedA, #smem, mutable>
    ttg.local_store %cst, %buf : tensor<128x16xf16, #blocked> -> !ttg.memdesc<128x16xf16, #sharedA, #smem, mutable>
    %ld = ttg.local_load %buf : !ttg.memdesc<128x16xf16, #sharedA, #smem, mutable> -> tensor<128x16xf16, #blocked>
    tt.return %ld : tensor<128x16xf16, #blocked>
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Negative test: no cluster barrier should be inserted for multiCTA TMA when the multicast is not set
  // CHECK-LABEL: @no_cluster_tma_without_multicast
  // CHECK: ttng.init_barrier
  // CHECK-NEXT: ttng.fence_mbarrier_init_release_cluster
  // CHECK-NEXT: ttng.cluster_arrive {relaxed = true}
  // CHECK-NEXT: ttng.cluster_wait
  // CHECK: tt.return
  tt.func @no_cluster_tma_without_multicast(%desc: !tt.tensordesc<tensor<64x128xf16, #nvmma>>) -> tensor<64x128xf16, #blocked> {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable>
    ttng.fence_mbarrier_init_release_cluster
    ttng.cluster_arrive {relaxed = true}
    ttng.cluster_wait
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %barrier, %true :
      !tt.tensordesc<tensor<64x128xf16, #nvmma>>, !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    ttng.wait_barrier %barrier, %c0 deps %buf :
      !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable>,
      !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    ttg.local_dealloc %buf : !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    ttg.local_dealloc %barrier : !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable>
    %buf2 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    ttg.local_store %cst, %buf2 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %ld = ttg.local_load %buf2 : !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable> -> tensor<64x128xf16, #blocked>
    tt.return %ld : tensor<64x128xf16, #blocked>
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // NB. Testing only. Note that in this program async_tma_copy_global
  //     and local_store are racing!
  // Even though we have a wait_barrier, we should still emit a cluster
  // barrier at the end of the kernel, as a in that wait just one CTA is waiting
  // for both the CTAs. It could be that CTA1 exits the kernel before CTA0,
  // otherwise!
  // CHECK-LABEL: @no_cluster_when_same_allocation
  // CHECK: ttng.init_barrier
  // CHECK-NEXT: ttng.fence_mbarrier_init_release_cluster
  // CHECK-NEXT: ttng.cluster_arrive {relaxed = true}
  // CHECK-NEXT: ttng.cluster_wait
  // CHECK: ttng.wait_barrier
  // CHECK: nvg.cluster_barrier {relaxed = false}
  // CHECK: tt.return
  tt.func @no_cluster_when_same_allocation(%desc: !tt.tensordesc<tensor<64x128xf16, #nvmma>>) -> tensor<64x128xf16, #blocked> {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable>
    ttng.fence_mbarrier_init_release_cluster
    ttng.cluster_arrive {relaxed = true}
    ttng.cluster_wait
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %barrier, %true {multicast} :
      !tt.tensordesc<tensor<64x128xf16, #nvmma>>, !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    ttng.wait_barrier %barrier, %c0 deps %buf :
      !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable>,
      !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    ttg.local_store %cst, %buf : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %ld = ttg.local_load %buf : !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable> -> tensor<64x128xf16, #blocked>
    tt.return %ld : tensor<64x128xf16, #blocked>
  }
}

// -----

#sharedA = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#sharedB = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#barrierTMA = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#barrierMMA = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1, CTASplitM = 2, twoCTAs = true>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttng.two-ctas" = true, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // Exact TTGIR shape for:
  // test_tma_mma_shared_inputs[True-True-ctas_per_cga1-reps0-warps2]
  // with smem allocations outside the loop.
  // If we fully manage the shared memory and the allocator does not create any new aliases,
  // then we shouldn't emit any cluster barriers
  // CHECK-LABEL: @example_matmul
  // CHECK: ttg.local_alloc
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK-NEXT: ttng.init_barrier
  // CHECK: ttng.tmem_alloc
  // CHECK-NEXT: ttng.fence_mbarrier_init_release_cluster
  // CHECK-NEXT: ttng.cluster_arrive {relaxed = true}
  // CHECK-NEXT: ttng.cluster_wait
  // CHECK: scf.for
  // CHECK: ttng.barrier_expect
  // CHECK-NOT: nvg.cluster_barrier {relaxed = false}
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  // CHECK-NOT: nvg.cluster_barrier {relaxed = false}
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  // CHECK-NOT: nvg.cluster_barrier {relaxed = false}
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: nvg.cluster_barrier {relaxed = false}
  // CHECK: ttng.tc_gen5_mma
  // CHECK: ttng.wait_barrier
  tt.func @example_matmul(%a_desc: !tt.tensordesc<tensor<256x16xf16, #sharedA>>, %b_desc: !tt.tensordesc<tensor<16x64xf16, #sharedB>>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : i32
    %c0_idx = arith.constant 0 : index
    %c4_idx = arith.constant 4 : index
    %c1_idx = arith.constant 1 : index
    %smem_a = ttg.local_alloc : () -> !ttg.memdesc<256x16xf16, #sharedA, #smem, mutable>
    %smem_b = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #sharedB, #smem, mutable>
    %bTMA = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrierTMA, #smem, mutable>
    %bMMA = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierMMA, #smem, mutable>
    ttng.init_barrier %bTMA, 1 : !ttg.memdesc<1xi64, #barrierTMA, #smem, mutable>
    ttng.init_barrier %bMMA, 1 : !ttg.memdesc<2xi64, #barrierMMA, #smem, mutable>
    %acc_tmem = ttng.tmem_alloc : () -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.fence_mbarrier_init_release_cluster
    ttng.cluster_arrive {relaxed = true}
    ttng.cluster_wait
    %phase_init = arith.constant 0 : i32
    %phase_tma = scf.for %k = %c0_idx to %c4_idx step %c1_idx iter_args(%phase = %phase_init) -> (i32) {
      %k_i32 = arith.index_cast %k : index to i32
      ttng.barrier_expect %bTMA, 5120, %true : !ttg.memdesc<1xi64, #barrierTMA, #smem, mutable>
      %offs = arith.muli %k_i32, %c16 : i32
      ttng.async_tma_copy_global_to_local %a_desc[%c0, %offs] %smem_a, %bTMA, %true {multicast} :
        !tt.tensordesc<tensor<256x16xf16, #sharedA>>, !ttg.memdesc<1xi64, #barrierTMA, #smem, mutable> -> !ttg.memdesc<256x16xf16, #sharedA, #smem, mutable>
      ttng.async_tma_copy_global_to_local %b_desc[%offs, %c0] %smem_b, %bTMA, %true {multicast} :
        !tt.tensordesc<tensor<16x64xf16, #sharedB>>, !ttg.memdesc<1xi64, #barrierTMA, #smem, mutable> -> !ttg.memdesc<16x64xf16, #sharedB, #smem, mutable>
      ttng.wait_barrier %bTMA, %phase, %true deps %smem_a, %smem_b :
        !ttg.memdesc<1xi64, #barrierTMA, #smem, mutable>,
        !ttg.memdesc<256x16xf16, #sharedA, #smem, mutable>,
        !ttg.memdesc<16x64xf16, #sharedB, #smem, mutable>
      %next_phase = arith.xori %phase, %c1 : i32
      %use_acc = arith.cmpi ne, %k_i32, %c0 : i32
      %mma_tok = ttng.tc_gen5_mma %smem_a, %smem_b, %acc_tmem[], %use_acc, %true, %bMMA[%true] {is_async, two_ctas, multicast} :
         !ttg.memdesc<256x16xf16, #sharedA, #smem, mutable>,
         !ttg.memdesc<16x64xf16, #sharedB, #smem, mutable>,
         !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>,
         !ttg.memdesc<2xi64, #barrierMMA, #smem, mutable>
      ttng.wait_barrier %bMMA, %phase, %true deps %smem_a, %smem_b :
        !ttg.memdesc<2xi64, #barrierMMA, #smem, mutable>,
        !ttg.memdesc<256x16xf16, #sharedA, #smem, mutable>,
        !ttg.memdesc<16x64xf16, #sharedB, #smem, mutable>
      scf.yield %next_phase : i32
    }
    ttg.local_dealloc %bTMA : !ttg.memdesc<1xi64, #barrierTMA, #smem, mutable>
    ttg.local_dealloc %bMMA : !ttg.memdesc<2xi64, #barrierMMA, #smem, mutable>
    ttg.local_dealloc %smem_a : !ttg.memdesc<256x16xf16, #sharedA, #smem, mutable>
    ttg.local_dealloc %smem_b : !ttg.memdesc<16x64xf16, #sharedB, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The wait just waits on the first CTA, so there should be a barrier in between
  // CHECK-LABEL: @cluster_barrier_between_lifetimes_same_offset
  // CHECK: ttng.init_barrier
  // CHECK-NEXT: ttng.fence_mbarrier_init_release_cluster
  // CHECK-NEXT: ttng.cluster_arrive {relaxed = true}
  // CHECK-NEXT: ttng.cluster_wait
  // CHECK: ttg.local_alloc
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  // CHECK: ttng.wait_barrier
  // CHECK: ttg.local_dealloc
  // CHECK: ttg.local_alloc
  // CHECK-NEXT: nvg.cluster_barrier {relaxed = false}
  // CHECK-NEXT: ttng.async_tma_copy_global_to_local
  tt.func @cluster_barrier_between_lifetimes_same_offset(%desc: !tt.tensordesc<tensor<64x128xf16, #nvmma>>) -> tensor<64x128xf16, #blocked> {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true

    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable>
    ttng.fence_mbarrier_init_release_cluster
    ttng.cluster_arrive {relaxed = true}
    ttng.cluster_wait
    // a lifetime start
    %a = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %a, %barrier, %true {multicast} :
      !tt.tensordesc<tensor<64x128xf16, #nvmma>>, !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    ttng.wait_barrier %barrier, %c0 deps %a :
      !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable>,
      !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %t = ttg.local_load %a : !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable> -> tensor<64x128xf16, #blocked>
    ttg.local_dealloc %a : !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    // a lifetime end

    // b lifetime start
    %b = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %b, %barrier, %true {multicast} :
      !tt.tensordesc<tensor<64x128xf16, #nvmma>>, !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    ttng.wait_barrier %barrier, %c0 deps %b :
      !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable>,
      !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %t2 = ttg.local_load %b : !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable> -> tensor<64x128xf16, #blocked>
    ttg.local_dealloc %b : !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    // b lifetime end

    tt.return %t2 : tensor<64x128xf16, #blocked>
  }
}
