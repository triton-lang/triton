// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-global-sanitizer | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: tt.func @instrumented
  tt.func @instrumented(%ptrs: tensor<128x!tt.ptr<f32>, #blocked>,
                        %mask: tensor<128xi1, #blocked>,
                        %other: tensor<128xf32, #blocked>,
                        %vals: tensor<128xf32, #blocked>) {
    // CHECK: tti.experimental_gsan_tensor_access %{{.*}}, false, %{{.*}}
    // CHECK-NEXT: %[[LD:.*]] = tt.load
    %0 = tt.load %ptrs, %mask, %other : tensor<128x!tt.ptr<f32>, #blocked>
    // CHECK: tti.experimental_gsan_tensor_access %{{.*}}, true, %{{.*}}
    // CHECK-NEXT: tt.store
    tt.store %ptrs, %vals, %mask : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @instrumented_atomic_poll
  tt.func @instrumented_atomic_poll(%ptr: !tt.ptr<i32>, %expected: i32) {
    // CHECK: %[[MATCHED:.*]] = tt.atomic_poll acquire, sys, %[[PTR:.*]], %{{.*}}
    // CHECK-NEXT: tti.experimental_gsan_atomic_poll acquire, sys, %[[PTR]], %[[MATCHED]] : !tt.ptr<i32>
    // CHECK-NEXT: ttg.barrier local
    %matched = tt.atomic_poll acquire, sys, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @instrumented_async_copy
  tt.func @instrumented_async_copy(%ptrs: tensor<128x!tt.ptr<f16>, #blocked>,
                                   %mask: tensor<128xi1, #blocked>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128xf16, #shared, #smem, mutable>
    // CHECK: tti.experimental_gsan_tensor_access %[[PTRS:.*]], false, %[[MASK:.*]] :
    // CHECK-NEXT: ttg.async_copy_global_to_local %[[PTRS]], {{.*}} mask %[[MASK]]
    %tok = ttg.async_copy_global_to_local %ptrs, %buf mask %mask : tensor<128x!tt.ptr<f16>, #blocked> -> <128xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @instrumented_async_tma_copy
  tt.func @instrumented_async_tma_copy(%desc: !tt.tensordesc<32x32xf32, #shared>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // CHECK: tti.experimental_gsan_tensordesc_info %arg0
    // CHECK: tti.experimental_gsan_tensor_access %{{.*}}, false, %{{.*}}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32] %buf, %barrier, %true : !tt.tensordesc<32x32xf32, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // CHECK: tti.experimental_gsan_tensordesc_info %arg0
    // CHECK: tti.experimental_gsan_tensor_access %{{.*}}, true, %{{.*}}
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0_i32, %c0_i32] %buf : !tt.tensordesc<32x32xf32, #shared>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#mbarrier_pair = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @mbarrier_release_acquire
  tt.func @mbarrier_release_acquire(%phase: i32, %pred: i1) {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 192 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_table_init %[[SCRATCH]], 1 : <i8>
    // CHECK-NEXT: ttng.cluster_barrier {relaxed = true}
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
    // CHECK: ttng.init_barrier %[[BARRIER:.*]], 1
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_init %[[SCRATCH]], %[[BARRIER]], 2
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
    // CHECK: tti.experimental_gsan_mbarrier_arrive %[[SCRATCH]], %[[BARRIER]], %{{.*}}, 1
    // CHECK-SAME: multicast = false, multicastMasks = array<i32>, sourceBroadcastMask = 0 : i32
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttng.arrive_barrier %[[BARRIER]], 1, %{{.*}}
    ttng.arrive_barrier %barrier, 1, %pred : !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
    // CHECK: ttng.wait_barrier %[[BARRIER]], %[[PHASE:.*]], %{{.*}}
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_wait %[[SCRATCH]], %[[BARRIER]], %[[PHASE]], %{{.*}}
    // CHECK-NEXT: ttg.barrier local
    ttng.wait_barrier %barrier, %phase, %pred : !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
    tt.return
  }
}

// -----

#mbarrier_pair = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @mbarrier_warp_specialized
  tt.func @mbarrier_warp_specialized(%pred: i1) {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
    // CHECK: ttg.warp_specialize(%[[BARRIER:.*]], %[[PRED:.*]], %[[SCRATCH]])
    ttg.warp_specialize(%barrier, %pred)
    default {
      ttg.warp_yield
    }
    // CHECK: partition0(%[[PARTITION_BARRIER:.*]]: !ttg.memdesc<1xi64, #{{.*}}, #smem, mutable>, %[[PARTITION_PRED:.*]]: i1, %[[PARTITION_SCRATCH:.*]]: !tt.ptr<i8>)
    partition0(%partition_barrier: !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>, %partition_pred: i1) num_warps(4) {
      // CHECK: tti.experimental_gsan_mbarrier_arrive %[[PARTITION_SCRATCH]], %[[PARTITION_BARRIER]], %[[PARTITION_PRED]], 1
      ttng.arrive_barrier %partition_barrier, 1, %partition_pred : !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
      %phase = arith.constant 0 : i32
      // CHECK: tti.experimental_gsan_mbarrier_wait %[[PARTITION_SCRATCH]], %[[PARTITION_BARRIER]], %{{.*}}, %[[PARTITION_PRED]]
      ttng.wait_barrier %partition_barrier, %phase, %partition_pred : !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>, i1) -> ()
    tt.return
  }
}

// -----

#mbarrier_local = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @mbarrier_tcgen5_commit
  tt.func @mbarrier_tcgen5_commit(%pred: i1) {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #mbarrier_local, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #mbarrier_local, #smem, mutable>
    // CHECK: tti.experimental_gsan_mbarrier_arrive %[[SCRATCH]], %[[BARRIER:.*]], %[[PRED:.*]], 1
    // CHECK-SAME: publishClock = false
    // CHECK-NEXT: ttng.tc_gen5_commit %[[BARRIER]], %[[PRED]]
    ttng.tc_gen5_commit %barrier, %pred : !ttg.memdesc<1xi64, #mbarrier_local, #smem, mutable>
    tt.return
  }
}

// -----

#mbarrier_local = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#shared_commit = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @mbarrier_tcgen5_commit_multicast
  tt.func @mbarrier_tcgen5_commit_multicast(%desc: !ttg.memdesc<128x128xf16, #shared_commit, #smem>, %wait_pred: i1) {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #mbarrier_local, #smem, mutable>
    // CHECK: ttng.init_barrier %[[BARRIER:.*]], 2
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_init %[[SCRATCH]], %[[BARRIER]], 2
    ttng.init_barrier %barrier, 2 : !ttg.memdesc<2xi64, #mbarrier_local, #smem, mutable>
    // CHECK: tti.experimental_gsan_mbarrier_arrive %[[SCRATCH]], %[[BARRIER:.*]], %{{.*}}, 1
    // CHECK-SAME: multicast = true, multicastMasks = array<i32: 1>, publishClock = false, sourceBroadcastMask = 0 : i32
    // CHECK-NEXT: ttng.tc_gen5_commit %[[BARRIER]] descs %{{.*}}
    ttng.tc_gen5_commit %barrier descs %desc : !ttg.memdesc<2xi64, #mbarrier_local, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared_commit, #smem>
    %phase = arith.constant 0 : i32
    // CHECK: ttng.wait_barrier %[[BARRIER]], %[[PHASE:.*]], %[[WAIT_PRED:.*]] :
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_wait %[[SCRATCH]], %[[BARRIER]], %[[PHASE]], %[[WAIT_PRED]]
    ttng.wait_barrier %barrier, %phase, %wait_pred : !ttg.memdesc<2xi64, #mbarrier_local, #smem, mutable>
    tt.return
  }
}

// -----

#mbarrier_partial = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0], [1]]}>
#tma_partial = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0], [0, 0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @mbarrier_multicast_partial_wait
  tt.func @mbarrier_multicast_partial_wait(%desc: !tt.tensordesc<256x128xf16, #tma_partial>, %wait_pred: i1) {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc
    %true = arith.constant true
    %zero = arith.constant 0 : i32
    %signal = ttg.local_alloc : () -> !ttg.memdesc<256x128xf16, #tma_partial, #smem, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #mbarrier_partial, #smem, mutable>
    // CHECK: ttng.init_barrier %[[BARRIER:.*]], 1
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_init %[[SCRATCH]], %[[BARRIER]], 2
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<2xi64, #mbarrier_partial, #smem, mutable>
    // CHECK: tti.experimental_gsan_mbarrier_arrive %[[SCRATCH]], %[[BARRIER]], %{{.*}}, 1
    // CHECK-SAME: multicast = false, multicastMasks = array<i32>, sourceBroadcastMask = 0 : i32
    ttng.barrier_expect %barrier, 32768, %true : !ttg.memdesc<2xi64, #mbarrier_partial, #smem, mutable>
    // CHECK: ttng.async_tma_copy_global_to_local %{{.*}}[%{{.*}}, %{{.*}}] %{{.*}}, %[[BARRIER]], %{{.*}} {multicast}
    ttng.async_tma_copy_global_to_local %desc[%zero, %zero] %signal, %barrier, %true {multicast} : !tt.tensordesc<256x128xf16, #tma_partial>, !ttg.memdesc<2xi64, #mbarrier_partial, #smem, mutable> -> !ttg.memdesc<256x128xf16, #tma_partial, #smem, mutable>
    %phase = arith.constant 0 : i32
    // CHECK: ttng.wait_barrier %[[BARRIER]], %[[PHASE:.*]], %[[WAIT_PRED:.*]] deps %{{.*}} :
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_wait %[[SCRATCH]], %[[BARRIER]], %[[PHASE]], %[[WAIT_PRED]]
    ttng.wait_barrier %barrier, %phase, %wait_pred deps %signal : !ttg.memdesc<2xi64, #mbarrier_partial, #smem, mutable>, !ttg.memdesc<256x128xf16, #tma_partial, #smem, mutable>
    tt.return
  }
}

// -----

#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#mbarrier_local = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, CGALayout = [[1, 0]], twoCTAs = true>

// Gluon represents two-CTA execution on the MMA op itself and does not add the
// module-level ttng.two-ctas attribute.
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @mbarrier_two_cta_mma_without_module_attr
  tt.func @mbarrier_two_cta_mma_without_module_attr() {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc
    %true = arith.constant true
    %a = ttg.local_alloc : () -> !ttg.memdesc<256x128xf16, #shared_a, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared_b, #smem, mutable>
    %d = ttng.tmem_alloc : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #mbarrier_local, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #mbarrier_local, #smem, mutable>
    // CHECK: tti.experimental_gsan_mbarrier_arrive %[[SCRATCH]], %[[BARRIER:.*]], %{{.*}}, 1
    // CHECK-SAME: multicast = true, multicastMasks = array<i32: 1>, publishClock = false, sourceBroadcastMask = 1 : i32
    // CHECK-NEXT: ttng.tc_gen5_mma %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[BARRIER]][%{{.*}}] {is_async, two_ctas}
    ttng.tc_gen5_mma %a, %b, %d, %true, %true, %barrier[%true] {is_async, two_ctas} :
      !ttg.memdesc<256x128xf16, #shared_a, #smem, mutable>,
      !ttg.memdesc<128x128xf16, #shared_b, #smem, mutable>,
      !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<1xi64, #mbarrier_local, #smem, mutable>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @cluster_barriers
  tt.func @cluster_barriers() {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 128 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_init %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier {relaxed = true}
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: ttng.cluster_barrier {relaxed = true}
    // CHECK-NOT: tti.experimental_gsan_cluster_barrier_sync
    ttng.cluster_barrier
    ttng.cluster_barrier {relaxed = true}
    tt.return
  }
}

// -----

#blockedSplitM = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#blockedSplitN = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[0, 1]]}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @cluster_barrier_equivalents
  tt.func @cluster_barrier_equivalents(%ptr: !tt.ptr<i32>) {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_init %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier {relaxed = true}
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: %{{.*}} = tti.experimental_gsan_atomic_rmw
    %one = arith.constant 1 : i32
    %release = tt.atomic_rmw add, release, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32

    // CHECK-NEXT: %{{.*}} = tti.experimental_gsan_atomic_rmw
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    %acquire = tt.atomic_rmw add, acquire, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32

    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: %{{.*}} = tti.experimental_gsan_atomic_rmw
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    %acq_rel = tt.atomic_rmw add, acq_rel, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32

    // CHECK-NEXT: %{{.*}} = tti.experimental_gsan_atomic_rmw
    %relaxed = tt.atomic_rmw add, relaxed, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32

    // CHECK: %{{.*}} = ttg.convert_layout
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    %value = arith.constant dense<0.000000e+00> : tensor<256x128xf16, #blockedSplitM>
    %converted = ttg.convert_layout %value : tensor<256x128xf16, #blockedSplitM> -> tensor<256x128xf16, #blockedSplitN>
    tt.return
  }
}

// -----

#blockedLocalBroadcast = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1]]}>
#blockedCTABroadcast = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[0]]}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @atomic_result_barrier_scope
  tt.func @atomic_result_barrier_scope(%ptr: !tt.ptr<i32>) {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_init %[[SCRATCH]] : <i8>
    %local_ptrs = tt.splat %ptr : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blockedLocalBroadcast>
    %local_ones = arith.constant dense<1> : tensor<32xi32, #blockedLocalBroadcast>

    // A CTA-local scratch barrier cannot supply the acquire cluster barrier,
    // even if shared-memory allocation has already attached an offset.
    // CHECK: %{{.*}} = tti.experimental_gsan_atomic_rmw
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    %local_acquire = tt.atomic_rmw add, acquire, gpu, %local_ptrs, %local_ones {allocation.offset = 0 : i32} : (tensor<32x!tt.ptr<i32>, #blockedLocalBroadcast>, tensor<32xi32, #blockedLocalBroadcast>) -> tensor<32xi32, #blockedLocalBroadcast>
    tt.store %local_ptrs, %local_acquire : tensor<32x!tt.ptr<i32>, #blockedLocalBroadcast>

    // A relaxed atomic with only a CTA-local result broadcast has no cluster
    // synchronization effect.
    // CHECK: %{{.*}} = tti.experimental_gsan_atomic_rmw
    // CHECK-NOT: tti.experimental_gsan_cluster_barrier_sync
    %local_relaxed = tt.atomic_rmw add, relaxed, gpu, %local_ptrs, %local_ones {allocation.offset = 0 : i32} : (tensor<32x!tt.ptr<i32>, #blockedLocalBroadcast>, tensor<32xi32, #blockedLocalBroadcast>) -> tensor<32xi32, #blockedLocalBroadcast>
    tt.store %local_ptrs, %local_relaxed : tensor<32x!tt.ptr<i32>, #blockedLocalBroadcast>

    %cluster_ptrs = tt.splat %ptr : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blockedCTABroadcast>
    %cluster_ones = arith.constant dense<1> : tensor<128xi32, #blockedCTABroadcast>
    // CHECK: %{{.*}} = tti.experimental_gsan_atomic_rmw
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    %cluster_relaxed = tt.atomic_rmw add, relaxed, gpu, %cluster_ptrs, %cluster_ones {allocation.offset = 0 : i32} : (tensor<128x!tt.ptr<i32>, #blockedCTABroadcast>, tensor<128xi32, #blockedCTABroadcast>) -> tensor<128xi32, #blockedCTABroadcast>
    tt.store %cluster_ptrs, %cluster_relaxed : tensor<128x!tt.ptr<i32>, #blockedCTABroadcast>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @instrumented_rank_reducing_tma_copy
  tt.func @instrumented_rank_reducing_tma_copy(%desc: !tt.tensordesc<1x1x1x32x32xf32, #shared>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // CHECK: tti.experimental_gsan_tensor_access %{{.*}}, false, %{{.*}}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32] %buf, %barrier, %true : !tt.tensordesc<1x1x1x32x32xf32, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // CHECK: tti.experimental_gsan_tensor_access %{{.*}}, true, %{{.*}}
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32] %buf : !tt.tensordesc<1x1x1x32x32xf32, #shared>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @instrumented_call_in_warp_specialize
  // CHECK-SAME: %[[STATE:[^:, )]+]]: !tt.ptr<i8>
  // CHECK-SAME: %[[STREAM_CLOCK:[^:, )]+]]: !tt.ptr<i32>
  // CHECK-SAME: %[[KERNEL_ID:[^:, )]+]]: i64
  tt.func public @instrumented_call_in_warp_specialize(%value: i32) {
    // CHECK: ttg.warp_specialize(%{{.*}}, %[[STATE]], %[[STREAM_CLOCK]], %[[KERNEL_ID]])
    ttg.warp_specialize(%value)
    default {
      ttg.warp_yield
    }
    // CHECK: partition0(%[[VALUE:[^:, )]+]]: i32, %[[PARTITION_STATE:[^:, )]+]]: !tt.ptr<i8>,
    // CHECK-SAME: %[[PARTITION_STREAM_CLOCK:[^:, )]+]]: !tt.ptr<i32>, %[[PARTITION_KERNEL_ID:[^:, )]+]]: i64) num_warps(4)
    partition0(%partition_value: i32) num_warps(4) {
      // CHECK: tt.call @identity(%[[VALUE]], %[[PARTITION_STATE]], %[[PARTITION_STREAM_CLOCK]], %[[PARTITION_KERNEL_ID]])
      // CHECK-SAME: : (i32, !tt.ptr<i8>, !tt.ptr<i32>, i64) -> i32
      %result = tt.call @identity(%partition_value) : (i32) -> i32
      ttg.warp_return
    } : (i32) -> ()
    tt.return
  }

  tt.func private @identity(%value: i32) -> i32 {
    tt.return %value : i32
  }
}

// -----

#blocked_rows_parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_rows = #ttg.slice<{dim = 0, parent = #blocked_rows_parent}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @instrumented_async_tma_gather_scatter
  tt.func @instrumented_async_tma_gather_scatter(%desc: !tt.tensordesc<1x32xf32, #shared>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %x_offsets = arith.constant dense<1> : tensor<32xi32, #blocked_rows>
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // CHECK: tti.experimental_gsan_tensordesc_info %arg0
    // CHECK: tti.experimental_gsan_tensor_access %{{.*}}, false, %{{.*}}
    // CHECK-NEXT: ttng.async_tma_gather
    ttng.async_tma_gather %desc[%x_offsets, %c0_i32] %buf, %barrier, %true : !tt.tensordesc<1x32xf32, #shared>, tensor<32xi32, #blocked_rows>, i32, !ttg.memdesc<1xi64, #bar, #smem, mutable>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, i1
    // CHECK: tti.experimental_gsan_tensordesc_info %arg0
    // CHECK: tti.experimental_gsan_tensor_access %{{.*}}, true, %{{.*}}
    // CHECK-NEXT: ttng.async_tma_scatter
    ttng.async_tma_scatter %desc[%x_offsets, %c0_i32] %buf : !tt.tensordesc<1x32xf32, #shared>, tensor<32xi32, #blocked_rows>, i32, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @instrumented_async_tma_copy_device_desc
  tt.func @instrumented_async_tma_copy_device_desc(%raw_desc: !tt.ptr<i8>,
                                                   %base: !tt.ptr<f32>,
                                                   %shape0: i32, %shape1: i32,
                                                   %stride0: i64) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    ttng.tensormap_create %raw_desc, %base, [%c32_i32, %c32_i32], [%shape1, %shape0], [%stride0], [%c1_i32, %c1_i32] {elem_type = 0 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 0 : i32} : (!tt.ptr<i8>, !tt.ptr<f32>, i32, i32, i32, i32, i64, i32, i32) -> ()
    // CHECK: %[[DESC:.*]] = ttng.reinterpret_tensor_descriptor %arg0
    %desc = ttng.reinterpret_tensor_descriptor %raw_desc : !tt.ptr<i8> to !tt.tensordesc<32x32xf32, #shared>
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // CHECK: tti.experimental_gsan_tensordesc_info %[[DESC]]
    // CHECK: tti.experimental_gsan_tensor_access %{{.*}}, false, %{{.*}}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32] %buf, %barrier, %true : !tt.tensordesc<32x32xf32, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}
