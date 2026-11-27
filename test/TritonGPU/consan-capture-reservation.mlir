// RUN: split-file %s %t
// RUN: not triton-opt %t/missing.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/missing.mlir --check-prefix=MISSING
// RUN: not triton-opt %t/too-small.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/too-small.mlir --check-prefix=SMALL
// RUN: not triton-opt %t/unsupported-callee.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/unsupported-callee.mlir --check-prefix=CALLEE
// RUN: not triton-opt %t/private-shared-access.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/private-shared-access.mlir --check-prefix=PRIVATE-ACCESS
// RUN: not triton-opt %t/private-mbarrier.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/private-mbarrier.mlir --check-prefix=PRIVATE-MBARRIER
// RUN: not triton-opt %t/private-amd-cluster-arrive.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/private-amd-cluster-arrive.mlir --check-prefix=AMD-ARRIVE
// RUN: not triton-opt %t/private-amd-cluster-wait.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/private-amd-cluster-wait.mlir --check-prefix=AMD-WAIT
// RUN: not triton-opt %t/private-cross-cta-scratch.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/private-cross-cta-scratch.mlir --check-prefix=CROSS-CTA
// RUN: not triton-opt %t/amd-cluster-wait-with-worker-warps.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/amd-cluster-wait-with-worker-warps.mlir --check-prefix=AMD-WORKERS
// RUN: not triton-opt %t/amd-cluster-wait-with-empty-partition.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/amd-cluster-wait-with-empty-partition.mlir --check-prefix=AMD-EMPTY-PARTITION
// RUN: not triton-opt %t/missing-target.mlir -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/missing-target.mlir --check-prefix=MISSING-TARGET
// RUN: not triton-opt %t/unsupported-target.mlir -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/unsupported-target.mlir --check-prefix=UNSUPPORTED-TARGET
// RUN: not triton-opt %t/thread-capacity.mlir -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/thread-capacity.mlir --check-prefix=THREAD-CAPACITY

//--- missing-target.mlir

// MISSING-TARGET: ConSan requires a ttg.target module attribute
module {}

//--- unsupported-target.mlir

// UNSUPPORTED-TARGET: unsupported ConSan target 'cpu'
module attributes {ttg.target = "cpu"} {}

//--- missing.mlir

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @missing_reservation() {
    // MISSING: WarpSpecialize op is missing 'consan.extra_capture_bytes'
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

//--- unsupported-callee.mlir

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func private @unsupported_callee() {
    // CALLEE: ConSan cannot summarize ttg.local_alloc in non-entry function @unsupported_callee
    // CALLEE-SAME: inline the function before ConSan or keep its body limited to register/global-memory operations and compiler-owned shared scratch
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %0 = ttg.local_load %buf : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    tt.return
  }

  tt.func public @entry() {
    tt.call @unsupported_callee() {allocation.offset = 0 : i32, allocation.size = 64 : i32} : () -> ()
    tt.return
  }
}

//--- private-shared-access.mlir

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func private @read_caller_descriptor(%incoming: !ttg.memdesc<16xi32, #shared, #smem, mutable>) {
    // PRIVATE-ACCESS: ConSan cannot summarize ttg.local_load in non-entry function @read_caller_descriptor
    %value = ttg.local_load %incoming : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    tt.return
  }

  tt.func public @entry() {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    tt.call @read_caller_descriptor(%buffer) : (!ttg.memdesc<16xi32, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

//--- private-mbarrier.mlir

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32, ttg.shared = 8 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func private @initialize_caller_barrier(%barrier: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // PRIVATE-MBARRIER: ConSan cannot summarize ttng.init_barrier in non-entry function @initialize_caller_barrier
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  tt.func public @entry() {
    %barrier = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.call @initialize_caller_barrier(%barrier) : (!ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

//--- private-amd-cluster-arrive.mlir

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "hip:gfx1250"} {
  tt.func private @private_cluster_arrive() {
    // AMD-ARRIVE: ConSan cannot summarize amdg.cluster_barrier_arrive in non-entry function @private_cluster_arrive
    amdg.cluster_barrier_arrive
    tt.return
  }

  tt.func public @entry() {
    tt.call @private_cluster_arrive() : () -> ()
    tt.return
  }
}

//--- private-amd-cluster-wait.mlir

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "hip:gfx1250"} {
  tt.func private @private_cluster_wait() {
    // AMD-WAIT: ConSan cannot summarize amdg.cluster_barrier_wait in non-entry function @private_cluster_wait
    amdg.cluster_barrier_wait
    tt.return
  }

  tt.func public @entry() {
    tt.call @private_cluster_wait() : () -> ()
    tt.return
  }
}

//--- amd-cluster-wait-with-worker-warps.mlir

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32, ttg.shared = 0 : i32, ttg.target = "hip:gfx1250"} {
  tt.func public @wait_with_worker_warps() {
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      amdg.cluster_barrier_arrive
      ttg.warp_return
    } : () -> ()
    // AMD-WORKERS: ConSan does not support amdg.cluster_barrier_wait outside warp specialization in a warp-specialized kernel
    // AMD-WORKERS-SAME: because only the default warps execute the wait and the worker warps leave the barrier completion unconsumed
    amdg.cluster_barrier_wait
    tt.return
  }
}

//--- amd-cluster-wait-with-empty-partition.mlir

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 12 : i32, ttg.shared = 0 : i32, ttg.target = "hip:gfx1250"} {
  tt.func public @wait_with_empty_partition() {
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      amdg.cluster_barrier_arrive
      // AMD-EMPTY-PARTITION: ConSan does not support amdg.cluster_barrier_wait in a warp_specialize op with empty partitions, including those added by the compiler for padding
      // AMD-EMPTY-PARTITION-SAME: their warps receive the barrier completion but do not execute a wait
      amdg.cluster_barrier_wait
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      amdg.cluster_barrier_wait
      ttg.warp_return
    }
    partition1() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

//--- private-cross-cta-scratch.mlir

#cross_src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#cross_dst = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 1]]}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func private @private_cross_cta_conversion(%value: tensor<8x32xi32, #cross_src>) {
    // CROSS-CTA: ConSan cannot summarize ttg.convert_layout in non-entry function @private_cross_cta_conversion
    %converted = ttg.convert_layout %value {allocation.offset = 0 : i32, allocation.size = 512 : i32}
        : tensor<8x32xi32, #cross_src> -> tensor<8x32xi32, #cross_dst>
    tt.return
  }

  tt.func public @entry(%value: tensor<8x32xi32, #cross_src>) {
    tt.call @private_cross_cta_conversion(%value) {allocation.offset = 0 : i32, allocation.size = 512 : i32}
        : (tensor<8x32xi32, #cross_src>) -> ()
    tt.return
  }
}

//--- too-small.mlir

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @small_reservation() {
    // SMALL: ConSan WarpSpecialize capture reservation is too small: reserved 0 bytes, but 1 captures require 8 bytes
    ttg.warp_specialize() attributes {consan.extra_capture_bytes = 0 : i32}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

//--- thread-capacity.mlir

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 16 : i32, ttg.shared = 32848 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32} {
  // Thirteen base regions use five tracking classes with Early A release and CLC.
  // THREAD-CAPACITY: ConSan requires 65 tracking identities for 13 base execution regions, but supports at most 64
  tt.func public @thread_capacity(%a_desc: !tt.tensordesc<128x64xf16, #shared>, %b_desc: !tt.tensordesc<64x128xf16, #shared>) {
    ttg.warp_specialize()
    default {
      %c0 = arith.constant 0 : i32
      %true = arith.constant true
      %false = arith.constant false
      %a = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %b = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %ready = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %full = ttg.local_alloc {allocation.offset = 32784 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %early = ttg.local_alloc {allocation.offset = 32800 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.init_barrier %ready, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.init_barrier %full, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.init_barrier %early, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.barrier_expect %ready, 32768, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.async_tma_copy_global_to_local %a_desc[%c0, %c0] %a, %ready, %true : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttng.async_tma_copy_global_to_local %b_desc[%c0, %c0] %b, %ready, %true : !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      ttng.wait_barrier %ready, %c0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %acc = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %mma = ttng.tc_gen5_mma %a, %b, %acc[], %false, %true {is_async} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_commit %early {early_a_release} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.tc_gen5_commit %full : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %early, %c0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %full, %c0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %clc_bar = ttg.local_alloc {allocation.offset = 32816 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %clc_result = ttg.local_alloc {allocation.offset = 32832 : i32} : () -> !ttg.memdesc<2xi64, #barrier, #smem, mutable>
      ttng.init_barrier %clc_bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.barrier_expect %clc_bar, 16, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.clc_try_cancel %clc_result, %clc_bar : !ttg.memdesc<2xi64, #barrier, #smem, mutable>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %clc_bar, %c0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      ttg.warp_return
    }
    partition1() num_warps(1) {
      ttg.warp_return
    }
    partition2() num_warps(1) {
      ttg.warp_return
    }
    partition3() num_warps(1) {
      ttg.warp_return
    }
    partition4() num_warps(1) {
      ttg.warp_return
    }
    partition5() num_warps(1) {
      ttg.warp_return
    }
    partition6() num_warps(1) {
      ttg.warp_return
    }
    partition7() num_warps(1) {
      ttg.warp_return
    }
    partition8() num_warps(1) {
      ttg.warp_return
    }
    partition9() num_warps(1) {
      ttg.warp_return
    }
    partition10() num_warps(1) {
      ttg.warp_return
    }
    partition11() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
