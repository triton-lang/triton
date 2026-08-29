// RUN: split-file %s %t
// RUN: not triton-opt %t/missing.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/missing.mlir --check-prefix=MISSING
// RUN: not triton-opt %t/too-small.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/too-small.mlir --check-prefix=SMALL
// RUN: not triton-opt %t/unsupported-callee.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/unsupported-callee.mlir --check-prefix=CALLEE
// RUN: not triton-opt %t/private-shared-access.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/private-shared-access.mlir --check-prefix=PRIVATE-ACCESS
// RUN: not triton-opt %t/private-mbarrier.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/private-mbarrier.mlir --check-prefix=PRIVATE-MBARRIER
// RUN: not triton-opt %t/private-amd-cluster-arrive.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/private-amd-cluster-arrive.mlir --check-prefix=AMD-ARRIVE
// RUN: not triton-opt %t/private-amd-cluster-wait.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/private-amd-cluster-wait.mlir --check-prefix=AMD-WAIT
// RUN: not triton-opt %t/private-cross-cta-scratch.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/private-cross-cta-scratch.mlir --check-prefix=CROSS-CTA

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
