// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-hoist-mbarrier-lifecycle --verify-diagnostics | FileCheck %s

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @hoist_loop_lifecycle
  // CHECK: %[[BAR:.*]] = ttg.local_alloc
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: scf.for {{.*}} iter_args(%[[PHASE:.*]] = %{{.*}}) -> (i32)
  // CHECK: ttng.async_tma_copy_global_to_local {{.*}} %[[BAR]], %true {multicast}
  // CHECK: ttng.wait_barrier %[[BAR]], %[[PHASE]]
  // CHECK-NEXT: %[[NEXT:.*]] = arith.xori %[[PHASE]],
  // CHECK: scf.yield %[[NEXT]]
  // CHECK: ttng.inval_barrier %[[BAR]]
  // CHECK-NEXT: tt.return
  tt.func @hoist_loop_lifecycle(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
    %c0 = arith.constant 0 : i32
    %i0 = arith.constant 0 : index
    %i4 = arith.constant 4 : index
    %i1 = arith.constant 1 : index
    %c4 = arith.constant 4 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    scf.for %i = %i0 to %i4 step %i1 {
      %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %bar, %true {multicast} :
        !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      ttng.wait_barrier %bar, %c0 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.inval_barrier %bar : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    }
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // expected-error @+1 {{cannot hoist mbarrier lifecycle: found 1 transaction(s) without a matching wait}}
  tt.func @error_transaction_without_wait(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buf0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %buf1 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf0, %bar, %true {multicast} :
      !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    ttng.wait_barrier %bar, %c0 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    // expected-note @+1 {{first transaction without a matching wait}}
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf1, %bar, %true {multicast} :
      !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @error_conditional_transaction_unconditional_wait(%desc: !tt.tensordesc<64x128xf16, #nvmma>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    scf.if %pred {
      ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %bar, %true {multicast} :
        !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    }
    // expected-error @+1 {{cannot hoist mbarrier lifecycle: transaction and wait must be in the same block}}
    ttng.wait_barrier %bar, %c0 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @error_predicated_transaction(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %bar, %false {multicast} :
      !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    // expected-error @+1 {{cannot hoist mbarrier lifecycle for predicated transactions}}
    ttng.wait_barrier %bar, %c0 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @hoist_loop_if_lifecycle
  // CHECK: %[[BAR:.*]] = ttg.local_alloc
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: scf.for {{.*}} iter_args(%[[PHASE:.*]] = %{{.*}}) -> (i32)
  // CHECK: scf.if {{.*}} -> (i32)
  // CHECK: ttng.wait_barrier %[[BAR]], %[[PHASE]]
  // CHECK-NEXT: %[[NEXT:.*]] = arith.xori %[[PHASE]],
  // CHECK: scf.yield %[[NEXT]] : i32
  // CHECK: scf.yield %[[PHASE]] : i32
  // CHECK: scf.yield {{.*}} : i32
  // CHECK: ttng.inval_barrier %[[BAR]]
  // CHECK-NEXT: tt.return
  tt.func @hoist_loop_if_lifecycle(%desc: !tt.tensordesc<64x128xf16, #nvmma>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %i0 = arith.constant 0 : index
    %i4 = arith.constant 4 : index
    %i1 = arith.constant 1 : index
    %true = arith.constant true
    scf.for %i = %i0 to %i4 step %i1 {
      %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      scf.if %pred {
        ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %bar, %true {multicast} :
          !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
        ttng.wait_barrier %bar, %c0 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      }
      ttng.inval_barrier %bar : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    }
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @hoist_no_loop_lifecycle
  // CHECK: %[[BAR:.*]] = ttg.local_alloc
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: ttng.async_tma_copy_global_to_local {{.*}} %[[BAR]], %true {multicast}
  // CHECK: ttng.wait_barrier %[[BAR]],
  // CHECK-NEXT: ttng.inval_barrier %[[BAR]]
  // CHECK-NEXT: tt.return
  tt.func @hoist_no_loop_lifecycle(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %bar, %true {multicast} :
      !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    ttng.wait_barrier %bar, %c0 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @error_two_transactions_one_wait(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buf0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %buf1 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    // expected-note @+1 {{first transaction covered by this wait}}
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf0, %bar, %true {multicast} :
      !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf1, %bar, %true {multicast} :
      !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    // expected-error @+1 {{cannot hoist mbarrier lifecycle: expected exactly one transaction before this wait, but found 2}}
    ttng.wait_barrier %bar, %c0 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    tt.return
  }
}
