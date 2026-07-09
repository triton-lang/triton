// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-hoist-mbarrier-lifecycle | FileCheck %s

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @hoist_loop_lifecycle
  // CHECK: %[[BAR:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
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
  // CHECK-LABEL: tt.func @hoist_loop_predicated_wait_lifecycle
  // CHECK: %[[BAR:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: scf.for {{.*}} iter_args(%[[PHASE:.*]] = %{{.*}}) -> (i32)
  // CHECK: ttng.wait_barrier %[[BAR]], %[[PHASE]], %[[PRED:.*]] :
  // CHECK-NEXT: %[[NEXT:.*]] = arith.xori %[[PHASE]],
  // CHECK-NEXT: %[[SELECTED:.*]] = arith.select %[[PRED]], %[[NEXT]], %[[PHASE]]
  // CHECK: scf.yield %[[SELECTED]]
  // CHECK: ttng.inval_barrier %[[BAR]]
  // CHECK-NEXT: tt.return
  tt.func @hoist_loop_predicated_wait_lifecycle(%desc: !tt.tensordesc<64x128xf16, #nvmma>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %i0 = arith.constant 0 : index
    %i4 = arith.constant 4 : index
    %i1 = arith.constant 1 : index
    %true = arith.constant true
    scf.for %i = %i0 to %i4 step %i1 {
      %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %bar, %true {multicast} :
        !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      ttng.wait_barrier %bar, %c0, %pred : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
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
  // CHECK-LABEL: tt.func @hoist_nested_loop_lifecycle
  // CHECK: %[[BAR:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: scf.for {{.*}} iter_args(%[[OUTER_PHASE:.*]] = %{{.*}}) -> (i32)
  // CHECK: %[[INNER_RESULT:.*]] = scf.for {{.*}} iter_args(%[[INNER_PHASE:.*]] = %[[OUTER_PHASE]]) -> (i32)
  // CHECK: ttng.wait_barrier %[[BAR]], %[[INNER_PHASE]]
  // CHECK-NEXT: %[[NEXT:.*]] = arith.xori %[[INNER_PHASE]],
  // CHECK: scf.yield %[[NEXT]] : i32
  // CHECK: scf.yield %[[INNER_RESULT]] : i32
  // CHECK: ttng.inval_barrier %[[BAR]]
  // CHECK-NEXT: tt.return
  tt.func @hoist_nested_loop_lifecycle(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
    %c0 = arith.constant 0 : i32
    %i0 = arith.constant 0 : index
    %i2 = arith.constant 2 : index
    %i4 = arith.constant 4 : index
    %i1 = arith.constant 1 : index
    %true = arith.constant true
    scf.for %i = %i0 to %i2 step %i1 {
      scf.for %j = %i0 to %i4 step %i1 {
        %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
        %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
        ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
        ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %bar, %true {multicast} :
          !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
        ttng.wait_barrier %bar, %c0 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
        ttng.inval_barrier %bar : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      }
    }
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @hoist_if_else_wait_lifecycle
  // CHECK: %[[BAR:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: scf.for {{.*}} iter_args(%[[PHASE:.*]] = %{{.*}}) -> (i32)
  // CHECK: %[[IF_RESULT:.*]] = scf.if {{.*}} -> (i32)
  // CHECK: ttng.wait_barrier %[[BAR]], %[[PHASE]]
  // CHECK-NEXT: %[[THEN_NEXT:.*]] = arith.xori %[[PHASE]],
  // CHECK: scf.yield %[[THEN_NEXT]] : i32
  // CHECK: } else {
  // CHECK: ttng.wait_barrier %[[BAR]], %[[PHASE]]
  // CHECK-NEXT: %[[ELSE_NEXT:.*]] = arith.xori %[[PHASE]],
  // CHECK: scf.yield %[[ELSE_NEXT]] : i32
  // CHECK: scf.yield %[[IF_RESULT]] : i32
  // CHECK: ttng.inval_barrier %[[BAR]]
  // CHECK-NEXT: tt.return
  tt.func @hoist_if_else_wait_lifecycle(%desc: !tt.tensordesc<64x128xf16, #nvmma>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %i0 = arith.constant 0 : index
    %i4 = arith.constant 4 : index
    %i1 = arith.constant 1 : index
    %true = arith.constant true
    scf.for %i = %i0 to %i4 step %i1 {
      %buf0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      %buf1 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      scf.if %pred {
        ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf0, %bar, %true {multicast} :
          !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
        ttng.wait_barrier %bar, %c0 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      } else {
        ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf1, %bar, %true {multicast} :
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
  // CHECK-LABEL: tt.func @hoist_nested_if_else_wait_lifecycle
  // CHECK: %[[BAR:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: scf.for {{.*}} iter_args(%[[PHASE:.*]] = %{{.*}}) -> (i32)
  // CHECK: %[[OUTER_RESULT:.*]] = scf.if {{.*}} -> (i32)
  // CHECK: %[[INNER_RESULT:.*]] = scf.if {{.*}} -> (i32)
  // CHECK: ttng.wait_barrier %[[BAR]], %[[PHASE]]
  // CHECK-NEXT: %[[INNER_THEN_NEXT:.*]] = arith.xori %[[PHASE]],
  // CHECK: scf.yield %[[INNER_THEN_NEXT]] : i32
  // CHECK: } else {
  // CHECK: ttng.wait_barrier %[[BAR]], %[[PHASE]]
  // CHECK-NEXT: %[[INNER_ELSE_NEXT:.*]] = arith.xori %[[PHASE]],
  // CHECK: scf.yield %[[INNER_ELSE_NEXT]] : i32
  // CHECK: scf.yield %[[INNER_RESULT]] : i32
  // CHECK: } else {
  // CHECK: ttng.wait_barrier %[[BAR]], %[[PHASE]]
  // CHECK-NEXT: %[[OUTER_ELSE_NEXT:.*]] = arith.xori %[[PHASE]],
  // CHECK: scf.yield %[[OUTER_ELSE_NEXT]] : i32
  // CHECK: scf.yield %[[OUTER_RESULT]] : i32
  // CHECK: ttng.inval_barrier %[[BAR]]
  // CHECK-NEXT: tt.return
  tt.func @hoist_nested_if_else_wait_lifecycle(%desc: !tt.tensordesc<64x128xf16, #nvmma>, %pred0: i1, %pred1: i1) {
    %c0 = arith.constant 0 : i32
    %i0 = arith.constant 0 : index
    %i4 = arith.constant 4 : index
    %i1 = arith.constant 1 : index
    %true = arith.constant true
    scf.for %i = %i0 to %i4 step %i1 {
      %buf0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      %buf1 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      %buf2 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      scf.if %pred0 {
        scf.if %pred1 {
          ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
          ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf0, %bar, %true {multicast} :
            !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
          ttng.wait_barrier %bar, %c0 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
        } else {
          ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
          ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf1, %bar, %true {multicast} :
            !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
          ttng.wait_barrier %bar, %c0 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
        }
      } else {
        ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf2, %bar, %true {multicast} :
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
  // CHECK-LABEL: tt.func @hoist_loop_if_lifecycle
  // CHECK: %[[BAR:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: scf.for {{.*}} iter_args(%[[PHASE:.*]] = %{{.*}}) -> (i32)
  // CHECK: scf.if
  // CHECK: ttng.wait_barrier %[[BAR]], %[[PHASE]]
  // CHECK: %[[NEXT:.*]] = arith.xori %[[PHASE]],
  // CHECK: scf.yield %[[NEXT]] : i32
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
  // CHECK: %[[BAR:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
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
  // CHECK-LABEL: tt.func @skip_no_loop_sequential_lifecycle
  // CHECK: %[[BAR:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: ttng.wait_barrier %[[BAR]],
  // CHECK-NEXT: ttng.inval_barrier %[[BAR]]
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: ttng.wait_barrier %[[BAR]],
  // CHECK-NEXT: ttng.inval_barrier %[[BAR]]
  // CHECK-NEXT: tt.return
  tt.func @skip_no_loop_sequential_lifecycle(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
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
  // CHECK-LABEL: tt.func @hoist_while_lifecycle
  // CHECK: %[[BAR:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: scf.while (%[[I:.*]] = %{{.*}}, %[[BEFORE_PHASE:.*]] = %{{.*}}) : (index, i32) -> (index, i32)
  // CHECK: scf.condition({{.*}}) %[[I]], %[[BEFORE_PHASE]] : index, i32
  // CHECK: ^bb0(%{{.*}}: index, %[[PHASE:.*]]: i32):
  // CHECK: ttng.wait_barrier %[[BAR]], %[[PHASE]]
  // CHECK-NEXT: %[[NEXT_PHASE:.*]] = arith.xori %[[PHASE]],
  // CHECK: scf.yield {{.*}}, %[[NEXT_PHASE]] : index, i32
  // CHECK: ttng.inval_barrier %[[BAR]]
  // CHECK-NEXT: tt.return
  tt.func @hoist_while_lifecycle(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
    %c0 = arith.constant 0 : i32
    %i0 = arith.constant 0 : index
    %i4 = arith.constant 4 : index
    %i1 = arith.constant 1 : index
    %true = arith.constant true
    %unused = scf.while (%i = %i0) : (index) -> (index) {
      %cond = arith.cmpi slt, %i, %i4 : index
      scf.condition(%cond) %i : index
    } do {
    ^bb0(%i_arg: index):
      %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %bar, %true {multicast} :
        !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      ttng.wait_barrier %bar, %c0 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.inval_barrier %bar : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      %next_i = arith.addi %i_arg, %i1 : index
      scf.yield %next_i : index
    }
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @hoist_alias_transaction_lifecycle
  // CHECK: %[[BAR:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: %[[VIEW:.*]] = ttg.memdesc_reinterpret %[[BAR]]
  // CHECK: %[[ALIAS:.*]] = arith.select %true, %[[VIEW]], %[[BAR]]
  // CHECK: ttng.async_tma_copy_global_to_local {{.*}} %[[ALIAS]], %true {multicast}
  // CHECK: ttng.wait_barrier %[[BAR]],
  // CHECK-NEXT: ttng.inval_barrier %[[BAR]]
  // CHECK-NEXT: tt.return
  tt.func @hoist_alias_transaction_lifecycle(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    %bar_view = ttg.memdesc_reinterpret %bar : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    %bar_alias = arith.select %true, %bar_view, %bar : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %bar_alias, %true {multicast} :
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
  // CHECK-LABEL: tt.func @hoist_while_if_else_wait_lifecycle
  // CHECK: %[[BAR:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: scf.while (%[[I:.*]] = %{{.*}}, %[[BEFORE_PHASE:.*]] = %{{.*}}) : (index, i32) -> (index, i32)
  // CHECK: scf.condition({{.*}}) %[[I]], %[[BEFORE_PHASE]] : index, i32
  // CHECK: ^bb0(%{{.*}}: index, %[[PHASE:.*]]: i32):
  // CHECK: %[[IF_RESULT:.*]] = scf.if {{.*}} -> (i32)
  // CHECK: ttng.wait_barrier %[[BAR]], %[[PHASE]]
  // CHECK-NEXT: %[[THEN_NEXT:.*]] = arith.xori %[[PHASE]],
  // CHECK: scf.yield %[[THEN_NEXT]] : i32
  // CHECK: } else {
  // CHECK: ttng.wait_barrier %[[BAR]], %[[PHASE]]
  // CHECK-NEXT: %[[ELSE_NEXT:.*]] = arith.xori %[[PHASE]],
  // CHECK: scf.yield %[[ELSE_NEXT]] : i32
  // CHECK: scf.yield {{.*}}, %[[IF_RESULT]] : index, i32
  // CHECK: ttng.inval_barrier %[[BAR]]
  // CHECK-NEXT: tt.return
  tt.func @hoist_while_if_else_wait_lifecycle(%desc: !tt.tensordesc<64x128xf16, #nvmma>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %i0 = arith.constant 0 : index
    %i4 = arith.constant 4 : index
    %i1 = arith.constant 1 : index
    %true = arith.constant true
    %unused = scf.while (%i = %i0) : (index) -> (index) {
      %cond = arith.cmpi slt, %i, %i4 : index
      scf.condition(%cond) %i : index
    } do {
    ^bb0(%i_arg: index):
      %buf0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      %buf1 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      scf.if %pred {
        ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf0, %bar, %true {multicast} :
          !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
        ttng.wait_barrier %bar, %c0 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      } else {
        ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf1, %bar, %true {multicast} :
          !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
        ttng.wait_barrier %bar, %c0 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      }
      ttng.inval_barrier %bar : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      %next_i = arith.addi %i_arg, %i1 : index
      scf.yield %next_i : index
    }
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @hoist_two_transactions_one_wait
  // CHECK: %[[BAR:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: ttng.async_tma_copy_global_to_local {{.*}} %[[BAR]], %true {multicast}
  // CHECK: ttng.async_tma_copy_global_to_local {{.*}} %[[BAR]], %true {multicast}
  // CHECK: ttng.wait_barrier %[[BAR]],
  // CHECK-NEXT: ttng.inval_barrier %[[BAR]]
  // CHECK-NEXT: tt.return
  tt.func @hoist_two_transactions_one_wait(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buf0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %buf1 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf0, %bar, %true {multicast} :
      !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf1, %bar, %true {multicast} :
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
  // CHECK-LABEL: tt.func @skip_block_arg_phase_lifecycle
  // CHECK: scf.for {{.*}} iter_args(%[[PHASE:.*]] =
  // CHECK: %[[BAR:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: ttng.wait_barrier %[[BAR]], %[[PHASE]]
  // CHECK: ttng.inval_barrier %[[BAR]]
  tt.func @skip_block_arg_phase_lifecycle(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
    %c0 = arith.constant 0 : i32
    %i0 = arith.constant 0 : index
    %i4 = arith.constant 4 : index
    %i1 = arith.constant 1 : index
    %true = arith.constant true
    scf.for %i = %i0 to %i4 step %i1 iter_args(%phase = %c0) -> (i32) {
      %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %bar, %true {multicast} :
        !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      ttng.wait_barrier %bar, %phase : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.inval_barrier %bar : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      scf.yield %phase : i32
    }
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @skip_nonzero_phase_lifecycle
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{[[:alnum:]_]+}} {
  // CHECK: %[[BAR:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
  // CHECK-NEXT: ttng.init_barrier %[[BAR]], 1
  // CHECK: ttng.wait_barrier %[[BAR]], %{{.*}}
  // CHECK: ttng.inval_barrier %[[BAR]]
  // CHECK: }
  // CHECK-NEXT: tt.return
  tt.func @skip_nonzero_phase_lifecycle(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
    %c0 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %i0 = arith.constant 0 : index
    %i4 = arith.constant 4 : index
    %i1 = arith.constant 1 : index
    %true = arith.constant true
    scf.for %i = %i0 to %i4 step %i1 {
      %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.barrier_expect %bar, 16384, %true : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %bar, %true {multicast} :
        !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
      ttng.wait_barrier %bar, %c1_i32 : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
      ttng.inval_barrier %bar : !ttg.memdesc<2xi64, #barrierEnc, #smem, mutable>
    }
    tt.return
  }
}
