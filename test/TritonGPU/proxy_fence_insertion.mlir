// RUN: triton-opt %s --convert-scf-to-cf -triton-nvidia-gpu-proxy-fence-insertion --split-input-file -allow-unregistered-dialect | FileCheck %s

#poll_blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#poll_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#poll_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#poll_smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // A successful polling loop acquires cp.async writes just like a blocking
  // wait. Forwarding them to TMA requires a generic-to-async proxy fence.
  // CHECK-LABEL: @copy_poll_then_tma_store
  // CHECK: ttng.barrier_test_wait_report
  // CHECK: ttng.fence_async_shared {bCluster = false}
  // CHECK-NEXT: ttng.async_tma_copy_local_to_global
  tt.func @copy_poll_then_tma_store(%src: tensor<128x!tt.ptr<i32>, #poll_blocked>, %desc: !tt.tensordesc<128xi32, #poll_shared>) {
    %zero = arith.constant 0 : i32
    %true = arith.constant true
    %false = arith.constant false
    %mem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128xi32, #poll_shared, #poll_smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 512 : i32} : () -> !ttg.memdesc<1xi64, #poll_bar, #poll_smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #poll_bar, #poll_smem, mutable>
    ttg.async_copy_global_to_local %src, %mem : tensor<128x!tt.ptr<i32>, #poll_blocked> -> <128xi32, #poll_shared, #poll_smem, mutable>
    ttng.async_copy_mbarrier_arrive %bar : !ttg.memdesc<1xi64, #poll_bar, #poll_smem, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #poll_bar, #poll_smem, mutable>
    %complete = scf.while (%done = %false) : (i1) -> i1 {
      %pending = arith.xori %done, %true : i1
      scf.condition(%pending) %done : i1
    } do {
    ^bb0(%previous: i1):
      %done, %report = ttng.barrier_test_wait_report %bar, %zero : !ttg.memdesc<1xi64, #poll_bar, #poll_smem, mutable> -> (i1, i1)
      scf.yield %done : i1
    }
    ttng.async_tma_copy_local_to_global %desc[%zero] %mem : !tt.tensordesc<128xi32, #poll_shared>, !ttg.memdesc<128xi32, #poll_shared, #poll_smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #poll_bar, #poll_smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: fence_write_after_read
  tt.func @fence_write_after_read(%arg0: !tt.tensordesc<64x64xf32, #shared>, %arg1: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) {
    // CHECK: ttg.local_load
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.async_tma_copy_global_to_local
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<32x64xf32, #shared, #smem, mutable>
    %1 = ttg.local_load %0 : !ttg.memdesc<32x64xf32, #shared, #smem, mutable> -> tensor<32x64xf32, #blocked>
    "test.keep"(%1) : (tensor<32x64xf32, #blocked>) -> ()
    %2 = ttg.local_alloc {allocation.offset = 32 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %2, %arg1, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: no_fence_read_after_read
  tt.func @no_fence_read_after_read(%desc: !tt.tensordesc<64x64xf32, #shared>) {
    %c0 = arith.constant 0 : i32
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    // CHECK: ttg.local_load
    // CHECK-NEXT: "test.keep"
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %buffer : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: inline_asm_no_proxy_effects
  tt.func @inline_asm_no_proxy_effects(%desc: !tt.tensordesc<64x64xf32, #shared>, %buffer: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>, %bar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.inline_asm
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttg.inline_asm "// opaque descriptor" {constraints = "r", pure = false} %buffer : (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: no_fence_for_disjoint_allocations
  tt.func @no_fence_for_disjoint_allocations(%desc: !tt.tensordesc<64x64xf32, #shared>, %bar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %read = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x64xf32, #shared, #smem, mutable>
    %write = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %value = ttg.local_load %read : !ttg.memdesc<32x64xf32, #shared, #smem, mutable> -> tensor<32x64xf32, #blocked>
    // CHECK: ttg.local_load
    // CHECK-NEXT: "test.keep"
    "test.keep"(%value) : (tensor<32x64xf32, #blocked>) -> ()
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %write, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: fence_write_after_write
  tt.func @fence_write_after_write(%value: tensor<64x64xf32, #blocked>, %desc: !tt.tensordesc<64x64xf32, #shared>, %bar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    // CHECK: ttg.local_store
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttg.local_store %value, %buffer : tensor<64x64xf32, #blocked> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#store_blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>
#store_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[0, 0]]}>
#store_barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#store_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // st.async executes asynchronously but accesses shared memory through the
  // generic proxy, so consuming its destination through TMA needs a fence.
  // CHECK-LABEL: @async_shared_store_uses_generic_proxy
  tt.func @async_shared_store_uses_generic_proxy(
      %value: tensor<32x32xf32, #store_blocked>,
      %desc: !tt.tensordesc<32x32xf32, #store_shared>) {
    %c0 = arith.constant 0 : i32
    %dst = ttg.local_alloc {allocation.offset = 0 : i32}
        : () -> !ttg.memdesc<32x32xf32, #store_shared, #store_smem, mutable>
    %completion = ttg.local_alloc {allocation.offset = 4096 : i32}
        : () -> !ttg.memdesc<2xi64, #store_barrier, #store_smem, mutable>
    ttng.init_barrier %completion, 1
        : !ttg.memdesc<2xi64, #store_barrier, #store_smem, mutable>
    %true = arith.constant true
    ttng.barrier_expect %completion, 4096, %true
        : !ttg.memdesc<2xi64, #store_barrier, #store_smem, mutable>
    // CHECK: ttng.async_shared_store
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.wait_barrier
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global
    ttng.async_shared_store %value, %dst, %completion
        : tensor<32x32xf32, #store_blocked>
        -> !ttg.memdesc<32x32xf32, #store_shared, #store_smem, mutable>,
           !ttg.memdesc<2xi64, #store_barrier, #store_smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    ttng.wait_barrier %completion, %c0 deps %dst : !ttg.memdesc<2xi64, #store_barrier, #store_smem, mutable>, !ttg.memdesc<32x32xf32, #store_shared, #store_smem, mutable>
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %dst
        : !tt.tensordesc<32x32xf32, #store_shared>,
          !ttg.memdesc<32x32xf32, #store_shared, #store_smem, mutable>
    tt.return
  }

  // The worker observes completion of the default role's write before the join.
  // The issuer's earlier fence cannot order that completion against parent TMA.
  // CHECK-LABEL: @sibling_observes_store_before_join
  tt.func @sibling_observes_store_before_join(%value: tensor<32x32xf32, #store_blocked>, %desc: !tt.tensordesc<32x32xf32, #store_shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %dst = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #store_shared, #store_smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<2xi64, #store_barrier, #store_smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #store_barrier, #store_smem, mutable>
    ttng.barrier_expect %bar, 4096, %true : !ttg.memdesc<2xi64, #store_barrier, #store_smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 4112 : i32, allocation.size = 12 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      ttng.async_shared_store %value, %dst, %bar : tensor<32x32xf32, #store_blocked> -> !ttg.memdesc<32x32xf32, #store_shared, #store_smem, mutable>, !ttg.memdesc<2xi64, #store_barrier, #store_smem, mutable>
      ttng.fence_async_shared {bCluster = false}
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<2xi64, #store_barrier, #store_smem, mutable>) num_warps(4) {
      %phase = arith.constant 0 : i32
      ttng.wait_barrier %b, %phase : !ttg.memdesc<2xi64, #store_barrier, #store_smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2xi64, #store_barrier, #store_smem, mutable>) -> ()
    // CHECK: ttg.warp_return
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %dst : !tt.tensordesc<32x32xf32, #store_shared>, !ttg.memdesc<32x32xf32, #store_shared, #store_smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    ttng.inval_barrier %bar : !ttg.memdesc<2xi64, #store_barrier, #store_smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @nested_fence_passthrough
  // CHECK-NOT: ttng.fence_async_shared
  // CHECK: tt.return
  tt.func private @nested_fence_passthrough(
      %buffer: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>) {
    tt.return
  }

  tt.func private @fenced_proxy_after_nested_call(
      %buffer: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<8x32xi32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    ttng.fence_async_shared {bCluster = false}
    tt.call @nested_fence_passthrough(%buffer) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: nested_call_after_fence_does_not_expose_proxy
  tt.func public @nested_call_after_fence_does_not_expose_proxy(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    // CHECK: ttg.local_load
    // CHECK-NEXT: "test.keep"
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    // CHECK-NEXT: tt.call @fenced_proxy_after_nested_call
    tt.call @fenced_proxy_after_nested_call(%buffer, %desc, %bar) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @callee_proxy_read_argument
  tt.func private @callee_proxy_read_argument(
      %src: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: caller_write_before_callee_proxy_read
  tt.func public @caller_write_before_callee_proxy_read(
      %value: tensor<8x32xi32, #blocked>,
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    // CHECK: ttg.local_store
    ttg.local_store %value, %buffer : tensor<8x32xi32, #blocked> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    // One fence in the caller covers both invocations.
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: tt.call @callee_proxy_read_argument
    // CHECK-NEXT: tt.call @callee_proxy_read_argument
    tt.call @callee_proxy_read_argument(%buffer, %desc) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>) -> ()
    tt.call @callee_proxy_read_argument(%buffer, %desc) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>) -> ()
    tt.return
  }

  // CHECK-LABEL: @callee_proxy_write_then_fence
  tt.func private @callee_proxy_write_then_fence(
      %dst: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<8x32xi32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    tt.return
  }

  // A fence after an async access cannot order generic accesses that precede
  // the call with that async access.
  // CHECK-LABEL: caller_read_before_callee_proxy_then_fence
  tt.func public @caller_read_before_callee_proxy_then_fence(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @callee_proxy_write_then_fence
    tt.call @callee_proxy_write_then_fence(%buffer, %desc, %bar) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }

  tt.func private @callee_fence_then_proxy_write(
      %dst: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<8x32xi32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: callee_fence_covers_caller_generic_access
  tt.func public @callee_fence_covers_caller_generic_access(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    // CHECK: ttg.local_load
    // CHECK-NEXT: tt.call @callee_fence_then_proxy_write
    tt.call @callee_fence_then_proxy_write(%buffer, %desc, %bar) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    tt.return
  }

  tt.func private @callee_read_then_fence(
      %src: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>) {
    %value = ttg.local_load %src : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    ttng.fence_async_shared {bCluster = false}
    tt.return
  }

  // CHECK-LABEL: callee_trailing_fence_clears_generic_summary
  tt.func public @callee_trailing_fence_clears_generic_summary(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: tt.call @callee_read_then_fence
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    tt.call @callee_read_then_fence(%buffer) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @callee_fence_then_read
  tt.func private @callee_fence_then_read(
      %src: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>) {
    ttng.fence_async_shared {bCluster = false}
    %value = ttg.local_load %src : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    // CHECK: ttg.local_load
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.return
    tt.return
  }

  // CHECK-LABEL: generic_effect_after_callee_fence_is_exported
  tt.func public @generic_effect_after_callee_fence_is_exported(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: tt.call @callee_fence_then_read
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    tt.call @callee_fence_then_read(%buffer) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @callee_all_paths_fence_before_proxy(
      %choose: i1,
      %dst: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<8x32xi32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    scf.if %choose {
      ttng.fence_async_shared {bCluster = false}
      ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    } else {
      ttng.fence_async_shared {bCluster = false}
      ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    }
    tt.return
  }

  // CHECK-LABEL: all_callee_paths_fence_caller_generic_access
  tt.func public @all_callee_paths_fence_caller_generic_access(
      %choose: i1, %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    // CHECK: ttg.local_load
    // CHECK-NEXT: "test.keep"
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    // CHECK-NEXT: tt.call @callee_all_paths_fence_before_proxy
    tt.call @callee_all_paths_fence_before_proxy(%choose, %buffer, %desc, %bar) : (i1, !ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }

  // CHECK-LABEL: @callee_one_path_fences_before_proxy
  tt.func private @callee_one_path_fences_before_proxy(
      %choose: i1,
      %dst: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<8x32xi32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    scf.if %choose {
      ttng.fence_async_shared {bCluster = false}
      // CHECK: ttng.fence_async_shared {bCluster = false}
      // CHECK-NEXT: ttng.async_tma_copy_global_to_local
      ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    } else {
      // CHECK: ttng.fence_async_shared {bCluster = false}
      // CHECK-NEXT: ttng.async_tma_copy_global_to_local
      ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    }
    tt.return
  }

  // CHECK-LABEL: one_unfenced_callee_path_requires_fence
  tt.func public @one_unfenced_callee_path_requires_fence(
      %choose: i1, %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @callee_one_path_fences_before_proxy
    tt.call @callee_one_path_fences_before_proxy(%choose, %buffer, %desc, %bar) : (i1, !ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#local = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#replicated = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 0]]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[1, 0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttng.two-ctas" = true, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: cta_local_scratch_requires_remote_proxy_fence
  tt.func @cta_local_scratch_requires_remote_proxy_fence(
      %source: tensor<16x32xi32, #src>,
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.convert_layout
    %converted = ttg.convert_layout %source {allocation.offset = 0 : i32, allocation.size = 1024 : i32} : tensor<16x32xi32, #src> -> tensor<16x32xi32, #local>
    "test.keep"(%converted) : (tensor<16x32xi32, #local>) -> ()
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16x32xi32, #shared, #smem, mutable>
    %remote = ttg.memdesc_subslice %parent [8, 0] : !ttg.memdesc<16x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %remote, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    tt.return
  }

  // Slicing columns leaves each row shard in its issuing CTA. The completion
  // mbarrier can target the CTA pair independently of the TMA payload.
  // CHECK-LABEL: @tma_subslice_along_local_dimension
  tt.func @tma_subslice_along_local_dimension(%source: tensor<16x64xi32, #src>,
                                               %desc: !tt.tensordesc<16x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %parent = ttg.local_alloc %source {allocation.offset = 0 : i32} : (tensor<16x64xi32, #src>) -> !ttg.memdesc<16x64xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %tile = ttg.memdesc_subslice %parent [0, 32] : !ttg.memdesc<16x64xi32, #shared, #smem, mutable> -> !ttg.memdesc<16x32xi32, #shared, #smem, mutable, 16x64>
    // CHECK: ttg.memdesc_subslice
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %tile, %bar, %true : !tt.tensordesc<16x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<16x32xi32, #shared, #smem, mutable, 16x64>
    tt.return
  }

  // Scalar scratch is owned by CTA 0. The tensor layout broadcasts bit 0,
  // so its scratch is also owned by CTA 0, never CTA 1.
  // CHECK-LABEL: atomic_scratch_uses_cta_owners
  tt.func @atomic_scratch_uses_cta_owners(
      %ptr: !tt.ptr<i32>,
      %ptrs: tensor<8x32x!tt.ptr<i32>, #replicated>,
      %cmp: tensor<8x32xi32, #replicated>,
      %values: tensor<8x32xi32, #replicated>,
      %indices: tensor<8x32xi32, #replicated>,
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %timeout = arith.constant 1 : i64
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16x32xi32, #shared, #smem, mutable>
    %owner0 = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<16x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    %nonowner1 = ttg.memdesc_subslice %parent [8, 0] : !ttg.memdesc<16x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %atomic_parent = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<16x32xi32, #shared, #smem, mutable>
    %atomic_dst = ttg.memdesc_subslice %atomic_parent [0, 0] : !ttg.memdesc<16x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>

    %scalar = tt.atomic_cas relaxed, gpu, %ptr, %c0, %c0 {allocation.offset = 0 : i32, allocation.size = 4 : i32} : (!tt.ptr<i32>, i32, i32) -> i32
    "test.keep"(%scalar) : (i32) -> ()
    // CHECK: tt.atomic_cas
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %nonowner1, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %owner0, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>

    %matched = tt.atomic_poll relaxed, gpu, %ptr, %c0 timeout %timeout {allocation.offset = 0 : i32, allocation.size = 4 : i32} : !tt.ptr<i32>, i32 -> i1
    "test.keep"(%matched) : (i1) -> ()
    // CHECK: tt.atomic_poll
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %nonowner1, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %owner0, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>

    %tensor = tt.atomic_cas relaxed, gpu, %ptrs, %cmp, %values {allocation.offset = 0 : i32, allocation.size = 1024 : i32} : (tensor<8x32x!tt.ptr<i32>, #replicated>, tensor<8x32xi32, #replicated>, tensor<8x32xi32, #replicated>) -> tensor<8x32xi32, #replicated>
    "test.keep"(%tensor) : (tensor<8x32xi32, #replicated>) -> ()
    // CHECK: tt.atomic_cas
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %nonowner1, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %owner0, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>

    %old = ttg.local_atomic_scatter_rmw add, %atomic_dst[%indices], %values {allocation.offset = 0 : i32, allocation.size = 1024 : i32, axis = 0 : i32} : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>, tensor<8x32xi32, #replicated>, tensor<8x32xi32, #replicated>) -> tensor<8x32xi32, #replicated>
    "test.keep"(%old) : (tensor<8x32xi32, #replicated>) -> ()
    // CHECK: ttg.local_atomic_scatter_rmw
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %nonowner1, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %owner0, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @callee_proxy_write_with_local_frame
  // CHECK-NOT: ttng.fence_async_shared
  // CHECK: ttng.async_tma_copy_global_to_local
  tt.func private @callee_proxy_write_with_local_frame(
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %dst = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @forward_proxy_write_with_local_frame
  tt.func private @forward_proxy_write_with_local_frame(
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: tt.call @callee_proxy_write_with_local_frame
    tt.call @callee_proxy_write_with_local_frame(%desc, %bar) {allocation.offset = 2048 : i32} : (!tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }

  // CHECK-LABEL: @callee_proxy_write_with_local_frame_disjoint
  // CHECK-NOT: ttng.fence_async_shared
  // CHECK: tt.return
  tt.func private @callee_proxy_write_with_local_frame_disjoint(
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %dst = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: callee_local_frame_does_not_alias_caller_buffer
  tt.func public @callee_local_frame_does_not_alias_caller_buffer(%desc: !tt.tensordesc<64x64xf32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 49152 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @callee_proxy_write_with_local_frame_disjoint
    tt.call @callee_proxy_write_with_local_frame_disjoint(%desc, %bar) {allocation.offset = 16384 : i32} : (!tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }

  // CHECK-LABEL: callee_local_proxy_summary_uses_call_frame_offset
  tt.func public @callee_local_proxy_summary_uses_call_frame_offset(%desc: !tt.tensordesc<64x64xf32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 49152 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: tt.call @callee_proxy_write_with_local_frame
    tt.call @callee_proxy_write_with_local_frame(%desc, %bar) {allocation.offset = 16384 : i32} : (!tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }

  // CHECK-LABEL: nested_callee_local_proxy_summary_composes_frame_offsets
  tt.func public @nested_callee_local_proxy_summary_composes_frame_offsets(%desc: !tt.tensordesc<64x64xf32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 6144 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 49152 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @forward_proxy_write_with_local_frame
    tt.call @forward_proxy_write_with_local_frame(%desc, %bar) {allocation.offset = 4096 : i32} : (!tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @callee_generic_read_from_local_frame
  tt.func private @callee_generic_read_from_local_frame() {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    // CHECK: ttg.local_load
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.return
    tt.return
  }

  tt.func private @forward_generic_read_from_local_frame() {
    tt.call @callee_generic_read_from_local_frame() {allocation.offset = 2048 : i32} : () -> ()
    tt.return
  }

  // CHECK-LABEL: callee_local_summary_uses_call_frame_offset
  tt.func public @callee_local_summary_uses_call_frame_offset(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: tt.call @callee_generic_read_from_local_frame
    tt.call @callee_generic_read_from_local_frame() {allocation.offset = 4096 : i32} : () -> ()
    %buffer = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: nested_callee_local_summary_composes_frame_offsets
  tt.func public @nested_callee_local_summary_composes_frame_offsets(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: tt.call @forward_generic_read_from_local_frame
    tt.call @forward_generic_read_from_local_frame() {allocation.offset = 4096 : i32} : () -> ()
    %buffer = ttg.local_alloc {allocation.offset = 6144 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: unknown_descriptors_are_conservative
  tt.func public @unknown_descriptors_are_conservative(
      %read: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %write: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.local_load
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.async_tma_copy_global_to_local
    %value = ttg.local_load %read : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %write, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: missing_proxy_fence_memdesc_index_alias_single
  tt.func @missing_proxy_fence_memdesc_index_alias_single(%arg0: !tt.tensordesc<64x64xf32, #shared>, %arg1: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) {
    // Keep the first fence to clear dependencies from local_alloc.
    // CHECK: ttng.fence_async_shared
    // CHECK: ttg.local_load
    // CHECK-NEXT: "test.keep"
    // CHECK-NEXT: ttng.fence_async_shared
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<1x64x64xf32, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %2 = ttg.local_load %1 : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%2) : (tensor<64x64xf32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %1, %arg1, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: async_proxy_after_async_proxy
  tt.func @async_proxy_after_async_proxy(%arg0: !tt.tensordesc<64x64xf32, #shared>, %arg1: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) {
    // CHECK: ttng.async_tma_copy_global_to_local
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %arg1, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    %2 = ttg.local_alloc {allocation.offset = 32 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %2, %arg1, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: missing_proxy_fence_local_store_before_async_tma_copy_local_to_global
  tt.func @missing_proxy_fence_local_store_before_async_tma_copy_local_to_global(%arg0: !tt.tensordesc<128x256xf32, #shared>, %arg1: tensor<128x256xf32, #blocked>) {
    // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
    // CHECK-NEXT: ttg.local_store
    // CHECK-NEXT: ttng.fence_async_shared
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<128x256xf32, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 1 : i32}
    ttg.local_store %arg1, %0 : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #shared, #smem, mutable>
    ttng.async_tma_copy_local_to_global %arg0[%c0_i32, %c0_i32] %0 : !tt.tensordesc<128x256xf32, #shared>, !ttg.memdesc<128x256xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: missing_proxy_fence_local_store_before_tmem_copy
  tt.func @missing_proxy_fence_local_store_before_tmem_copy(%arg0: tensor<128x4xi8, #blocked>,
      %arg1: !ttg.memdesc<128x4xi8, #tmem, #ttng.tensor_memory, mutable>) {
    // CHECK: ttg.local_store
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.tmem_copy
    %0 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<128x4xi8, #shared1, #smem, mutable>
    ttg.local_store %arg0, %0 : tensor<128x4xi8, #blocked> -> !ttg.memdesc<128x4xi8, #shared1, #smem, mutable>
    ttng.tmem_copy %0, %arg1 : !ttg.memdesc<128x4xi8, #shared1, #smem, mutable>, !ttg.memdesc<128x4xi8, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked_a = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#mma_a = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#mma_b = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: missing_proxy_fence_async_copy_before_mma
  tt.func @missing_proxy_fence_async_copy_before_mma(%arg0: tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked_a>,
      %arg1: tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked_b>,
      %arg2: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>,
      %arg3: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    %false = arith.constant false
    %true = arith.constant true
    // CHECK: ttg.async_copy_global_to_local
    // CHECK-NEXT: ttg.async_copy_global_to_local
    // CHECK-NEXT: ttg.async_commit_group
    // CHECK-NEXT: ttg.async_wait
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.tc_gen5_mma
    %0 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 32 : i32} : () -> !ttg.memdesc<32x64xf8E4M3FN, #mma_b, #smem, mutable>
    %2 = ttg.async_copy_global_to_local %arg0, %0 : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked_a> -> !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>
    %3 = ttg.async_copy_global_to_local %arg1, %1 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked_b> -> !ttg.memdesc<32x64xf8E4M3FN, #mma_b, #smem, mutable>
    %4 = ttg.async_commit_group tokens %2, %3
    %5 = ttg.async_wait %4 {num = 0 : i32}
    ttng.tc_gen5_mma %0, %1, %arg2, %false, %true, %arg3[%true] {is_async} : !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>, !ttg.memdesc<32x64xf8E4M3FN, #mma_b, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // A copy issued by the caller becomes visible inside the noinline callee.
  // CHECK-LABEL: @wait_then_tma
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
  // CHECK-NEXT: ttng.async_tma_copy_local_to_global
  tt.func private @wait_then_tma(%src: !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>, %desc: !tt.tensordesc<64x32xf8E4M3FN, #mma_a>) {
    %c0 = arith.constant 0 : i32
    %wait = ttg.async_wait {num = 0 : i32}
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src : !tt.tensordesc<64x32xf8E4M3FN, #mma_a>, !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @copy_then_call
  // CHECK: ttg.async_commit_group
  // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
  // CHECK-NEXT: tt.call @wait_then_tma
  tt.func @copy_then_call(%ptrs: tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked_a>, %desc: !tt.tensordesc<64x32xf8E4M3FN, #mma_a>) {
    %src = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>
    %copy = ttg.async_copy_global_to_local %ptrs, %src : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked_a> -> !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>
    %commit = ttg.async_commit_group
    ttng.fence_async_shared {bCluster = false}
    tt.call @wait_then_tma(%src, %desc) : (!ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>, !tt.tensordesc<64x32xf8E4M3FN, #mma_a>) -> ()
    tt.return
  }

  // A copy issued by a noinline callee can complete in its caller.
  // CHECK-LABEL: @issue_copy
  // CHECK: ttg.async_commit_group
  // CHECK-NEXT: tt.return
  tt.func private @issue_copy(%ptrs: tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked_a>, %src: !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>) {
    %copy = ttg.async_copy_global_to_local %ptrs, %src : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked_a> -> !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>
    %commit = ttg.async_commit_group
    tt.return
  }

  // CHECK-LABEL: @call_then_wait
  // CHECK: tt.call @issue_copy
  // CHECK-NEXT: ttg.async_wait
  // CHECK-NEXT: ttng.async_tma_copy_local_to_global
  // CHECK-NEXT: ttng.async_tma_store_wait
  // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
  // CHECK-NEXT: ttng.async_tma_copy_local_to_global
  tt.func @call_then_wait(%value: tensor<64x32xf8E4M3FN, #blocked_a>, %ptrs: tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked_a>, %desc: !tt.tensordesc<64x32xf8E4M3FN, #mma_a>) {
    %c0 = arith.constant 0 : i32
    %src = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>
    %other = ttg.local_alloc %value {allocation.offset = 4096 : i32} : (tensor<64x32xf8E4M3FN, #blocked_a>) -> !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    tt.call @issue_copy(%ptrs, %src) : (tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked_a>, !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>) -> ()
    %wait = ttg.async_wait {num = 0 : i32}
    // Only %src was copied. The fence before the call still covers %other.
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %other : !tt.tensordesc<64x32xf8E4M3FN, #mma_a>, !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src : !tt.tensordesc<64x32xf8E4M3FN, #mma_a>, !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}

// -----

#shared_clc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: clc_try_cancel_single_cta_fence
  tt.func @clc_try_cancel_single_cta_fence() {
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<1xi64, #shared_clc, #smem, mutable>
    %response = ttng.clc_load_result %result : !ttg.memdesc<2xi64, #shared_clc, #smem, mutable> -> i128
    "test.keep"(%response) : (i128) -> ()
    // CHECK: ttng.clc_load_result
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.clc_try_cancel
    ttng.clc_try_cancel %result, %barrier :
      !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>,
      !ttg.memdesc<1xi64, #shared_clc, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: clc_try_cancel_single_cta_after_cluster_fence
  tt.func @clc_try_cancel_single_cta_after_cluster_fence() {
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<1xi64, #shared_clc, #smem, mutable>
    %response = ttng.clc_load_result %result : !ttg.memdesc<2xi64, #shared_clc, #smem, mutable> -> i128
    "test.keep"(%response) : (i128) -> ()
    // CHECK: ttng.clc_load_result
    // CHECK: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.clc_try_cancel
    ttng.fence_async_shared {bCluster = true}
    ttng.clc_try_cancel %result, %barrier :
      !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>,
      !ttg.memdesc<1xi64, #shared_clc, #smem, mutable>
    tt.return
  }
}

// -----

#shared_clc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: clc_try_cancel_multi_cta_fence
  tt.func @clc_try_cancel_multi_cta_fence() {
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %response = ttng.clc_load_result %result : !ttg.memdesc<2xi64, #shared_clc, #smem, mutable> -> i128
    "test.keep"(%response) : (i128) -> ()
    // CHECK: ttng.clc_load_result
    // CHECK: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.clc_try_cancel
    ttng.clc_try_cancel %result, %barrier :
      !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>,
      !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: clc_try_cancel_multi_cta_after_cta_fence
  tt.func @clc_try_cancel_multi_cta_after_cta_fence() {
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %response = ttng.clc_load_result %result : !ttg.memdesc<2xi64, #shared_clc, #smem, mutable> -> i128
    "test.keep"(%response) : (i128) -> ()
    // CHECK: ttng.clc_load_result
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.clc_try_cancel
    ttng.fence_async_shared {bCluster = false}
    ttng.clc_try_cancel %result, %barrier :
      !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>,
      !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: clc_try_cancel_multi_cta_after_cluster_fence
  tt.func @clc_try_cancel_multi_cta_after_cluster_fence() {
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %response = ttng.clc_load_result %result : !ttg.memdesc<2xi64, #shared_clc, #smem, mutable> -> i128
    "test.keep"(%response) : (i128) -> ()
    // CHECK: ttng.clc_load_result
    // CHECK: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.clc_try_cancel
    ttng.fence_async_shared {bCluster = true}
    ttng.clc_try_cancel %result, %barrier :
      !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>,
      !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: multicast_tma_after_cluster_barrier_and_cta_fence
  tt.func @multicast_tma_after_cluster_barrier_and_cta_fence(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %result : !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable> -> tensor<64x128xf16, #blocked>
    "test.keep"(%value) : (tensor<64x128xf16, #blocked>) -> ()
    // CHECK: ttg.local_load
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.cluster_barrier
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %result, %barrier, %true {multicast} :
      !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: non_multicast_tma_after_cluster_barrier_and_cta_fence
  tt.func @non_multicast_tma_after_cluster_barrier_and_cta_fence(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %result : !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable> -> tensor<64x128xf16, #blocked>
    "test.keep"(%value) : (tensor<64x128xf16, #blocked>) -> ()
    // CHECK: ttg.local_load
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NOT: ttng.fence_async_shared {bCluster = true}
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.cluster_barrier
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %result, %barrier, %true :
      !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[0, 0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0], CGALayout = [[0, 0]]}>
#offsets = #ttg.slice<{dim = 0, parent = #blocked}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: multicast_tma_gather_after_cluster_barrier_and_cta_fence
  tt.func @multicast_tma_gather_after_cluster_barrier_and_cta_fence(%desc: !tt.tensordesc<1x32xf32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %offsets = arith.constant dense<0> : tensor<32xi32, #offsets>
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %result : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    "test.keep"(%value) : (tensor<32x32xf32, #blocked>) -> ()
    // CHECK: ttg.local_load
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.async_tma_gather
    ttng.cluster_barrier
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_gather %desc[%offsets, %c0] %result, %barrier, %true {multicast} :
      !tt.tensordesc<1x32xf32, #shared>, tensor<32xi32, #offsets>, i32, !ttg.memdesc<1xi64, #barrier, #smem, mutable>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, i1
    tt.return
  }
}

// -----

#sharedA = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#sharedB = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 2], order = [0, 1], CGALayout = [[1, 0]]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, CGALayout = [[1, 0]], twoCTAs = true>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 8 : i32, "ttng.two-ctas" = true, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: two_cta_mma_after_cluster_barrier_and_cta_fence
  tt.func @two_cta_mma_after_cluster_barrier_and_cta_fence(%value: tensor<256x32xf16, #blocked>) {
    %true = arith.constant true
    %a = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<256x32xf16, #sharedA, #smem, mutable>
    %b = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<32x128xf16, #sharedB, #smem, mutable>
    %acc = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 24576 : i32} : () -> !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    ttg.local_store %value, %a : tensor<256x32xf16, #blocked> -> !ttg.memdesc<256x32xf16, #sharedA, #smem, mutable>
    // CHECK: ttg.local_store
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.tc_gen5_mma
    ttng.cluster_barrier
    ttng.fence_async_shared {bCluster = false}
    ttng.tc_gen5_mma %a, %b, %acc, %true, %true, %barrier[%true] {is_async, two_ctas} :
      !ttg.memdesc<256x32xf16, #sharedA, #smem, mutable>, !ttg.memdesc<32x128xf16, #sharedB, #smem, mutable>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[0, 0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, CGALayout = [[0, 0]]>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttng.two-ctas" = true, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: two_cta_tmem_copy_after_cluster_barrier_and_cta_fence
  tt.func @two_cta_tmem_copy_after_cluster_barrier_and_cta_fence(%value: tensor<128x128xf32, #blocked>) {
    %src = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %dst = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttg.local_store %value, %src : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    // CHECK: ttg.local_store
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.tmem_copy
    ttng.cluster_barrier
    ttng.fence_async_shared {bCluster = false}
    ttng.tmem_copy %src, %dst : !ttg.memdesc<128x128xf32, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @invalidated_barrier_storage_requires_proxy_fence
  tt.func @invalidated_barrier_storage_requires_proxy_fence(
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %completion: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %old_barrier = ttg.local_alloc {allocation.offset = 0 : i32}
        : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %old_barrier, 1
        : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.inval_barrier %old_barrier
        : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.local_dealloc %old_barrier
        : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %payload = ttg.local_alloc {allocation.offset = 0 : i32}
        : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %payload,
        %completion, %true : !tt.tensordesc<64x64xf32, #shared>,
        !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @disjoint_invalidated_barrier_needs_no_proxy_fence
  tt.func @disjoint_invalidated_barrier_needs_no_proxy_fence(
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %completion: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %old_barrier = ttg.local_alloc {allocation.offset = 16384 : i32}
        : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %payload = ttg.local_alloc {allocation.offset = 0 : i32}
        : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.init_barrier %old_barrier, 1
        : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttng.inval_barrier
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.inval_barrier %old_barrier
        : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %payload,
        %completion, %true : !tt.tensordesc<64x64xf32, #shared>,
        !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @barrier_arrival_is_not_a_generic_proxy_access
  tt.func @barrier_arrival_is_not_a_generic_proxy_access(
      %desc: !tt.tensordesc<64x64xf32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %completion = ttg.local_alloc {allocation.offset = 16384 : i32}
        : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %payload = ttg.local_alloc {allocation.offset = 0 : i32}
        : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.init_barrier %completion, 1
        : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttng.arrive_barrier
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.arrive_barrier %completion, 1, %true
        : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %payload,
        %completion, %true : !tt.tensordesc<64x64xf32, #shared>,
        !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: disjoint_pipeline_stages_do_not_require_proxy_fence
  tt.func @disjoint_pipeline_stages_do_not_require_proxy_fence(%desc: !tt.tensordesc<64x64xf32, #shared>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %second = ttg.memdesc_index %parent[%c1] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    // CHECK: ttg.local_load
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    %value = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %second, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: selected_pipeline_stage_may_alias
  tt.func @selected_pipeline_stage_may_alias(%desc: !tt.tensordesc<64x64xf32, #shared>, %choose: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %second = ttg.memdesc_index %parent[%c1] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %selected = arith.select %choose, %first, %second : !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    // CHECK: ttg.local_load
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    %value = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %selected, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: aliasing_callee_arguments_require_proxy_fence
  tt.func private @aliasing_callee_arguments_require_proxy_fence(
      %read: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %write: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.local_load
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.async_tma_copy_global_to_local
    %value = ttg.local_load %read : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %write, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  tt.func public @call_aliasing_callee_arguments(%desc: !tt.tensordesc<64x64xf32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.call @aliasing_callee_arguments_require_proxy_fence(%buffer, %buffer, %desc, %bar) : (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: disjoint_callee_contexts_join_conservatively
  tt.func private @disjoint_callee_contexts_join_conservatively(
      %read: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %write: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.local_load
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.async_tma_copy_global_to_local
    %value = ttg.local_load %read : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %write, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  tt.func public @call_disjoint_callee_contexts(%desc: !tt.tensordesc<64x64xf32, #shared>) {
    %first = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %second = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.call @disjoint_callee_contexts_join_conservatively(%first, %second, %desc, %bar) : (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.call @disjoint_callee_contexts_join_conservatively(%second, %first, %desc, %bar) : (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked_src = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: scratch_reuse_requires_proxy_fence
  tt.func @scratch_reuse_requires_proxy_fence(%source: tensor<128x32xf16, #blocked_src>, %desc: !tt.tensordesc<64x64xf32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.convert_layout
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    %converted = ttg.convert_layout %source {allocation.offset = 0 : i32, allocation.size = 2048 : i32} : tensor<128x32xf16, #blocked_src> -> tensor<128x32xf16, #dot>
    "test.keep"(%converted) : (tensor<128x32xf16, #dot>) -> ()
    %dst = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: disjoint_scratch_does_not_require_proxy_fence
  tt.func @disjoint_scratch_does_not_require_proxy_fence(%desc: !tt.tensordesc<64x64xf32, #shared>, %ptr: !tt.ptr<i8>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %dst = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 20480 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttng.tensormap_create
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.tensormap_create %ptr, %ptr, [%c0], [%c0], [], [%c0] {allocation.offset = 0 : i32, allocation.size = 128 : i32, elem_type = 3 : i32, fill_mode = 1 : i32, interleave_layout = 0 : i32, swizzle_mode = 2 : i32} : (!tt.ptr<i8>, !tt.ptr<i8>, i32, i32, i32) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[1, 0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#shared_replicated = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[0, 0]]}>
#replicated = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: same_bytes_in_different_ctas_do_not_alias
  tt.func @same_bytes_in_different_ctas_do_not_alias(%desc: !tt.tensordesc<8x32xi32, #shared>, %choose: i1) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %local = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<16x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    %remote = ttg.memdesc_subslice %parent [8, 0] : !ttg.memdesc<16x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    %selected = arith.select %choose, %remote, %remote : !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    // CHECK: ttg.local_load
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    %value = ttg.local_load %local : !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %selected, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    tt.return
  }

  // A replicated allocation includes CTA 1's storage, unlike the fixed
  // CTA 0 view above. Reinterpretation preserves its per-CTA allocation size.
  // CHECK-LABEL: replicated_shared_aliases_remote_cta
  tt.func @replicated_shared_aliases_remote_cta(%desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %replicated = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared_replicated, #smem, mutable>
    %parent = ttg.memdesc_reinterpret %replicated : !ttg.memdesc<8x32xi32, #shared_replicated, #smem, mutable> -> !ttg.memdesc<16x32xi32, #shared, #smem, mutable>
    %remote = ttg.memdesc_subslice %parent [8, 0] : !ttg.memdesc<16x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttg.local_load
    %value = ttg.local_load %replicated : !ttg.memdesc<8x32xi32, #shared_replicated, #smem, mutable> -> tensor<8x32xi32, #replicated>
    "test.keep"(%value) : (tensor<8x32xi32, #replicated>) -> ()
    // CHECK: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %remote, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @return_aliasing_stage(
      %parent: !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>)
      -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable> {
    %c0 = arith.constant 0 : i32
    %stage = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return %stage : !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
  }

  // CHECK-LABEL: returned_descriptor_preserves_physical_region
  tt.func public @returned_descriptor_preserves_physical_region(%desc: !tt.tensordesc<64x64xf32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    // CHECK: ttg.local_load
    %value = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    %returned = tt.call @return_aliasing_stage(%parent) : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %returned, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: nested_proxy_leaf
  tt.func private @nested_proxy_leaf(
      %write: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %write, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  tt.func private @nested_proxy_middle(
      %write: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    tt.call @nested_proxy_leaf(%write, %desc, %bar) : (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }

  // CHECK-LABEL: nested_callee_proxy_access_is_visible
  tt.func public @nested_callee_proxy_access_is_visible(%desc: !tt.tensordesc<64x64xf32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @nested_proxy_middle
    tt.call @nested_proxy_middle(%buffer, %desc, %bar) : (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: control_flow_join_preserves_proxy_hazards
  tt.func public @control_flow_join_preserves_proxy_hazards(%desc: !tt.tensordesc<64x64xf32, #shared>, %choose: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %second = ttg.memdesc_index %parent[%c1] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    cf.cond_br %choose, ^left, ^right
  ^left:
    %left = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%left) : (tensor<64x64xf32, #blocked>) -> ()
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: cf.br ^[[JOIN:bb[0-9]+]]
    cf.br ^merge
  ^right:
    %right = ttg.local_load %second : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%right) : (tensor<64x64xf32, #blocked>) -> ()
    // Only this arm reads the stage overwritten at the join.
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: cf.br ^[[JOIN]]
    cf.br ^merge
  ^merge:
    // CHECK: ^[[JOIN]]:
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %second, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: disjoint_control_flow_join_avoids_proxy_fence
  tt.func public @disjoint_control_flow_join_avoids_proxy_fence(%desc: !tt.tensordesc<64x64xf32, #shared>, %choose: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %second = ttg.memdesc_index %parent[%c1] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    cf.cond_br %choose, ^left, ^right
  ^left:
    %left = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%left) : (tensor<64x64xf32, #blocked>) -> ()
    cf.br ^merge
  ^right:
    %right = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%right) : (tensor<64x64xf32, #blocked>) -> ()
    cf.br ^merge
  ^merge:
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %second, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked_src = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The NVIDIA allocator reserves 2,048 bytes for this conversion. The generic
  // allocation analysis incorrectly models it as 8,192 bytes.
  // CHECK-LABEL: exact_nvidia_scratch_size_avoids_proxy_fence
  tt.func @exact_nvidia_scratch_size_avoids_proxy_fence(
      %source: tensor<128x32xf16, #blocked_src>,
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.convert_layout
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    %converted = ttg.convert_layout %source {allocation.offset = 0 : i32, allocation.size = 2048 : i32} : tensor<128x32xf16, #blocked_src> -> tensor<128x32xf16, #dot>
    "test.keep"(%converted) : (tensor<128x32xf16, #dot>) -> ()
    %dst = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: scratch_write_before_async_proxy_read_requires_fence
  tt.func @scratch_write_before_async_proxy_read_requires_fence(
      %source: tensor<128x32xf16, #blocked_src>,
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    // CHECK: ttg.convert_layout
    %converted = ttg.convert_layout %source {allocation.offset = 0 : i32, allocation.size = 2048 : i32} : tensor<128x32xf16, #blocked_src> -> tensor<128x32xf16, #dot>
    "test.keep"(%converted) : (tensor<128x32xf16, #dot>) -> ()
    %src = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: consan_warp_specialization_captures_require_proxy_fence
  tt.func @consan_warp_specialization_captures_require_proxy_fence(
      %capture: i32, %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.warp_specialize
    ttg.warp_specialize(%capture) attributes {allocation.offset = 0 : i32, allocation.size = 260 : i32, "consan.extra_capture_bytes" = 256 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%arg: i32) num_warps(4) {
      ttg.warp_return
    } : (i32) -> ()
    %dst = ttg.local_alloc {allocation.offset = 128 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: initialized_allocation_requires_proxy_fence
  tt.func @initialized_allocation_requires_proxy_fence(
      %value: tensor<8x32xi32, #blocked>,
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.local_alloc
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    %dst = ttg.local_alloc %value {allocation.offset = 0 : i32} : (tensor<8x32xi32, #blocked>) -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: atomic_read_write_requires_proxy_fence
  tt.func @atomic_read_write_requires_proxy_fence(
      %values: tensor<8x32xi32, #blocked>,
      %indices: tensor<8x32xi32, #blocked>,
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %dst = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttg.local_atomic_scatter_rmw
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    %old = ttg.local_atomic_scatter_rmw add, %dst[%indices], %values {allocation.offset = 4096 : i32, allocation.size = 1024 : i32, axis = 0 : i32} : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, tensor<8x32xi32, #blocked>, tensor<8x32xi32, #blocked>) -> tensor<8x32xi32, #blocked>
    "test.keep"(%old) : (tensor<8x32xi32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#inner = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 1, partitionDim = 0, partitionLayout = #inner}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @callee_generic_read_from_second_partition
  // CHECK: ttg.local_load
  // CHECK-NEXT: tt.return
  tt.func private @callee_generic_read_from_second_partition() -> tensor<2xi64, #blocked> {
    %parent = ttg.local_alloc {allocation.offset = [0 : i32, 1024 : i32]} : () -> !ttg.memdesc<4xi64, #partitioned, #smem, mutable>
    %second = ttg.memdesc_subslice %parent [2] : !ttg.memdesc<4xi64, #partitioned, #smem, mutable> -> !ttg.memdesc<2xi64, #partitioned, #smem, mutable, 4>
    %result = ttg.local_load %second : !ttg.memdesc<2xi64, #partitioned, #smem, mutable, 4> -> tensor<2xi64, #blocked>
    tt.return %result : tensor<2xi64, #blocked>
  }

  // CHECK-LABEL: @callee_generic_read_from_second_partition_disjoint
  // CHECK-NOT: ttng.fence_async_shared
  // CHECK: tt.return
  tt.func private @callee_generic_read_from_second_partition_disjoint() -> tensor<2xi64, #blocked> {
    %parent = ttg.local_alloc {allocation.offset = [0 : i32, 1024 : i32]} : () -> !ttg.memdesc<4xi64, #partitioned, #smem, mutable>
    %second = ttg.memdesc_subslice %parent [2] : !ttg.memdesc<4xi64, #partitioned, #smem, mutable> -> !ttg.memdesc<2xi64, #partitioned, #smem, mutable, 4>
    %result = ttg.local_load %second : !ttg.memdesc<2xi64, #partitioned, #smem, mutable, 4> -> tensor<2xi64, #blocked>
    tt.return %result : tensor<2xi64, #blocked>
  }

  // CHECK-LABEL: callee_partition_frame_does_not_alias_caller_buffer
  tt.func public @callee_partition_frame_does_not_alias_caller_buffer() -> tensor<2xi64, #blocked> {
    // CHECK-NOT: ttng.fence_async_shared
    %result = tt.call @callee_generic_read_from_second_partition_disjoint() {allocation.offset = 3072 : i32} : () -> tensor<2xi64, #blocked>
    %buffer = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<2xi64, #inner, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<1xi64, #inner, #smem, mutable>
    // CHECK: ttng.clc_try_cancel
    // CHECK-NEXT: tt.return
    ttng.clc_try_cancel %buffer, %bar : !ttg.memdesc<2xi64, #inner, #smem, mutable>, !ttg.memdesc<1xi64, #inner, #smem, mutable>
    tt.return %result : tensor<2xi64, #blocked>
  }

  // CHECK-LABEL: callee_partition_summary_translates_selected_base
  tt.func public @callee_partition_summary_translates_selected_base() -> tensor<2xi64, #blocked> {
    // CHECK: tt.call @callee_generic_read_from_second_partition
    %result = tt.call @callee_generic_read_from_second_partition() {allocation.offset = 3072 : i32} : () -> tensor<2xi64, #blocked>
    // The second partition starts at 3072 + 1024 = 4096 in the caller.
    %buffer = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<2xi64, #inner, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #inner, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.clc_try_cancel
    ttng.clc_try_cancel %buffer, %bar : !ttg.memdesc<2xi64, #inner, #smem, mutable>, !ttg.memdesc<1xi64, #inner, #smem, mutable>
    tt.return %result : tensor<2xi64, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: structured_control_flow_preserves_aliasing_views
  tt.func @structured_control_flow_preserves_aliasing_views(
      %desc: !tt.tensordesc<64x64xf32, #shared>, %choose: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %second = ttg.memdesc_index %parent[%c1] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %selected = scf.if %choose -> (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>) {
      scf.yield %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    } else {
      scf.yield %second : !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    }
    // CHECK: ttg.local_load
    %value = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %selected, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: loop_carried_views_remain_conservative
  tt.func @loop_carried_views_remain_conservative(
      %desc: !tt.tensordesc<64x64xf32, #shared>, %limit: index) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %second = ttg.memdesc_index %parent[%c1] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %selected = scf.for %iv = %i0 to %limit step %i1 iter_args(%view = %first) -> (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>) {
      scf.yield %second : !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    }
    // CHECK: ttg.local_load
    %value = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %selected, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[0, 0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: callee_multicast_proxy_write
  tt.func private @callee_multicast_proxy_write(
      %write: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<8x32xi32, #shared>,
      %bar: !ttg.memdesc<2xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %write, %bar, %true {multicast} : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<2xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: callee_multicast_requires_cluster_fence
  tt.func public @callee_multicast_requires_cluster_fence(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    // CHECK: ttng.fence_async_shared {bCluster = false}
    ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: tt.call @callee_multicast_proxy_write
    tt.call @callee_multicast_proxy_write(%buffer, %desc, %bar) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<2xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: shared_callee_contexts_refresh_after_fence_insertion
  tt.func private @shared_callee_contexts_refresh_after_fence_insertion(
      %read: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>,
      %write: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<8x32xi32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %value = ttg.local_load %read : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %write, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: caller_reuses_fenced_callee_without_extra_fences
  tt.func public @caller_reuses_fenced_callee_without_extra_fences(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x8x32xi32, #shared, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x8x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %second = ttg.memdesc_index %parent[%c1] : !ttg.memdesc<2x8x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @shared_callee_contexts_refresh_after_fence_insertion
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @shared_callee_contexts_refresh_after_fence_insertion
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @shared_callee_contexts_refresh_after_fence_insertion
    tt.call @shared_callee_contexts_refresh_after_fence_insertion(%first, %second, %desc, %bar) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.call @shared_callee_contexts_refresh_after_fence_insertion(%second, %first, %desc, %bar) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.call @shared_callee_contexts_refresh_after_fence_insertion(%first, %second, %desc, %bar) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }

  // CHECK-LABEL: second_kernel_context_requires_shared_callee_fence
  tt.func public @second_kernel_context_requires_shared_callee_fence(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %first = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @shared_callee_contexts_refresh_after_fence_insertion
    tt.call @shared_callee_contexts_refresh_after_fence_insertion(%first, %first, %desc, %bar) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[0, 0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // A single cluster fence covers both the local and the multicast TMA access.
  // CHECK-LABEL: cluster_fence_covers_local_then_multicast
  tt.func @cluster_fence_covers_local_then_multicast(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 1024 : i32} : () -> !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    // CHECK-NOT: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<2xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true {multicast} : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<2xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: cluster_fence_covers_later_cta_access
  tt.func @cluster_fence_covers_later_cta_access(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 1024 : i32} : () -> !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    // CHECK: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true {multicast} : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<2xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<2xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: joined_cta_fences_preserve_cluster_hazard
  tt.func @joined_cta_fences_preserve_cluster_hazard(
      %desc: !tt.tensordesc<8x32xi32, #shared>, %choose: i1) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 1024 : i32} : () -> !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    cf.cond_br %choose, ^cta, ^cluster
  ^cta:
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: cf.br
    ttng.fence_async_shared {bCluster = false}
    cf.br ^merge
  ^cluster:
    ttng.fence_async_shared {bCluster = true}
    cf.br ^merge
  ^merge:
    // CHECK: ^bb3:
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true {multicast} : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<2xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @callee_mixed_proxy_scopes
  tt.func private @callee_mixed_proxy_scopes(
      %buffer: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<8x32xi32, #shared>,
      %bar: !ttg.memdesc<2xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<2xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true {multicast} : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<2xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: mixed_callee_scopes_require_only_cluster_fence
  tt.func public @mixed_callee_scopes_require_only_cluster_fence(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 1024 : i32} : () -> !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @callee_mixed_proxy_scopes
    tt.call @callee_mixed_proxy_scopes(%buffer, %desc, %bar) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<2xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The TMA page reuses worker scratch but does not overlap the captures.
  // CHECK-LABEL: @worker_scratch_reuse_requires_proxy_fence
  tt.func @worker_scratch_reuse_requires_proxy_fence(%input: !tt.ptr<i32>, %desc: !tt.tensordesc<64xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    ttg.warp_specialize(%input) attributes {allocation.offset = 0 : i32, allocation.size = 8 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%in: !tt.ptr<i32>) num_warps(4) {
      %ptrs = tt.splat %in : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #src>
      %value = tt.load %ptrs : tensor<128x!tt.ptr<i32>, #src>
      %converted = ttg.convert_layout %value {allocation.offset = 0 : i32, allocation.size = 512 : i32} : tensor<128xi32, #src> -> tensor<128xi32, #dst>
      tt.print "converted" {hex = false, isSigned = array<i32: 1>} : %converted : tensor<128xi32, #dst>
      // CHECK: ttg.warp_return
      ttg.warp_return
    } : (!tt.ptr<i32>) -> ()
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %page = ttg.memdesc_index %buffer[%c1] : !ttg.memdesc<2x64xi32, #shared, #smem, mutable> -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 512 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.barrier_expect %bar, 256, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0] %page, %bar, %true : !tt.tensordesc<64xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %c0, %true deps %page : !ttg.memdesc<1xi64, #barrier, #smem, mutable>, !ttg.memdesc<64xi32, #shared, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.return
  }

  // One fence before the launch covers both workers reading the parent's tile.
  // CHECK-LABEL: @parent_store_before_worker_tma
  tt.func @parent_store_before_worker_tma(%value: tensor<64xi32, #src>, %desc: !tt.tensordesc<64xi32, #shared>) {
    %buffer = ttg.local_alloc %value {allocation.offset = 0 : i32} : (tensor<64xi32, #src>) -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttg.warp_specialize
    ttg.warp_specialize(%buffer, %desc) attributes {allocation.offset = 256 : i32, allocation.size = 20 : i32, warpGroupStartIds = array<i32: 4, 6>}
    default {
      ttg.warp_yield
    }
    partition0(%src: !ttg.memdesc<64xi32, #shared, #smem, mutable>, %out: !tt.tensordesc<64xi32, #shared>) num_warps(2) {
      %c0 = arith.constant 0 : i32
      ttng.async_tma_copy_local_to_global %out[%c0] %src : !tt.tensordesc<64xi32, #shared>, !ttg.memdesc<64xi32, #shared, #smem, mutable>
      ttng.async_tma_store_wait {pendings = 0 : i32}
      ttg.warp_return
    }
    partition1(%src: !ttg.memdesc<64xi32, #shared, #smem, mutable>, %out: !tt.tensordesc<64xi32, #shared>) num_warps(2) {
      %c64 = arith.constant 64 : i32
      ttng.async_tma_copy_local_to_global %out[%c64] %src : !tt.tensordesc<64xi32, #shared>, !ttg.memdesc<64xi32, #shared, #smem, mutable>
      ttng.async_tma_store_wait {pendings = 0 : i32}
      ttg.warp_return
    } : (!ttg.memdesc<64xi32, #shared, #smem, mutable>, !tt.tensordesc<64xi32, #shared>) -> ()
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.return
    tt.return
  }

  // Both roles call the same helper on disjoint shared frames and output tiles.
  // CHECK-LABEL: @store_role_tile
  // CHECK: ttg.local_alloc
  // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
  // CHECK-NEXT: ttng.async_tma_copy_local_to_global
  tt.func private @store_role_tile(%out: !tt.tensordesc<64xi32, #shared>, %offset: i32) attributes {noinline = true} {
    %value = arith.constant dense<1> : tensor<64xi32, #src>
    %tile = ttg.local_alloc %value {allocation.offset = 0 : i32} : (tensor<64xi32, #src>) -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    ttng.async_tma_copy_local_to_global %out[%offset] %tile : !tt.tensordesc<64xi32, #shared>, !ttg.memdesc<64xi32, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @two_roles_share_fenced_callee
  tt.func @two_roles_share_fenced_callee(%desc: !tt.tensordesc<64xi32, #shared>) {
    ttg.warp_specialize(%desc) attributes {allocation.offset = 512 : i32, allocation.size = 8 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %c0 = arith.constant 0 : i32
      tt.call @store_role_tile(%desc, %c0) {allocation.offset = 0 : i32} : (!tt.tensordesc<64xi32, #shared>, i32) -> ()
      ttg.warp_yield
    }
    partition0(%out: !tt.tensordesc<64xi32, #shared>) num_warps(4) {
      %c64 = arith.constant 64 : i32
      tt.call @store_role_tile(%out, %c64) {allocation.offset = 256 : i32} : (!tt.tensordesc<64xi32, #shared>, i32) -> ()
      ttg.warp_return
    } : (!tt.tensordesc<64xi32, #shared>) -> ()
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.return
    tt.return
  }

  // The parent fences the initialized tile before the worker acquires it.
  // CHECK-LABEL: @parent_fence_before_worker_tma
  tt.func @parent_fence_before_worker_tma(%value: tensor<64xi32, #src>, %desc: !tt.tensordesc<64xi32, #shared>) {
    %buffer = ttg.local_alloc %value {allocation.offset = 0 : i32} : (tensor<64xi32, #src>) -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttg.warp_specialize
    ttg.warp_specialize(%buffer, %desc) attributes {allocation.offset = 256 : i32, allocation.size = 20 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%src: !ttg.memdesc<64xi32, #shared, #smem, mutable>, %out: !tt.tensordesc<64xi32, #shared>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      // CHECK-NOT: ttng.fence_async_shared
      // CHECK: ttng.async_tma_copy_local_to_global
      ttng.async_tma_copy_local_to_global %out[%c0] %src : !tt.tensordesc<64xi32, #shared>, !ttg.memdesc<64xi32, #shared, #smem, mutable>
      ttng.async_tma_store_wait {pendings = 0 : i32}
      ttg.warp_return
    } : (!ttg.memdesc<64xi32, #shared, #smem, mutable>, !tt.tensordesc<64xi32, #shared>) -> ()
    tt.return
  }

  // The default warps complete their own copy after entering the region.
  // Their fence before the wait cannot order the completed write against TMA.
  // CHECK-LABEL: @copy_completed_in_default_region
  tt.func @copy_completed_in_default_region(%input: !tt.ptr<i32>, %desc: !tt.tensordesc<64xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %offsets = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #src>
    %base = tt.splat %input : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #src>
    %ptrs = tt.addptr %base, %offsets : tensor<64x!tt.ptr<i32>, #src>, tensor<64xi32, #src>
    %tile = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    %copy = ttg.async_copy_global_to_local %ptrs, %tile : tensor<64x!tt.ptr<i32>, #src> -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    %commit = ttg.async_commit_group
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      // CHECK: ttng.fence_async_shared {bCluster = false}
      // CHECK-NEXT: {{.*}}ttg.async_wait
      // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
      // CHECK-NEXT: ttng.async_tma_copy_local_to_global
      ttng.fence_async_shared {bCluster = false}
      %wait = ttg.async_wait {num = 0 : i32}
      ttng.async_tma_copy_local_to_global %desc[%c0] %tile : !tt.tensordesc<64xi32, #shared>, !ttg.memdesc<64xi32, #shared, #smem, mutable>
      ttng.async_tma_store_wait {pendings = 0 : i32}
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }

  // A worker inside the parent's loop inherits its loop depth.
  // CHECK-LABEL: @worker_tma_in_parent_loop
  tt.func @worker_tma_in_parent_loop(%value: tensor<64xi32, #src>, %desc: !tt.tensordesc<64xi32, #shared>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %tile = ttg.local_alloc %value {allocation.offset = 0 : i32} : (tensor<64xi32, #src>) -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: cf.br
    scf.for %offset = %c0 to %n step %c64 : i32 {
      ttg.warp_specialize(%tile, %desc, %offset) attributes {allocation.offset = 256 : i32, allocation.size = 24 : i32, warpGroupStartIds = array<i32: 4>}
      default {
        ttg.warp_yield
      }
      partition0(%src: !ttg.memdesc<64xi32, #shared, #smem, mutable>, %out: !tt.tensordesc<64xi32, #shared>, %i: i32) num_warps(4) {
        ttng.async_tma_copy_local_to_global %out[%i] %src : !tt.tensordesc<64xi32, #shared>, !ttg.memdesc<64xi32, #shared, #smem, mutable>
        ttng.async_tma_store_wait {pendings = 0 : i32}
        ttg.warp_return
      } : (!ttg.memdesc<64xi32, #shared, #smem, mutable>, !tt.tensordesc<64xi32, #shared>, i32) -> ()
    }
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.return
    tt.return
  }

  // The default warps fence the parent's write on every path to the TMA store.
  // CHECK-LABEL: @default_fence_covers_parent_store
  tt.func @default_fence_covers_parent_store(%value: tensor<64xi32, #src>, %desc: !tt.tensordesc<64xi32, #shared>, %status: !tt.ptr<i32>, %length: i32) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // CHECK: ttg.local_alloc
    %tile = ttg.local_alloc %value {allocation.offset = 0 : i32} : (tensor<64xi32, #src>) -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    // CHECK-NOT: ttng.fence_async_shared
    scf.for %offset = %c0 to %length step %c64 : i32 {
      // CHECK: ttg.warp_specialize
      ttg.warp_specialize(%status) attributes {allocation.offset = 256 : i32, allocation.size = 8 : i32, warpGroupStartIds = array<i32: 4>}
      default {
        // CHECK: default {
        // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
        ttng.fence_async_shared {bCluster = false}
        ttg.warp_yield
      }
      partition0(%out: !tt.ptr<i32>) num_warps(4) {
        %one = arith.constant 1 : i32
        tt.store %out, %one : !tt.ptr<i32>
        ttg.warp_return
      } : (!tt.ptr<i32>) -> ()
      // CHECK-NOT: ttng.fence_async_shared
      // CHECK: ttng.async_tma_copy_local_to_global
      ttng.async_tma_copy_local_to_global %desc[%offset] %tile : !tt.tensordesc<64xi32, #shared>, !ttg.memdesc<64xi32, #shared, #smem, mutable>
      ttng.async_tma_store_wait {pendings = 0 : i32}
    }
    tt.return
  }

  tt.func private @publish_tile() {
    ttng.fence_async_shared {bCluster = false}
    tt.return
  }

  // The default warps publish the parent's write inside a noinline call.
  // CHECK-LABEL: @default_callee_fence_covers_parent_store
  tt.func @default_callee_fence_covers_parent_store(%value: tensor<64xi32, #src>, %desc: !tt.tensordesc<64xi32, #shared>, %status: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %tile = ttg.local_alloc %value {allocation.offset = 0 : i32} : (tensor<64xi32, #src>) -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @publish_tile
    ttg.warp_specialize(%status) attributes {allocation.offset = 256 : i32, allocation.size = 8 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      tt.call @publish_tile() : () -> ()
      ttg.warp_yield
    }
    partition0(%out: !tt.ptr<i32>) num_warps(4) {
      %one = arith.constant 1 : i32
      tt.store %out, %one : !tt.ptr<i32>
      ttg.warp_return
    } : (!tt.ptr<i32>) -> ()
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0] %tile : !tt.tensordesc<64xi32, #shared>, !ttg.memdesc<64xi32, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }

  // When the default warps can skip their fence, the parent fences before the loop.
  // CHECK-LABEL: @conditional_default_fence_requires_parent_fence
  tt.func @conditional_default_fence_requires_parent_fence(%value: tensor<64xi32, #src>, %desc: !tt.tensordesc<64xi32, #shared>, %status: !tt.ptr<i32>, %length: i32, %fence: i1) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    // CHECK: ttg.local_alloc
    %tile = ttg.local_alloc %value {allocation.offset = 0 : i32} : (tensor<64xi32, #src>) -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: cf.br
    scf.for %offset = %c0 to %length step %c64 : i32 {
      // CHECK: ttg.warp_specialize
      ttg.warp_specialize(%status) attributes {allocation.offset = 256 : i32, allocation.size = 8 : i32, warpGroupStartIds = array<i32: 4>}
      default {
        // CHECK: cf.cond_br
        // CHECK: tt.call @publish_tile
        scf.if %fence {
          tt.call @publish_tile() : () -> ()
        }
        ttg.warp_yield
      }
      partition0(%out: !tt.ptr<i32>) num_warps(4) {
        %one = arith.constant 1 : i32
        tt.store %out, %one : !tt.ptr<i32>
        ttg.warp_return
      } : (!tt.ptr<i32>) -> ()
      // CHECK-NOT: ttng.fence_async_shared
      // CHECK: ttng.async_tma_copy_local_to_global
      ttng.async_tma_copy_local_to_global %desc[%offset] %tile : !tt.tensordesc<64xi32, #shared>, !ttg.memdesc<64xi32, #shared, #smem, mutable>
      ttng.async_tma_store_wait {pendings = 0 : i32}
    }
    tt.return
  }

  // The default warps publish the tile on the first launch. Workers read it
  // after the join on each subsequent launch.
  // CHECK-LABEL: @worker_reads_tile_published_on_first_launch
  tt.func @worker_reads_tile_published_on_first_launch(%value: tensor<64xi32, #src>, %desc: !tt.tensordesc<64xi32, #shared>, %length: i32) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %tile = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    scf.for %offset = %c0 to %length step %c64 : i32 {
      %first = arith.cmpi eq, %offset, %c0 : i32
      ttg.warp_specialize(%tile, %desc, %offset) attributes {allocation.offset = 256 : i32, allocation.size = 24 : i32, warpGroupStartIds = array<i32: 4>}
      default {
        scf.if %first {
          // CHECK: ttg.local_store
          // CHECK-NEXT: tt.call @publish_tile
          ttg.local_store %value, %tile : tensor<64xi32, #src> -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
          tt.call @publish_tile() : () -> ()
        }
        ttg.warp_yield
      }
      partition0(%src: !ttg.memdesc<64xi32, #shared, #smem, mutable>, %out: !tt.tensordesc<64xi32, #shared>, %out_offset: i32) num_warps(4) {
        %zero = arith.constant 0 : i32
        %ready = arith.cmpi ne, %out_offset, %zero : i32
        scf.if %ready {
          // CHECK-NOT: ttng.fence_async_shared
          // CHECK: ttng.async_tma_copy_local_to_global
          ttng.async_tma_copy_local_to_global %out[%out_offset] %src : !tt.tensordesc<64xi32, #shared>, !ttg.memdesc<64xi32, #shared, #smem, mutable>
          ttng.async_tma_store_wait {pendings = 0 : i32}
        }
        ttg.warp_return
      } : (!ttg.memdesc<64xi32, #shared, #smem, mutable>, !tt.tensordesc<64xi32, #shared>, i32) -> ()
    }
    tt.return
  }
}

// -----

#rows = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#replica = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[0, 0]]}>
!rows = tensor<32x32xf32, #rows>
!tile = !ttg.memdesc<32x32xf32, #replica, #ttg.shared_memory, mutable>
!desc = !tt.tensordesc<32x32xf32, #replica>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  // Each CTA writes both replicas. The TMA issuer reads its own replica.
  // CHECK-LABEL: @peer_store_to_tma
  tt.func @peer_store_to_tma(%value: !rows, %out: !desc) {
    %c0 = arith.constant 0 : i32
    %tile = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !tile

    // A writer's CTA fence misses the peer's replica. Fence it after acquire.
    // CHECK:      ttg.local_store
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.cluster_barrier{{$}}
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global
    ttg.local_store %value, %tile : !rows -> !tile
    ttng.fence_async_shared {bCluster = false}
    ttng.cluster_barrier
    ttng.async_tma_copy_local_to_global %out[%c0, %c0] %tile : !desc, !tile
    ttng.async_tma_store_wait {pendings = 0 : i32}
    ttng.cluster_barrier

    // A cluster fence on the writer covers both replicas.
    // CHECK:      ttg.local_store
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.cluster_barrier{{$}}
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global
    ttg.local_store %value, %tile : !rows -> !tile
    ttng.fence_async_shared {bCluster = true}
    ttng.cluster_barrier
    ttng.async_tma_copy_local_to_global %out[%c0, %c0] %tile : !desc, !tile
    ttng.async_tma_store_wait {pendings = 0 : i32}
    ttng.cluster_barrier

    // After acquire the local fence on the reader covers the peer's writes.
    // CHECK:      ttg.local_store
    // CHECK-NEXT: ttng.cluster_barrier{{$}}
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global
    ttg.local_store %value, %tile : !rows -> !tile
    ttng.cluster_barrier
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %out[%c0, %c0] %tile : !desc, !tile
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }

  tt.func private @local_proxy_fence() {
    ttng.fence_async_shared {bCluster = false}
    tt.return
  }

  // CHECK-LABEL: @copy_replica
  // CHECK: ttng.fence_async_shared {bCluster = false}
  // CHECK-NEXT: ttng.async_tma_copy_local_to_global
  tt.func private @copy_replica(%tile: !tile, %out: !desc) {
    %c0 = arith.constant 0 : i32
    ttng.async_tma_copy_local_to_global %out[%c0, %c0] %tile : !desc, !tile
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }

  // The CTA fence in the first call precedes the acquire. The TMA still
  // needs a fence after the full cluster barrier.
  // CHECK-LABEL: @peer_store_across_calls
  // CHECK: ttg.local_store
  // CHECK-NEXT: ttng.cluster_barrier {relaxed = true}
  // CHECK-NEXT: tt.call @local_proxy_fence
  // CHECK-NEXT: ttng.cluster_barrier{{$}}
  // CHECK-NEXT: tt.call @copy_replica
  tt.func @peer_store_across_calls(%value: !rows, %out: !desc) {
    %tile = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !tile
    ttg.local_store %value, %tile : !rows -> !tile
    ttng.cluster_barrier {relaxed = true}
    tt.call @local_proxy_fence() : () -> ()
    ttng.cluster_barrier
    tt.call @copy_replica(%tile, %out) : (!tile, !desc) -> ()
    tt.return
  }
}
