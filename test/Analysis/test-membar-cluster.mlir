// RUN: triton-opt %s -split-input-file -test-print-membar | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32} {
  // Two lifetimes alias at offset 0; second is multicast TMA => cluster barrier.
  // CHECK-LABEL: alias_async_then_multicast
  tt.func @alias_async_then_multicast(%desc: !tt.tensordesc<tensor<32x32xf16, #shared1>>) {
    %pred = arith.constant true
    %c0 = arith.constant 0 : i32
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %dst0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared1, #smem, mutable>
    // First lifetime: single-CTA async copy and consume token.
    ttng.barrier_expect %bar, 2048, %pred : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst0, %bar, %pred {multicast = false} : !tt.tensordesc<tensor<32x32xf16, #shared1>>, !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf16, #shared1, #smem, mutable>
    ttng.wait_barrier %bar, %c0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: ttg.local_load
    %t0 = ttg.local_load %dst0 : !ttg.memdesc<32x32xf16, #shared1, #smem, mutable> -> tensor<32x32xf16, #blocked>
    // Second lifetime aliases offset 0 and writes multicast to all CTAs.
    %dst1 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared1, #smem, mutable>
    // CHECK: ttng.barrier_expect
    // CHECK: ttng.cluster_arrive
    // CHECK: ttng.cluster_wait
    ttng.barrier_expect %bar, 2048, %pred : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local {{.*}} {multicast = true}
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst1, %bar, %pred {multicast = true} : !tt.tensordesc<tensor<32x32xf16, #shared1>>, !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf16, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

// Async cp then multicast alias (needs cluster barrier)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: async_cp_then_multicast_alias
  tt.func @async_cp_then_multicast_alias(%gptr: !tt.ptr<f16>, %desc: !tt.tensordesc<tensor<32x32xf16, #shared1>>, %desc2: !tt.tensordesc<tensor<32x32xf16, #shared2>>) {
    %pred = arith.constant true
    %c0 = arith.constant 0 : i32
    %gptr_tensor = tt.splat %gptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %a = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared1, #smem, mutable>
    %cp = ttg.async_copy_global_to_local %gptr_tensor, %a : tensor<32x32x!tt.ptr<f16>, #blocked> -> !ttg.memdesc<32x32xf16, #shared1, #smem, mutable>
    %tok = ttg.async_commit_group tokens %cp
    %tok2 = ttg.async_wait %tok {num = 0 : i32}
    // CHECK: ttg.local_load
    %ld = ttg.local_load %a token %tok2 : !ttg.memdesc<32x32xf16, #shared1, #smem, mutable> -> tensor<32x32xf16, #blocked>
    %b = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared2, #smem, mutable>
    // CHECK: ttng.barrier_expect
    // CHECK: ttng.cluster_arrive
    // CHECK: ttng.cluster_wait
    ttng.barrier_expect %bar, 2048, %pred : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local {{.*}} {multicast = true}
    ttng.async_tma_copy_global_to_local %desc2[%c0, %c0] %b, %bar, %pred {multicast = true} : !tt.tensordesc<tensor<32x32xf16, #shared2>>, !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf16, #shared2, #smem, mutable>
    ttng.wait_barrier %bar, %c0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Async cp + reinterpet & multicast on same allocation (no cluster barrier)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: async_cp_and_multicast_same_alloc
  tt.func @async_cp_and_multicast_same_alloc(%gptr: !tt.ptr<f16>, %desc: !tt.tensordesc<tensor<32x32xf16, #shared2>>) {
    %pred = arith.constant true
    %c0 = arith.constant 0 : i32
    %gptr_tensor = tt.splat %gptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared1, #smem, mutable>
    %cp = ttg.async_copy_global_to_local %gptr_tensor, %buf : tensor<32x32x!tt.ptr<f16>, #blocked> -> !ttg.memdesc<32x32xf16, #shared1, #smem, mutable>
    %tok = ttg.async_commit_group tokens %cp
    %tok2 = ttg.async_wait %tok {num = 0 : i32}
    %ld = ttg.local_load %buf token %tok2 : !ttg.memdesc<32x32xf16, #shared1, #smem, mutable> -> tensor<32x32xf16, #blocked>
    %buf_reint = ttg.memdesc_reinterpret %buf : !ttg.memdesc<32x32xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf16, #shared2, #smem, mutable>

    ttng.barrier_expect %bar, 2048, %pred : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: ttng.barrier_expect
    // CHECK-NOT: ttng.cluster_arrive
    // CHECK-NOT: ttng.cluster_wait
    // CHECK: ttng.async_tma_copy_global_to_local {{.*}} {multicast = true}
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf_reint, %bar, %pred {multicast = true} : !tt.tensordesc<tensor<32x32xf16, #shared2>>, !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf16, #shared2, #smem, mutable>
    ttng.wait_barrier %bar, %c0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Distributed convert alias (needs cluster barrier)
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0], CGALayout = [[1, 0], [0, 1]]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0], CGALayout = [[0, 1], [1, 0]]}>

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_alias_same_offset
  tt.func @convert_alias_same_offset() -> (tensor<2x2xf16, #blocked2>, tensor<2x2xf16, #blocked2>) {
    %c0 = arith.constant 0.000000e+00 : f16
    %src = tt.splat %c0 : f16 -> tensor<2x2xf16, #blocked1>
    // CHECK: ttg.convert_layout
    %cvt0 = ttg.convert_layout %src {allocation.offset = 0 : i32} : tensor<2x2xf16, #blocked1> -> tensor<2x2xf16, #blocked2>
    // CHECK: ttng.cluster_arrive
    // CHECK: ttng.cluster_wait
    // CHECK: ttg.convert_layout
    %cvt1 = ttg.convert_layout %src {allocation.offset = 0 : i32} : tensor<2x2xf16, #blocked1> -> tensor<2x2xf16, #blocked2>
    tt.return %cvt0, %cvt1 : tensor<2x2xf16, #blocked2>, tensor<2x2xf16, #blocked2>
  }
}
