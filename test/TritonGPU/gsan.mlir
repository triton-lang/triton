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
  tt.func public @instrumented_call_in_warp_specialize(%value: i32) {
    // CHECK: ttg.warp_specialize(%{{.*}}, %[[STATE]])
    ttg.warp_specialize(%value)
    default {
      ttg.warp_yield
    }
    // CHECK: partition0(%[[VALUE:[^:, )]+]]: i32, %[[PARTITION_STATE:[^:, )]+]]: !tt.ptr<i8>) num_warps(4)
    partition0(%partition_value: i32) num_warps(4) {
      // CHECK: tt.call @identity(%[[VALUE]], %[[PARTITION_STATE]]) : (i32, !tt.ptr<i8>) -> i32
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
