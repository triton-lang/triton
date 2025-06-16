// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: @single_local_alloc
  tt.func public @single_local_alloc() {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0>} : tensor<1xi64, #blocked>
    // CHECK-DAG: %[[STATE:.*]] = arith.constant dense<0> : tensor<1xi8, #blocked>
    // CHECK-DAG: %[[BARRIERS:.*]] = arith.constant dense<0> : tensor<1xi64, #blocked>

    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: @two_local_alloc
  tt.func public @two_local_alloc() {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0, 65536>} : tensor<2xi64, #blocked>
    // CHECK-DAG: %[[STATE:.*]] = arith.constant dense<0> : tensor<2xi8, #blocked>
    // CHECK-DAG: %[[BARRIERS:.*]] = arith.constant dense<0> : tensor<2xi64, #blocked>

    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: @three_local_alloc
  tt.func public @three_local_alloc() {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0, 65536, 131072, 0>} : tensor<4xi64, #blocked>
    // CHECK-DAG: %[[STATE:.*]] = arith.constant dense<0> : tensor<4xi8, #blocked>
    // CHECK-DAG: %[[BARRIERS:.*]] = arith.constant dense<0> : tensor<4xi64, #blocked>

    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %2 = ttg.local_alloc {allocation.offset = 131072 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: @three_sub_bufs
  tt.func public @three_sub_bufs() {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0, 32768, 65536, 0>} : tensor<4xi64, #blocked>
    // CHECK-DAG: %[[STATE:.*]] = arith.constant dense<0> : tensor<4xi8, #blocked>
    // CHECK-DAG: %[[BARRIERS:.*]] = arith.constant dense<0> : tensor<4xi64, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<3x32x32xf32, #shared, #smem, mutable>
    %1 = ttg.memdesc_subview %0[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: @async_tma_copy_global_to_local
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>) {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0, 65536>} : tensor<2xi64, #blocked>
    // CHECK-DAG: %[[STATE:.*]] = arith.constant dense<0> : tensor<2xi8, #blocked>
    // CHECK-DAG: %[[BARRIERS:.*]] = arith.constant dense<0> : tensor<2xi64, #blocked>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: %[[OUT_STATES:.*]], %[[OUT_BARRIERS:.*]] = tti.experimental_check_async_write_with_mbar_shared {{.*}}, {{.*}}{%[[BUFFERS]], %[[STATE]], %[[BARRIERS]]}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %bar, %true : !tt.tensordesc<tensor<32x32xf32, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: @wait_barrier
  tt.func public @wait_barrier(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>) {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0, 65536>} : tensor<2xi64, #blocked>
    // CHECK-DAG: %[[STATE:.*]] = arith.constant dense<0> : tensor<2xi8, #blocked>
    // CHECK-DAG: %[[BARRIERS:.*]] = arith.constant dense<0> : tensor<2xi64, #blocked>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: %[[OUT_STATES:.*]], %[[OUT_BARRIERS:.*]] = tti.experimental_check_wait_mbar {{.*}}{%[[STATE]], %[[BARRIERS]]}
    ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}
