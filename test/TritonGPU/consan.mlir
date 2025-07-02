// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: @single_local_alloc
  tt.func public @single_local_alloc() {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0>} : tensor<1xi64, #blocked>
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<1xi64, #blocked>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    // CHECK: %[[SPLAT:.*]] = tt.splat %[[WRT_BARS_GLOB]] : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>, #blocked>
    // CHECK: %[[RANGE:.*]] = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #blocked>
    // CHECK: %[[ADD:.*]] = tt.addptr %[[SPLAT]], %[[RANGE]] : tensor<1x!tt.ptr<i64>, #blocked>, tensor<1xi32, #blocked>
    // CHECK: tt.store %[[ADD]], %cst : tensor<1x!tt.ptr<i64>, #blocked>
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0, 4096>} : tensor<2xi64, #blocked>
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<2xi64, #blocked>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 16 : i32} : !tt.ptr<i64>
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0, 4096, 8192, 0>} : tensor<4xi64, #blocked>
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<4xi64, #blocked>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 32 : i32} : !tt.ptr<i64>
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %2 = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0, 4096, 8192, 0>} : tensor<4xi64, #blocked>
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<4xi64, #blocked>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 32 : i32} : !tt.ptr<i64>
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
  // CHECK: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: @async_tma_copy_global_to_local
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>) {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0>} : tensor<1xi64, #blocked>
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 65536>} : tensor<1xi64, #blocked>
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<1xi64, #blocked>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_check_outstanding_writes {{.*}}{%[[BUFFERS]], %[[WRT_BARS_GLOB]](tensor<1xi64, #blocked>)}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %bar, %true : !tt.tensordesc<tensor<32x32xf32, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: @wait_barrier
  tt.func public @wait_barrier(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>) {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0>} : tensor<1xi64, #blocked>
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 65536>} : tensor<1xi64, #blocked>
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<1xi64, #blocked>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_clear_write_barrier {{.*}}{%[[WRT_BARS_GLOB]](tensor<1xi64, #blocked>)}
    ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}
