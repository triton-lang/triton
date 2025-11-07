// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BUFS_L:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: #[[BUFS_THREADS_L:.*]] = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
  // CHECK: #[[BUFS_BARS_L:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
  // CHECK: @single_local_alloc
  tt.func public @single_local_alloc() {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared_mem : tensor<1xi64, #[[BUFS_L]]>
    // CHECK: %[[WRITE_VISIBILITY:.*]] = arith.constant dense<0> : tensor<1xi64, #[[BUFS_L]]>
    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    // CHECK: %[[READ_VISIBILITY:.*]] = arith.constant dense<0> : tensor<1x64xi64, #[[BUFS_THREADS_L]]>
    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 512 : i32} : !tt.ptr<i64>
    // CHECK: %[[WRITE_TRACKING:.*]] = arith.constant dense<0> : tensor<1x1xi8, #[[BUFS_BARS_L]]>
    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[READ_TRACKING:.*]] = arith.constant dense<0> : tensor<1x1xi64, #[[BUFS_BARS_L]]>
    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @two_local_alloc
  tt.func public @two_local_alloc() {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0, 4096], shared_mem : tensor<2xi64,
    // CHECK: %[[WRITE_VISIBILITY:.*]] = arith.constant dense<0> : tensor<2xi64,
    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 16 : i32} : !tt.ptr<i64>
    // CHECK: %[[READ_VISIBILITY:.*]] = arith.constant dense<0> : tensor<2x64xi64,
    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 1024 : i32} : !tt.ptr<i64>
    // CHECK: %[[WRITE_TRACKING:.*]] = arith.constant dense<0> : tensor<2x1xi8,
    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK: %[[READ_TRACKING:.*]] = arith.constant dense<0> : tensor<2x1xi64,
    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 16 : i32} : !tt.ptr<i64>
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @three_local_alloc
  tt.func public @three_local_alloc() {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0, 4096, 8192, 0], shared_mem : tensor<4xi64,
    // CHECK: %[[WRITE_VISIBILITY:.*]] = arith.constant dense<0> : tensor<4xi64,
    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 32 : i32} : !tt.ptr<i64>
    // CHECK: %[[READ_VISIBILITY:.*]] = arith.constant dense<0> : tensor<4x64xi64,
    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 2048 : i32} : !tt.ptr<i64>
    // CHECK: %[[WRITE_TRACKING:.*]] = arith.constant dense<0> : tensor<4x1xi8,
    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 4 : i32} : !tt.ptr<i8>
    // CHECK: %[[READ_TRACKING:.*]] = arith.constant dense<0> : tensor<4x1xi64,
    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 32 : i32} : !tt.ptr<i64>
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %2 = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 12288 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @three_sub_bufs
  tt.func public @three_sub_bufs() {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0, 4096, 8192, 0], shared_mem : tensor<4xi64,
    // CHECK: %[[WRITE_VISIBILITY:.*]] = arith.constant dense<0> : tensor<4xi64,
    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 32 : i32} : !tt.ptr<i64>
    // CHECK: %[[READ_VISIBILITY:.*]] = arith.constant dense<0> : tensor<4x64xi64,
    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 2048 : i32} : !tt.ptr<i64>
    // CHECK: %[[WRITE_TRACKING:.*]] = arith.constant dense<0> : tensor<4x1xi8,
    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 4 : i32} : !tt.ptr<i8>
    // CHECK: %[[READ_TRACKING:.*]] = arith.constant dense<0> : tensor<4x1xi64,
    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 32 : i32} : !tt.ptr<i64>
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<3x32x32xf32, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<3x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[READ_BARS_L:.*]] = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
  // CHECK: @read_bars_alloc
  tt.func public @read_bars_alloc() {
    // CHECK: %[[READ_BARS_G:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 8 : i32} : !tt.ptr<i8>
    // CHECK: %[[SPLAT:.*]] = tt.splat %[[READ_BARS_G]] : !tt.ptr<i8> -> tensor<2x4x!tt.ptr<i8>, #[[READ_BARS_L]]>
    // CHECK: %[[RANGE:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #[[READ_BARS_L]]}>
    // CHECK: %[[STRIDE:.*]] = arith.constant dense<1> : tensor<2xi32, #ttg.slice<{dim = 1, parent = #[[READ_BARS_L]]}>
    // CHECK: %[[OFFS:.*]] = arith.muli %[[RANGE]], %[[STRIDE]]
    // CHECK: %[[EXP:.*]] = tt.expand_dims %[[OFFS]] {axis = 1 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #[[READ_BARS_L]]}>> -> tensor<2x1xi32, #[[READ_BARS_L]]>
    // CHECK: %[[BROAD:.*]] = tt.broadcast %[[EXP]] : tensor<2x1xi32, #[[READ_BARS_L]]> -> tensor<2x4xi32, #[[READ_BARS_L]]>
    // CHECK: %[[PTR0:.*]] = tt.addptr %[[SPLAT]], %[[BROAD]] : tensor<2x4x!tt.ptr<i8>, #[[READ_BARS_L]]>, tensor<2x4xi32, #[[READ_BARS_L]]>
    // CHECK: %[[RANGE:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #[[READ_BARS_L]]}>
    // CHECK: %[[STRIDE:.*]] = arith.constant dense<2> : tensor<4xi32, #ttg.slice<{dim = 0, parent = #[[READ_BARS_L]]}>
    // CHECK: %[[OFFS:.*]] = arith.muli %[[RANGE]], %[[STRIDE]]
    // CHECK: %[[EXP:.*]] = tt.expand_dims %[[OFFS]] {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #[[READ_BARS_L]]}>> -> tensor<1x4xi32, #[[READ_BARS_L]]>
    // CHECK: %[[BROAD:.*]] = tt.broadcast %[[EXP]] : tensor<1x4xi32, #[[READ_BARS_L]]> -> tensor<2x4xi32, #[[READ_BARS_L]]>
    // CHECK: %[[PTR1:.*]] = tt.addptr %[[PTR0]], %[[BROAD]] : tensor<2x4x!tt.ptr<i8>, #[[READ_BARS_L]]>, tensor<2x4xi32, #[[READ_BARS_L]]>
    // CHECK: tt.store %[[PTR1]], {{.*}} : tensor<2x4x!tt.ptr<i8>, #[[READ_BARS_L]]>
    %c0 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<4x1xi64, #shared1, #smem, mutable>
    %bar_sub = ttg.memdesc_index %bar[%c0] : !ttg.memdesc<4x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar_sub, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %buf_sub = ttg.memdesc_index %0[%c0] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>

    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BUFS_L:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: @tmem_alloc
  tt.func public @tmem_alloc() {
    // CHECK-DAG: %[[TMEM_BUFS:.*]] = tti.experimental_buffer_pointers [0], tensor_mem : tensor<1xi64, #[[BUFS_L]]>
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_pointers [4096], shared_mem : tensor<1xi64, #[[BUFS_L]]>
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_tma_copy_global_to_local
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>) {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared_mem : tensor<1xi64
    // CHECK-DAG: %[[WRITE_VISIBILITY:.*]] = arith.constant dense<0> : tensor<1xi64,
    // CHECK-DAG: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[READ_VISIBILITY:.*]] = arith.constant dense<0> : tensor<1x64xi64,
    // CHECK-DAG: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 512 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_pointers [65536], shared_mem : tensor<1xi64
    // CHECK-DAG: %[[WRITE_TRACKING:.*]] = arith.constant dense<0> : tensor<1x1xi8,
    // CHECK-DAG: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK-DAG: %[[READ_TRACKING:.*]] = arith.constant dense<0> : tensor<1x1xi64,
    // CHECK-DAG: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_init_barrier_state
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_set_write_visibility
    // CHECK: tt.call @__triton_consan_clear_write_tracking
    // CHECK: tt.call @__triton_consan_clear_read_visibility
    // CHECK: tt.call @__triton_consan_clear_read_tracking
    // CHECK: tt.call @__triton_consan_track_visible_writes
    // CHECK: tt.call @__triton_consan_verify_barrier_arrive
    // CHECK: tt.call @__triton_consan_update_barrier_state
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %bar, %true : !tt.tensordesc<tensor<32x32xf32, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_tma_copy_local_to_global
  tt.func public @async_tma_copy_local_to_global(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>, %ptr: tensor<128x128x!tt.ptr<f16>, #blocked>, %acc: tensor<128x128xf16, #mma>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>
    ttng.warp_group_dot %shmem, %shmem, %acc : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #mma>
    // CHECK: ttng.warp_group_dot

    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits
    ttng.async_tma_copy_local_to_global %arg0[%c0_i32, %c0_i32] %0 : !tt.tensordesc<tensor<32x32xf32, #shared>>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_tma_gather
  tt.func public @async_tma_gather(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>, %ptr: tensor<128x128x!tt.ptr<f16>, #blocked>, %acc: tensor<128x128xf16, #mma>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %x_offsets = arith.constant dense<1> : tensor<32xi32>
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>
    ttng.warp_group_dot %shmem, %shmem, %acc : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #mma>
    // CHECK: ttng.warp_group_dot

    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_set_write_visibility
    // CHECK: tt.call @__triton_consan_clear_write_tracking
    // CHECK: tt.call @__triton_consan_clear_read_visibility
    // CHECK: tt.call @__triton_consan_clear_read_tracking
    // CHECK: tt.call @__triton_consan_track_visible_writes
    ttng.async_tma_gather %arg0[%x_offsets, %c0_i32] %0, %bar, %true : !tt.tensordesc<tensor<32x32xf32, #shared>>, tensor<32xi32>, i32, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, i1
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_tma_scatter
  tt.func public @async_tma_scatter(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>, %ptr: tensor<128x128x!tt.ptr<f16>, #blocked>, %acc: tensor<128x128xf16, #mma>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %x_offsets = arith.constant dense<1> : tensor<32xi32>
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>
    ttng.warp_group_dot %shmem, %shmem, %acc : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #mma>
    // CHECK: ttng.warp_group_dot

    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits
    ttng.async_tma_scatter %arg0[%x_offsets, %c0_i32] %0 : !tt.tensordesc<tensor<32x32xf32, #shared>>, tensor<32xi32>, i32, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @wait_barrier
  tt.func public @wait_barrier(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>) {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared_mem : tensor<1xi64, #blocked>
    // CHECK-DAG: %[[WRITE_VISIBILITY:.*]] = arith.constant dense<0> : tensor<1xi64,
    // CHECK-DAG: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[READ_VISIBILITY:.*]] = arith.constant dense<0> : tensor<1x64xi64,
    // CHECK-DAG: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 512 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_pointers [65536], shared_mem : tensor<1xi64, #blocked>
    // CHECK-DAG: %[[WRITE_TRACKING:.*]] = arith.constant dense<0> : tensor<1x1xi8,
    // CHECK-DAG: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK-DAG: %[[READ_TRACKING:.*]] = arith.constant dense<0> : tensor<1x1xi64,
    // CHECK-DAG: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK-DAG: tt.call @__triton_consan_set_waiting
    // CHECK-DAG: tt.call @__triton_consan_check_all_active_waiting
    // CHECK: ttng.wait_barrier
    ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_transfer_visible_writes{{.*}}%[[BARRIERS]], %[[WRITE_VISIBILITY_GLOB]], %[[WRITE_TRACKING_GLOB]]
    // CHECK: tt.call @__triton_consan_transfer_visible_reads{{.*}}%[[BARRIERS]], %[[READ_VISIBILITY_GLOB]], %[[READ_TRACKING_GLOB]]
    // CHECK: tt.call @__triton_consan_clear_waiting
    // CHECK: tti.experimental_lock_release
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @arrive_barrier
  tt.func public @arrive_barrier(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>) {
    // CHECK-DAG: %[[BSTATE_INIT:.*]] = arith.constant dense<0> : tensor<1xi32, #{{.*}}>
    // CHECK-DAG: %[[BSTATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 4 : i32, nbytes = 4 : i32} : !tt.ptr<i32>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_init_barrier_state
    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_track_visible_writes
    // CHECK: tt.call @__triton_consan_track_visible_reads
    // CHECK: tt.call @__triton_consan_verify_barrier_arrive
    // CHECK: tt.call @__triton_consan_update_barrier_state
    // CHECK: tti.experimental_lock_release
    ttng.arrive_barrier %bar, 2, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tcgen5_mma
  tt.func public @tcgen5_mma(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>) {
    // CHECK-DAG: %[[SM_BUFS:.*]] = tti.experimental_buffer_pointers [0, 32768], shared_mem : tensor<2xi64
    // CHECK-DAG: %[[SM_WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 16 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[SM_READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 1024 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[TM_BUFS:.*]] = tti.experimental_buffer_pointers [0], tensor_mem : tensor<1xi64
    // CHECK-DAG: %[[TM_WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[TM_READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 512 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_pointers [65536], shared_mem : tensor<1xi64

    // CHECK-DAG: %[[SM_WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK-DAG: %[[SM_READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 16 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[TM_WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK-DAG: %[[TM_READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>

    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[A:.*]], %true, %[[TC_BIT]], %[[SM_BUFS]], %[[SM_WRITE_VISIBILITY_GLOB]]
    // CHECK: %[[TC_MASK:.*]] = arith.constant 4294967296 : i64
    // CHECK: tt.call @__triton_consan_set_read_visibility{{.*}}%[[A]], %true, %[[TC_MASK]], %[[SM_BUFS]], %[[SM_READ_VISIBILITY_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[B:.*]], %true, %[[TC_BIT]], %[[SM_BUFS]], %[[SM_WRITE_VISIBILITY_GLOB]]
    // CHECK: %[[TC_MASK:.*]] = arith.constant 4294967296 : i64
    // CHECK: tt.call @__triton_consan_set_read_visibility{{.*}}%[[B]], %true, %[[TC_MASK]], %[[SM_BUFS]], %[[SM_READ_VISIBILITY_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[ACC:.*]], %true, %[[TC_BIT]], %[[TM_BUFS]], %[[TM_WRITE_VISIBILITY_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}%[[ACC]], %true, %[[TC_BIT]], %[[TM_BUFS]], %[[TM_READ_VISIBILITY_GLOB]]
    // CHECK: %[[TC_MASK:.*]] = arith.constant 4294967296 : i64
    // CHECK: tt.call @__triton_consan_set_write_visibility{{.*}}%[[ACC]], %true, %[[TC_MASK]], %[[TM_BUFS]], %[[TM_WRITE_VISIBILITY_GLOB]]
    // CHECK: tt.call @__triton_consan_clear_write_tracking{{.*}}%[[ACC]], %true, %[[TM_BUFS]], %[[TM_WRITE_TRACKING_GLOB]]
    // CHECK: tt.call @__triton_consan_clear_read_visibility{{.*}}%[[ACC]], %true, %[[TM_BUFS]], %[[TM_READ_VISIBILITY_GLOB]]
    // CHECK: tt.call @__triton_consan_clear_read_tracking{{.*}}%[[ACC]], %true, %[[TM_BUFS]], %[[TM_READ_TRACKING_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_track_visible_writes{{.*}}%[[BAR:.*]], {{.*}}, %[[TC_BIT]], %[[BARRIERS]], %[[SM_WRITE_VISIBILITY_GLOB]], %[[SM_WRITE_TRACKING_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_track_visible_reads{{.*}}%[[BAR]], {{.*}}, %[[TC_BIT]], %[[BARRIERS]], %[[SM_READ_VISIBILITY_GLOB]], %[[SM_READ_TRACKING_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_track_visible_writes{{.*}}%[[BAR]], {{.*}}, %[[TC_BIT]], %[[BARRIERS]], %[[TM_WRITE_VISIBILITY_GLOB]], %[[TM_WRITE_TRACKING_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_track_visible_reads{{.*}}%[[BAR]], {{.*}}, %[[TC_BIT]], %[[BARRIERS]], %[[TM_READ_VISIBILITY_GLOB]], %[[TM_READ_TRACKING_GLOB]]
    // CHECK: ttng.tc_gen5_mma %[[A]], %[[B]], %[[ACC]][], {{.*}}, {{.*}}, %[[BAR]]
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %result = ttng.tmem_alloc  {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %true = arith.constant true
    ttng.tc_gen5_mma %0, %1, %result[], %true, %true, %bar[%true] {is_async} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tcgen5_mma_lhs_in_tmem
  tt.func public @tcgen5_mma_lhs_in_tmem(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>) {
    // CHECK-DAG: %[[SM_BUFS:.*]] = tti.experimental_buffer_pointers [32768], shared_mem : tensor<1xi64
    // CHECK-DAG: %[[SM_WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[SM_READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 512 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[TM_BUFS:.*]] = tti.experimental_buffer_pointers [0, 128], tensor_mem : tensor<2xi64
    // CHECK-DAG: %[[TM_WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 16 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[TM_READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 1024 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_pointers [65536], shared_mem : tensor<1xi64

    // CHECK-DAG: %[[SM_WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK-DAG: %[[SM_READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[TM_WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK-DAG: %[[TM_READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 16 : i32} : !tt.ptr<i64>

    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[A:.*]], %true, %[[TC_BIT]], %[[TM_BUFS]], %[[TM_WRITE_VISIBILITY_GLOB]]
    // CHECK: %[[TC_MASK:.*]] = arith.constant 4294967296 : i64
    // CHECK: tt.call @__triton_consan_set_read_visibility{{.*}}%[[A]], %true, %[[TC_MASK]], %[[TM_BUFS]], %[[TM_READ_VISIBILITY_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[B:.*]], %true, %[[TC_BIT]], %[[SM_BUFS]], %[[SM_WRITE_VISIBILITY_GLOB]]
    // CHECK: %[[TC_MASK:.*]] = arith.constant 4294967296 : i64
    // CHECK: tt.call @__triton_consan_set_read_visibility{{.*}}%[[B]], %true, %[[TC_MASK]], %[[SM_BUFS]], %[[SM_READ_VISIBILITY_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[ACC:.*]], %true, %[[TC_BIT]], %[[TM_BUFS]], %[[TM_WRITE_VISIBILITY_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}%[[ACC]], %true, %[[TC_BIT]], %[[TM_BUFS]], %[[TM_READ_VISIBILITY_GLOB]]
    // CHECK: %[[TC_MASK:.*]] = arith.constant 4294967296 : i64
    // CHECK: tt.call @__triton_consan_set_write_visibility{{.*}}%[[ACC]], %true, %[[TC_MASK]], %[[TM_BUFS]], %[[TM_WRITE_VISIBILITY_GLOB]]
    // CHECK: tt.call @__triton_consan_clear_write_tracking{{.*}}%[[ACC]], %true, %[[TM_BUFS]], %[[TM_WRITE_TRACKING_GLOB]]
    // CHECK: tt.call @__triton_consan_clear_read_visibility{{.*}}%[[ACC]], %true, %[[TM_BUFS]], %[[TM_READ_VISIBILITY_GLOB]]
    // CHECK: tt.call @__triton_consan_clear_read_tracking{{.*}}%[[ACC]], %true, %[[TM_BUFS]], %[[TM_READ_TRACKING_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_track_visible_writes{{.*}}%[[BAR:.*]], {{.*}}, %[[TC_BIT]], %[[BARRIERS]], %[[SM_WRITE_VISIBILITY_GLOB]], %[[SM_WRITE_TRACKING_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_track_visible_reads{{.*}}%[[BAR:.*]], {{.*}}, %[[TC_BIT]], %[[BARRIERS]], %[[SM_READ_VISIBILITY_GLOB]], %[[SM_READ_TRACKING_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_track_visible_writes{{.*}}%[[BAR:.*]], {{.*}}, %[[TC_BIT]], %[[BARRIERS]], %[[TM_WRITE_VISIBILITY_GLOB]], %[[TM_WRITE_TRACKING_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 32 : i32
    // CHECK: tt.call @__triton_consan_track_visible_reads{{.*}}%[[BAR:.*]], {{.*}}, %[[TC_BIT]], %[[BARRIERS]], %[[TM_READ_VISIBILITY_GLOB]], %[[TM_READ_TRACKING_GLOB]]
    // CHECK: tt.call @__triton_consan_verify_barrier_arrive
    // CHECK: tt.call @__triton_consan_update_barrier_state
    // CHECK: tti.experimental_lock_release
    // CHECK: ttng.tc_gen5_mma %[[A]], %[[B]], %[[ACC]][], {{.*}}, {{.*}}, %[[BAR]]
    %c0_i32 = arith.constant 0 : i32
    %0 = ttng.tmem_alloc  {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>
    %1 = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %result = ttng.tmem_alloc  {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %true = arith.constant true
    ttng.tc_gen5_mma %0, %1, %result[], %true, %true, %bar[%true] {is_async} : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tcgen5_commit
  tt.func public @tcgen5_commit(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>) {

    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %result = ttng.tmem_alloc  {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_init_barrier_state
    %true = arith.constant true
    // CHECK: tt.call @__triton_consan_track_visible_writes
    // CHECK: tt.call @__triton_consan_track_visible_reads
    // CHECK: tt.call @__triton_consan_track_visible_writes
    // CHECK: tt.call @__triton_consan_track_visible_reads
    // CHECK: tt.call @__triton_consan_verify_barrier_arrive
    // CHECK: tt.call @__triton_consan_update_barrier_state
    ttng.tc_gen5_commit %bar : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_copy_global_to_local
  tt.func public @async_copy_global_to_local(%ptr: tensor<128x128x!tt.ptr<f16>, #blocked>) {
    // CHECK: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared_mem : tensor<1xi64
    // CHECK: %[[WRITE_COMMITS:.*]] = arith.constant dense<0> : tensor<1x16xi8
    // CHECK: %[[WRT_COMMITS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 16 : i32} : !tt.ptr<i8>

    // CHECK: tt.call @__triton_consan_verify_write_visibility_nw1
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_check_outstanding_commits{{.*}}(%[[A:.*]], {{.*}}, %[[THREAD_BIT]], %[[BUFFERS]], %[[WRT_COMMITS_GLOB]]
    // CHECK: tt.call @__triton_consan_verify_read_visibility_nw1
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_stage_access_for_commit_nw1{{.*}}(%[[A]], {{.*}}, %[[THREAD_BIT]], %[[BUFFERS]], %[[WRT_COMMITS_GLOB]]
    // CHECK: ttg.async_copy_global_to_local %{{.*}}, %[[A]]

    %shmem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_copy_global_to_local_with_barriers
  tt.func public @async_copy_global_to_local_with_barriers(%ptr: tensor<128x128x!tt.ptr<f16>, #blocked>) {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared_mem : tensor<1xi64
    // CHECK-DAG: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 512 : i32} : !tt.ptr<i64>
    // CHECK-DAG: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK-DAG: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>

    // CHECK-DAG: %[[WRT_COMMITS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 16 : i32} : !tt.ptr<i8>

    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_check_outstanding_commits{{.*}}(%[[A:.*]], {{.*}}, %[[THREAD_BIT]], %[[BUFFERS]], %[[WRT_COMMITS_GLOB]]
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_stage_access_for_commit{{.*}}(%[[A]], {{.*}}, %[[THREAD_BIT]], %[[BUFFERS]], %[[WRT_COMMITS_GLOB]]
    // CHECK: ttg.async_copy_global_to_local %{{.*}}, %[[A]]
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_commit_group
  tt.func public @async_commit_group() {
    // CHECK: tt.call @__triton_consan_commit_accesses
    %shmem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_commit_group
  tt.func public @async_commit_group() {
    // CHECK: tti.experimental_lock_acquire
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 4295032833 : i64
    // CHECK: %[[OUTSTANDING_NUM:.*]] = arith.constant 42 : i32
    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_writes{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]], %[[OUTSTANDING_NUM]]
    %shmem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.async_wait {num = 42 : i32}
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tmem_load
  tt.func public @tmem_load() {
    %result = ttng.tmem_alloc  {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    ttng.tmem_load %result : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf16>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @warp_group_dot
  tt.func public @warp_group_dot(%acc: tensor<128x128xf16, #mma>) {
    // CHECK-DAG: %[[SM_BUFS:.*]] = tti.experimental_buffer_pointers [0, 32768], shared_mem : tensor<2xi64
    // CHECK-DAG: %[[SM_WGMMA_READS:.*]] = arith.constant dense<0> : tensor<2x16xi8
    // CHECK-DAG: %[[SM_WGMMA_WRITES_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 32 : i32} : !tt.ptr<i8>
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_stage_access_for_commit{{.*}}(%[[A:.*]], {{.*}}, %[[THREAD_BIT]], %[[SM_BUFS]], %[[SM_WGMMA_WRITES_GLOB]]
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_stage_access_for_commit{{.*}}(%[[B:.*]], {{.*}}, %[[THREAD_BIT]], %[[SM_BUFS]], %[[SM_WGMMA_WRITES_GLOB]]
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_commit_accesses{{.*}}(%[[THREAD_BIT]], {{.*}}, %[[SM_WGMMA_WRITES_GLOB]]
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %true = arith.constant true
    ttng.warp_group_dot %0, %1, %acc, %true {isAsync = true} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #mma>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @warp_group_dot_sync
  tt.func public @warp_group_dot_sync(%acc: tensor<128x128xf16, #mma>) {
    // CHECK-DAG: %[[SM_BUFS:.*]] = tti.experimental_buffer_pointers [0, 32768], shared_mem : tensor<2xi64
    // CHECK-DAG: %[[SM_WGMMA_READS:.*]] = arith.constant dense<0> : tensor<2x16xi8
    // CHECK-DAG: %[[SM_WGMMA_WRITES_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 32 : i32} : !tt.ptr<i8>

    // CHECK: "before_dot"
    // CHECK-NOT: tt.call @__triton_consan_stage_access_for_commit
    // CHECK-NOT: tt.call @__triton_consan_commit_accesses
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %true = arith.constant true
    "before_dot"() : () -> ()
    ttng.warp_group_dot %0, %1, %acc, %true {isAsync = false} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #mma>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @warp_group_dot_wait
  tt.func public @warp_group_dot_wait(%acc: tensor<128x128xf16, #mma>) {
    // Dummy buffer just to make the pass run
    %dummy = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_reads
    ttng.warp_group_dot_wait %acc { pendings = 42 : i32 } : tensor<128x128xf16, #mma>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @local_alloc_with_src
  tt.func public @local_alloc_with_src(%acc: tensor<128x128xf16, #mma>) {
    // CHECK: %[[BUF:.*]] = ttg.local_alloc
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}(%[[BUF]]
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}(%[[BUF]]
    %buf = ttg.local_alloc %acc {allocation.offset = 0 : i32} : (tensor<128x128xf16, #mma>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tmem_alloc_with_src
  tt.func public @tmem_alloc_with_src(%acc: tensor<128x128xf16, #blocked>) {
    // CHECK: %[[BUF:.*]] = ttng.tmem_alloc
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}(%[[BUF]]
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}(%[[BUF]]
    %buf = ttng.tmem_alloc %acc { tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32 } : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @local_load_barriers
  tt.func public @local_load_barriers() {
    // CHECK: tti.experimental_buffer_pointers
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_set_read_visibility
    // CHECK: tti.experimental_lock_release
    ttg.local_load %buf : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @local_load_barriers
  tt.func public @local_load_barriers_cp_async(%ptr: tensor<128x128x!tt.ptr<f16>, #blocked>) {
    // CHECK: tti.experimental_buffer_pointers
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>

    // CHECK: ttg.async_copy_global_to_local

    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits
    // CHECK: tt.call @__triton_consan_set_read_visibility
    // CHECK: tti.experimental_lock_release
    // CHECK: ttg.local_load
    ttg.local_load %buf : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @local_store_barriers_cp_async_wgmma
  tt.func public @local_store_barriers_cp_async_wgmma(%ptr: tensor<128x128x!tt.ptr<f16>, #blocked>, %acc: tensor<128x128xf16, #mma>) {
    // CHECK: tti.experimental_buffer_pointers
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>
    ttng.warp_group_dot %shmem, %shmem, %acc : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #mma>
    // CHECK: ttng.warp_group_dot

    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits
    // CHECK: tt.call @__triton_consan_set_write_visibility
    // CHECK: tt.call @__triton_consan_clear_write_tracking
    // CHECK: tt.call @__triton_consan_clear_read_visibility
    // CHECK: tt.call @__triton_consan_clear_read_tracking
    // CHECK: tti.experimental_lock_release
    // CHECK: ttg.local_store
    ttg.local_store %acc, %buf : tensor<128x128xf16, #mma> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @ws_allocation
  tt.func public @ws_allocation(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>) {
    // CHECK-DAG: tti.experimental_buffer_pointers [65536], shared_mem : tensor<1xi64,
    // CHECK-DAG: tti.experimental_buffer_pointers [0], shared_mem : tensor<1xi64
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 8590065666 : i64
    // CHECK: tt.call @__triton_consan_copy_write_visibility{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]]
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 8590065666 : i64
    // CHECK: tt.call @__triton_consan_copy_read_visibility{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]]
    ttg.warp_specialize(%smem, %bar) attributes {actualRegisters = array<i32: 480, 32>, allocation.offset = 512 : i32, requestedRegisters = array<i32: 32>, warpGroupStartIds = array<i32: 4>}
    default {
      // CHECK: tti.experimental_lock_acquire
      // CHECK: tt.call @__triton_consan_verify_write_visibility
      // CHECK: tt.call @__triton_consan_set_read_visibility
      // CHECK: tti.experimental_lock_release
      ttg.local_load %smem : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
      ttg.warp_yield
    }
    partition0(%arg1: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) num_warps(4) {
      // CHECK: partition0
      // CHECK-DAG: tti.experimental_buffer_pointers [65536], shared_mem : tensor<1xi64,
      // CHECK-DAG: tti.experimental_buffer_pointers [0], shared_mem : tensor<1xi64
      // CHECK: tti.experimental_lock_acquire
      // CHECK: tt.call @__triton_consan_verify_write_visibility
      // CHECK: tt.call @__triton_consan_set_read_visibility
      // CHECK: tti.experimental_lock_release
      ttg.local_load %arg1 : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
      ttg.warp_return
    } : (!ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @ws_buf_ptrs_default
  tt.func public @ws_buf_ptrs_default(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>) {
    // CHECK-DAG: tti.experimental_buffer_pointers [0, 32768, 65536, 0], shared_mem
    // CHECK-DAG: tti.experimental_buffer_pointers [65536], shared_mem
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 8590065666 : i64
    // CHECK: tt.call @__triton_consan_copy_write_visibility{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]]
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 8590065666 : i64
    // CHECK: tt.call @__triton_consan_copy_read_visibility{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]]
    ttg.warp_specialize(%smem, %bar) attributes {actualRegisters = array<i32: 480, 32>, allocation.offset = 512 : i32, requestedRegisters = array<i32: 32>, warpGroupStartIds = array<i32: 4>}
    default {
      %c0_i32 = arith.constant 0 : i32
      %1 = ttg.memdesc_index %smem[%c0_i32] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg1: !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) num_warps(4) {
      ttg.warp_return
    } : (!ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>) -> ()
    tt.return
  }
}

// -----


#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @ws_buf_ptrs_partition0
  tt.func public @ws_buf_ptrs_partition0(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>) {
    // CHECK-DAG: tti.experimental_buffer_pointers [0, 32768, 65536, 0], shared_mem
    // CHECK-DAG: tti.experimental_buffer_pointers [65536], shared_mem
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 8590065666 : i64
    // CHECK: tt.call @__triton_consan_copy_write_visibility{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]]
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 8590065666 : i64
    // CHECK: tt.call @__triton_consan_copy_read_visibility{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]]
    ttg.warp_specialize(%smem, %bar) attributes {actualRegisters = array<i32: 480, 32>, allocation.offset = 512 : i32, requestedRegisters = array<i32: 32>, warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%arg1: !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) num_warps(4) {
      %c0_i32 = arith.constant 0 : i32
      %1 = ttg.memdesc_index %arg1[%c0_i32] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @ws_wait_barrier
  tt.func public @ws_wait_barrier() {
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.warp_specialize(%smem, %bar) attributes {actualRegisters = array<i32: 480, 32>, allocation.offset = 512 : i32, requestedRegisters = array<i32: 32>, warpGroupStartIds = array<i32: 4>}
    default {
      // CHECK: tti.experimental_lock_acquire
      // CHECK: tt.call @__triton_consan_set_waiting
      // CHECK: %[[ACTIVE_MASK:.*]] = arith.constant 5 : i32
      // CHECK: tt.call @__triton_consan_check_all_active_waiting{{.*}}(%[[ACTIVE_MASK]], {{.*}}, {{.*}}, {{.*}})
      // CHECK: tti.experimental_lock_release
      %c0_i32 = arith.constant 0 : i32
      %true = arith.constant true
      ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg1: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) num_warps(4) {
      // CHECK: partition0
      // CHECK: tti.experimental_lock_acquire
      // CHECK: tt.call @__triton_consan_set_waiting
      // CHECK: %[[ACTIVE_MASK:.*]] = arith.constant 5 : i32
      // CHECK: tt.call @__triton_consan_check_all_active_waiting{{.*}}(%[[ACTIVE_MASK]], {{.*}}, {{.*}}, {{.*}})
      // CHECK: tti.experimental_lock_release
      %c0_i32 = arith.constant 0 : i32
      %true = arith.constant true
      ttng.wait_barrier %arg2, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>) -> ()
    tt.return
  }
}
