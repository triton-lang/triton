// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BUFS_L:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: #[[WRT_BARS_L:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
  // CHECK: @single_local_alloc
  tt.func public @single_local_alloc() {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared : tensor<1xi64, #[[BUFS_L]]>
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<1x1xi8, #[[WRT_BARS_L]]>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[SPLAT:.*]] = tt.splat %[[WRT_BARS_GLOB]] : !tt.ptr<i8> -> tensor<1x1x!tt.ptr<i8>, #[[WRT_BARS_L]]>
    // CHECK: %[[RANGE0:.*]] = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #ttg.slice<{dim = 1, parent = #[[WRT_BARS_L]]}>>
    // CHECK: %[[STRIDE0:.*]] = arith.constant dense<1> : tensor<1xi32, #ttg.slice<{dim = 1, parent = #[[WRT_BARS_L]]}>>
    // CHECK: %[[OFFS0:.*]] = arith.muli %[[RANGE0]], %[[STRIDE0]]
    // CHECK: %[[EXP0:.*]] = tt.expand_dims %[[OFFS0]] {axis = 1 : i32} : tensor<1xi32, #ttg.slice<{dim = 1, parent = #[[WRT_BARS_L]]}>> -> tensor<1x1xi32, #[[WRT_BARS_L]]>
    // CHECK: %[[ADD0:.*]] = tt.addptr %[[SPLAT]], %[[EXP0]] : tensor<1x1x!tt.ptr<i8>, #[[WRT_BARS_L]]>, tensor<1x1xi32, #[[WRT_BARS_L]]>
    // CHECK: %[[RANGE1:.*]] = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #ttg.slice<{dim = 0, parent = #[[WRT_BARS_L]]}>
    // CHECK: %[[STRIDE1:.*]] = arith.constant dense<1> : tensor<1xi32, #ttg.slice<{dim = 0, parent = #[[WRT_BARS_L]]}>
    // CHECK: %[[OFFS1:.*]] = arith.muli %[[RANGE1]], %[[STRIDE1]]
    // CHECK: %[[EXP1:.*]] = tt.expand_dims %[[OFFS1]] {axis = 0 : i32} : tensor<1xi32, #ttg.slice<{dim = 0, parent = #[[WRT_BARS_L]]}>> -> tensor<1x1xi32, #[[WRT_BARS_L]]>
    // CHECK: %[[ADD1:.*]] = tt.addptr %[[ADD0]], %[[EXP1]] : tensor<1x1x!tt.ptr<i8>, #[[WRT_BARS_L]]>, tensor<1x1xi32, #[[WRT_BARS_L]]>
    // CHECK: tt.store %[[ADD1]], %cst : tensor<1x1x!tt.ptr<i8>, #[[WRT_BARS_L]]>
    // CHECK: %[[WRITE_STATE:.*]] = arith.constant dense<0> : tensor<1xi8, #[[BUFS_L]]>
    // CHECK: %[[WRT_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: tt.store {{.*}}, %[[WRITE_STATE]]
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0, 4096], shared : tensor<2xi64,
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<2x1xi8,
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK: %[[WRITE_STATE:.*]] = arith.constant dense<0> : tensor<2xi8,
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0, 4096, 8192, 0], shared : tensor<4xi64,
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<4x1xi8,
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 4 : i32} : !tt.ptr<i8>
    // CHECK: %[[WRITE_STATE:.*]] = arith.constant dense<0> : tensor<4xi8,
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0, 4096, 8192, 0], shared : tensor<4xi64,
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<4x1xi8,
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 4 : i32} : !tt.ptr<i8>
    // CHECK: %[[WRITE_STATE:.*]] = arith.constant dense<0> : tensor<4xi8,
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<3x32x32xf32, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0, %c0_i32 : !ttg.memdesc<3x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
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
    %bar_sub = ttg.memdesc_index %bar, %c0 : !ttg.memdesc<4x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar_sub, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %buf_sub = ttg.memdesc_index %0, %c0 : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>

    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BUFS_L:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: @tmem_alloc
  tt.func public @tmem_alloc() {
    // CHECK-DAG: %[[TMEM_BUFS:.*]] = tti.experimental_buffer_pointers [0], tensor : tensor<1xi64, #[[BUFS_L]]>
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [4096], shared : tensor<1xi64, #[[BUFS_L]]>
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<1x1xi8,
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[WRITE_STATE:.*]] = arith.constant dense<0> : tensor<1xi8,
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared : tensor<1xi64
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_pointers [65536], shared : tensor<1xi64
    // CHECK: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<1x1xi8
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[WRITE_STATE:.*]] = arith.constant dense<0> : tensor<1xi8,
    // CHECK: %[[WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[READ_BARS:.*]] = arith.constant dense<0> : tensor<1x1xi8
    // CHECK: %[[READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[OUTSTANDING_COMMITS:.*]] = arith.constant dense<0> : tensor<1xi8
    // CHECK: %[[OUTSTANDING_COMMITS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_check_outstanding_writes {{.*}}{%[[BUFFERS]], %[[WRT_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>), %[[WRITE_STATE_GLOB]](tensor<1xi8, #{{.*}}>)}, %true pipelined false
    // CHECK: tti.experimental_check_write_commit {{.*}}{%[[BUFFERS]], %[[OUTSTANDING_COMMITS_GLOB]](tensor<1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_check_outstanding_reads {{.*}}{%[[BUFFERS]], %[[READ_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_mark_as_write {{.*}}{%[[BUFFERS]], %[[WRITE_STATE_GLOB]](tensor<1xi8, #{{.*}}>)}, %true pipelined false
    // CHECK: tti.experimental_commit_write_with_barrier {{.*}}{%[[BARRIERS]], %[[WRT_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>), %[[WRITE_STATE_GLOB]](tensor<1xi8, #{{.*}}>)}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %bar, %true : !tt.tensordesc<tensor<32x32xf32, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared : tensor<1xi64, #blocked>
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_pointers [65536], shared : tensor<1xi64, #blocked>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_clear_write_barrier {{.*}}{%[[BARRIERS]], %[[WRT_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>), %[[WRITE_STATE_GLOB]](tensor<1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_clear_read_barrier {{.*}}{%[[BARRIERS]], %[[READ_BARS_GLOB]](tensor<1x1xi8, #blocked1>)}
    ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared : tensor<1xi64, #blocked>
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_pointers [65536], shared : tensor<1xi64, #blocked>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_check_barrier_writes_cleared {{.*}}{%[[BARRIERS]], %[[WRT_BARS_GLOB]](tensor<1x1xi8, #blocked1>)}
    ttng.barrier_expect %bar, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tcgen5_mma
  tt.func public @tcgen5_mma(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>) {
    // CHECK-DAG: %[[SM_BUFS:.*]] = tti.experimental_buffer_pointers [0, 32768], shared : tensor<2xi64
    // CHECK-DAG: %[[TM_BUFS:.*]] = tti.experimental_buffer_pointers [0], tensor : tensor<1xi64
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_pointers [65536], shared : tensor<1xi64
    // CHECK: %[[SM_WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK: %[[SM_WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK: %[[SM_READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK: %[[TM_WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[TM_WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[TM_READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[WRT_COMMITS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>

    // CHECK: tti.experimental_check_outstanding_writes %[[A:.*]]{%[[SM_BUFS]], %[[SM_WRT_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>), %[[SM_WRITE_STATE_GLOB]](tensor<2xi8, #{{.*}}>)}, %true pipelined false
    // CHECK: tti.experimental_check_write_commit %[[A]]{%[[SM_BUFS]], %[[WRT_COMMITS_GLOB]](tensor<2xi8, #blocked>)}
    // CHECK: tti.experimental_mark_as_read %[[A]], %[[BAR:.*]]{%[[SM_BUFS]], %[[BARRIERS]], %[[SM_READ_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_check_outstanding_writes %[[B:.*]]{%[[SM_BUFS]], %[[SM_WRT_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>), %[[SM_WRITE_STATE_GLOB]](tensor<2xi8, #{{.*}}>)}, %true pipelined false
    // CHECK: tti.experimental_check_write_commit %[[B]]{%[[SM_BUFS]], %[[WRT_COMMITS_GLOB]](tensor<2xi8, #blocked>)}
    // CHECK: tti.experimental_mark_as_read %[[B]], %[[BAR:.*]]{%[[SM_BUFS]], %[[BARRIERS]], %[[SM_READ_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_check_outstanding_writes %[[ACC:.*]]{%[[TM_BUFS]], %[[TM_WRT_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>), %[[TM_WRITE_STATE_GLOB]](tensor<1xi8, #{{.*}}>)}, %true pipelined true
    // CHECK: tti.experimental_check_outstanding_reads %[[ACC]]{%[[TM_BUFS]], %[[TM_READ_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_mark_as_write %[[ACC]]{%[[TM_BUFS]], %[[TM_WRITE_STATE_GLOB]](tensor<1xi8, #{{.*}}>)}, {{.*}} pipelined true
    // CHECK: tti.experimental_commit_write_with_barrier {{.*}}{%[[BARRIERS]], %[[TM_WRT_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>), %[[TM_WRITE_STATE_GLOB]](tensor<1xi8, #{{.*}}>)}
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
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = false>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tcgen5_mma_lhs_in_tmem
  tt.func public @tcgen5_mma_lhs_in_tmem(%arg0: !tt.tensordesc<tensor<32x32xf32, #shared>>) {
    // CHECK-DAG: %[[SM_BUFS:.*]] = tti.experimental_buffer_pointers [32768], shared : tensor<1xi64
    // CHECK-DAG: %[[TM_BUFS:.*]] = tti.experimental_buffer_pointers [0, 128], tensor : tensor<2xi64
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_pointers [65536], shared : tensor<1xi64
    // CHECK: %[[SM_WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[SM_WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[SM_READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[TM_WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK: %[[TM_WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK: %[[TM_READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK: %[[WRT_COMMITS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>

    // CHECK: tti.experimental_check_outstanding_writes %[[A:.*]]{%[[TM_BUFS]], %[[TM_WRT_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>), %[[TM_WRITE_STATE_GLOB]](tensor<2xi8, #{{.*}}>)}, %true pipelined false
    // CHECK: tti.experimental_mark_as_read %[[A]], %[[BAR:.*]]{%[[TM_BUFS]], %[[BARRIERS]], %[[TM_READ_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_check_outstanding_writes %[[B:.*]]{%[[SM_BUFS]], %[[SM_WRT_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>), %[[SM_WRITE_STATE_GLOB]](tensor<1xi8, #{{.*}}>)}, %true pipelined false
    // CHECK: tti.experimental_check_write_commit %[[B]]{%[[SM_BUFS]], %[[WRT_COMMITS_GLOB]](tensor<1xi8, #blocked>)}
    // CHECK: tti.experimental_mark_as_read %[[B]], %[[BAR:.*]]{%[[SM_BUFS]], %[[BARRIERS]], %[[SM_READ_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_check_outstanding_writes %[[ACC:.*]]{%[[TM_BUFS]], %[[TM_WRT_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>), %[[TM_WRITE_STATE_GLOB]](tensor<2xi8, #{{.*}}>)}, %true pipelined true
    // CHECK: tti.experimental_check_outstanding_reads %[[ACC]]{%[[TM_BUFS]], %[[TM_READ_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_mark_as_write %[[ACC]]{%[[TM_BUFS]], %[[TM_WRITE_STATE_GLOB]](tensor<2xi8, #{{.*}}>)}, {{.*}} pipelined true
    // CHECK: tti.experimental_commit_write_with_barrier {{.*}}{%[[BARRIERS]], %[[TM_WRT_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>), %[[TM_WRITE_STATE_GLOB]](tensor<2xi8, #{{.*}}>)}
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

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_copy_local_to_global
  tt.func public @async_copy_local_to_global(%ptr: tensor<128x128x!tt.ptr<f16>, #blocked>) {
    // CHECK: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared : tensor<1xi64
    // CHECK: %[[WRITE_COMMITS:.*]] = arith.constant dense<0> : tensor<1xi8
    // CHECK: %[[WRT_COMMITS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>

    // CHECK-NOT: tti.experimental_check_outstanding_writes
    // CHECK-NOT: tti.experimental_check_outstanding_reads
    // CHECK: tti.experimental_check_write_commit %[[A:.*]]{%[[BUFFERS]], %[[WRT_COMMITS_GLOB]](tensor<1xi8,
    // CHECK: tti.experimental_stage_write_for_commit %[[A]]{%[[BUFFERS]], %[[WRT_COMMITS_GLOB]](tensor<1xi8,
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
  // CHECK-LABEL: @async_copy_local_to_global_with_barriers
  tt.func public @async_copy_local_to_global_with_barriers(%ptr: tensor<128x128x!tt.ptr<f16>, #blocked>) {
    // CHECK: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared : tensor<1xi64
    // CHECK: %[[SM_WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[SM_WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[SM_READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[WRT_COMMITS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>

    // CHECK: tti.experimental_check_outstanding_writes
    // CHECK: tti.experimental_check_write_commit %[[A:.*]]{%[[BUFFERS]], %[[WRT_COMMITS_GLOB]](tensor<1xi8,
    // CHECK: tti.experimental_check_outstanding_reads
    // CHECK: tti.experimental_stage_write_for_commit %[[A]]{%[[BUFFERS]], %[[WRT_COMMITS_GLOB]](tensor<1xi8,
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
    // CHECK: tti.experimental_commit_writes
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
    // CHECK: tti.experimental_clear_write_commits{{.*}}, 42 : !tt.ptr<i8>
    %shmem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.async_wait {num = 42 : i32}
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tmem_load
  tt.func public @tmem_load() {
    %result = ttng.tmem_alloc  {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_check_outstanding_writes
    ttng.tmem_load %result : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf16>
    tt.return
  }
}
