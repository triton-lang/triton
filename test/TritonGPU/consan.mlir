// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BUFS_L:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: #[[WRT_BARS_L:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
  // CHECK: @single_local_alloc
  tt.func public @single_local_alloc() {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared_mem : tensor<1xi64, #[[BUFS_L]]>
    // CHECK: %[[WRITE_STATE:.*]] = arith.constant dense<0> : tensor<1xi8, #[[BUFS_L]]>
    // CHECK: %[[WRT_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: tt.store {{.*}}, %[[WRITE_STATE]]
    // CHECK: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<1x1xi8, #[[WRT_BARS_L]]>
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
    // CHECK: tt.store %[[ADD1]], %[[WRITE_BARS]] : tensor<1x1x!tt.ptr<i8>, #[[WRT_BARS_L]]>
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
    // CHECK: %[[WRITE_STATE:.*]] = arith.constant dense<0> : tensor<2xi8,
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<2x1xi8,
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
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
    // CHECK: %[[WRITE_STATE:.*]] = arith.constant dense<0> : tensor<4xi8,
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 4 : i32} : !tt.ptr<i8>
    // CHECK: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<4x1xi8,
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
    // CHECK: %[[WRITE_STATE:.*]] = arith.constant dense<0> : tensor<4xi8,
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<4x1xi8,
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 4 : i32} : !tt.ptr<i8>
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
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BUFS_L:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: @tmem_alloc
  tt.func public @tmem_alloc() {
    // CHECK: %[[TMEM_BUFS:.*]] = tti.experimental_buffer_pointers [0], tensor_mem : tensor<1xi64, #[[BUFS_L]]>
    // CHECK: %[[TM_WRITE_STATE:.*]] = arith.constant dense<0> : tensor<1xi8,
    // CHECK: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [4096], shared_mem : tensor<1xi64, #[[BUFS_L]]>
    // CHECK: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<1x1xi8,
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
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
    // CHECK: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared_mem : tensor<1xi64
    // CHECK: %[[WRITE_STATE:.*]] = arith.constant dense<0> : tensor<1xi8,
    // CHECK: %[[WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[BARRIERS:.*]] = tti.experimental_buffer_pointers [65536], shared_mem : tensor<1xi64
    // CHECK: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<1x1xi8
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[READ_BARS:.*]] = arith.constant dense<0> : tensor<1x1xi8
    // CHECK: %[[READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_check_write_state {{.*}}{%[[BUFFERS]], %[[WRT_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>), %[[WRITE_STATE_GLOB]](tensor<1xi8, #{{.*}}>)}, %true pipelined false
    // CHECK: tti.experimental_check_read_barriers {{.*}}{%[[BUFFERS]], %[[READ_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_set_write_state {{.*}}{%[[BUFFERS]], %[[WRITE_STATE_GLOB]](tensor<1xi8, #{{.*}}>)}, %true pipelined false
    // CHECK: tti.experimental_commit_write_with_barrier {{.*}}{%[[BARRIERS]], %[[WRT_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>), %[[WRITE_STATE_GLOB]](tensor<1xi8, #{{.*}}>)}
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

    // CHECK: tti.experimental_check_write_state
    // CHECK: tti.experimental_check_outstanding_commits
    // CHECK-NOT: tti.experimental_check_read_barriers
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

    // CHECK: tti.experimental_check_write_state
    // CHECK: tti.experimental_check_read_barriers
    // CHECK: tti.experimental_set_write_state
    // CHECK: tti.experimental_commit_write_with_barrier
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

    // CHECK: tti.experimental_check_write_state
    // CHECK: tti.experimental_check_outstanding_commits
    // CHECK-NOT: tti.experimental_check_read_barriers
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
    // CHECK: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared_mem : tensor<1xi64, #blocked>
    // CHECK: %[[WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[BARRIERS:.*]] = tti.experimental_buffer_pointers [65536], shared_mem : tensor<1xi64, #blocked>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared_mem : tensor<1xi64, #blocked>
    // CHECK: %[[WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_pointers [65536], shared_mem : tensor<1xi64, #blocked>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
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
    // CHECK-DAG: %[[SM_BUFS:.*]] = tti.experimental_buffer_pointers [0, 32768], shared_mem : tensor<2xi64
    // CHECK: %[[SM_WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK-DAG: %[[TM_BUFS:.*]] = tti.experimental_buffer_pointers [0], tensor_mem : tensor<1xi64
    // CHECK: %[[TM_WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_pointers [65536], shared_mem : tensor<1xi64
    // CHECK: %[[SM_WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK: %[[SM_READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK: %[[TM_WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[TM_READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>

    // CHECK: tti.experimental_check_write_state %[[A:.*]]{%[[SM_BUFS]], %[[SM_WRT_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>), %[[SM_WRITE_STATE_GLOB]](tensor<2xi8, #{{.*}}>)}, %true pipelined false
    // CHECK: tti.experimental_set_read_barrier %[[A]], %[[BAR:.*]]{%[[SM_BUFS]], %[[BARRIERS]], %[[SM_READ_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_check_write_state %[[B:.*]]{%[[SM_BUFS]], %[[SM_WRT_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>), %[[SM_WRITE_STATE_GLOB]](tensor<2xi8, #{{.*}}>)}, %true pipelined false
    // CHECK: tti.experimental_set_read_barrier %[[B]], %[[BAR:.*]]{%[[SM_BUFS]], %[[BARRIERS]], %[[SM_READ_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_check_write_state %[[ACC:.*]]{%[[TM_BUFS]], %[[TM_WRT_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>), %[[TM_WRITE_STATE_GLOB]](tensor<1xi8, #{{.*}}>)}, %true pipelined true
    // CHECK: tti.experimental_check_read_barriers %[[ACC]]{%[[TM_BUFS]], %[[TM_READ_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_set_write_state %[[ACC]]{%[[TM_BUFS]], %[[TM_WRITE_STATE_GLOB]](tensor<1xi8, #{{.*}}>)}, {{.*}} pipelined true
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
    // CHECK: %[[SM_BUFS:.*]] = tti.experimental_buffer_pointers [32768], shared_mem : tensor<1xi64
    // CHECK: %[[SM_WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[TM_BUFS:.*]] = tti.experimental_buffer_pointers [0, 128], tensor_mem : tensor<2xi64
    // CHECK: %[[TM_WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK: %[[BARRIERS:.*]] = tti.experimental_buffer_pointers [65536], shared_mem : tensor<1xi64
    // CHECK: %[[SM_WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[SM_READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[TM_WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK: %[[TM_READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>

    // CHECK: tti.experimental_check_write_state %[[A:.*]]{%[[TM_BUFS]], %[[TM_WRT_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>), %[[TM_WRITE_STATE_GLOB]](tensor<2xi8, #{{.*}}>)}, %true pipelined false
    // CHECK: tti.experimental_set_read_barrier %[[A]], %[[BAR:.*]]{%[[TM_BUFS]], %[[BARRIERS]], %[[TM_READ_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_check_write_state %[[B:.*]]{%[[SM_BUFS]], %[[SM_WRT_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>), %[[SM_WRITE_STATE_GLOB]](tensor<1xi8, #{{.*}}>)}, %true pipelined false
    // CHECK: tti.experimental_set_read_barrier %[[B]], %[[BAR:.*]]{%[[SM_BUFS]], %[[BARRIERS]], %[[SM_READ_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_check_write_state %[[ACC:.*]]{%[[TM_BUFS]], %[[TM_WRT_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>), %[[TM_WRITE_STATE_GLOB]](tensor<2xi8, #{{.*}}>)}, %true pipelined true
    // CHECK: tti.experimental_check_read_barriers %[[ACC]]{%[[TM_BUFS]], %[[TM_READ_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_set_write_state %[[ACC]]{%[[TM_BUFS]], %[[TM_WRITE_STATE_GLOB]](tensor<2xi8, #{{.*}}>)}, {{.*}} pipelined true
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
  // CHECK-LABEL: @async_copy_global_to_local
  tt.func public @async_copy_global_to_local(%ptr: tensor<128x128x!tt.ptr<f16>, #blocked>) {
    // CHECK: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared_mem : tensor<1xi64
    // CHECK: %[[WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[WRITE_COMMITS:.*]] = arith.constant dense<0> : tensor<1xi8
    // CHECK: %[[WRT_COMMITS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>

    // CHECK-NOT: tti.experimental_check_write_state
    // CHECK-NOT: tti.experimental_check_read_barriers
    // CHECK: tti.experimental_check_outstanding_commits %[[A:.*]]{%[[BUFFERS]], %[[WRT_COMMITS_GLOB]](tensor<1xi8,
    // CHECK: tti.experimental_stage_access_for_commit %[[A]]{%[[BUFFERS]], %[[WRT_COMMITS_GLOB]](tensor<1xi8,
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
    // CHECK: %[[BUFFERS:.*]] = tti.experimental_buffer_pointers [0], shared_mem : tensor<1xi64
    // CHECK: %[[SM_WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[SM_WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[SM_READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    // CHECK: %[[WRT_COMMITS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>

    // CHECK: tti.experimental_check_write_state
    // CHECK: tti.experimental_check_outstanding_commits %[[A:.*]]{%[[BUFFERS]], %[[WRT_COMMITS_GLOB]](tensor<1xi8,
    // CHECK: tti.experimental_check_read_barriers
    // CHECK: tti.experimental_stage_access_for_commit %[[A]]{%[[BUFFERS]], %[[WRT_COMMITS_GLOB]](tensor<1xi8,
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
    // CHECK: tti.experimental_commit_accesses
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
    // CHECK: tti.experimental_clear_outstanding_commits{{.*}}, 42 : !tt.ptr<i8>
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
    // CHECK: tti.experimental_check_write_state
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
    // CHECK: %[[SM_BUFS:.*]] = tti.experimental_buffer_pointers [0, 32768], shared_mem : tensor<2xi64
    // CHECK: %[[WRITE_STATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>
    // CHECK: %[[SM_WGMMA_READS:.*]] = arith.constant dense<0> : tensor<2xi8
    // CHECK: %[[SM_WGMMA_WRITES_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>

    // CHECK: tti.experimental_stage_access_for_commit %[[A:.*]]{%[[SM_BUFS]], %[[SM_WGMMA_WRITES_GLOB]](tensor<2xi8, #{{.*}}>)}
    // CHECK: tti.experimental_stage_access_for_commit %[[B:.*]]{%[[SM_BUFS]], %[[SM_WGMMA_WRITES_GLOB]](tensor<2xi8, #{{.*}}>)}
    // CHECK: tti.experimental_commit_accesses{%[[SM_WGMMA_WRITES_GLOB]](tensor<2xi8, #{{.*}}>)}
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
    // CHECK: %[[SM_BUFS:.*]] = tti.experimental_buffer_pointers [0, 32768], shared_mem : tensor<2xi64
    // CHECK: %[[SM_WGMMA_READS:.*]] = arith.constant dense<0> : tensor<2xi8
    // CHECK: %[[SM_WGMMA_WRITES_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>

    // CHECK: "before_dot"
    // CHECK-NOT: tti.experimental_stage_access_for_commit
    // CHECK-NOT: tti.experimental_commit_accesses
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
    // CHECK: tti.experimental_clear_outstanding_commits
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
    // CHECK: tti.experimental_check_write_state %[[BUF]]
    // CHECK: tti.experimental_check_read_barriers %[[BUF]]
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
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tmem_alloc_with_src
  tt.func public @tmem_alloc_with_src(%acc: tensor<128x128xf16, #blocked>) {
    // CHECK: %[[BUF:.*]] = ttng.tmem_alloc
    // CHECK: tti.experimental_check_write_state %[[BUF]]
    // CHECK: tti.experimental_check_read_barriers %[[BUF]]
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
  // CHECK-LABEL: @local_load_no_checks
  tt.func public @local_load_no_checks() {
    // CHECK: tti.experimental_buffer_pointers
    // No barriers, no async ops, no checks
    // CHECK-NOT: tti.
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
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
  tt.func public @local_load_barriers() {
    // CHECK: tti.experimental_buffer_pointers
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
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

    // CHECK: tti.experimental_check_write_state
    // CHECK: tti.experimental_check_outstanding_commits
    // CHECK-NOT: tti.experimental_check_read_barriers
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

    // CHECK: tti.experimental_check_write_state
    // CHECK: tti.experimental_check_outstanding_commits
    // CHECK: tti.experimental_check_read_barriers
    // CHECK: tti.experimental_check_outstanding_commits
    // CHECK: ttg.local_store
    ttg.local_store %acc, %buf : tensor<128x128xf16, #mma> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    tt.return
  }
}
