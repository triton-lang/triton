// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BUFS_L:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
  // CHECK: @single_local_alloc
  tt.func public @single_local_alloc() {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0>} : tensor<1xi64, #[[BUFS_L]]>
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<1xi64, #[[BUFS_L]]>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    // CHECK: %[[SPLAT:.*]] = tt.splat %[[WRT_BARS_GLOB]] : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>, #[[BUFS_L]]>
    // CHECK: %[[RANGE:.*]] = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #[[BUFS_L]]>
    // CHECK: %[[STRIDE0:.*]] = arith.constant dense<1> : tensor<1xi32, #[[BUFS_L]]>
    // CHECK: %[[OFFS0:.*]] = arith.muli %[[RANGE]], %[[STRIDE0]]
    // CHECK: %[[ADD:.*]] = tt.addptr %[[SPLAT]], %[[OFFS0]] : tensor<1x!tt.ptr<i64>, #[[BUFS_L]]>, tensor<1xi32, #[[BUFS_L]]>
    // CHECK: tt.store %[[ADD]], %cst : tensor<1x!tt.ptr<i64>, #[[BUFS_L]]>
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0, 4096>} : tensor<2xi64, #blocked>
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<2xi64, #blocked>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 16 : i32} : !tt.ptr<i64>
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
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
  // CHECK-LABEL: @three_local_alloc
  tt.func public @three_local_alloc() {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0, 4096, 8192, 0>} : tensor<4xi64, #blocked>
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<4xi64, #blocked>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 32 : i32} : !tt.ptr<i64>
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %2 = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
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
  // CHECK-LABEL: @three_sub_bufs
  tt.func public @three_sub_bufs() {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0, 4096, 8192, 0>} : tensor<4xi64, #blocked>
    // CHECK-DAG: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<4xi64, #blocked>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 32 : i32} : !tt.ptr<i64>
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<3x32x32xf32, #shared, #smem, mutable>
    %1 = ttg.memdesc_subview %0[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
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
    // CHECK: tt.store %[[PTR1]], %cst_1 : tensor<2x4x!tt.ptr<i8>, #[[READ_BARS_L]]>
    %c0 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<4x1xi64, #shared1, #smem, mutable>
    %bar_sub = ttg.memdesc_subview %bar[%c0, %c0] : !ttg.memdesc<4x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar_sub, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %buf_sub = ttg.memdesc_subview %0[%c0, %c0, %c0] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>

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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0>} : tensor<1xi64
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 65536>} : tensor<1xi64
    // CHECK: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<1xi64
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    // CHECK: %[[READ_BARS:.*]] = arith.constant dense<0> : tensor<1x1xi8
    // CHECK: %[[READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_check_outstanding_writes {{.*}}{%[[BUFFERS]], %[[WRT_BARS_GLOB]](tensor<1xi64, #{{.*}}>)}
    // CHECK: tti.experimental_check_outstanding_reads {{.*}}{%[[BUFFERS]], %[[READ_BARS_GLOB]](tensor<1x1xi8, #{{.*}}>)}
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0>} : tensor<1xi64, #blocked>
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 65536>} : tensor<1xi64, #blocked>
    // CHECK: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<1xi64, #blocked>
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 8 : i32} : !tt.ptr<i64>
    // CHECK: %[[READ_BARS:.*]] = arith.constant dense<0> : tensor<1x1xi8
    // CHECK: %[[READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 1 : i32} : !tt.ptr<i8>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_clear_write_barrier {{.*}}{%[[WRT_BARS_GLOB]](tensor<1xi64, #blocked>)}
    // CHECK: tti.experimental_clear_read_barrier {{.*}}{%[[BARRIERS]], %[[READ_BARS_GLOB]](tensor<1x1xi8, #blocked1>)}
    ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0, 32768>} : tensor<2xi64
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 65536>} : tensor<1xi64
    // CHECK: %[[WRITE_BARS:.*]] = arith.constant dense<0> : tensor<2xi64
    // CHECK: %[[WRT_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 16 : i32} : !tt.ptr<i64>
    // CHECK: %[[READ_BARS:.*]] = arith.constant dense<0> : tensor<2x1xi8
    // CHECK: %[[READ_BARS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 1 : i32, nbytes = 2 : i32} : !tt.ptr<i8>

    // CHECK: tti.experimental_check_outstanding_writes %[[A:.*]]{%[[BUFFERS]], %[[WRT_BARS_GLOB]](tensor<2xi64, #{{.*}}>)}
    // CHECK: tti.experimental_mark_as_read %[[A]], %[[BAR:.*]]{%[[BUFFERS]], %[[BARRIERS]], %[[READ_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>)}
    // CHECK: tti.experimental_check_outstanding_writes %[[B:.*]]{%[[BUFFERS]], %[[WRT_BARS_GLOB]](tensor<2xi64, #{{.*}}>)}
    // CHECK: tti.experimental_mark_as_read %[[B]], %[[BAR:.*]]{%[[BUFFERS]], %[[BARRIERS]], %[[READ_BARS_GLOB]](tensor<2x1xi8, #{{.*}}>)}
    // CHECK: ttng.tc_gen5_mma %[[A]], %[[B]], {{.*}}, {{.*}}, {{.*}}, %[[BAR]]
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %true = arith.constant true
    ttng.tc_gen5_mma %0, %1, %result[], %true, %true, %bar[%true] {is_async} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
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
    // CHECK: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0>} : tensor<1xi64
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
    // CHECK: %[[BUFFERS:.*]] = tti.experimental_shared_buffer_pointers {offsets = array<i32: 0>} : tensor<1xi64
    // CHECK: %[[WRITE_COMMITS:.*]] = arith.constant dense<0> : tensor<1xi8
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
