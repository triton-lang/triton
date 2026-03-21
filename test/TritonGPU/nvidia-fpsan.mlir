// RUN: split-file %s %t
// RUN: triton-opt %t/success.mlir -split-input-file -allow-unregistered-dialect -tritoninstrument-fp-sanitizer | FileCheck %t/success.mlir

//--- success.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tmem_load_store
  tt.func public @tmem_load_store() {
    // CHECK: ttg.global_scratch_alloc
    // CHECK: tt.store
    // CHECK: tt.load
    // CHECK-NOT: ttng.tmem_load
    // CHECK-NOT: ttng.tmem_store
    %true = arith.constant true
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %buf = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %zero, %buf, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %val = ttng.tmem_load %buf : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    ttng.tmem_store %val, %buf, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tmem_copy_subslice
  tt.func public @tmem_copy_subslice() {
    // CHECK: ttg.global_scratch_alloc
    // CHECK: ttng.arrive_barrier
    // CHECK-NOT: ttng.tmem_copy
    // CHECK-NOT: ttng.tmem_subslice
    %src = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %dst = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %sub = ttng.tmem_subslice %dst {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.tmem_copy %src, %sub, %bar : !ttg.memdesc<128x128xf32, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tmem_subslice_load_store
  tt.func public @tmem_subslice_load_store() {
    // CHECK: ttg.global_scratch_alloc
    // CHECK-NOT: ttng.tmem_subslice
    // CHECK-NOT: ttng.tmem_load
    // CHECK-NOT: ttng.tmem_store
    %true = arith.constant true
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %buf = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %sub = ttng.tmem_subslice %buf {N = 128 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    ttng.tmem_store %zero, %sub, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    %val = ttng.tmem_load %sub : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<128x128xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tcgen05_mma
  tt.func public @tcgen05_mma() {
    // CHECK: ttg.global_scratch_alloc
    // CHECK: scf.for
    // CHECK: ttng.arrive_barrier
    // CHECK-NOT: ttng.tc_gen5_mma
    %true = arith.constant true
    %a = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %d = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.tc_gen5_mma %a, %b, %d, %true, %true, %bar[%true] {is_async} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tcgen05_mma_pred
  tt.func public @tcgen05_mma_pred() {
    // CHECK: ttg.global_scratch_alloc
    // CHECK: scf.for
    // CHECK: ttng.arrive_barrier
    // CHECK-NOT: ttng.tc_gen5_mma
    %false = arith.constant false
    %a = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %d = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.tc_gen5_mma %a, %b, %d, %false, %false, %bar[%false] {is_async} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tmem_memdesc_index
  tt.func public @tmem_memdesc_index() {
    // CHECK: ttg.global_scratch_alloc
    // CHECK-NOT: ttng.tmem_load
    // CHECK-NOT: ttng.tmem_store
    // CHECK-NOT: ttg.memdesc_index
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %buf = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %view = ttg.memdesc_index %buf[%c0_i32] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %zero, %view, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %val = ttng.tmem_load %view : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tmem_memdesc_index_nonzero
  tt.func public @tmem_memdesc_index_nonzero() {
    // CHECK: ttg.global_scratch_alloc
    // CHECK-NOT: ttng.tmem_load
    // CHECK-NOT: ttng.tmem_store
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %buf = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %view = ttg.memdesc_index %buf[%c1_i32] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %zero, %view, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %val = ttng.tmem_load %view : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tmem_reinterpret
  tt.func public @tmem_reinterpret() {
    // CHECK: ttg.global_scratch_alloc
    // CHECK-NOT: ttng.tmem_load
    // CHECK-NOT: ttng.tmem_store
    %true = arith.constant true
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %buf = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %view = ttg.memdesc_reinterpret %buf : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %zero, %view, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %val = ttng.tmem_load %view : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @ws_partition_tmem_load
  tt.func public @ws_partition_tmem_load() {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc
    // CHECK: ttg.warp_specialize(%{{.*}}, %{{.*}}, %{{.*}}, %[[SCRATCH]])
    // CHECK: partition0(%{{.*}}: !ttg.memdesc<1xi64, #{{[^,>]+}}, #smem, mutable>, %{{.*}}: !ttg.memdesc<128x128xf32, #{{[^,>]+}}, #smem, mutable>, %{{.*}}: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %[[SCRATCH_ARG:.*]]: !tt.ptr<f32>) num_warps(4)
    // CHECK: %[[PTRS:.*]] = tt.splat %[[SCRATCH_ARG]] : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked>
    // CHECK: tt.load
    // CHECK: ttg.local_store
    // CHECK-NOT: ttng.tmem_load
    %bar = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %smem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %buf = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttg.warp_specialize(%bar, %smem, %buf) attributes {actualRegisters = array<i32: 32, 32>, allocation.offset = 512 : i32, requestedRegisters = array<i32: 32>, warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<128x128xf32, #shared, #smem, mutable>, %arg2: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(4) {
      %val = ttng.tmem_load %arg2 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      ttg.local_store %val, %arg1 : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) -> ()
    tt.return
  }
}
