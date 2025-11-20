// RUN: triton-opt %s -mlir-disable-threading -test-print-buffer-region -verify-diagnostics -o /dev/null

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @single_local_alloc() {
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // expected-remark @below {{Shared: [0, 4096]}}
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }

  tt.func public @multiple_local_allocs() {
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // expected-remark @below {{Shared: [0, 4096]}}
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    // expected-remark @below {{Shared: [4096, 4096]}}
    ttg.local_load %1 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }

  tt.func public @memdesc_index_multiple_access(%idx: i32) {
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    %view = ttg.memdesc_index %0[%idx] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // expected-remark @below {{Shared: [4096, 4096], [0, 4096]}}
    ttg.local_load %view : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }

  tt.func public @local_store_updates_region() {
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    // expected-remark @below {{Shared: [0, 4096]}}
    ttg.local_store %cst, %0 : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }

  tt.func public @tensor_memory_regions() {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %true = arith.constant true
    %tm = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-remark @below {{Tensor: [0, 128]}}
    ttng.tmem_load %tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32>
    // expected-remark @below {{Tensor: [0, 128]}}
    ttng.tmem_store %cst, %tm, %true : tensor<128x128xf32> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  tt.func public @tensor_memory_indexed(%idx: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %true = arith.constant true
    %tm = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %view = ttg.memdesc_index %tm[%idx] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-remark @below {{Tensor: [128, 128], [0, 128]}}
    ttng.tmem_load %view : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32>
    // expected-remark @below {{Tensor: [128, 128], [0, 128]}}
    ttng.tmem_store %cst, %view, %true : tensor<128x128xf32> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  tt.func public @barrier_regions() {
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // expected-remark @below {{Barrier: [8192, 8]}}
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }

  tt.func public @barrier_indexed(%idx: i32) {
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    %view = ttg.memdesc_index %bar[%idx] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // expected-remark @below {{Barrier: [8192, 8], [8200, 8]}}
    ttng.init_barrier %view, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}
