// RUN: triton-opt %s -split-input-file -mlir-disable-threading -test-print-buffer-region -verify-diagnostics -o /dev/null

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @single_local_alloc() {
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // expected-remark @below {{Buffers: [0, 4096]}}
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }

  // expected-remark @below {{All Shared Regions: [0, 4096]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @multiple_local_allocs() {
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // expected-remark @below {{Buffers: [0, 4096]}}
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    // expected-remark @below {{Buffers: [4096, 4096]}}
    ttg.local_load %1 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }

  // expected-remark @below {{All Shared Regions: [0, 4096], [4096, 4096]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @memdesc_index_multiple_access(%idx: i32) {
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32x32xf32, #shared, #smem, mutable>
    %sub = ttg.memdesc_subslice %0 [3, 0, 0] : !ttg.memdesc<8x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<3x32x32xf32, #shared, #smem, mutable, 8x32x32>
    %view = ttg.memdesc_index %sub[%idx] : !ttg.memdesc<3x32x32xf32, #shared, #smem, mutable, 8x32x32> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // expected-remark @below {{Buffers: [12288, 4096], [16384, 4096], [20480, 4096]}}
    ttg.local_load %view : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }

  // expected-remark @below {{All Shared Regions: [12288, 4096], [16384, 4096], [20480, 4096]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#shared_holey = #ttg.shared_linear<{offset = [[0, 1], [0, 0], [1, 0], [2, 0]], block = []}, alignment = 16>
#shared_dense = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [1, 0], [2, 0]], block = []}, alignment = 16>
#shared_dense_half = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [1, 0]], block = []}, alignment = 16>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 128 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @shared_zero_bases_belong_to_allocation() {
    %alloc = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x2xi32, #shared_holey, #smem, mutable>
    // expected-remark @below {{Buffers: [0, 64]}}
    ttg.local_load %alloc : !ttg.memdesc<4x2xi32, #shared_holey, #smem, mutable> -> tensor<4x2xi32>
    %view = ttg.memdesc_reinterpret %alloc : !ttg.memdesc<4x2xi32, #shared_holey, #smem, mutable> -> !ttg.memdesc<4x4xi32, #shared_dense, #smem, mutable>
    // expected-remark @below {{Buffers: [0, 64]}}
    ttg.local_load %view : !ttg.memdesc<4x4xi32, #shared_dense, #smem, mutable> -> tensor<4x4xi32>
    tt.return
  }

  tt.func public @shared_reinterpret_smaller_holey_region() {
    %alloc = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x2xi32, #shared_holey, #smem, mutable>
    %view = ttg.memdesc_reinterpret %alloc : !ttg.memdesc<4x2xi32, #shared_holey, #smem, mutable> -> !ttg.memdesc<2x4xi32, #shared_dense_half, #smem, mutable>
    // expected-remark @below {{Buffers: [0, 32]}}
    ttg.local_load %view : !ttg.memdesc<2x4xi32, #shared_dense_half, #smem, mutable> -> tensor<2x4xi32>
    tt.return
  }

  tt.func public @shared_zero_bases_belong_to_subslices() {
    %alloc = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x2xi32, #shared_holey, #smem, mutable>
    %lower = ttg.memdesc_subslice %alloc [0, 0] : !ttg.memdesc<4x2xi32, #shared_holey, #smem, mutable> -> !ttg.memdesc<2x2xi32, #shared_holey, #smem, mutable, 4x2>
    %upper = ttg.memdesc_subslice %alloc [2, 0] : !ttg.memdesc<4x2xi32, #shared_holey, #smem, mutable> -> !ttg.memdesc<2x2xi32, #shared_holey, #smem, mutable, 4x2>
    // expected-remark @below {{Buffers: [0, 32]}}
    ttg.local_load %lower : !ttg.memdesc<2x2xi32, #shared_holey, #smem, mutable, 4x2> -> tensor<2x2xi32>
    // expected-remark @below {{Buffers: [32, 32]}}
    ttg.local_load %upper : !ttg.memdesc<2x2xi32, #shared_holey, #smem, mutable, 4x2> -> tensor<2x2xi32>
    tt.return
  }

  tt.func public @shared_zero_bases_pipeline_stages(%index: i32) {
    %alloc = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x4x2xi32, #shared_holey, #smem, mutable>
    %view = ttg.memdesc_index %alloc[%index] : !ttg.memdesc<2x4x2xi32, #shared_holey, #smem, mutable> -> !ttg.memdesc<4x2xi32, #shared_holey, #smem, mutable>
    // expected-remark @below {{Buffers: [0, 64], [64, 64]}}
    ttg.local_load %view : !ttg.memdesc<4x2xi32, #shared_holey, #smem, mutable> -> tensor<4x2xi32>
    tt.return
  }

  tt.func public @shared_zero_bases_pipeline_subslice() {
    %zero = arith.constant 0 : i32
    %alloc = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x4x2xi32, #shared_holey, #smem, mutable>
    %stage = ttg.memdesc_subslice %alloc [1, 0, 0] : !ttg.memdesc<2x4x2xi32, #shared_holey, #smem, mutable> -> !ttg.memdesc<1x4x2xi32, #shared_holey, #smem, mutable, 2x4x2>
    %view = ttg.memdesc_index %stage[%zero] : !ttg.memdesc<1x4x2xi32, #shared_holey, #smem, mutable, 2x4x2> -> !ttg.memdesc<4x2xi32, #shared_holey, #smem, mutable>
    // expected-remark @below {{Buffers: [64, 64]}}
    ttg.local_load %view : !ttg.memdesc<4x2xi32, #shared_holey, #smem, mutable> -> tensor<4x2xi32>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 3584 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @gapped_pipeline_stage_subslice_region() {
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<7x16x16xf16, #shared, #smem, mutable>
    %stages = ttg.memdesc_subslice %parent [3, 8, 0] : !ttg.memdesc<7x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<2x8x16xf16, #shared, #smem, mutable, 7x16x16>
    %view = ttg.memdesc_reinterpret %stages : !ttg.memdesc<2x8x16xf16, #shared, #smem, mutable, 7x16x16> -> !ttg.memdesc<8x16xf16, #shared, #smem, mutable>
    // expected-remark @below {{Buffers: [1792, 256]}}
    ttg.local_load %view : !ttg.memdesc<8x16xf16, #shared, #smem, mutable> -> tensor<8x16xf16, #blocked>
    tt.return
  }

  // expected-remark @below {{All Shared Regions: [1792, 256]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#padded = #ttg.padded_shared<[4:+2] {order = [1, 0], shape = [4, 4]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 184 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @padded_shared_reinterpret_region() {
    %alloc = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x4xf16, #padded, #smem, mutable>
    %view = ttg.memdesc_reinterpret %alloc : !ttg.memdesc<4x4xf16, #padded, #smem, mutable> -> !ttg.memdesc<4x4xbf16, #padded, #smem, mutable>
    // expected-remark @below {{Buffers: [0, 44]}}
    ttg.local_load %view : !ttg.memdesc<4x4xbf16, #padded, #smem, mutable> -> tensor<4x4xbf16>
    tt.return
  }

  tt.func public @padded_shared_buffer_regions() {
    %alloc = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x4xi32, #padded, #smem, mutable>
    // expected-remark @below {{Buffers: [0, 88]}}
    ttg.local_load %alloc : !ttg.memdesc<4x4xi32, #padded, #smem, mutable> -> tensor<4x4xi32>
    tt.return
  }

  tt.func public @padded_shared_subslice_regions() {
    %alloc = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x4xi32, #padded, #smem, mutable>
    %lower = ttg.memdesc_subslice %alloc [0, 0] : !ttg.memdesc<4x4xi32, #padded, #smem, mutable> -> !ttg.memdesc<2x4xi32, #padded, #smem, mutable, 4x4>
    %upper = ttg.memdesc_subslice %alloc [2, 0] : !ttg.memdesc<4x4xi32, #padded, #smem, mutable> -> !ttg.memdesc<2x4xi32, #padded, #smem, mutable, 4x4>
    // expected-remark @below {{Buffers: [0, 40]}}
    ttg.local_load %lower : !ttg.memdesc<2x4xi32, #padded, #smem, mutable, 4x4> -> tensor<2x4xi32>
    // expected-remark @below {{Buffers: [48, 40]}}
    ttg.local_load %upper : !ttg.memdesc<2x4xi32, #padded, #smem, mutable, 4x4> -> tensor<2x4xi32>
    tt.return
  }

  tt.func public @padded_shared_pipeline_stages(%index: i32) {
    %alloc = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x4x4xi32, #padded, #smem, mutable>
    %view = ttg.memdesc_index %alloc[%index] : !ttg.memdesc<2x4x4xi32, #padded, #smem, mutable> -> !ttg.memdesc<4x4xi32, #padded, #smem, mutable>
    // expected-remark @below {{Buffers: [0, 88], [96, 88]}}
    ttg.local_load %view : !ttg.memdesc<4x4xi32, #padded, #smem, mutable> -> tensor<4x4xi32>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @local_store_updates_region() {
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    // expected-remark @below {{Buffers: [0, 4096]}}
    ttg.local_store %cst, %0 : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }

  // expected-remark @below {{All Shared Regions: [0, 4096]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @tensor_memory_regions() {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %true = arith.constant true
    %tm = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-remark @below {{Buffers: [0, 128]}}
    ttng.tmem_load %tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32>
    // expected-remark @below {{Buffers: [0, 128]}}
    ttng.tmem_store %cst, %tm, %true : tensor<128x128xf32> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  tt.func public @tensor_memory_reinterpret_smaller_region() {
    %tm = ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %view = ttg.memdesc_reinterpret %tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    // expected-remark @below {{Buffers: [128, 64]}}
    ttng.tmem_load %view : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf16>
    tt.return
  }

  // expected-remark @below {{All Tensor Regions: [0, 128], [128, 64]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @tensor_memory_indexed(%idx: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %true = arith.constant true
    %tm = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<5x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %stages = ttng.tmem_subslice %tm {offset = 2 : i32, dim = 0 : i32} : !ttg.memdesc<5x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable, 5x128x128>
    %view = ttg.memdesc_index %stages[%idx] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable, 5x128x128> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-remark @below {{Buffers: [256, 128], [384, 128]}}
    ttng.tmem_load %view : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32>
    // expected-remark @below {{Buffers: [256, 128], [384, 128]}}
    ttng.tmem_store %cst, %view, %true : tensor<128x128xf32> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  // expected-remark @below {{All Tensor Regions: [256, 128], [384, 128]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @tensor_memory_subslice_interleaved() {
    %tm = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %sub = ttng.tmem_subslice %tm {offset = 32 : i32} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable, 64x128>
    // expected-remark @below {{Buffers: [1048576, 32]}}
    ttng.tmem_load %sub : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable, 64x128> -> tensor<64x32xf32>
    tt.return
  }

  // expected-remark @below {{All Tensor Regions: [1048576, 32]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }

}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @tensor_memory_row_subslice() {
    %tm = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %sub = ttng.tmem_subslice %tm {offset = 128 : i32, dim = 0 : i32} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    // expected-remark @below {{Buffers: [128, 128]}}
    ttng.tmem_load %sub : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 256x128> -> tensor<128x128xf32>
    tt.return
  }

  // expected-remark @below {{All Tensor Regions: [128, 128]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @tensor_memory_row_subslice_interleaved() {
    %tm = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %sub = ttng.tmem_subslice %tm {offset = 64 : i32, dim = 0 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    // expected-remark @below {{Buffers: [1048576, 256]}}
    ttng.tmem_load %sub : !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> tensor<64x256xf32>
    tt.return
  }

  // expected-remark @below {{All Tensor Regions: [1048576, 256]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @barrier_regions() {
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // expected-remark @below {{Buffers: [8192, 8]}}
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }

  // expected-remark @below {{All Barrier Regions: [8192, 8]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @barrier_indexed(%idx: i32) {
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    %view = ttg.memdesc_index %bar[%idx] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // expected-remark @below {{Buffers: [8192, 8], [8200, 8]}}
    ttng.init_barrier %view, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }

  // expected-remark @below {{All Barrier Regions: [8192, 8], [8200, 8]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @tcgen5_commit_barrier_regions() {
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // expected-remark @below {{Buffers: [8192, 8]}}
    ttng.tc_gen5_commit %bar : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }

  // expected-remark @below {{All Barrier Regions: [8192, 8]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @tmem_copy_regions() {
    %src = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %dst = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-remark @below {{Buffers: [0, 65536]}}
    ttng.tmem_copy %src, %dst : !ttg.memdesc<128x128xf32, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  // expected-remark @below {{All Shared Regions: [0, 65536]}}
  // expected-remark @below {{All Tensor Regions: [0, 128]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @cf_block_arg() {
    %alloc = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    cf.br ^use(%alloc : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>)
  ^use(%arg0: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>):
    // expected-remark @below {{Buffers: [16384, 4096]}}
    ttg.local_load %arg0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    cf.br ^exit
  ^exit:
    tt.return
  }

  // expected-remark @below {{All Shared Regions: [16384, 4096]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @cf_if_same_size(%cond: i1) {
    %alloc_then = ttg.local_alloc {allocation.offset = 20480 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %alloc_else = ttg.local_alloc {allocation.offset = 24576 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    cf.cond_br %cond, ^then(%alloc_then : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>), ^else(%alloc_else : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>)
  ^then(%arg_then: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>):
    cf.br ^merge(%arg_then : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>)
  ^else(%arg_else: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>):
    cf.br ^merge(%arg_else : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>)
  ^merge(%phi: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>):
    // expected-remark @below {{Buffers: [20480, 4096], [24576, 4096]}}
    ttg.local_load %phi : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    cf.br ^exit
  ^exit:
    tt.return
  }

  // expected-remark @below {{All Shared Regions: [20480, 4096], [24576, 4096]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @cf_memdesc_index_select(%cond: i1) {
    %alloc_multi = ttg.local_alloc {allocation.offset = 28672 : i32} : () -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    %alloc_simple = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    %view = ttg.memdesc_index %alloc_multi[%c0] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    cf.cond_br %cond, ^use_view(%view : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>), ^use_simple(%alloc_simple : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>)
  ^use_view(%arg_view: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>):
    cf.br ^merge(%arg_view : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>)
  ^use_simple(%arg_simple: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>):
    cf.br ^merge(%arg_simple : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>)
  ^merge(%phi: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>):
    // expected-remark @below {{Buffers: [4096, 4096], [28672, 4096], [32768, 4096]}}
    ttg.local_load %phi : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    cf.br ^exit
  ^exit:
    tt.return
  }

  // expected-remark @below {{All Shared Regions: [4096, 4096], [28672, 4096], [32768, 4096]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @cf_loop_carried() {
    %alloc = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %trip = arith.constant 1 : index
    cf.br ^loop(%alloc, %trip : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, index)
  ^loop(%arg_alloc: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, %iv: index):
    // expected-remark @below {{Buffers: [32768, 4096]}}
    ttg.local_load %arg_alloc : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cond = arith.cmpi eq, %iv, %c0 : index
    %next = arith.subi %iv, %c1 : index
    cf.cond_br %cond, ^exit, ^loop(%arg_alloc, %next : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, index)
  ^exit:
    tt.return
  }

  // expected-remark @below {{All Shared Regions: [32768, 4096]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @cf_pessimistic_join(%cond: i1, %incoming: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    %alloc = ttg.local_alloc {allocation.offset = 36864 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    cf.cond_br %cond, ^has_alloc(%alloc : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>), ^no_alloc(%incoming : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>)
  ^has_alloc(%arg: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>):
    cf.br ^merge(%arg : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>)
  ^no_alloc(%arg_in: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>):
    cf.br ^merge(%arg_in : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>)
  ^merge(%phi: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>):
    // expected-remark @below {{Buffers: [36864, 4096]}}
    ttg.local_load %phi : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }

  // expected-remark @below {{All Shared Regions: [36864, 4096]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @cf_overwrite_before_merge(%cond: i1) {
    %alloc_a = ttg.local_alloc {allocation.offset = 40960 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %alloc_b = ttg.local_alloc {allocation.offset = 45056 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    cf.cond_br %cond, ^path_a(%alloc_a : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>), ^path_b(%alloc_a : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>)
  ^path_a(%arg_a: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>):
    cf.br ^merge(%arg_a : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>)
  ^path_b(%arg_from_entry: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>):
    cf.br ^merge(%alloc_b : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>)
  ^merge(%phi: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>):
    // expected-remark @below {{Buffers: [40960, 4096], [45056, 4096]}}
    ttg.local_load %phi : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }

  // expected-remark @below {{All Shared Regions: [40960, 4096], [45056, 4096]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @select_shared_memory_regions(%cond: i1) {
    %alloc_a = ttg.local_alloc {allocation.offset = 57344 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %alloc_b = ttg.local_alloc {allocation.offset = 61440 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %selected = arith.select %cond, %alloc_a, %alloc_b : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // expected-remark @below {{Buffers: [57344, 4096], [61440, 4096]}}
    ttg.local_load %selected : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }

  // expected-remark @below {{All Shared Regions: [57344, 4096], [61440, 4096]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @select_tensor_memory_regions(%cond: i1) {
    %tm0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %tm1 = ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %selected = arith.select %cond, %tm0, %tm1 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-remark @below {{Buffers: [0, 128], [128, 128]}}
    ttng.tmem_load %selected : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32>
    tt.return
  }

  // expected-remark @below {{All Tensor Regions: [0, 128], [128, 128]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked_ws = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @warp_specialize_propagation() {
    %smem = ttg.local_alloc {allocation.offset = 49152 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 53248 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.warp_specialize(%smem, %bar) attributes {actualRegisters = array<i32: 64, 16>, allocation.offset = 512 : i32, requestedRegisters = array<i32: 16>, warpGroupStartIds = array<i32: 0>} default {
      // expected-remark @below {{Buffers: [49152, 4096]}}
      ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked_ws>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, %arg1: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) num_warps(4) {
      // expected-remark @below {{Buffers: [49152, 4096]}}
      ttg.local_load %arg0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked_ws>
      ttg.warp_return
    } : (!ttg.memdesc<32x32xf32, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>) -> ()
    tt.return
  }

  // expected-remark @below {{All Shared Regions: [49152, 4096]}}
  tt.func private @print_all_regions() attributes {test.print_all_used_regions} {
    tt.return
  }
}
