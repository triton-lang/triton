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
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    %view = ttg.memdesc_index %0[%idx] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // expected-remark @below {{Buffers: [0, 4096], [4096, 4096]}}
    ttg.local_load %view : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
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

  // expected-remark @below {{All Tensor Regions: [0, 128]}}
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
    %tm = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %view = ttg.memdesc_index %tm[%idx] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-remark @below {{Buffers: [0, 128], [128, 128]}}
    ttng.tmem_load %view : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32>
    // expected-remark @below {{Buffers: [0, 128], [128, 128]}}
    ttng.tmem_store %cst, %view, %true : tensor<128x128xf32> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  // expected-remark @below {{All Tensor Regions: [0, 128], [128, 128]}}
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
