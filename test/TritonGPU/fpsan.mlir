// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-fp-sanitizer | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
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
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
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
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tmem_subslice_load_store
  tt.func public @tmem_subslice_load_store() {
    // CHECK: ttg.global_scratch_alloc
    // CHECK-NOT: ttng.tmem_subslice
    // CHECK-NOT: ttng.tmem_load
    // CHECK-NOT: ttng.tmem_store
    %true = arith.constant true
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %buf = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %sub = ttng.tmem_subslice %buf {N = 1 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %zero, %sub, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %val = ttng.tmem_load %sub : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
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
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
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
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
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
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
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

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot_operand_a = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot_operand_b = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @dot_emulation
  tt.func public @dot_emulation() -> tensor<16x16xf32, #blocked> {
    // CHECK: scf.for
    // CHECK-NOT: tt.dot
    // CHECK-NOT: ttg.convert_layout
    %cst = arith.constant 1.000000e+00 : f16
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %a = tt.splat %cst : f16 -> tensor<16x16xf16, #dot_operand_a>
    %b = tt.splat %cst : f16 -> tensor<16x16xf16, #dot_operand_b>
    %out = tt.dot %a, %b, %zero : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #blocked>
    tt.return %out : tensor<16x16xf32, #blocked>
  }
}

// -----

// CHECK-LABEL: @binary_ops
tt.func public @binary_ops(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: tt.bitcast
  // CHECK: arith.addi
  // CHECK: arith.subi
  // CHECK: arith.muli
  // CHECK-NOT: arith.addf
  // CHECK-NOT: arith.subf
  // CHECK-NOT: arith.mulf
  %add = arith.addf %a, %b : tensor<4xf32>
  %sub = arith.subf %a, %b : tensor<4xf32>
  %mul = arith.mulf %a, %b : tensor<4xf32>
  %sum = arith.addf %add, %sub : tensor<4xf32>
  %out = arith.mulf %sum, %mul : tensor<4xf32>
  tt.return %out : tensor<4xf32>
}

// -----

// CHECK-LABEL: @div_rem_ops
tt.func public @div_rem_ops(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: tt.bitcast
  // CHECK: arith.xori
  // CHECK: arith.muli
  // CHECK-NOT: arith.divf
  // CHECK-NOT: arith.remf
  %div = arith.divf %a, %b : tensor<4xf32>
  %rem = arith.remf %a, %b : tensor<4xf32>
  %out = arith.addf %div, %rem : tensor<4xf32>
  tt.return %out : tensor<4xf32>
}

// -----

// CHECK-LABEL: @fma_op
tt.func public @fma_op(%a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: arith.muli
  // CHECK: arith.addi
  // CHECK-NOT: math.fma
  %fma = math.fma %a, %b, %c : tensor<4xf32>
  tt.return %fma : tensor<4xf32>
}

// -----

// CHECK-LABEL: @unary_ops
tt.func public @unary_ops(%a: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: tt.bitcast
  // CHECK: arith.xori
  // CHECK: arith.xori
  // CHECK-NOT: math.exp
  // CHECK-NOT: math.log
  // CHECK-NOT: math.sqrt
  %e = math.exp %a : tensor<4xf32>
  %l = math.log %e : tensor<4xf32>
  %s = math.sqrt %l : tensor<4xf32>
  tt.return %s : tensor<4xf32>
}

// -----

// CHECK-LABEL: @cast_extf
tt.func public @cast_extf(%a: tensor<4xf16>) -> tensor<4xf32> {
  // CHECK: tt.bitcast
  // CHECK: arith.extui
  // CHECK: arith.shli
  // CHECK-NOT: arith.extf
  %0 = arith.extf %a : tensor<4xf16> to tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @cast_truncf
tt.func public @cast_truncf(%a: tensor<4xf32>) -> tensor<4xf16> {
  // CHECK: tt.bitcast
  // CHECK: arith.shrui
  // CHECK: arith.trunci
  // CHECK-NOT: arith.truncf
  %0 = arith.truncf %a : tensor<4xf32> to tensor<4xf16>
  tt.return %0 : tensor<4xf16>
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
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
