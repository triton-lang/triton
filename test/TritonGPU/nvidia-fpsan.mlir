// RUN: split-file %s %t
// RUN: triton-opt %t/success.mlir -split-input-file -allow-unregistered-dialect -tritoninstrument-fp-sanitizer -triton-nvidia-check-matmul-two-cta | FileCheck %t/success.mlir

//--- success.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_reduce = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#red = #ttg.slice<{dim = 1, parent = #blocked_reduce}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tmem_load_store
  tt.func public @tmem_load_store() {
    // CHECK: ttg.global_scratch_alloc
    // CHECK: tt.store
    // CHECK-NEXT: ttg.barrier global_read|global_write
    // CHECK: tt.load
    // CHECK-NOT: ttng.tmem_load
    // CHECK-NOT: ttng.tmem_store
    %true = arith.constant true
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %buf = ttng.tmem_alloc %zero {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %val = ttng.tmem_load %buf : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    ttng.tmem_store %val, %buf, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @tmem_store_predicate(
  // CHECK-SAME: %[[PRED_ARG:.*]]: i1
  tt.func public @tmem_store_predicate(%pred: i1) {
    // CHECK: %[[OLD:.*]] = tt.load
    // CHECK: %[[PRED:.*]] = tt.splat %[[PRED_ARG]]
    // CHECK: %[[SELECTED:.*]] = arith.select %[[PRED]], %{{.*}}, %[[OLD]]
    // CHECK: tt.store {{.*}}, %[[SELECTED]]
    // CHECK-NOT: ttng.tmem_store
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %one = arith.constant dense<1.0> : tensor<128x128xf32, #blocked>
    %buf = ttng.tmem_alloc %zero {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %one, %buf, %pred : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @tmem_load_reduce
  tt.func public @tmem_load_reduce() -> tensor<128xf32, #red> {
    // CHECK: %[[LOADED:.*]] = tt.load
    // CHECK: %[[VALUE:.*]] = tti.experimental_fpsan_unembed %[[LOADED]]
    // CHECK: %[[ABS:.*]] = math.absf %[[VALUE]]
    // CHECK: %[[PAYLOAD:.*]] = tti.experimental_fpsan_embed %[[ABS]]
    // CHECK: %[[REDUCED:.*]] = "tt.reduce"(%[[PAYLOAD]]) <{axis = 1 : i32}>
    // CHECK: arith.maxsi
    // CHECK: tt.reduce.return
    // CHECK: %[[RED_VALUE:.*]] = tti.experimental_fpsan_unembed %[[REDUCED]]
    // CHECK: tt.return %[[RED_VALUE]]
    // CHECK-NOT: ttng.tmem_load
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked_reduce>
    %buf = ttng.tmem_alloc %zero {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked_reduce>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %result, %red = ttng.tmem_load %buf {redOp = #ttng.redOp<max>, abs = true, NaN = true} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_reduce>, tensor<128xf32, #red>
    tt.return %red : tensor<128xf32, #red>
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
    // CHECK-NOT: ttng.tmem_copy
    // CHECK-NOT: ttng.tmem_subslice
    %src = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %dst = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %sub = ttng.tmem_subslice %dst {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_copy %src, %sub : !ttg.memdesc<128x128xf32, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
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
    // CHECK: ttg.barrier global_read|global_write
    // CHECK-NEXT: scf.for
    // CHECK: tti.experimental_local_gather
    // CHECK: tti.experimental_local_gather
    // CHECK-COUNT-32: tti.dot_i8
    // CHECK-NOT: tti.dot_i8
    // CHECK: tt.store
    // CHECK: ttg.barrier global_read|global_write
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
#tmem_a = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_d = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tcgen05_mma_tmem_a_shared_b
  tt.func public @tcgen05_mma_tmem_a_shared_b() {
    // CHECK: ttg.global_scratch_alloc
    // CHECK: tt.store
    // CHECK: ttg.barrier global_read|global_write
    // CHECK: tti.experimental_local_gather
    // CHECK: tti.dot_i8
    // CHECK-NOT: ttng.tc_gen5_mma
    %true = arith.constant true
    %a = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem_a, #ttng.tensor_memory, mutable>
    %b = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %d = ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem_d, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.tc_gen5_mma %a, %b, %d, %true, %true, %bar[%true] {is_async} : !ttg.memdesc<128x128xf16, #tmem_a, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem_d, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
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
    // CHECK: ttg.barrier global_read|global_write
    // CHECK-NEXT: scf.for
    // CHECK: tt.store
    // CHECK: ttg.barrier global_read|global_write
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

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#sharedT = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tcgen05_mma_scaled
  tt.func public @tcgen05_mma_scaled() {
    // CHECK-COUNT-3: ttg.global_scratch_alloc
    // CHECK-NOT: ttg.global_scratch_alloc
    // CHECK: ttg.barrier global_read|global_write
    // CHECK-NEXT: scf.for
    // CHECK: tti.experimental_local_gather {{.*}}#ttg.dot_op<{{.*}}opIdx = 0{{.*}}kWidth = 2
    // CHECK: tti.experimental_local_gather {{.*}}#ttg.dot_op<{{.*}}opIdx = 1{{.*}}kWidth = 2
    // CHECK-COUNT-32: tti.dot_i8
    // CHECK-NOT: tti.dot_i8
    // CHECK: ttg.barrier global_read|global_write
    // CHECK: tt.store
    // CHECK: ttg.barrier global_read|global_write
    // CHECK: ttng.arrive_barrier
    // CHECK-NOT: ttng.tc_gen5_mma_scaled
    %true = arith.constant true
    %a = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x256xi8, #shared, #ttg.shared_memory, mutable>
    %b = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<256x64xi8, #sharedT, #ttg.shared_memory, mutable>
    %d = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %a_scale = ttng.tmem_alloc : () -> !ttg.memdesc<128x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory, mutable>
    %b_scale = ttng.tmem_alloc : () -> !ttg.memdesc<64x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    ttng.tc_gen5_mma_scaled %a, %b, %d, %a_scale, %b_scale, %true, %true lhs = e2m1 rhs = e2m1, %bar[%true] {is_async} : !ttg.memdesc<128x256xi8, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<256x64xi8, #sharedT, #ttg.shared_memory, mutable>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
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
    // CHECK: partition0(%{{.*}}: !ttg.memdesc<1xi64, #{{[^,>]+}}, #smem, mutable>, %{{.*}}: !ttg.memdesc<128x128xf32, #{{[^,>]+}}, #smem, mutable>, %{{.*}}: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %[[SCRATCH_ARG:.*]]: !tt.ptr<i32>) num_warps(4)
    // CHECK: %[[PTRS:.*]] = tt.splat %[[SCRATCH_ARG]] : !tt.ptr<i32> -> tensor<128x128x!tt.ptr<i32>, #blocked>
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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, CGALayout = [[1, 0]], twoCTAs = true>
// CHECK: module attributes {{.*}}"ttng.two-ctas" = true
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 1 : i32} {
    // CHECK-LABEL: @tcgen05_mma_two_ctas
  tt.func public @tcgen05_mma_two_ctas() {
    // CHECK: ttg.global_scratch_alloc {{.*}}shared_cluster_state
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: scf.for
    // CHECK: tti.experimental_local_gather
    // CHECK: tti.experimental_local_gather
    // CHECK: tt.store
    // CHECK: ttg.barrier global_read|global_write
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: ttng.arrive_barrier
    // CHECK-NOT: ttng.tc_gen5_mma
    %true = arith.constant true
    %a = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<256x128xf16, #shared_a, #smem, mutable>
    %b = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared_b, #smem, mutable>
    %d = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.tc_gen5_mma %a, %b, %d, %true, %true, %bar[%true] {is_async, two_ctas} : !ttg.memdesc<256x128xf16, #shared_a, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared_b, #smem, mutable>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#shared_copy = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4], [0, 8], [0, 16]], block = [[0, 0]]}, alignment = 16>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory
#blocked_multibuffer = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#blocked_copy = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[0, 0]]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, CGALayout = [[1, 0]], twoCTAs = true>
#tmem_copy_alias = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, CGALayout = [[0, 0]]>
#tmem_scales = #ttng.tensor_memory_scales_encoding<CGALayout = [[0, 0]]>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @enable_two_ctas() {
    %true = arith.constant true
    %a = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<256x128xf16, #shared_a, #smem, mutable>
    %b = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared_b, #smem, mutable>
    %d = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a, %b, %d, %true, %true {two_ctas} : !ttg.memdesc<256x128xf16, #shared_a, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared_b, #smem, mutable>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @tmem_multibuffer_two_ctas
  tt.func public @tmem_multibuffer_two_ctas(%idx: i32) {
    // CHECK: ttg.global_scratch_alloc {{.*}}shared_cluster_state
    // CHECK: tt.store {{.*}} {ignore_cta} : tensor<256x128x!tt.ptr<i32>
    // CHECK-NOT: ttng.tmem_load
    // CHECK-NOT: ttng.tmem_store
    // CHECK-NOT: ttg.memdesc_index
    %true = arith.constant true
    %zero = arith.constant dense<0.0> : tensor<256x128xf32, #blocked_multibuffer>
    %buf = ttng.tmem_alloc : () -> !ttg.memdesc<2x256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %view = ttg.memdesc_index %buf[%idx] : !ttg.memdesc<2x256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %zero, %view, %true : tensor<256x128xf32, #blocked_multibuffer> -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %val = ttng.tmem_load %view : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #blocked_multibuffer>
    tt.return
  }

  // CHECK-LABEL: @tmem_copy_commit_two_ctas
  tt.func public @tmem_copy_commit_two_ctas() -> tensor<128x128xi8, #blocked_copy> {
    // CHECK: ttg.global_scratch_alloc {{.*}}nbytes = 4096{{.*}}shared_cluster_state
    // CHECK-NOT: ttng.tmem_copy
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: {{.*}} = ttg.local_load
    // CHECK: tt.store
    // CHECK: ttg.barrier global_read|global_write
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK: ttg.barrier global_read|global_write
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK: ttng.arrive_barrier
    // CHECK: tt.load
    // CHECK: tt.trans
    // CHECK: tt.broadcast
    // CHECK: ttg.convert_layout
    // CHECK-NOT: ttng.tmem_load
    // CHECK-NOT: ttng.tc_gen5_commit
    // CHECK-NOT: ttg.global_scratch_alloc
    // CHECK-NOT: tt.store
    %true = arith.constant true
    %src = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x32xi8, #shared_copy, #smem, mutable>
    %dst = ttng.tmem_alloc : () -> !ttg.memdesc<128x32xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %alias = ttg.memdesc_reinterpret %dst : !ttg.memdesc<128x32xi8, #tmem_scales, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xi8, #tmem_copy_alias, #ttng.tensor_memory, mutable>
    ttng.tmem_copy %src, %dst : !ttg.memdesc<128x32xi8, #shared_copy, #smem, mutable>, !ttg.memdesc<128x32xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %bar, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %val = ttng.tmem_load %alias : !ttg.memdesc<128x128xi8, #tmem_copy_alias, #ttng.tensor_memory, mutable> -> tensor<128x128xi8, #blocked_copy>
    tt.return %val : tensor<128x128xi8, #blocked_copy>
  }
}

// -----

#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, CGALayout = [[1, 0]]}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, CGALayout = [[0, 1]]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, CGALayout = [[1, 0]], twoCTAs = true>
#tmem_scale_a = #ttng.tensor_memory_scales_encoding<CGALayout = [[1, 0]]>
#tmem_scale_b = #ttng.tensor_memory_scales_encoding<CGALayout = [[0, 0]]>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tcgen05_mma_scaled_two_ctas
  tt.func public @tcgen05_mma_scaled_two_ctas() {
    // CHECK-COUNT-3: ttg.global_scratch_alloc {{.*}}shared_cluster_state
    // CHECK-NOT: ttg.global_scratch_alloc
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: scf.for
    // CHECK: tti.experimental_local_gather {{.*}}#ttg.dot_op<{{.*}}opIdx = 0{{.*}}kWidth = 2
    // CHECK: tti.experimental_local_gather {{.*}}#ttg.dot_op<{{.*}}opIdx = 1{{.*}}kWidth = 2
    // CHECK: tt.store
    // CHECK: ttg.barrier global_read|global_write
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: ttng.arrive_barrier
    // CHECK-NOT: ttng.tc_gen5_mma_scaled
    %true = arith.constant true
    %a = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<256x256xi8, #shared_a, #smem, mutable>
    %b = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<256x128xi8, #shared_b, #smem, mutable>
    %d = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %a_scale = ttng.tmem_alloc : () -> !ttg.memdesc<256x8xf8E4M3FN, #tmem_scale_a, #ttng.tensor_memory, mutable>
    %b_scale = ttng.tmem_alloc : () -> !ttg.memdesc<128x8xf8E4M3FN, #tmem_scale_b, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.tc_gen5_mma_scaled %a, %b, %d, %a_scale, %b_scale, %true, %true lhs = e2m1 rhs = e2m1, %bar[%true] {is_async, two_ctas} : !ttg.memdesc<256x256xi8, #shared_a, #smem, mutable>, !ttg.memdesc<256x128xi8, #shared_b, #smem, mutable>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x8xf8E4M3FN, #tmem_scale_a, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xf8E4M3FN, #tmem_scale_b, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}
