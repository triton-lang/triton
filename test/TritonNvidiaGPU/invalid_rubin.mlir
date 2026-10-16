// RUN: triton-opt --split-input-file %s --verify-diagnostics

// Test: tmem_load_sparsify source must be f32
#cdata_sparse = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata_sparse = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_sparse = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_load_sparsify_requires_f32() {
    %src = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem_sparse, #ttng.tensor_memory, mutable>
    // expected-error @below {{source element type must be f32}}
    %cdata, %mdata = ttng.tmem_load_sparsify %src {rowOp = #ttng.rowOp<max>} :
      !ttg.memdesc<128x128xi32, #tmem_sparse, #ttng.tensor_memory, mutable> ->
      tensor<128x64xf32, #cdata_sparse>, tensor<128x4xi32, #mdata_sparse>
    tt.return
  }
}

// -----

// Test: tmem_load_sparsify requires N >= 4
#cdata_sparse = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata_sparse = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_sparse = #ttng.tensor_memory_encoding<blockM = 128, blockN = 2, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_load_sparsify_requires_min_n() {
    %src = ttng.tmem_alloc : () -> !ttg.memdesc<128x2xf32, #tmem_sparse, #ttng.tensor_memory, mutable>
    // expected-error @below {{N must be >= 4}}
    %cdata, %mdata = ttng.tmem_load_sparsify %src {rowOp = #ttng.rowOp<max>} :
      !ttg.memdesc<128x2xf32, #tmem_sparse, #ttng.tensor_memory, mutable> ->
      tensor<128x1xf32, #cdata_sparse>, tensor<128x1xi32, #mdata_sparse>
    tt.return
  }
}

// -----

// Test: tmem_load_sparsify validates cdata shape
#cdata_sparse_bad = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata_sparse = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_sparse = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_load_sparsify_bad_cdata_shape() {
    %src = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem_sparse, #ttng.tensor_memory, mutable>
    // expected-error @below {{cdata shape must be [M, N/2]}}
    %cdata, %mdata = ttng.tmem_load_sparsify %src {rowOp = #ttng.rowOp<max>} :
      !ttg.memdesc<128x128xf32, #tmem_sparse, #ttng.tensor_memory, mutable> ->
      tensor<128x32xf32, #cdata_sparse_bad>, tensor<128x4xi32, #mdata_sparse>
    tt.return
  }
}

// -----

// Test: tmem_load_sparsify validates mdata element type
#cdata_sparse = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata_sparse_bad = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_sparse = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_load_sparsify_bad_mdata_type() {
    %src = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem_sparse, #ttng.tensor_memory, mutable>
    // expected-error @below {{mdata element type must be i32}}
    %cdata, %mdata = ttng.tmem_load_sparsify %src {rowOp = #ttng.rowOp<max>} :
      !ttg.memdesc<128x128xf32, #tmem_sparse, #ttng.tensor_memory, mutable> ->
      tensor<128x64xf32, #cdata_sparse>, tensor<128x4xf32, #mdata_sparse_bad>
    tt.return
  }
}

// -----

// Test: tmem_load_sparsify NaN requires reduction result
#cdata_sparse = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata_sparse = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_sparse = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_load_sparsify_nan_requires_red() {
    %src = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem_sparse, #ttng.tensor_memory, mutable>
    // expected-error @below {{'NaN' requires reduction result}}
    %cdata, %mdata = ttng.tmem_load_sparsify %src {rowOp = #ttng.rowOp<max>, NaN = true} :
      !ttg.memdesc<128x128xf32, #tmem_sparse, #ttng.tensor_memory, mutable> ->
      tensor<128x64xf32, #cdata_sparse>, tensor<128x4xi32, #mdata_sparse>
    tt.return
  }
}

// -----

// Test: tmem_load_sparsify token form is rejected
#cdata_sparse = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata_sparse = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_sparse = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_load_sparsify_token_rejected() {
    %src, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem_sparse, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // expected-error @below {{tmem_load_sparsify does not support tokens yet; token support will be added when RemoveTMEMTokens pass is updated}}
    %cdata, %mdata, %tok2 = ttng.tmem_load_sparsify %src[%tok] {rowOp = #ttng.rowOp<max>} :
      !ttg.memdesc<128x128xf32, #tmem_sparse, #ttng.tensor_memory, mutable> ->
      tensor<128x64xf32, #cdata_sparse>, tensor<128x4xi32, #mdata_sparse>
    tt.return
  }
}

// -----

// Test: sp_decompress cdata type must be f16/bf16/fp8
#cdata_sp = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata_sp = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#dense_sp = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sp_decompress_requires_f16_bf16() {
    %cdata = arith.constant dense<0.0> : tensor<128x64xf32, #cdata_sp>
    %mdata = arith.constant dense<0> : tensor<128x4xi32, #mdata_sp>
    // expected-error @below {{cdata element type must be f16, bf16, float8_e5m2, or float8_e4m3fn}}
    %dense = ttng.sp_decompress %cdata, %mdata :
      tensor<128x64xf32, #cdata_sp>, tensor<128x4xi32, #mdata_sp> -> tensor<128x128xf32, #dense_sp>
    tt.return
  }
}

// -----

// Test: sp_decompress accepts fp8 dtypes
#cdata_sp = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata_sp = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#dense_sp = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sp_decompress_accepts_fp8() {
    %cdata_e5 = arith.constant dense<0.0> : tensor<128x64xf8E5M2, #cdata_sp>
    %cdata_e4 = arith.constant dense<0.0> : tensor<128x64xf8E4M3FN, #cdata_sp>
    %mdata = arith.constant dense<0> : tensor<128x4xi32, #mdata_sp>
    %dense_e5 = ttng.sp_decompress %cdata_e5, %mdata :
      tensor<128x64xf8E5M2, #cdata_sp>, tensor<128x4xi32, #mdata_sp> -> tensor<128x128xf8E5M2, #dense_sp>
    %dense_e4 = ttng.sp_decompress %cdata_e4, %mdata :
      tensor<128x64xf8E4M3FN, #cdata_sp>, tensor<128x4xi32, #mdata_sp> -> tensor<128x128xf8E4M3FN, #dense_sp>
    tt.return
  }
}

// -----

// Test: sp_decompress mdata type must be i32
#cdata_sp = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata_sp_bad = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#dense_sp = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sp_decompress_requires_i32_mdata() {
    %cdata = arith.constant dense<0.0> : tensor<128x64xf16, #cdata_sp>
    %mdata = arith.constant dense<0.0> : tensor<128x4xf16, #mdata_sp_bad>
    // expected-error @below {{mdata element type must be i32}}
    %dense = ttng.sp_decompress %cdata, %mdata :
      tensor<128x64xf16, #cdata_sp>, tensor<128x4xf16, #mdata_sp_bad> -> tensor<128x128xf16, #dense_sp>
    tt.return
  }
}

// -----

// Test: sp_decompress validates result shape
#cdata_sp = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata_sp = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#dense_sp_bad = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sp_decompress_bad_result_shape() {
    %cdata = arith.constant dense<0.0> : tensor<128x64xf16, #cdata_sp>
    %mdata = arith.constant dense<0> : tensor<128x4xi32, #mdata_sp>
    // expected-error @below {{result shape must be [M, 2 * N_compressed]}}
    %dense = ttng.sp_decompress %cdata, %mdata :
      tensor<128x64xf16, #cdata_sp>, tensor<128x4xi32, #mdata_sp> -> tensor<128x64xf16, #dense_sp_bad>
    tt.return
  }
}

// -----

// Test: sp_decompress requires compressed N >= 2
#cdata_small = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata_small = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#dense_small = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sp_decompress_requires_min_compressed_n() {
    %cdata = arith.constant dense<0.0> : tensor<128x1xf16, #cdata_small>
    %mdata = arith.constant dense<0> : tensor<128x1xi32, #mdata_small>
    // expected-error @below {{compressed N dimension must be >= 2}}
    %dense = ttng.sp_decompress %cdata, %mdata :
      tensor<128x1xf16, #cdata_small>, tensor<128x1xi32, #mdata_small> -> tensor<128x2xf16, #dense_small>
    tt.return
  }
}

// -----

// Test: tc_gen5_mma_sp rejects barriers in synchronous mode
#shared_a_sp = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_b_sp = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared_bar_sp = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem_a_sp = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_d_sp = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_m_sp = #ttng.tensor_memory_encoding<blockM = 128, blockN = 4, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tc_gen5_mma_sp_sync_with_barrier_rejected(
      %a: !ttg.memdesc<128x64xf16, #shared_a_sp, #ttg.shared_memory>,
      %b: !ttg.memdesc<128x128xf16, #shared_b_sp, #ttg.shared_memory>,
      %m: !ttg.memdesc<128x4xi32, #tmem_m_sp, #ttng.tensor_memory>,
      %d: !ttg.memdesc<128x128xf32, #tmem_d_sp, #ttng.tensor_memory, mutable>,
      %useAcc: i1, %pred: i1,
      %barrier: !ttg.memdesc<1xi64, #shared_bar_sp, #ttg.shared_memory, mutable>, %barrierPred: i1) {
    // expected-error @below {{The op is synchronous but a barrier is present.}}
    ttng.tc_gen5_mma_sp %a, %b, %m, %d, %useAcc, %pred, %barrier[%barrierPred] :
      !ttg.memdesc<128x64xf16, #shared_a_sp, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf16, #shared_b_sp, #ttg.shared_memory>,
      !ttg.memdesc<128x4xi32, #tmem_m_sp, #ttng.tensor_memory>,
      !ttg.memdesc<128x128xf32, #tmem_d_sp, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<1xi64, #shared_bar_sp, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

// Test: tc_gen5_mma_sp rejects f32 A dtype
#shared_a_sp = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 32}>
#shared_b_sp = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 32}>
#tmem_m_sp = #ttng.tensor_memory_encoding<blockM = 128, blockN = 4, colStride = 1>
#tmem_d_sp = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tc_gen5_mma_sp_bad_lhs_dtype(
      %a: !ttg.memdesc<128x64xf32, #shared_a_sp, #ttg.shared_memory>,
      %b: !ttg.memdesc<128x128xf32, #shared_b_sp, #ttg.shared_memory>,
      %m: !ttg.memdesc<128x4xi32, #tmem_m_sp, #ttng.tensor_memory>,
      %d: !ttg.memdesc<128x128xf32, #tmem_d_sp, #ttng.tensor_memory, mutable>,
      %useAcc: i1, %pred: i1) {
    // expected-error @below {{unsupported LHS operand dtype}}
    ttng.tc_gen5_mma_sp %a, %b, %m, %d, %useAcc, %pred :
      !ttg.memdesc<128x64xf32, #shared_a_sp, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf32, #shared_b_sp, #ttg.shared_memory>,
      !ttg.memdesc<128x4xi32, #tmem_m_sp, #ttng.tensor_memory>,
      !ttg.memdesc<128x128xf32, #tmem_d_sp, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Test: tc_gen5_mma_sp checks packed K relation
#shared_a_sp = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_b_sp = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#tmem_m_sp = #ttng.tensor_memory_encoding<blockM = 128, blockN = 4, colStride = 1>
#tmem_d_sp = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tc_gen5_mma_sp_bad_k_relation(
      %a: !ttg.memdesc<128x64xf16, #shared_a_sp, #ttg.shared_memory>,
      %b: !ttg.memdesc<32x128xf16, #shared_b_sp, #ttg.shared_memory>,
      %m: !ttg.memdesc<128x4xi32, #tmem_m_sp, #ttng.tensor_memory>,
      %d: !ttg.memdesc<128x128xf32, #tmem_d_sp, #ttng.tensor_memory, mutable>,
      %useAcc: i1, %pred: i1) {
    // expected-error @below {{expected the last dimension of the first operand to be equal to the second-to-last dimension of the second operand}}
    ttng.tc_gen5_mma_sp %a, %b, %m, %d, %useAcc, %pred :
      !ttg.memdesc<128x64xf16, #shared_a_sp, #ttg.shared_memory>,
      !ttg.memdesc<32x128xf16, #shared_b_sp, #ttg.shared_memory>,
      !ttg.memdesc<128x4xi32, #tmem_m_sp, #ttng.tensor_memory>,
      !ttg.memdesc<128x128xf32, #tmem_d_sp, #ttng.tensor_memory, mutable>
    tt.return
  }
}
