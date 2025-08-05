// RUN: triton-opt --split-input-file %s --verify-diagnostics

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @alloc_tensor_memory() {
    // expected-error @+1 {{uninitialized alloc must have a mutable memdesc type}}
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @alloc_tensor_memory() {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %0 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    // expected-error @+1 {{Cannot store into an immutable alloc}}
    ttng.tmem_store %cst, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#scales = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#tmem = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @alloc_tensor_memory(%arg: !ttg.memdesc<1x512xi8, #shared1, #ttg.shared_memory, mutable>) {
    %cst = arith.constant dense<0> : tensor<128x4xi8, #scales>
    %0 = ttng.tmem_alloc %cst : (tensor<128x4xi8, #scales>) -> !ttg.memdesc<128x4xi8, #tmem, #ttng.tensor_memory>
    // expected-error @+1 {{Cannot copy into an immutable alloc}}
    ttng.tmem_copy %arg, %0,  : (!ttg.memdesc<1x512xi8, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<128x4xi8, #tmem, #ttng.tensor_memory>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @async_tma_gather(%desc: !tt.tensordesc<tensor<1x128xbf16, #shared>>, %x_offsets: tensor<32xi32, #blocked>, %y_offset: i32,
                          %bar: !ttg.memdesc<2xi32, #shared1, #ttg.shared_memory, mutable>,
                          %result: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>,
                          %pred: i1) {
  // expected-error @below {{barrier allocation must be a descriptor of 1xi64 type}}
  ttng.async_tma_gather %desc[%x_offsets, %y_offset] %result, %bar, %pred : !tt.tensordesc<tensor<1x128xbf16, #shared>>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<2xi32, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>, i1
  tt.return
}
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @async_tma_gather(%desc: !tt.tensordesc<tensor<1x128xbf16, #shared>>, %x_offsets: tensor<32xi32, #blocked>, %y_offset: i32,
                          %bar: !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>,
                          %result: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory>,
                          %pred: i1) {
  // expected-error @below {{cannot store into immutable memory}}
  ttng.async_tma_gather %desc[%x_offsets, %y_offset] %result, %bar, %pred : !tt.tensordesc<tensor<1x128xbf16, #shared>>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory>, i1
  tt.return
}
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_non_tmem_source() {
    %md = ttg.local_alloc : () -> !ttg.memdesc<128x128xi32, #shared1, #ttg.shared_memory, mutable>
    // expected-error @+1 {{The source must be a tensor memory buffer.}}
    %sub = ttng.tmem_subslice %md {N = 0 : i32} : !ttg.memdesc<128x128xi32, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xi32, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_rank_not_2() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The result must be a 2D tensor memory buffer.}}
    %sub = ttng.tmem_subslice %md {N = 0 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_rows_mismatch() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The result must have the same number of rows as the source.}}
    %sub = ttng.tmem_subslice %md {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_element_type_mismatch() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The source and result must have the same element type.}}
    %sub = ttng.tmem_subslice %md {N = 0 : i32} : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_alloc_shape_mismatch() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The source and result must have the same alloc shape.}}
    %sub = ttng.tmem_subslice %md {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_offset_alignment_invalid() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The split offset may not touch the tile}}
    %sub = ttng.tmem_subslice %md {N = 32 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_offset_exceed() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The split offset may not exceed the source shape}}
    %sub = ttng.tmem_subslice %md {N = 128 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}
