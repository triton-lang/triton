// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-tmem-barrier-insertion --convert-triton-gpu-to-llvm=compute-capability=107 -cse | FileCheck %s

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#cdata = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_spcompress_max
  // CHECK: tcgen05.ld.spcompress.sync.aligned.32x32b.x{{[0-9]+}}.max.sp::2:4.f32.b2
  // CHECK: tcgen05.wait <load>
  tt.func public @tensor_memory_ld_spcompress_max() {
    %src = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %cdata, %mdata = ttng.tmem_load_sparsify %src {rowOp = #ttng.rowOp<max>} :
      !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> ->
      tensor<128x64xf32, #cdata>, tensor<128x4xi32, #mdata>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#cdata = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_spcompress_maxabs
  // CHECK: tcgen05.ld.spcompress.sync.aligned.32x32b.x{{[0-9]+}}.max.sp::2:4.abs.f32.b2
  // CHECK: tcgen05.wait <load>
  tt.func public @tensor_memory_ld_spcompress_maxabs() {
    %src = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %cdata, %mdata = ttng.tmem_load_sparsify %src {rowOp = #ttng.rowOp<max>, abs = true} :
      !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> ->
      tensor<128x64xf32, #cdata>, tensor<128x4xi32, #mdata>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#cdata = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_spcompress_minabs
  // CHECK: tcgen05.ld.spcompress.sync.aligned.32x32b.x{{[0-9]+}}.min.sp::2:4.abs.f32.b2
  // CHECK: tcgen05.wait <load>
  tt.func public @tensor_memory_ld_spcompress_minabs() {
    %src = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %cdata, %mdata = ttng.tmem_load_sparsify %src {rowOp = #ttng.rowOp<min>, abs = true} :
      !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> ->
      tensor<128x64xf32, #cdata>, tensor<128x4xi32, #mdata>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#cdata = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#red = #ttg.slice<{dim = 1, parent = #cdata}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_red_spcompress_max_nan
  // CHECK: tcgen05.ld.red.spcompress.sync.aligned.32x32b.x{{[0-9]+}}.max.sp::2:4.NaN.f32.b2
  // CHECK: tcgen05.wait <load>
  tt.func public @tensor_memory_ld_red_spcompress_max_nan() {
    %src = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %cdata, %mdata, %red = ttng.tmem_load_sparsify %src {rowOp = #ttng.rowOp<max>, NaN = true} :
      !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> ->
      tensor<128x64xf32, #cdata>, tensor<128x4xi32, #mdata>, tensor<128xf32, #red>
    tt.return
  }
}

// -----

// Test that N=256 sparsify tiles into two .x128 instructions (PTX max is .x128).
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#cdata = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_spcompress_max_n256
  // CHECK: tcgen05.ld.spcompress.sync.aligned.32x32b.x128.max.sp::2:4.f32.b2
  // CHECK: tcgen05.ld.spcompress.sync.aligned.32x32b.x128.max.sp::2:4.f32.b2
  // CHECK: tcgen05.wait <load>
  tt.func public @tensor_memory_ld_spcompress_max_n256() {
    %src = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %cdata, %mdata = ttng.tmem_load_sparsify %src {rowOp = #ttng.rowOp<max>} :
      !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> ->
      tensor<128x128xf32, #cdata>, tensor<128x8xi32, #mdata>
    tt.return
  }
}

// -----

// Test N=256 sparsify with reduction also tiles.
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#cdata = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#red = #ttg.slice<{dim = 1, parent = #cdata}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_red_spcompress_max_n256
  // CHECK: tcgen05.ld.red.spcompress.sync.aligned.32x32b.x128.max.sp::2:4.f32.b2
  // CHECK: tcgen05.ld.red.spcompress.sync.aligned.32x32b.x128.max.sp::2:4.f32.b2
  // CHECK: tcgen05.wait <load>
  tt.func public @tensor_memory_ld_red_spcompress_max_n256() {
    %src = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %cdata, %mdata, %red = ttng.tmem_load_sparsify %src {rowOp = #ttng.rowOp<max>} :
      !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> ->
      tensor<128x128xf32, #cdata>, tensor<128x8xi32, #mdata>, tensor<128xf32, #red>
    tt.return
  }
}

// -----

// Test N=512 sparsify tiles into four .x128 instructions (for fp8 K=512).
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 512, colStride = 1>
#cdata = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_spcompress_max_n512
  // CHECK: tcgen05.ld.spcompress.sync.aligned.32x32b.x128.max.sp::2:4.f32.b2
  // CHECK: tcgen05.ld.spcompress.sync.aligned.32x32b.x128.max.sp::2:4.f32.b2
  // CHECK: tcgen05.ld.spcompress.sync.aligned.32x32b.x128.max.sp::2:4.f32.b2
  // CHECK: tcgen05.ld.spcompress.sync.aligned.32x32b.x128.max.sp::2:4.f32.b2
  // CHECK: tcgen05.wait <load>
  tt.func public @tensor_memory_ld_spcompress_max_n512() {
    %src = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x512xf32, #tmem, #ttng.tensor_memory, mutable>
    %cdata, %mdata = ttng.tmem_load_sparsify %src {rowOp = #ttng.rowOp<max>} :
      !ttg.memdesc<128x512xf32, #tmem, #ttng.tensor_memory, mutable> ->
      tensor<128x256xf32, #cdata>, tensor<128x16xi32, #mdata>
    tt.return
  }
}
