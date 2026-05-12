// RUN: triton-opt --split-input-file %s --verify-diagnostics

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @alloc_tensor_memory() {
    // expected-error @+1 {{uninitialized alloc must have a mutable memdesc type}}
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_layout_cta_mismatch() {
    // expected-error @+1 {{Layout has 1 CTAs per CGA, but the context requires 2 CTAs per CGA.}}
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
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
  tt.func public @alloc_tensor_memory(%arg: !ttg.memdesc<128x4xi8, #shared1, #ttg.shared_memory, mutable>) {
    %cst = arith.constant dense<0> : tensor<128x4xi8, #scales>
    %0 = ttng.tmem_alloc %cst : (tensor<128x4xi8, #scales>) -> !ttg.memdesc<128x4xi8, #tmem, #ttng.tensor_memory>
    // expected-error @+1 {{Cannot copy into an immutable alloc}}
    ttng.tmem_copy %arg, %0 : !ttg.memdesc<128x4xi8, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<128x4xi8, #tmem, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @init_barrier_zero_count() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // expected-error @+1 {{count must be greater than or equal to 1}}
    ttng.init_barrier %bar, 0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @async_tma_gather(%desc: !tt.tensordesc<1x128xbf16, #shared>, %x_offsets: tensor<32xi32, #blocked>, %y_offset: i32,
                          %bar: !ttg.memdesc<2xi32, #shared1, #ttg.shared_memory, mutable>,
                          %result: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>,
                          %pred: i1) {
  // expected-error @below {{barrier allocation must be a descriptor of Nxi64 type with N <= number of CTAs}}
  ttng.async_tma_gather %desc[%x_offsets, %y_offset] %result, %bar, %pred : !tt.tensordesc<1x128xbf16, #shared>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<2xi32, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>, i1
  tt.return
}
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {
tt.func @async_tma_gather(%desc: !tt.tensordesc<1x128xbf16, #shared>, %x_offsets: tensor<32xi32, #blocked>, %y_offset: i32,
                          %bar: !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>,
                          %result: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory>,
                          %pred: i1) {
  // expected-error @below {{cannot store into immutable memory}}
  ttng.async_tma_gather %desc[%x_offsets, %y_offset] %result, %bar, %pred : !tt.tensordesc<1x128xbf16, #shared>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory>, i1
  tt.return
}
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32} {
tt.func @wgmma(%a: tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>, %b: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, %c: tensor<128x128xf16, #mma>) {
  // expected-error @below {{in-register LHS operand must have a kWidth of 2 but got 1}}
  %0 = ttng.warp_group_dot %a, %b, %c : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> -> tensor<128x128xf16, #mma>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<1x256x32xf32, #shared>) -> tensor<256x32xf32, #blocked> {
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // expected-error @below {{TMA descriptor must have NVMMA shared layout}}
    ttng.async_tma_copy_global_to_local %arg0[%c32_i32, %c32_i32, %c32_i32] %0, %1, %true : !tt.tensordesc<1x256x32xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<1x256x32xf32, #shared>) -> tensor<256x32xf32, #blocked> {
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{TMA descriptor layout must not be transposed}}
    ttng.async_tma_copy_global_to_local %arg0[%c32_i32, %c32_i32, %c32_i32] %0, %1, %true : !tt.tensordesc<1x256x32xf32, #shared>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma32 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#nvmma64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared_mbar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<1x256x64xf32, #nvmma32>) {
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x64xf32, #nvmma64, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_mbar, #smem, mutable>
    // expected-error @below {{TMA descriptor layout must match shared layout}}
    ttng.async_tma_copy_global_to_local %arg0[%c32_i32, %c32_i32, %c32_i32] %0, %1, %true : !tt.tensordesc<1x256x64xf32, #nvmma32>, !ttg.memdesc<1xi64, #shared_mbar, #smem, mutable> -> !ttg.memdesc<256x64xf32, #nvmma64, #smem, mutable>
    tt.return
  }
}
// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttng.two-ctas" = false, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_copy_global_to_local_requires_1d_barrier_layout(
      %arg0: !tt.tensordesc<64x128xf16, #nvmma>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // expected-error @below {{TMA barrier cga_layout must be [[1]], got [[0]]}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0], [0, 1]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1], [2]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, "ttng.two-ctas" = true, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_copy_global_to_local_requires_two_cta_barrier_layout(
      %arg0: !tt.tensordesc<64x128xf16, #nvmma>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<4xi64, #barrier, #smem, mutable>
    // expected-error @below {{TMA barrier cga_layout must be [[0], [1]], got [[1], [2]]}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<4xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_im2col_missing_offsets(%arg0: !ttng.tensordesc_im2col<64x128xf16, #nvmma_128>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{IM2COL mode requires offsets to be provided}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] %0, %1, %true : !ttng.tensordesc_im2col<64x128xf16, #nvmma_128>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_im2col_wrong_offset_count(%arg0: !ttng.tensordesc_im2col<64x128xf16, #nvmma_128>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i16 = arith.constant 1 : i16
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{IM2COL mode with 4D coordinates requires 2 offsets, but got 1}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] offsets = [%c1_i16] %0, %1, %true : !ttng.tensordesc_im2col<64x128xf16, #nvmma_128>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_tiled_with_offsets(%arg0: !tt.tensordesc<64x128xf16, #nvmma_128>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i16 = arith.constant 1 : i16
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{TILED mode does not support offsets}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] offsets = [%c1_i16] %0, %1, %true : !tt.tensordesc<64x128xf16, #nvmma_128>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_im2col_2d_invalid(%arg0: !ttng.tensordesc_im2col<64x128xf16, #nvmma_128>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{IM2COL mode requires at least 3D coordinates, but got 2D}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !ttng.tensordesc_im2col<64x128xf16, #nvmma_128>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem_f16 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @tcgen5(%a: !ttg.memdesc<128x128xbf16, #shared, #ttg.shared_memory>,
                  %b: !ttg.memdesc<128x256xbf16, #shared1, #ttg.shared_memory>,
                  %c: !ttg.memdesc<128x256xf16, #tmem_f16, #ttng.tensor_memory, mutable>,
                  %accUse: i1,
                  %pred: i1,
                  %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                  %barrierPred: i1) {
    // expected-error @below {{unsupported accumulator dtype for operand types 'bf16' and 'bf16', accumulator dtype is 'f16' but must be one of ['f32']}}
    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred, %barrier[%barrierPred] {is_async} :
       !ttg.memdesc<128x128xbf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xbf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf16, #tmem_f16, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1, CGALayout = [[0, 1]]>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @tcgen5_completion_barrier_cga_layout(
      %a: !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
      %b: !ttg.memdesc<16x128xf16, #shared1, #ttg.shared_memory>,
      %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %accUse: i1,
      %pred: i1,
      %bar: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory>,
      %barPred: i1) {
    // expected-error @below {{completion barrier cga_layout must be}}
    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred, %bar[%barPred] {is_async} :
       !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<16x128xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory>
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @tcgen5_commit_completion_barrier_cga_layout(
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %pred: i1) {
    // expected-error @below {{completion barrier cga_layout must be}}
    ttng.tc_gen5_commit %bar, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#sharedT = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @tcgen5_mma_scaled_sync_with_barrier(
      %a: !ttg.memdesc<128x256xi8, #shared, #ttg.shared_memory>,
      %b: !ttg.memdesc<256x64xi8, #sharedT, #ttg.shared_memory>,
      %c: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>,
      %scale_a: !ttg.memdesc<128x8xf8E4M3FN, #shared1, #ttg.shared_memory>,
      %scale_b: !ttg.memdesc<64x8xf8E4M3FN, #shared1, #ttg.shared_memory>,
      %useAcc: i1,
      %pred: i1,
      %bar: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>,
      %barPred: i1) {
    // expected-error @below {{The op is synchronous but a barrier is present.}}
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e2m1 rhs = e2m1, %bar[%barPred] :
      !ttg.memdesc<128x256xi8, #shared, #ttg.shared_memory>,
      !ttg.memdesc<256x64xi8, #sharedT, #ttg.shared_memory>,
      !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<128x8xf8E4M3FN, #shared1, #ttg.shared_memory>,
      !ttg.memdesc<64x8xf8E4M3FN, #shared1, #ttg.shared_memory>,
      !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared_clc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0], [0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0], [0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @clc_try_cancel_completion_barrier_cga_layout(
      %result: !ttg.memdesc<2xi64, #shared_clc, #smem>,
      %mbar: !ttg.memdesc<1xi64, #barrier, #smem>) {
    // expected-error @below {{completion barrier cga_layout must be}}
    ttng.clc_try_cancel %result, %mbar :
      !ttg.memdesc<2xi64, #shared_clc, #smem>, !ttg.memdesc<1xi64, #barrier, #smem>
    tt.return
  }
}

// -----

#shared_clc_bad = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @clc_try_cancel_result_cga_layout_bases_nonzero(
      %result: !ttg.memdesc<2xi64, #shared_clc_bad, #smem>,
      %mbar: !ttg.memdesc<1xi64, #barrier, #smem>) {
    // expected-error @below {{Expected CLC result buffer cga_layout bases to be all zeros. Got [[1]]}}
    ttng.clc_try_cancel %result, %mbar :
      !ttg.memdesc<2xi64, #shared_clc_bad, #smem>, !ttg.memdesc<1xi64, #barrier, #smem>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @fence_mbarrier_init_release_cluster_invalid() {
    // expected-error @below {{requires ttg.num-ctas > 1}}
    ttng.fence_mbarrier_init_release_cluster
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @cluster_arrive_invalid() {
    // expected-error @below {{requires ttg.num-ctas > 1}}
    ttng.cluster_arrive
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @cluster_wait_invalid() {
    // expected-error @below {{requires ttg.num-ctas > 1}}
    ttng.cluster_wait
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @cluster_arrive_in_default_region_invalid() {
    ttg.warp_specialize()
    default {
      // expected-error @below {{cannot be used inside `ttg.warp_specialize`}}
      ttng.cluster_arrive
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @cluster_wait_in_partition_invalid() {
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      // expected-error @below {{cannot be used inside `ttg.warp_specialize`}}
      ttng.cluster_wait
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @cluster_barrier_invalid() {
    // expected-error @below {{requires ttg.num-ctas > 1}}
    ttng.cluster_barrier
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @cluster_barrier_in_default_region_invalid() {
    ttg.warp_specialize()
    default {
      // expected-error @below {{cannot be used inside `ttg.warp_specialize`}}
      ttng.cluster_barrier
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @cluster_barrier_in_partition_invalid() {
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      // expected-error @below {{cannot be used inside `ttg.warp_specialize`}}
      ttng.cluster_barrier
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @init_barrier_in_default_region_invalid() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.warp_specialize()
    default {
      // expected-error @below {{cannot be used inside `ttg.warp_specialize`}}
      ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

// expected-error @+1 {{LinearEncodingAttr requires a permutation matrix layout after removing broadcast bases}}
#linear = #ttg.linear<{register = [[0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1]], warp = [[16, 0], [8, 0]], block = []}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @invalid_linear_layout(%arg0: tensor<32x64xi32, #linear>) {
    tt.return
  }
}

// -----

// Test that reduction with warps split across N dimension is rejected
// 128x256 with 8 warps -> warpsPerCTA = [4, 2] (2 warps in N)
#blocked_split = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked_red = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#tmem_warp_split = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_red_warp_split_rejected() {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked_split>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x256xf32, #blocked_split>) -> !ttg.memdesc<128x256xf32, #tmem_warp_split, #ttng.tensor_memory, mutable>
    // expected-error @below {{tmem_load reduction with N dimension sharded across threads is not supported.}}
    %result, %red = ttng.tmem_load %0 {redOp = #ttng.redOp<min>} : !ttg.memdesc<128x256xf32, #tmem_warp_split, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked_split>, tensor<128xf32, #blocked_red>
    tt.return
  }
}

// -----

// Test that reduction with N shared across threads is rejected
#blocked_split = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_red = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#bm64_bn128 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_red_16x32bx2_atom_rejected() {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked_split>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<64x128xf32, #blocked_split>) -> !ttg.memdesc<64x128xf32, #bm64_bn128, #ttng.tensor_memory, mutable>
    // expected-error @below {{tmem_load reduction with N dimension sharded across threads is not supported.}}
    %result, %red = ttng.tmem_load %0 {redOp = #ttng.redOp<min>} : !ttg.memdesc<64x128xf32, #bm64_bn128, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked_split>, tensor<64xf32, #blocked_red>
    tt.return
  }
}

// -----

// Test: abs requires redOp to be set
#blocked_abs = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_abs = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_abs_requires_redop() {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked_abs>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked_abs>) -> !ttg.memdesc<128x128xf32, #tmem_abs, #ttng.tensor_memory, mutable>
    // expected-error @below {{'abs' requires 'redOp' to be set}}
    %result = ttng.tmem_load %0 {abs = true} : !ttg.memdesc<128x128xf32, #tmem_abs, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_abs>
    tt.return
  }
}

// -----

// Test: NaN requires redOp to be set
#blocked_nan = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_nan = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_nan_requires_redop() {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked_nan>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked_nan>) -> !ttg.memdesc<128x128xf32, #tmem_nan, #ttng.tensor_memory, mutable>
    // expected-error @below {{'NaN' requires 'redOp' to be set}}
    %result = ttng.tmem_load %0 {NaN = true} : !ttg.memdesc<128x128xf32, #tmem_nan, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_nan>
    tt.return
  }
}

// -----

// Test: abs requires f32 element type
#blocked_abs_i32 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_red_abs_i32 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tmem_abs_i32 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_abs_requires_f32() {
    %cst_0 = arith.constant dense<0> : tensor<128x128xi32, #blocked_abs_i32>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xi32, #blocked_abs_i32>) -> !ttg.memdesc<128x128xi32, #tmem_abs_i32, #ttng.tensor_memory, mutable>
    // expected-error @below {{'abs' requires floating-point element type (f32)}}
    %result, %red = ttng.tmem_load %0 {redOp = #ttng.redOp<min>, abs = true} : !ttg.memdesc<128x128xi32, #tmem_abs_i32, #ttng.tensor_memory, mutable> -> tensor<128x128xi32, #blocked_abs_i32>, tensor<128xi32, #blocked_red_abs_i32>
    tt.return
  }
}

// -----

// Test: NaN requires f32 element type
#blocked_nan_i32 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_red_nan_i32 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tmem_nan_i32 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_nan_requires_f32() {
    %cst_0 = arith.constant dense<0> : tensor<128x128xi32, #blocked_nan_i32>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xi32, #blocked_nan_i32>) -> !ttg.memdesc<128x128xi32, #tmem_nan_i32, #ttng.tensor_memory, mutable>
    // expected-error @below {{'NaN' requires floating-point element type (f32)}}
    %result, %red = ttng.tmem_load %0 {redOp = #ttng.redOp<min>, NaN = true} : !ttg.memdesc<128x128xi32, #tmem_nan_i32, #ttng.tensor_memory, mutable> -> tensor<128x128xi32, #blocked_nan_i32>, tensor<128xi32, #blocked_red_nan_i32>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma_no_broadcast = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_copy_multicast_requires_broadcast(%arg0: !tt.tensordesc<64x128xf16, #nvmma_no_broadcast>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_no_broadcast, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared_bar, #smem, mutable>
    // expected-error @below {{multicast requires the shared layout to broadcast across CTAs}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true {multicast} : !tt.tensordesc<64x128xf16, #nvmma_no_broadcast>, !ttg.memdesc<2xi64, #shared_bar, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_no_broadcast, #smem, mutable>
    tt.return
  }
}

// -----

#blocked_broadcast = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[0]]}>
#nvmma_no_broadcast = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_gather_multicast_requires_broadcast(%arg0: !tt.tensordesc<1x128xf16, #nvmma_no_broadcast>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %x_offsets = arith.constant dense<0> : tensor<32xi32, #blocked_broadcast>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    %result = ttg.local_alloc : () -> !ttg.memdesc<32x128xf16, #nvmma_no_broadcast, #smem, mutable>
    // expected-error @below {{multicast requires the shared layout to broadcast across CTAs}}
    ttng.async_tma_gather %arg0[%x_offsets, %c0_i32] %result, %bar, %true {multicast} : !tt.tensordesc<1x128xf16, #nvmma_no_broadcast>, tensor<32xi32, #blocked_broadcast>, i32, !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>, !ttg.memdesc<32x128xf16, #nvmma_no_broadcast, #smem, mutable>, i1
    tt.return
  }
}

// -----

#blocked_broadcast_parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 0], [0, 0]]}>
#blocked_broadcast = #ttg.slice<{dim = 0, parent = #blocked_broadcast_parent}>
#nvmma_partial_broadcast = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0], [0, 0]]}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0], [0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_gather_multicast_requires_matching_x_offset_cga(%arg0: !tt.tensordesc<1x128xf16, #nvmma_partial_broadcast>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %x_offsets = arith.constant dense<0> : tensor<32xi32, #blocked_broadcast>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    %result = ttg.local_alloc : () -> !ttg.memdesc<32x128xf16, #nvmma_partial_broadcast, #smem, mutable>
    // expected-error @below {{x offsets must have the same row CGA layout as the memdesc}}
    ttng.async_tma_gather %arg0[%x_offsets, %c0_i32] %result, %bar, %true {multicast} : !tt.tensordesc<1x128xf16, #nvmma_partial_broadcast>, tensor<32xi32, #blocked_broadcast>, i32, !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>, !ttg.memdesc<32x128xf16, #nvmma_partial_broadcast, #smem, mutable>, i1
    tt.return
  }
}

// -----

#blocked_split_parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 1]]}>
#blocked_split = #ttg.slice<{dim = 0, parent = #blocked_split_parent}>
#nvmma_broadcast = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_gather_multicast_requires_uniform_x_offsets(%arg0: !tt.tensordesc<1x128xf16, #nvmma_broadcast>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %x_offsets = arith.constant dense<0> : tensor<32xi32, #blocked_split>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    %result = ttg.local_alloc : () -> !ttg.memdesc<32x128xf16, #nvmma_broadcast, #smem, mutable>
    // expected-error @below {{x offsets must have the same row CGA layout as the memdesc}}
    ttng.async_tma_gather %arg0[%x_offsets, %c0_i32] %result, %bar, %true {multicast} : !tt.tensordesc<1x128xf16, #nvmma_broadcast>, tensor<32xi32, #blocked_split>, i32, !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>, !ttg.memdesc<32x128xf16, #nvmma_broadcast, #smem, mutable>, i1
    tt.return
  }

}

// -----

// Test invalid TensorDescIm2ColType: rank-3 blockType (must be rank-2)
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // expected-error @below {{TensorDescIm2ColType requires rank-2 shape, got rank 3}}
  tt.func @tensordesc_im2col_wrong_rank(%desc: !ttng.tensordesc_im2col<32x64x128xf16>) {
    tt.return
  }
}

// -----

#shared_bad = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0], [2], [1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @wait_barrier_invalid_cga_layout(%bar: !ttg.memdesc<4xi64, #shared_bad, #smem, mutable>, %phase: i32) {
    // expected-error @below {{broadcasted cluster barriers require bases to be the sequence}}
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<4xi64, #shared_bad, #smem, mutable>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @memdesc_reinterpret_changed_storage_size_tmem(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) {
    // expected-error @+1 {{source and result must have the same logical storage size}}
    %0 = ttg.memdesc_reinterpret %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_non_tmem_source() {
    %md = ttg.local_alloc : () -> !ttg.memdesc<128x128xi32, #shared1, #ttg.shared_memory, mutable>
    // expected-error @+1 {{The source must be a tensor memory buffer.}}
    %sub = ttng.tmem_subslice %md {N = 0 : i32} : !ttg.memdesc<128x128xi32, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xi32, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_rank_not_2() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The result must be a 2D tensor memory buffer.}}
    %sub = ttng.tmem_subslice %md {N = 0 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_rows_mismatch() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The result must have the same number of rows as the source.}}
    %sub = ttng.tmem_subslice %md {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_element_type_mismatch() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The source and result must have the same element type.}}
    %sub = ttng.tmem_subslice %md {N = 0 : i32} : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_alloc_shape_mismatch() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The source and result must have the same alloc shape.}}
    %sub = ttng.tmem_subslice %md {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_offset_alignment_invalid() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The split offset may not touch the tile}}
    %sub = ttng.tmem_subslice %md {N = 32 : i32} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_offset_exceed() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The split offset may not exceed the source shape}}
    %sub = ttng.tmem_subslice %md {N = 128 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_reduce_rejects_unsupported_kind(%arg0: !tt.tensordesc<32x32xf32, #shared>, %x: i32) {
    %src = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // expected-error @below {{unsupported reduce kind inc for element type 'f32'}}
    ttng.async_tma_reduce inc, %arg0[%x, %x] %src : !tt.tensordesc<32x32xf32, #shared>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_gather_requires_legal_x_offsets(%arg0: !tt.tensordesc<1x128xf16, #shared>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %x_offsets = arith.constant dense<0> : tensor<32xi32, #blocked>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    %result = ttg.local_alloc : () -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    // expected-error @below {{x offsets must have at least 4 contiguous elements per thread}}
    ttng.async_tma_gather %arg0[%x_offsets, %c0_i32] %result, %bar, %true : !tt.tensordesc<1x128xf16, #shared>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>, !ttg.memdesc<32x128xf16, #shared, #smem, mutable>, i1
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 64}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_reduce_rejects_signed_i64_add(%arg0: !tt.tensordesc<32x32xsi64, #shared>, %x: i32) {
    %src = ttg.local_alloc : () -> !ttg.memdesc<32x32xi64, #shared, #smem, mutable>
    // expected-error @below {{unsupported reduce kind add for element type 'si64'}}
    ttng.async_tma_reduce add, %arg0[%x, %x] %src : !tt.tensordesc<32x32xsi64, #shared>, !ttg.memdesc<32x32xi64, #shared, #smem, mutable>
    tt.return
  }
}
