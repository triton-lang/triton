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

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @async_tma_gather(%desc: !tt.tensordesc<tensor<1x128xbf16, #shared>>, %x_offsets: tensor<32xi32, #blocked>, %y_offset: i32,
                          %bar: !ttg.memdesc<2xi32, #shared1, #ttg.shared_memory, mutable>,
                          %result: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>,
                          %pred: i1) {
  // expected-error @below {{barrier allocation must be a descriptor of Nxi64 type with N <= number of CTAs}}
  ttng.async_tma_gather %desc[%x_offsets, %y_offset] %result, %bar, %pred : !tt.tensordesc<tensor<1x128xbf16, #shared>>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<2xi32, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>, i1
  tt.return
}
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {
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
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<tensor<1x256x32xf32, #shared>>) -> tensor<256x32xf32, #blocked> {
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // expected-error @below {{TMA descriptor must have NVMMA shared layout}}
    ttng.async_tma_copy_global_to_local %arg0[%c32_i32, %c32_i32, %c32_i32] %0, %1, %true : !tt.tensordesc<tensor<1x256x32xf32, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<tensor<1x256x32xf32, #shared>>) -> tensor<256x32xf32, #blocked> {
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{TMA descriptor layout must not be transposed}}
    ttng.async_tma_copy_global_to_local %arg0[%c32_i32, %c32_i32, %c32_i32] %0, %1, %true : !tt.tensordesc<tensor<1x256x32xf32, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma32 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#nvmma64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared_mbar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<tensor<1x256x64xf32, #nvmma32>>) {
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x64xf32, #nvmma64, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_mbar, #smem, mutable>
    // expected-error @below {{TMA descriptor layout must match shared layout}}
    ttng.async_tma_copy_global_to_local %arg0[%c32_i32, %c32_i32, %c32_i32] %0, %1, %true : !tt.tensordesc<tensor<1x256x64xf32, #nvmma32>>, !ttg.memdesc<1xi64, #shared_mbar, #smem, mutable> -> !ttg.memdesc<256x64xf32, #nvmma64, #smem, mutable>
    tt.return
  }
}
// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_im2col_missing_offsets(%arg0: !tt.tensordesc<tensor<64x128xf16, #nvmma_128>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{IM2COL mode requires offsets to be provided}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] %0, %1, %true tensorMode = im2col : !tt.tensordesc<tensor<64x128xf16, #nvmma_128>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_im2col_wrong_offset_count(%arg0: !tt.tensordesc<tensor<64x128xf16, #nvmma_128>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i16 = arith.constant 1 : i16
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{IM2COL mode with 4D coordinates requires 2 offsets, but got 1}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] offsets = [%c1_i16] %0, %1, %true tensorMode = im2col : !tt.tensordesc<tensor<64x128xf16, #nvmma_128>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_tiled_with_offsets(%arg0: !tt.tensordesc<tensor<64x128xf16, #nvmma_128>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i16 = arith.constant 1 : i16
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{TILED mode does not support offsets}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] offsets = [%c1_i16] %0, %1, %true : !tt.tensordesc<tensor<64x128xf16, #nvmma_128>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_im2col_2d_invalid(%arg0: !tt.tensordesc<tensor<64x128xf16, #nvmma_128>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{IM2COL mode requires at least 3D coordinates, but got 2D}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true tensorMode = im2col : !tt.tensordesc<tensor<64x128xf16, #nvmma_128>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
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
    ttng.cluster_arrive {relaxed = false}
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

// expected-error @+1 {{After removing the zero bases the layout must be bijective}}
#linear = #ttg.linear<{register = [[0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1]], warp = [[16, 0], [8, 0]], block = []}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @invalid_linear_layout(%arg0: tensor<32x64xi32, #linear>) {
    tt.return
  }
}
