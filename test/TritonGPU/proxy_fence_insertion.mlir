// RUN: triton-opt %s -triton-nvidia-gpu-proxy-fence-insertion='use-buffer-region-alias-analysis=false' --split-input-file -allow-unregistered-dialect | FileCheck %s --check-prefixes=CHECK,LEGACY
// RUN: triton-opt %s -triton-nvidia-gpu-proxy-fence-insertion --split-input-file -allow-unregistered-dialect | FileCheck %s --check-prefixes=CHECK,REGION

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: fence_write_after_read
  tt.func @fence_write_after_read(%arg0: !tt.tensordesc<64x64xf32, #shared>, %arg1: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) {
    // CHECK: ttg.local_load
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.async_tma_copy_global_to_local
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<32x64xf32, #shared, #smem, mutable>
    %1 = ttg.local_load %0 : !ttg.memdesc<32x64xf32, #shared, #smem, mutable> -> tensor<32x64xf32, #blocked>
    "test.keep"(%1) : (tensor<32x64xf32, #blocked>) -> ()
    %2 = ttg.local_alloc {allocation.offset = 32 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %2, %arg1, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @nested_fence_passthrough(
      %buffer: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>) {
    tt.return
  }

  tt.func private @fenced_proxy_after_nested_call(
      %buffer: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<8x32xi32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    ttng.fence_async_shared {bCluster = false}
    tt.call @nested_fence_passthrough(%buffer) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: nested_call_after_fence_does_not_expose_proxy
  tt.func public @nested_call_after_fence_does_not_expose_proxy(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    // CHECK: ttg.local_load
    // CHECK-NEXT: "test.keep"
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    // CHECK-NEXT: tt.call @fenced_proxy_after_nested_call
    tt.call @fenced_proxy_after_nested_call(%buffer, %desc, %bar) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#dst = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[0, 1]]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[1, 0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: cross_cta_scratch_requires_remote_proxy_fence
  tt.func @cross_cta_scratch_requires_remote_proxy_fence(
      %source: tensor<16x32xi32, #src>,
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.convert_layout
    %converted = ttg.convert_layout %source {allocation.offset = 0 : i32} : tensor<16x32xi32, #src> -> tensor<16x32xi32, #dst>
    "test.keep"(%converted) : (tensor<16x32xi32, #dst>) -> ()
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16x32xi32, #shared, #smem, mutable>
    %remote = ttg.memdesc_subslice %parent [8, 0] : !ttg.memdesc<16x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %remote, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @callee_proxy_write_with_local_frame(
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %dst = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: callee_local_frame_does_not_alias_caller_buffer
  tt.func public @callee_local_frame_does_not_alias_caller_buffer(%desc: !tt.tensordesc<64x64xf32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 49152 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @callee_proxy_write_with_local_frame
    tt.call @callee_proxy_write_with_local_frame(%desc, %bar) {allocation.offset = 16384 : i32} : (!tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @callee_generic_read_from_local_frame() {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    tt.return
  }

  tt.func private @forward_generic_read_from_local_frame() {
    tt.call @callee_generic_read_from_local_frame() {allocation.offset = 2048 : i32} : () -> ()
    tt.return
  }

  // CHECK-LABEL: callee_local_summary_uses_call_frame_offset
  tt.func public @callee_local_summary_uses_call_frame_offset(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: tt.call @callee_generic_read_from_local_frame
    tt.call @callee_generic_read_from_local_frame() {allocation.offset = 4096 : i32} : () -> ()
    %buffer = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: nested_callee_local_summary_composes_frame_offsets
  tt.func public @nested_callee_local_summary_composes_frame_offsets(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: tt.call @forward_generic_read_from_local_frame
    tt.call @forward_generic_read_from_local_frame() {allocation.offset = 4096 : i32} : () -> ()
    %buffer = ttg.local_alloc {allocation.offset = 6144 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buffer, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: unknown_descriptors_are_conservative
  tt.func public @unknown_descriptors_are_conservative(
      %read: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %write: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.local_load
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.async_tma_copy_global_to_local
    %value = ttg.local_load %read : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %write, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: missing_proxy_fence_memdesc_index_alias_single
  tt.func @missing_proxy_fence_memdesc_index_alias_single(%arg0: !tt.tensordesc<64x64xf32, #shared>, %arg1: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) {
    // Keep the first fence to clear dependencies from local_alloc.
    // CHECK: ttng.fence_async_shared
    // CHECK: ttg.local_load
    // CHECK-NEXT: "test.keep"
    // CHECK-NEXT: ttng.fence_async_shared
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<1x64x64xf32, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %2 = ttg.local_load %1 : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%2) : (tensor<64x64xf32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %1, %arg1, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: async_proxy_after_async_proxy
  tt.func @async_proxy_after_async_proxy(%arg0: !tt.tensordesc<64x64xf32, #shared>, %arg1: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) {
    // CHECK: ttng.async_tma_copy_global_to_local
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %arg1, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    %2 = ttg.local_alloc {allocation.offset = 32 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %2, %arg1, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: missing_proxy_fence_local_store_before_async_tma_copy_local_to_global
  tt.func @missing_proxy_fence_local_store_before_async_tma_copy_local_to_global(%arg0: !tt.tensordesc<128x256xf32, #shared>, %arg1: tensor<128x256xf32, #blocked>) {
    // CHECK: ttng.async_tma_store_wait {pendings = 1 : i32}
    // CHECK-NEXT: ttg.local_store
    // CHECK-NEXT: ttng.fence_async_shared
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<128x256xf32, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 1 : i32}
    ttg.local_store %arg1, %0 : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #shared, #smem, mutable>
    ttng.async_tma_copy_local_to_global %arg0[%c0_i32, %c0_i32] %0 : !tt.tensordesc<128x256xf32, #shared>, !ttg.memdesc<128x256xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: missing_proxy_fence_local_store_before_tmem_copy
  tt.func @missing_proxy_fence_local_store_before_tmem_copy(%arg0: tensor<128x4xi8, #blocked>,
      %arg1: !ttg.memdesc<128x4xi8, #tmem, #ttng.tensor_memory, mutable>) {
    // CHECK: ttg.local_store
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.tmem_copy
    %0 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<128x4xi8, #shared1, #smem, mutable>
    ttg.local_store %arg0, %0 : tensor<128x4xi8, #blocked> -> !ttg.memdesc<128x4xi8, #shared1, #smem, mutable>
    ttng.tmem_copy %0, %arg1 : !ttg.memdesc<128x4xi8, #shared1, #smem, mutable>, !ttg.memdesc<128x4xi8, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked_a = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#mma_a = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#mma_b = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: missing_proxy_fence_async_copy_before_mma
  tt.func @missing_proxy_fence_async_copy_before_mma(%arg0: tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked_a>,
      %arg1: tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked_b>,
      %arg2: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>,
      %arg3: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    %false = arith.constant false
    %true = arith.constant true
    // CHECK: ttg.async_copy_global_to_local
    // CHECK-NEXT: ttg.async_copy_global_to_local
    // CHECK-NEXT: ttg.async_commit_group
    // CHECK-NEXT: ttg.async_wait
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.tc_gen5_mma
    %0 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 32 : i32} : () -> !ttg.memdesc<32x64xf8E4M3FN, #mma_b, #smem, mutable>
    %2 = ttg.async_copy_global_to_local %arg0, %0 : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked_a> -> !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>
    %3 = ttg.async_copy_global_to_local %arg1, %1 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked_b> -> !ttg.memdesc<32x64xf8E4M3FN, #mma_b, #smem, mutable>
    %4 = ttg.async_commit_group tokens %2, %3
    %5 = ttg.async_wait %4 {num = 0 : i32}
    ttng.tc_gen5_mma %0, %1, %arg2, %false, %true, %arg3[%true] {is_async} : !ttg.memdesc<64x32xf8E4M3FN, #mma_a, #smem, mutable>, !ttg.memdesc<32x64xf8E4M3FN, #mma_b, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared_clc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: clc_try_cancel_single_cta_fence
  tt.func @clc_try_cancel_single_cta_fence() {
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<1xi64, #shared_clc, #smem, mutable>
    %response = ttng.clc_load_result %result : !ttg.memdesc<2xi64, #shared_clc, #smem, mutable> -> i128
    "test.keep"(%response) : (i128) -> ()
    // CHECK: ttng.clc_load_result
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.clc_try_cancel
    ttng.clc_try_cancel %result, %barrier :
      !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>,
      !ttg.memdesc<1xi64, #shared_clc, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: clc_try_cancel_single_cta_after_cluster_fence
  tt.func @clc_try_cancel_single_cta_after_cluster_fence() {
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<1xi64, #shared_clc, #smem, mutable>
    %response = ttng.clc_load_result %result : !ttg.memdesc<2xi64, #shared_clc, #smem, mutable> -> i128
    "test.keep"(%response) : (i128) -> ()
    // CHECK: ttng.clc_load_result
    // CHECK: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.clc_try_cancel
    ttng.fence_async_shared {bCluster = true}
    ttng.clc_try_cancel %result, %barrier :
      !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>,
      !ttg.memdesc<1xi64, #shared_clc, #smem, mutable>
    tt.return
  }
}

// -----

#shared_clc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: clc_try_cancel_multi_cta_fence
  tt.func @clc_try_cancel_multi_cta_fence() {
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %response = ttng.clc_load_result %result : !ttg.memdesc<2xi64, #shared_clc, #smem, mutable> -> i128
    "test.keep"(%response) : (i128) -> ()
    // CHECK: ttng.clc_load_result
    // CHECK: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.clc_try_cancel
    ttng.clc_try_cancel %result, %barrier :
      !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>,
      !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: clc_try_cancel_multi_cta_after_cta_fence
  tt.func @clc_try_cancel_multi_cta_after_cta_fence() {
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %response = ttng.clc_load_result %result : !ttg.memdesc<2xi64, #shared_clc, #smem, mutable> -> i128
    "test.keep"(%response) : (i128) -> ()
    // CHECK: ttng.clc_load_result
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.clc_try_cancel
    ttng.fence_async_shared {bCluster = false}
    ttng.clc_try_cancel %result, %barrier :
      !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>,
      !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: clc_try_cancel_multi_cta_after_cluster_fence
  tt.func @clc_try_cancel_multi_cta_after_cluster_fence() {
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %response = ttng.clc_load_result %result : !ttg.memdesc<2xi64, #shared_clc, #smem, mutable> -> i128
    "test.keep"(%response) : (i128) -> ()
    // CHECK: ttng.clc_load_result
    // CHECK: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.clc_try_cancel
    ttng.fence_async_shared {bCluster = true}
    ttng.clc_try_cancel %result, %barrier :
      !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>,
      !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: multicast_tma_after_cluster_barrier_and_cta_fence
  tt.func @multicast_tma_after_cluster_barrier_and_cta_fence(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %result : !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable> -> tensor<64x128xf16, #blocked>
    "test.keep"(%value) : (tensor<64x128xf16, #blocked>) -> ()
    // CHECK: ttg.local_load
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.cluster_barrier
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %result, %barrier, %true {multicast} :
      !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: non_multicast_tma_after_cluster_barrier_and_cta_fence
  tt.func @non_multicast_tma_after_cluster_barrier_and_cta_fence(%desc: !tt.tensordesc<64x128xf16, #nvmma>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %result : !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable> -> tensor<64x128xf16, #blocked>
    "test.keep"(%value) : (tensor<64x128xf16, #blocked>) -> ()
    // CHECK: ttg.local_load
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NOT: ttng.fence_async_shared {bCluster = true}
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.cluster_barrier
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %result, %barrier, %true :
      !tt.tensordesc<64x128xf16, #nvmma>, !ttg.memdesc<2xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[0, 0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0], CGALayout = [[0, 0]]}>
#offsets = #ttg.slice<{dim = 0, parent = #blocked}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: multicast_tma_gather_after_cluster_barrier_and_cta_fence
  tt.func @multicast_tma_gather_after_cluster_barrier_and_cta_fence(%desc: !tt.tensordesc<1x32xf32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %offsets = arith.constant dense<0> : tensor<32xi32, #offsets>
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %result : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    "test.keep"(%value) : (tensor<32x32xf32, #blocked>) -> ()
    // CHECK: ttg.local_load
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.async_tma_gather
    ttng.cluster_barrier
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_gather %desc[%offsets, %c0] %result, %barrier, %true {multicast} :
      !tt.tensordesc<1x32xf32, #shared>, tensor<32xi32, #offsets>, i32, !ttg.memdesc<1xi64, #barrier, #smem, mutable>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, i1
    tt.return
  }
}

// -----

#sharedA = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#sharedB = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 2], order = [0, 1], CGALayout = [[1, 0]]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, CGALayout = [[1, 0]], twoCTAs = true>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 8 : i32, "ttng.two-ctas" = true, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: two_cta_mma_after_cluster_barrier_and_cta_fence
  tt.func @two_cta_mma_after_cluster_barrier_and_cta_fence(%value: tensor<256x32xf16, #blocked>) {
    %true = arith.constant true
    %a = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<256x32xf16, #sharedA, #smem, mutable>
    %b = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<32x128xf16, #sharedB, #smem, mutable>
    %acc = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 24576 : i32} : () -> !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    ttg.local_store %value, %a : tensor<256x32xf16, #blocked> -> !ttg.memdesc<256x32xf16, #sharedA, #smem, mutable>
    // CHECK: ttg.local_store
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.tc_gen5_mma
    ttng.cluster_barrier
    ttng.fence_async_shared {bCluster = false}
    ttng.tc_gen5_mma %a, %b, %acc, %true, %true, %barrier[%true] {is_async, two_ctas} :
      !ttg.memdesc<256x32xf16, #sharedA, #smem, mutable>, !ttg.memdesc<32x128xf16, #sharedB, #smem, mutable>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[0, 0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, CGALayout = [[0, 0]]>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttng.two-ctas" = true, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: two_cta_tmem_copy_after_cluster_barrier_and_cta_fence
  tt.func @two_cta_tmem_copy_after_cluster_barrier_and_cta_fence(%value: tensor<128x128xf32, #blocked>) {
    %src = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %dst = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttg.local_store %value, %src : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    // CHECK: ttg.local_store
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.fence_async_shared {bCluster = true}
    // CHECK-NEXT: ttng.tmem_copy
    ttng.cluster_barrier
    ttng.fence_async_shared {bCluster = false}
    ttng.tmem_copy %src, %dst : !ttg.memdesc<128x128xf32, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: disjoint_pipeline_stages_do_not_require_proxy_fence
  tt.func @disjoint_pipeline_stages_do_not_require_proxy_fence(%desc: !tt.tensordesc<64x64xf32, #shared>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %second = ttg.memdesc_index %parent[%c1] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    // CHECK: ttg.local_load
    // LEGACY: ttng.fence_async_shared {bCluster = false}
    // REGION-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    %value = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %second, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: selected_pipeline_stage_may_alias
  tt.func @selected_pipeline_stage_may_alias(%desc: !tt.tensordesc<64x64xf32, #shared>, %choose: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %second = ttg.memdesc_index %parent[%c1] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %selected = arith.select %choose, %first, %second : !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    // CHECK: ttg.local_load
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    %value = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %selected, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: aliasing_callee_arguments_require_proxy_fence
  tt.func private @aliasing_callee_arguments_require_proxy_fence(
      %read: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %write: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.local_load
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.async_tma_copy_global_to_local
    %value = ttg.local_load %read : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %write, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  tt.func public @call_aliasing_callee_arguments(%desc: !tt.tensordesc<64x64xf32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.call @aliasing_callee_arguments_require_proxy_fence(%buffer, %buffer, %desc, %bar) : (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: disjoint_callee_contexts_join_conservatively
  tt.func private @disjoint_callee_contexts_join_conservatively(
      %read: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %write: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.local_load
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.async_tma_copy_global_to_local
    %value = ttg.local_load %read : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %write, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  tt.func public @call_disjoint_callee_contexts(%desc: !tt.tensordesc<64x64xf32, #shared>) {
    %first = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %second = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.call @disjoint_callee_contexts_join_conservatively(%first, %second, %desc, %bar) : (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.call @disjoint_callee_contexts_join_conservatively(%second, %first, %desc, %bar) : (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked_src = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: scratch_reuse_requires_proxy_fence
  tt.func @scratch_reuse_requires_proxy_fence(%source: tensor<128x32xf16, #blocked_src>, %desc: !tt.tensordesc<64x64xf32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.convert_layout
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    %converted = ttg.convert_layout %source {allocation.offset = 0 : i32} : tensor<128x32xf16, #blocked_src> -> tensor<128x32xf16, #dot>
    "test.keep"(%converted) : (tensor<128x32xf16, #dot>) -> ()
    %dst = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: disjoint_scratch_does_not_require_proxy_fence
  tt.func @disjoint_scratch_does_not_require_proxy_fence(%desc: !tt.tensordesc<64x64xf32, #shared>, %ptr: !tt.ptr<i8>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %dst = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 20480 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttng.tensormap_create
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.tensormap_create %ptr, %ptr, [%c0], [%c0], [], [%c0] {allocation.offset = 0 : i32, elem_type = 3 : i32, fill_mode = 1 : i32, interleave_layout = 0 : i32, swizzle_mode = 2 : i32} : (!tt.ptr<i8>, !tt.ptr<i8>, i32, i32, i32) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[1, 0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: same_bytes_in_different_ctas_do_not_alias
  tt.func @same_bytes_in_different_ctas_do_not_alias(%desc: !tt.tensordesc<8x32xi32, #shared>, %choose: i1) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %local = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<16x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    %remote = ttg.memdesc_subslice %parent [8, 0] : !ttg.memdesc<16x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    %selected = arith.select %choose, %remote, %remote : !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    // CHECK: ttg.local_load
    // LEGACY: ttng.fence_async_shared {bCluster = false}
    // REGION-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    %value = ttg.local_load %local : !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %selected, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable, 16x32>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @return_aliasing_stage(
      %parent: !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>)
      -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable> {
    %c0 = arith.constant 0 : i32
    %stage = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return %stage : !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
  }

  // CHECK-LABEL: returned_descriptor_preserves_physical_region
  tt.func public @returned_descriptor_preserves_physical_region(%desc: !tt.tensordesc<64x64xf32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    // CHECK: ttg.local_load
    %value = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    %returned = tt.call @return_aliasing_stage(%parent) : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %returned, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: nested_proxy_leaf
  tt.func private @nested_proxy_leaf(
      %write: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %write, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  tt.func private @nested_proxy_middle(
      %write: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    tt.call @nested_proxy_leaf(%write, %desc, %bar) : (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }

  // CHECK-LABEL: nested_callee_proxy_access_is_visible
  tt.func public @nested_callee_proxy_access_is_visible(%desc: !tt.tensordesc<64x64xf32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @nested_proxy_middle
    tt.call @nested_proxy_middle(%buffer, %desc, %bar) : (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>, !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: control_flow_join_preserves_proxy_hazards
  tt.func public @control_flow_join_preserves_proxy_hazards(%desc: !tt.tensordesc<64x64xf32, #shared>, %choose: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %second = ttg.memdesc_index %parent[%c1] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    cf.cond_br %choose, ^left, ^right
  ^left:
    %left = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%left) : (tensor<64x64xf32, #blocked>) -> ()
    cf.br ^merge
  ^right:
    %right = ttg.local_load %second : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%right) : (tensor<64x64xf32, #blocked>) -> ()
    cf.br ^merge
  ^merge:
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %second, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: disjoint_control_flow_join_avoids_proxy_fence
  tt.func public @disjoint_control_flow_join_avoids_proxy_fence(%desc: !tt.tensordesc<64x64xf32, #shared>, %choose: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %second = ttg.memdesc_index %parent[%c1] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    cf.cond_br %choose, ^left, ^right
  ^left:
    %left = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%left) : (tensor<64x64xf32, #blocked>) -> ()
    cf.br ^merge
  ^right:
    %right = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%right) : (tensor<64x64xf32, #blocked>) -> ()
    cf.br ^merge
  ^merge:
    // LEGACY: ttng.fence_async_shared {bCluster = false}
    // REGION-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %second, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked_src = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The NVIDIA allocator reserves 2,048 bytes for this conversion. The generic
  // allocation analysis incorrectly models it as 8,192 bytes.
  // CHECK-LABEL: exact_nvidia_scratch_size_avoids_proxy_fence
  tt.func @exact_nvidia_scratch_size_avoids_proxy_fence(
      %source: tensor<128x32xf16, #blocked_src>,
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.convert_layout
    // LEGACY: ttng.fence_async_shared {bCluster = false}
    // REGION-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_global_to_local
    %converted = ttg.convert_layout %source {allocation.offset = 0 : i32} : tensor<128x32xf16, #blocked_src> -> tensor<128x32xf16, #dot>
    "test.keep"(%converted) : (tensor<128x32xf16, #dot>) -> ()
    %dst = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: scratch_write_before_async_proxy_read_requires_fence
  tt.func @scratch_write_before_async_proxy_read_requires_fence(
      %source: tensor<128x32xf16, #blocked_src>,
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    // CHECK: ttg.convert_layout
    %converted = ttg.convert_layout %source {allocation.offset = 0 : i32} : tensor<128x32xf16, #blocked_src> -> tensor<128x32xf16, #dot>
    "test.keep"(%converted) : (tensor<128x32xf16, #dot>) -> ()
    %src = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: consan_warp_specialization_captures_require_proxy_fence
  tt.func @consan_warp_specialization_captures_require_proxy_fence(
      %capture: i32, %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.warp_specialize
    ttg.warp_specialize(%capture) attributes {allocation.offset = 0 : i32, "consan.extra_capture_bytes" = 256 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%arg: i32) num_warps(4) {
      ttg.warp_return
    } : (i32) -> ()
    %dst = ttg.local_alloc {allocation.offset = 128 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: initialized_allocation_requires_proxy_fence
  tt.func @initialized_allocation_requires_proxy_fence(
      %value: tensor<8x32xi32, #blocked>,
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.local_alloc
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    %dst = ttg.local_alloc %value {allocation.offset = 0 : i32} : (tensor<8x32xi32, #blocked>) -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: atomic_read_write_requires_proxy_fence
  tt.func @atomic_read_write_requires_proxy_fence(
      %values: tensor<8x32xi32, #blocked>,
      %indices: tensor<8x32xi32, #blocked>,
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %dst = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: ttg.local_atomic_scatter_rmw
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    %old = ttg.local_atomic_scatter_rmw add, %dst[%indices], %values {allocation.offset = 4096 : i32, axis = 0 : i32} : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, tensor<8x32xi32, #blocked>, tensor<8x32xi32, #blocked>) -> tensor<8x32xi32, #blocked>
    "test.keep"(%old) : (tensor<8x32xi32, #blocked>) -> ()
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#inner = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 1, partitionDim = 0, partitionLayout = #inner}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @callee_proxy_write_to_second_partition(
      %bar: !ttg.memdesc<1xi64, #inner, #smem, mutable>) {
    %parent = ttg.local_alloc {allocation.offset = [0 : i32, 1024 : i32]} : () -> !ttg.memdesc<4xi64, #partitioned, #smem, mutable>
    %second = ttg.memdesc_subslice %parent [2] : !ttg.memdesc<4xi64, #partitioned, #smem, mutable> -> !ttg.memdesc<2xi64, #partitioned, #smem, mutable, 4>
    ttng.clc_try_cancel %second, %bar : !ttg.memdesc<2xi64, #partitioned, #smem, mutable, 4>, !ttg.memdesc<1xi64, #inner, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: callee_partition_frame_does_not_alias_caller_buffer
  tt.func public @callee_partition_frame_does_not_alias_caller_buffer() {
    %buffer = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<2xi64, #inner, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<1xi64, #inner, #smem, mutable>
    // CHECK: ttng.clc_load_result
    %result = ttng.clc_load_result %buffer : !ttg.memdesc<2xi64, #inner, #smem, mutable> -> i128
    "test.keep"(%result) : (i128) -> ()
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @callee_proxy_write_to_second_partition
    tt.call @callee_proxy_write_to_second_partition(%bar) {allocation.offset = 3072 : i32} : (!ttg.memdesc<1xi64, #inner, #smem, mutable>) -> ()
    tt.return
  }

  tt.func private @callee_generic_read_from_second_partition() {
    %parent = ttg.local_alloc {allocation.offset = [0 : i32, 1024 : i32]} : () -> !ttg.memdesc<4xi64, #partitioned, #smem, mutable>
    %second = ttg.memdesc_subslice %parent [2] : !ttg.memdesc<4xi64, #partitioned, #smem, mutable> -> !ttg.memdesc<2xi64, #partitioned, #smem, mutable, 4>
    %result = ttng.clc_load_result %second : !ttg.memdesc<2xi64, #partitioned, #smem, mutable, 4> -> i128
    "test.keep"(%result) : (i128) -> ()
    tt.return
  }

  // CHECK-LABEL: callee_partition_summary_translates_selected_base
  tt.func public @callee_partition_summary_translates_selected_base() {
    // CHECK: tt.call @callee_generic_read_from_second_partition
    tt.call @callee_generic_read_from_second_partition() {allocation.offset = 3072 : i32} : () -> ()
    %buffer = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<2xi64, #inner, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #inner, #smem, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.clc_try_cancel
    ttng.clc_try_cancel %buffer, %bar : !ttg.memdesc<2xi64, #inner, #smem, mutable>, !ttg.memdesc<1xi64, #inner, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: structured_control_flow_preserves_aliasing_views
  tt.func @structured_control_flow_preserves_aliasing_views(
      %desc: !tt.tensordesc<64x64xf32, #shared>, %choose: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %second = ttg.memdesc_index %parent[%c1] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %selected = scf.if %choose -> (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>) {
      scf.yield %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    } else {
      scf.yield %second : !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    }
    // CHECK: ttg.local_load
    %value = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %selected, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: loop_carried_views_remain_conservative
  tt.func @loop_carried_views_remain_conservative(
      %desc: !tt.tensordesc<64x64xf32, #shared>, %limit: index) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %second = ttg.memdesc_index %parent[%c1] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %selected = scf.for %iv = %i0 to %limit step %i1 iter_args(%view = %first) -> (!ttg.memdesc<64x64xf32, #shared, #smem, mutable>) {
      scf.yield %second : !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    }
    // CHECK: ttg.local_load
    %value = ttg.local_load %first : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %selected, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#scale_shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [128, 0]]}, alignment = 128>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#reshape = #ttg.blocked<{sizePerThread = [1, 1, 1, 2, 4], threadsPerWarp = [1, 1, 16, 2, 1], warpsPerCTA = [2, 1, 2, 1, 1], order = [4, 3, 2, 1, 0]}>
#transpose = #ttg.blocked<{sizePerThread = [1, 2, 1, 1, 4], threadsPerWarp = [1, 2, 16, 1, 1], warpsPerCTA = [2, 1, 2, 1, 1], order = [4, 1, 2, 3, 0]}>
#scale = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0]], lane = [[64, 0], [1, 0], [2, 0], [4, 0], [8, 0]], warp = [[16, 0], [128, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: shared_mma_scales_require_async_proxy_fence
  tt.func @shared_mma_scales_require_async_proxy_fence(
      %values: tensor<2x512xi8, #blocked>) {
    %true = arith.constant true
    %reshaped = tt.reshape %values : tensor<2x512xi8, #blocked> -> tensor<2x1x32x4x4xi8, #reshape>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 3, 2, 1, 4>} : tensor<2x1x32x4x4xi8, #reshape> -> tensor<2x4x32x1x4xi8, #transpose>
    %scales = tt.reshape %transposed : tensor<2x4x32x1x4xi8, #transpose> -> tensor<256x4xi8, #scale>
    %aScale = ttg.local_alloc %scales {allocation.offset = 0 : i32} : (tensor<256x4xi8, #scale>) -> !ttg.memdesc<256x4xi8, #scale_shared, #smem>
    %bScale = ttg.local_alloc %scales {allocation.offset = 1024 : i32} : (tensor<256x4xi8, #scale>) -> !ttg.memdesc<256x4xi8, #scale_shared, #smem>
    %a = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem, mutable>
    %b = ttg.local_alloc {allocation.offset = 20480 : i32} : () -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem, mutable>
    %acc = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.tc_gen5_mma_scaled
    ttng.tc_gen5_mma_scaled %a, %b, %acc, %aScale, %bScale, %true, %true lhs = e5m2 rhs = e5m2 : !ttg.memdesc<128x128xf8E5M2, #shared, #smem, mutable>, !ttg.memdesc<128x128xf8E5M2, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x4xi8, #scale_shared, #smem>, !ttg.memdesc<256x4xi8, #scale_shared, #smem>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[0, 0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: callee_multicast_proxy_write
  tt.func private @callee_multicast_proxy_write(
      %write: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<8x32xi32, #shared>,
      %bar: !ttg.memdesc<2xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttng.fence_async_shared {bCluster = true}
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %write, %bar, %true {multicast} : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<2xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: callee_multicast_requires_cluster_fence
  tt.func public @callee_multicast_requires_cluster_fence(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    %value = ttg.local_load %buffer : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    // CHECK: ttng.fence_async_shared {bCluster = false}
    ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: tt.call @callee_multicast_proxy_write
    tt.call @callee_multicast_proxy_write(%buffer, %desc, %bar) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<2xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: shared_callee_contexts_refresh_after_fence_insertion
  tt.func private @shared_callee_contexts_refresh_after_fence_insertion(
      %read: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>,
      %write: !ttg.memdesc<8x32xi32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<8x32xi32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %value = ttg.local_load %read : !ttg.memdesc<8x32xi32, #shared, #smem, mutable> -> tensor<8x32xi32, #blocked>
    "test.keep"(%value) : (tensor<8x32xi32, #blocked>) -> ()
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %write, %bar, %true : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: caller_reuses_fenced_callee_without_extra_fences
  tt.func public @caller_reuses_fenced_callee_without_extra_fences(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x8x32xi32, #shared, #smem, mutable>
    %first = ttg.memdesc_index %parent[%c0] : !ttg.memdesc<2x8x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %second = ttg.memdesc_index %parent[%c1] : !ttg.memdesc<2x8x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @shared_callee_contexts_refresh_after_fence_insertion
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @shared_callee_contexts_refresh_after_fence_insertion
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @shared_callee_contexts_refresh_after_fence_insertion
    tt.call @shared_callee_contexts_refresh_after_fence_insertion(%first, %second, %desc, %bar) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.call @shared_callee_contexts_refresh_after_fence_insertion(%second, %first, %desc, %bar) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.call @shared_callee_contexts_refresh_after_fence_insertion(%first, %second, %desc, %bar) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }

  // CHECK-LABEL: second_kernel_context_requires_shared_callee_fence
  tt.func public @second_kernel_context_requires_shared_callee_fence(
      %desc: !tt.tensordesc<8x32xi32, #shared>) {
    %first = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: tt.call @shared_callee_contexts_refresh_after_fence_insertion
    tt.call @shared_callee_contexts_refresh_after_fence_insertion(%first, %first, %desc, %bar) : (!ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !ttg.memdesc<8x32xi32, #shared, #smem, mutable>, !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}
