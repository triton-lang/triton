// RUN: triton-opt %s -triton-nvidia-gpu-proxy-fence-insertion --split-input-file -allow-unregistered-dialect | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: fence_write_after_read
  tt.func @fence_write_after_read(%arg0: !tt.tensordesc<64x64xf32, #shared>, %arg1: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) {
    // CHECK: ttg.local_load
    // CHECK: ttng.fence_async_shared
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
    // CHECK-NEXT: ttng.fence_async_shared
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
    // CHECK-NEXT: ttng.fence_async_shared
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
