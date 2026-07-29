// RUN: triton-opt %s -triton-nvidia-gpu-proxy-fence-insertion --split-input-file -allow-unregistered-dialect | FileCheck %s

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
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
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
    %dst = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
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
  // CHECK-LABEL: unknown_descriptors_require_proxy_fence
  tt.func public @unknown_descriptors_require_proxy_fence(
      %read: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %write: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
      %desc: !tt.tensordesc<64x64xf32, #shared>,
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: ttg.local_load
    %value = ttg.local_load %read : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %write, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
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
    %value = ttg.local_load %read : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    "test.keep"(%value) : (tensor<64x64xf32, #blocked>) -> ()
    // CHECK: ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
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

#blocked_src = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
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
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src : !tt.tensordesc<8x32xi32, #shared>, !ttg.memdesc<8x32xi32, #shared, #smem, mutable>
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
    // CHECK-NEXT: ttng.tc_gen5_mma_scaled
    ttng.tc_gen5_mma_scaled %a, %b, %acc, %aScale, %bScale, %true, %true lhs = e5m2 rhs = e5m2 : !ttg.memdesc<128x128xf8E5M2, #shared, #smem, mutable>, !ttg.memdesc<128x128xf8E5M2, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x4xi8, #scale_shared, #smem>, !ttg.memdesc<256x4xi8, #scale_shared, #smem>
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
}
