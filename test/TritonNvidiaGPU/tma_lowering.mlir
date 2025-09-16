// RUN: triton-opt %s -split-input-file --triton-nvidia-tma-lowering | FileCheck %s
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tma_load
// CHECK: ttg.local_alloc : ()
// CHECK: ttg.local_alloc : ()
// CHECK: ttng.init_barrier
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.wait_barrier
// CHECK: ttng.inval_barrier
// CHECK: ttg.local_load
  tt.func public @tma_load(%arg0: !tt.tensordesc<tensor<128x64xf16, #nvmma_128>>, %arg1: i32) -> tensor<128x64xf16, #blocked> {
    %l = tt.descriptor_load %arg0[%arg1, %arg1] : !tt.tensordesc<tensor<128x64xf16, #nvmma_128>> -> tensor<128x64xf16, #blocked>
    tt.return %l : tensor<128x64xf16, #blocked>
  }
}

// -----
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tma_store
//       CHECK: ttg.local_alloc {{.*}} -> !ttg.memdesc<128x256xf32, #shared, #smem>
//       CHECK: ttng.fence_async_shared {bCluster = false}
//       CHECK: ttng.async_tma_copy_local_to_global
  tt.func public @tma_store(%arg0: !tt.tensordesc<tensor<128x256xf32, #nvmma_128>>, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: tensor<128x256xf32, #blocked>) {
    tt.descriptor_store %arg0[%arg1, %arg1], %arg2 : !tt.tensordesc<tensor<128x256xf32, #nvmma_128>>, tensor<128x256xf32, #blocked>
    tt.return
  }
}

// -----
#nvmma_32 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: make_tensor_descriptor
  // CHECK: %0 = arith.extsi %arg2 : i32 to i64
  // CHECK: %1 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
  // CHECK: ttng.tensormap_create %1, %arg0, [%c32_i32, %c8_i32], [%arg2, %arg1], [%0], [%c1_i32, %c1_i32] {elem_type = 0 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 1 : i32} : (!tt.ptr<i8>, !tt.ptr<i8>, i32, i32, i32, i32, i64, i32, i32) -> ()
  // CHECK: ttng.tensormap_fenceproxy_acquire %1 : !tt.ptr<i8>
  // CHECK: ttng.reinterpret_tensor_descriptor %1 : !tt.ptr<i8> to !tt.tensordesc<tensor<8x32xi8, #shared>>
  tt.func public @make_tensor_descriptor(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32} ) -> !tt.tensordesc<tensor<8x32xi8, #nvmma_32>> {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<32> : tensor<8x1xi32>
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.extsi %arg2 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : !tt.ptr<i8>, !tt.tensordesc<tensor<8x32xi8, #nvmma_32>>
    tt.return %1 : !tt.tensordesc<tensor<8x32xi8, #nvmma_32>>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @tma_gather
tt.func @tma_gather(%arg0: !tt.tensordesc<tensor<1x128xbf16, #nvmma_128>>, %arg1: tensor<32xi32, #blocked>, %arg2: i32) -> tensor<32x128xbf16, #blocked1> {
  // CHECK: [[RESULT:%.*]] = ttg.local_alloc
  // CHECK: [[BARRIER:%.*]] = ttg.local_alloc
  // CHECK: ttng.init_barrier [[BARRIER]]
  // CHECK: ttng.async_tma_gather %arg0[%arg1, %arg2] [[RESULT]], [[BARRIER]], %true
  // CHECK: ttng.wait_barrier [[BARRIER]]
  // CHECK: ttng.inval_barrier [[BARRIER]]
  // CHECK: [[OUT:%.*]] = ttg.local_load [[RESULT]]
  %0 = tt.descriptor_gather %arg0[%arg1, %arg2] : (!tt.tensordesc<tensor<1x128xbf16, #nvmma_128>>, tensor<32xi32, #blocked>, i32) -> tensor<32x128xbf16, #blocked1>
  // CHECK: return [[OUT]]
  tt.return %0 : tensor<32x128xbf16, #blocked1>
}

// CHECK-LABEL: @tma_scatter
tt.func @tma_scatter(%arg0: !tt.tensordesc<tensor<1x128xbf16, #nvmma_128>>, %arg1: tensor<32xi32, #blocked>, %arg2: i32, %arg3: tensor<32x128xbf16, #blocked1>) {
  // CHECK-NEXT: [[SRC:%.*]] = ttg.local_alloc %arg3
  // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
  // CHECK-NEXT: ttng.async_tma_scatter %arg0[%arg1, %arg2] [[SRC]]
  // CHECK-NEXT: ttng.async_tma_store_wait
  tt.descriptor_scatter %arg0[%arg1, %arg2], %arg3 : !tt.tensordesc<tensor<1x128xbf16, #nvmma_128>>, tensor<32xi32, #blocked>, i32, tensor<32x128xbf16, #blocked1>
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 2, 0]}>
// CHECK: #[[$SHARED:.+]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABLE: @rank_reducing_load
  tt.func public @rank_reducing_load(%arg0: !tt.tensordesc<tensor<1x256x32xf32, #shared>>) -> tensor<256x32xf32, #blocked> {
      %c32_i32 = arith.constant 32 : i32
      // CHECK: %[[A:.+]] = ttg.local_alloc : () -> !ttg.memdesc<256x32xf32, #[[$SHARED]], #smem, mutable>
      // CHECK: tng.async_tma_copy_global_to_local %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] %[[A]],
      %l = tt.descriptor_load %arg0[%c32_i32, %c32_i32, %c32_i32] : !tt.tensordesc<tensor<1x256x32xf32, #shared>> -> tensor<256x32xf32, #blocked>
      tt.return %l : tensor<256x32xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tma_load_alloc_user
  tt.func public @tma_load_alloc_user(%arg0: !tt.tensordesc<tensor<64x64xf32, #shared>>, %arg1: i32) -> (tensor<64x64xf32, #blocked>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable>) {
    %0 = tt.descriptor_load %arg0[%arg1, %arg1, %arg1] : !tt.tensordesc<tensor<64x64xf32, #shared>> -> tensor<64x64xf32, #blocked>
    // CHECK: %[[A:.+]] = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32
    // CHECK: ttng.async_tma_copy_global_to_local {{.*}} %[[A]],
    %1 = ttg.local_alloc %0 : (tensor<64x64xf32, #blocked>) -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    // CHECK: %[[L:.+]] = ttg.local_load %[[A]] :
    // CHECK: tt.return %[[L]], %[[A]] :
    tt.return %0, %1 : tensor<64x64xf32, #blocked>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#shared2 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tma_load_double_use
  tt.func public @tma_load_double_use(%arg0: !tt.tensordesc<tensor<64x32xf32, #shared>>, %arg1: !tt.tensordesc<tensor<64x64xf32, #shared1>>) -> tensor<64x32xf32, #mma1> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma1>
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    // CHECK: %[[A:.+]] = ttg.local_alloc : () -> !ttg.memdesc<64x32xf32
    %0 = tt.descriptor_load %arg0[%c64_i32, %c32_i32] : !tt.tensordesc<tensor<64x32xf32, #shared>> -> tensor<64x32xf32, #blocked>
    // CHECK: %[[B:.+]] = ttg.local_load %[[A]]
    // CHECK: %[[C:.+]] = ttg.local_alloc %[[B]]
    %1 = ttg.local_alloc %0 : (tensor<64x32xf32, #blocked>) -> !ttg.memdesc<64x32xf32, #shared1, #smem>
    // CHECK: %[[D:.+]] = ttg.memdesc_trans %[[C]]
    %2 = ttg.memdesc_trans %1 {order = array<i32: 1, 0>} : !ttg.memdesc<64x32xf32, #shared1, #smem> -> !ttg.memdesc<32x64xf32, #shared2, #smem>
    %3 = ttg.local_alloc %0 : (tensor<64x32xf32, #blocked>) -> !ttg.memdesc<64x32xf32, #shared, #smem>
    // CHECK: %[[E:.+]] = ttg.local_load %[[D]]
    %4 = ttg.local_load %2 : !ttg.memdesc<32x64xf32, #shared2, #smem> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    // CHECK: %[[F:.+]] = ttg.local_load %[[A]]
    %5 = ttg.local_load %3 : !ttg.memdesc<64x32xf32, #shared, #smem> -> tensor<64x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    // CHECK: %[[G:.+]] = tt.dot %[[E]], %[[F]]
    %6 = tt.dot %4, %5, %cst, inputPrecision = tf32 : tensor<32x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    // CHECK: %[[H:.+]] = ttg.local_alloc %[[G]]
    %7 = ttg.local_alloc %6 : (tensor<32x32xf32, #mma>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    // CHECK: {{.*}} = ttng.warp_group_dot %[[A]], %[[H]]
    %8 = ttng.warp_group_dot %3, %7, %cst_0 {isAsync = true} : !ttg.memdesc<64x32xf32, #shared, #smem> * !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<64x32xf32, #mma1>
    %9:3 = ttng.warp_group_dot_wait %8, %3, %7 {pendings = 0 : i32} : tensor<64x32xf32, #mma1>, !ttg.memdesc<64x32xf32, #shared, #smem>, !ttg.memdesc<32x32xf32, #shared, #smem>
    tt.return %9 : tensor<64x32xf32, #mma1>
  }
}
