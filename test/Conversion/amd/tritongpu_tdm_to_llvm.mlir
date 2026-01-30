// RUN: triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_load
  tt.func public @tdm_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c_offset = arith.constant 0 : i32
    %c_pred = arith.constant 1 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <tensor<64x64xf16, #shared>>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK-COUNT-4: llvm.insertelement{{.*}} : vector<4xi32>
    // CHECK-COUNT-8: llvm.insertelement{{.*}} : vector<8xi32>
    // CHECK: llvm.amdgcn.tensor.load.to.lds.d2{{.*}} : (vector<4xi32>, vector<8xi32>, i32) -> ()
    %2 = amdg.async_tdm_copy_global_to_local %0[%c_offset, %c_offset] into %1, %c_pred : !tt.tensordesc<tensor<64x64xf16, #shared>> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: rocdl.s.wait.tensorcnt 0
    %3 = amdg.async_tdm_wait  {num = 0 : i32}
    %4 = ttg.local_load %1 : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_store
  tt.func public @tdm_store(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c_offset = arith.constant 0 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <tensor<64x64xf16, #shared>>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %2 = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #blocked>
    ttg.local_store %2, %1 : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK-COUNT-4: llvm.insertelement{{.*}} : vector<4xi32>
    // CHECK-COUNT-8: llvm.insertelement{{.*}} : vector<8xi32>
    // CHECK: llvm.amdgcn.tensor.store.from.lds.d2{{.*}} : (vector<4xi32>, vector<8xi32>, i32) -> ()
    amdg.async_tdm_copy_local_to_global %0[%c_offset, %c_offset] from %1: !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !tt.tensordesc<tensor<64x64xf16, #shared>>
    // CHECK: rocdl.s.wait.tensorcnt 0
    %3 = amdg.async_tdm_wait  {num = 0 : i32}
    tt.return
  }
}

// -----

// Check that CTA offsets are computed and applied to base pointer for multi-cta layouts
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_load_multi_cta
  tt.func public @tdm_load_multi_cta(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c_offset = arith.constant 0 : i32
    %c_pred = arith.constant 1 : i32

    // CHECK-DAG: %[[STRIDE0:.*]] = llvm.mlir.constant(128 : i64) : i64
    // CHECK-DAG: %[[STRIDE1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG: llvm.call_intrinsic "llvm.amdgcn.cluster.workgroup.id.x"
    // CHECK-DAG: %[[STRIDE0_TRUNC:.*]] = llvm.trunc %[[STRIDE0]] : i64 to i32
    // CHECK: %[[OFFSET_DIM0:.*]] = llvm.mul{{.*}}%[[STRIDE0_TRUNC]]
    // CHECK: %[[OFFSET_TMP1:.*]] = llvm.add{{.*}}%[[OFFSET_DIM0]]
    // CHECK: %[[OFFSET_DIM1:.*]] = llvm.mul{{.*}}%[[STRIDE1]]
    // CHECK: %[[TOTAL_OFFSET:.*]] = llvm.add %[[OFFSET_TMP1]], %[[OFFSET_DIM1]]
    // CHECK: %[[ADJUSTED_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[TOTAL_OFFSET]]]
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <tensor<64x64xf16, #shared>>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>

    // CHECK: llvm.amdgcn.tensor.load.to.lds.d2{{.*}} : (vector<4xi32>, vector<8xi32>, i32) -> ()
    %2 = amdg.async_tdm_copy_global_to_local %0[%c_offset, %c_offset] into %1, %c_pred : !tt.tensordesc<tensor<64x64xf16, #shared>> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Check that CTA offsets are computed and applied to base pointer for multi-cta layouts (store)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1]]}>
#blocked_store = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_store_multi_cta
  tt.func public @tdm_store_multi_cta(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c_offset = arith.constant 0 : i32

    // CHECK-DAG: %[[STRIDE0:.*]] = llvm.mlir.constant(128 : i64) : i64
    // CHECK-DAG: %[[STRIDE1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG: llvm.call_intrinsic "llvm.amdgcn.cluster.workgroup.id.x"
    // CHECK-DAG: %[[STRIDE0_TRUNC:.*]] = llvm.trunc %[[STRIDE0]] : i64 to i32
    // CHECK: %[[OFFSET_DIM0:.*]] = llvm.mul{{.*}}%[[STRIDE0_TRUNC]]
    // CHECK: %[[OFFSET_TMP1:.*]] = llvm.add{{.*}}%[[OFFSET_DIM0]]
    // CHECK: %[[OFFSET_DIM1:.*]] = llvm.mul{{.*}}%[[STRIDE1]]
    // CHECK: %[[TOTAL_OFFSET:.*]] = llvm.add %[[OFFSET_TMP1]], %[[OFFSET_DIM1]]
    // CHECK: %[[ADJUSTED_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[TOTAL_OFFSET]]]
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <tensor<64x64xf16, #shared>>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: llvm.amdgcn.tensor.store.from.lds.d2{{.*}} : (vector<4xi32>, vector<8xi32>, i32) -> ()
    amdg.async_tdm_copy_local_to_global %0[%c_offset, %c_offset] from %1: !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !tt.tensordesc<tensor<64x64xf16, #shared>>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1], [0, 2], [0, 0], [0, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 16 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_load_multicast
  tt.func public @tdm_load_multicast(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c_offset = arith.constant 0 : i32
    %c_pred = arith.constant 1 : i32

    // Check we compute the multicast mask and used it in the second group of SGPRs (vector<8xi32>)
    // CHECK-DAG: %[[GROUP_MASK:.*]] = llvm.mlir.constant(4369 : i32) : i32
    // CHECK-DAG: %[[NON_FREE_BITS:.*]] = llvm.mlir.constant(-13 : i32) : i32
    // CHECK-DAG: %[[CTA_ID:.*]] = {{.*}}llvm.amdgcn.cluster.workgroup.id.x
    // CHECK: %[[SHIFT_AMOUNT:.*]] = llvm.and %[[CTA_ID]], %[[NON_FREE_BITS]]
    // CHECK: %[[CTA_MASK:.*]] = llvm.shl %[[GROUP_MASK]], %[[SHIFT_AMOUNT]]
    // Combine with other values
    // CHECK: %[[TMP:.*]] = llvm.or %{{.*}}, %[[CTA_MASK]]
    // CHECK: %[[TMP2:.*]] = llvm.and %[[TMP]]
    // CHECK-NOT: llvm.insertelement{{.*}} : vector<8xi32>
    // CHECK: llvm.insertelement %[[TMP2]]
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <tensor<64x64xf16, #shared>>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>


    // CHECK: llvm.amdgcn.tensor.load.to.lds.d2{{.*}} : (vector<4xi32>, vector<8xi32>, i32) -> ()
    %2 = amdg.async_tdm_copy_global_to_local %0[%c_offset, %c_offset] into %1, %c_pred : !tt.tensordesc<tensor<64x64xf16, #shared>> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [64, 64]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_prefetch_regular
  tt.func public @tdm_prefetch_regular(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c_offset = arith.constant 0 : i32
    %c_pred = arith.constant true
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <tensor<64x64xf16, #shared>>

    // CHECK-DAG: %[[NON_SPECULATIVE_BITS:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG: %[[SPECULATIVE_BITS:.*]] = llvm.mlir.constant(9 : i32) : i32

    // CHECK: llvm.amdgcn.global.prefetch{{.*}}%[[NON_SPECULATIVE_BITS]]
    amdg.tdm_prefetch %0[%c_offset, %c_offset], %c_pred, speculative = false : !tt.tensordesc<tensor<64x64xf16, #shared>>

    // CHECK: llvm.amdgcn.global.prefetch{{.*}}%[[SPECULATIVE_BITS]]
    amdg.tdm_prefetch %0[%c_offset, %c_offset], %c_pred, speculative = true : !tt.tensordesc<tensor<64x64xf16, #shared>>
    tt.return
  }
}
