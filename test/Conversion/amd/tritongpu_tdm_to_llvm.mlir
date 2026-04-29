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
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK-COUNT-4: llvm.insertelement{{.*}} : vector<4xi32>
    // CHECK-COUNT-8: llvm.insertelement{{.*}} : vector<8xi32>
    // CHECK: "llvm.amdgcn.tensor.load.to.lds"({{.+}}) : (vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32>, vector<8xi32>, i32) -> ()
    %2 = amdg.async_tdm_copy_global_to_local %0[%c_offset, %c_offset] into %1, pred = %c_pred : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: rocdl.s.wait.tensorcnt 0
    %3 = amdg.async_tdm_intrinsic_wait  {count = 0 : i32}
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
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %2 = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #blocked>
    ttg.local_store %2, %1 : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK-COUNT-4: llvm.insertelement{{.*}} : vector<4xi32>
    // CHECK-COUNT-8: llvm.insertelement{{.*}} : vector<8xi32>
    // CHECK: "llvm.amdgcn.tensor.store.from.lds"({{.+}}) : (vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32>, vector<8xi32>, i32) -> ()
    amdg.async_tdm_copy_local_to_global %0[%c_offset, %c_offset] from %1: !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x64xf16, #shared>
    // CHECK: rocdl.s.wait.tensorcnt 0
    %3 = amdg.async_tdm_intrinsic_wait  {count = 0 : i32}
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
    // CHECK: %[[STRIDE0_TRUNC:.*]] = llvm.trunc %[[STRIDE0]] : i64 to i32
    // CHECK: %[[OFFSET_DIM0:.*]] = llvm.mul{{.*}}%[[STRIDE0_TRUNC]]
    // CHECK: %[[OFFSET_TMP1:.*]] = llvm.add{{.*}}%[[OFFSET_DIM0]]
    // CHECK: %[[OFFSET_DIM1:.*]] = llvm.mul{{.*}}%[[STRIDE1]]
    // CHECK: %[[TOTAL_OFFSET:.*]] = llvm.add %[[OFFSET_TMP1]], %[[OFFSET_DIM1]]
    // CHECK: %[[ADJUSTED_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[TOTAL_OFFSET]]]
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>

    // CHECK: "llvm.amdgcn.tensor.load.to.lds"({{.+}}) : (vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32>, vector<8xi32>, i32) -> ()
    %2 = amdg.async_tdm_copy_global_to_local %0[%c_offset, %c_offset] into %1, pred = %c_pred : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
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
    // CHECK-DAG: rocdl.cluster.workgroup.id.x
    // CHECK-DAG: %[[STRIDE0_TRUNC:.*]] = llvm.trunc %[[STRIDE0]] : i64 to i32
    // CHECK: %[[OFFSET_DIM0:.*]] = llvm.mul{{.*}}%[[STRIDE0_TRUNC]]
    // CHECK: %[[OFFSET_TMP1:.*]] = llvm.add{{.*}}%[[OFFSET_DIM0]]
    // CHECK: %[[OFFSET_DIM1:.*]] = llvm.mul{{.*}}%[[STRIDE1]]
    // CHECK: %[[TOTAL_OFFSET:.*]] = llvm.add %[[OFFSET_TMP1]], %[[OFFSET_DIM1]]
    // CHECK: %[[ADJUSTED_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[TOTAL_OFFSET]]]
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: "llvm.amdgcn.tensor.store.from.lds"({{.+}}) : (vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32>, vector<8xi32>, i32) -> ()
    amdg.async_tdm_copy_local_to_global %0[%c_offset, %c_offset] from %1: !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x64xf16, #shared>
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
    // CHECK-DAG: %[[CTA_ID:.*]] = rocdl.cluster.workgroup.id.x
    // CHECK: %[[SHIFT_AMOUNT:.*]] = llvm.and %[[CTA_ID]], %[[NON_FREE_BITS]]
    // CHECK: %[[CTA_MASK:.*]] = llvm.shl %[[GROUP_MASK]], %[[SHIFT_AMOUNT]]
    // Combine with other values
    // CHECK: %[[TMP:.*]] = llvm.or %{{.*}}, %[[CTA_MASK]]
    // CHECK: %[[TMP2:.*]] = llvm.and %[[TMP]]
    // CHECK-NOT: llvm.insertelement{{.*}} : vector<8xi32>
    // CHECK: llvm.insertelement %[[TMP2]]
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>


    // CHECK: "llvm.amdgcn.tensor.load.to.lds"({{.+}}) : (vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32>, vector<8xi32>, i32) -> ()
    %2 = amdg.async_tdm_copy_global_to_local %0[%c_offset, %c_offset] into %1, pred = %c_pred : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
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
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>

    // CHECK: rocdl.global.prefetch %{{.*}} scope 8
    amdg.tdm_prefetch %0[%c_offset, %c_offset], %c_pred, speculative = false : !tt.tensordesc<64x64xf16, #shared>

    // CHECK: rocdl.global.prefetch %{{.*}} scope 9
    amdg.tdm_prefetch %0[%c_offset, %c_offset], %c_pred, speculative = true : !tt.tensordesc<64x64xf16, #shared>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_wait_inst
  tt.func public @tdm_wait_inst() {
    // CHECK: rocdl.s.wait.tensorcnt 0
    %3 = amdg.async_tdm_intrinsic_wait  {count = 0 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.padded_shared<[64:+4] {order = [1, 0], shape = [128, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_2d_with_padding
  tt.func public @tdm_2d_with_padding(
    %tensorDesc: !tt.tensordesc<128x64xf16>,
    %memDesc: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  ) {
    %c0_i32 = arith.constant 0 : i32
    amdg.async_tdm_copy_local_to_global %tensorDesc[%c0_i32, %c0_i32] from %memDesc: !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !tt.tensordesc<128x64xf16>
    // CHECK: "llvm.amdgcn.tensor.store.from.lds"({{.+}}) : (vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32>, vector<8xi32>, i32) -> ()
    tt.return
  }
}

// -----

#shared_5d = #ttg.padded_shared<[16:+4] {order = [4, 3, 2, 1, 0], shape = [8, 8, 8, 16, 16]}>
#smem_5d = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_5d_with_padding
  tt.func public @tdm_5d_with_padding(
    %tensorDesc: !tt.tensordesc<8x8x8x16x16xf16>,
    %memDesc: !ttg.memdesc<8x8x8x16x16xf16, #shared_5d, #smem_5d, mutable>
  ) {
    %c0_i32 = arith.constant 0 : i32
    amdg.async_tdm_copy_local_to_global %tensorDesc[%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32] from %memDesc: !ttg.memdesc<8x8x8x16x16xf16, #shared_5d, #smem_5d, mutable> -> !tt.tensordesc<8x8x8x16x16xf16>
    // CHECK: "llvm.amdgcn.tensor.store.from.lds"({{.+}}) : (vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32>, vector<8xi32>, i32) -> ()
    tt.return
  }
}

// -----

// Scatter with padded shared layout: padding interval = innermost block dim.
#idx_parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#idx_layout = #ttg.slice<{dim = 1, parent = #idx_parent}>
#shared_scatter_pad = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [8, 64]}>
#smem_scatter_pad = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_scatter_with_padding
  tt.func public @tdm_scatter_with_padding(
    %tensorDesc: !tt.tensordesc<8x64xf16>,
    %memDesc: !ttg.memdesc<8x64xf16, #shared_scatter_pad, #smem_scatter_pad, mutable>,
    %row_indices: tensor<8xi32, #idx_layout>
  ) {
    %c0_i32 = arith.constant 0 : i32
    amdg.async_tdm_scatter %tensorDesc[%row_indices, %c0_i32] from %memDesc : tensor<8xi32, #idx_layout>, !ttg.memdesc<8x64xf16, #shared_scatter_pad, #smem_scatter_pad, mutable> -> !tt.tensordesc<8x64xf16>
    // CHECK: "llvm.amdgcn.tensor.store.from.lds"({{.+}}) : (vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32>, vector<8xi32>, i32) -> ()
    tt.return
  }
}

// -----

// Partial TDM copy: warp_used_hint = 0x0F picks K=4 active warps out of 8.
// The warp sublayout of the TDM LinearLayout is an identity over K=4 warps
// (low 2 bits of warpId drive offsets), so the free-variable mask for the
// "warp" input dim is 0b100 (bit 2 is redundant).  Predication is
// (warpId & 4) == 0, i.e. warps 0..3 are active and warps 4..7 issue a
// no-op TDM.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_load_warp_used_hint_predication
  tt.func public @tdm_load_warp_used_hint_predication(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 256 : i32
    %c_stride0 = arith.constant 256 : i64
    %c_stride1 = arith.constant 1 : i64
    %c_offset = arith.constant 0 : i32
    %c_pred = arith.constant 1 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <256x64xf16, #shared>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    // CHECK-DAG: %[[FREE_MASK:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[MASKED:.*]] = llvm.and %{{.*}}, %[[FREE_MASK]] : i32
    // CHECK: %[[IS_ACTIVE:.*]] = llvm.icmp "eq" %[[MASKED]], %[[ZERO]] : i32
    // CHECK: %[[LAYOUT_PRED:.*]] = llvm.select %[[IS_ACTIVE]], %{{.*}}, %{{.*}} : i1, i32
    // CHECK: llvm.and %{{.*}}, %[[LAYOUT_PRED]] : i32
    // CHECK: "llvm.amdgcn.tensor.load.to.lds"
    %2 = amdg.async_tdm_copy_global_to_local %0[%c_offset, %c_offset] into %1, pred = %c_pred {warp_used_hint = 15 : i32} : !tt.tensordesc<256x64xf16, #shared> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Partial TDM copy with partitioned layout.  warp_used_hint = 0x03 selects
// K=2 active warps out of 4, matching numLogicalPieces (2) so the copy
// fits in a single TDM instruction.  The redundant warp bit is bit 1 of
// warpId (free mask = 2).
#shared_inner = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 1, partitionDim = 0, partitionLayout = #shared_inner}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_load_warp_used_hint_partitioned
  tt.func public @tdm_load_warp_used_hint_partitioned(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 256 : i32
    %c_stride0 = arith.constant 256 : i64
    %c_stride1 = arith.constant 1 : i64
    %c_offset = arith.constant 0 : i32
    %c_pred = arith.constant 1 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <128x16xf16, #partitioned>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>
    // CHECK-DAG: %[[FREE_MASK:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[MASKED:.*]] = llvm.and %{{.*}}, %[[FREE_MASK]] : i32
    // CHECK: llvm.icmp "eq" %[[MASKED]], %[[ZERO]] : i32
    // CHECK: "llvm.amdgcn.tensor.load.to.lds"
    %2 = amdg.async_tdm_copy_global_to_local %0[%c_offset, %c_offset] into %1, pred = %c_pred {warp_used_hint = 3 : i32} : !tt.tensordesc<128x16xf16, #partitioned> -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>
    tt.return
  }
}

// -----

// Implicit merge: two adjacent TDM copies with disjoint hints whose
// union is a verifier-legal axis-aligned coset (0x0F | 0xF0 == 0xFF, K=8)
// must collapse into ONE `llvm.amdgcn.tensor.load.to.lds` intrinsic.  The
// merged emit lifts both members' descriptors and selects between them
// per-wave; we only check the resulting intrinsic count and the
// presence of a uniform `llvm.select` chain (the per-wave member
// selector).
#shared_merge = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_merge = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_merge_disjoint_legal_union
  tt.func public @tdm_merge_disjoint_legal_union(
    %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}
  ) {
    %c_shape = arith.constant 256 : i32
    %c_stride0 = arith.constant 256 : i64
    %c_stride1 = arith.constant 1 : i64
    %c_offset = arith.constant 0 : i32
    %c_pred = arith.constant 1 : i32
    %a_desc = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared_merge>
    %b_desc = tt.make_tensor_descriptor %arg1, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared_merge>
    %a_buf = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared_merge, #smem_merge, mutable>
    %b_buf = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared_merge, #smem_merge, mutable>
    // CHECK-COUNT-1: "llvm.amdgcn.tensor.load.to.lds"
    // CHECK-NOT: "llvm.amdgcn.tensor.load.to.lds"
    // CHECK: rocdl.s.wait.tensorcnt 0
    %ta = amdg.async_tdm_copy_global_to_local %a_desc[%c_offset, %c_offset] into %a_buf, pred = %c_pred {warp_used_hint = 15 : i32} : !tt.tensordesc<64x64xf16, #shared_merge> -> !ttg.memdesc<64x64xf16, #shared_merge, #smem_merge, mutable>
    %tb = amdg.async_tdm_copy_global_to_local %b_desc[%c_offset, %c_offset] into %b_buf, pred = %c_pred {warp_used_hint = 240 : i32} : !tt.tensordesc<64x64xf16, #shared_merge> -> !ttg.memdesc<64x64xf16, #shared_merge, #smem_merge, mutable>
    %w = amdg.async_tdm_intrinsic_wait {count = 0 : i32}
    tt.return
  }
}

// -----

// Implicit merge negative case: one op carries `warp_used_hint`, the
// other does not.  The merge analysis requires every member to have a
// hint, so the pair is NOT fused; the lowering emits TWO
// `llvm.amdgcn.tensor.load.to.lds` intrinsics.
#shared_no_merge = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_no_merge = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_no_merge_one_hinted
  tt.func public @tdm_no_merge_one_hinted(
    %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}
  ) {
    %c_shape = arith.constant 256 : i32
    %c_stride0 = arith.constant 256 : i64
    %c_stride1 = arith.constant 1 : i64
    %c_offset = arith.constant 0 : i32
    %c_pred = arith.constant 1 : i32
    %a_desc = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared_no_merge>
    %b_desc = tt.make_tensor_descriptor %arg1, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared_no_merge>
    %a_buf = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared_no_merge, #smem_no_merge, mutable>
    %b_buf = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared_no_merge, #smem_no_merge, mutable>
    // CHECK-COUNT-2: "llvm.amdgcn.tensor.load.to.lds"
    // CHECK-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %ta = amdg.async_tdm_copy_global_to_local %a_desc[%c_offset, %c_offset] into %a_buf, pred = %c_pred {warp_used_hint = 15 : i32} : !tt.tensordesc<64x64xf16, #shared_no_merge> -> !ttg.memdesc<64x64xf16, #shared_no_merge, #smem_no_merge, mutable>
    %tb = amdg.async_tdm_copy_global_to_local %b_desc[%c_offset, %c_offset] into %b_buf, pred = %c_pred : !tt.tensordesc<64x64xf16, #shared_no_merge> -> !ttg.memdesc<64x64xf16, #shared_no_merge, #smem_no_merge, mutable>
    tt.return
  }
}

// -----

// Implicit merge negative case: hints overlap (0xFF & 0x0F != 0), so the
// pair is not pairwise disjoint and the merge analysis declines.  Two
// intrinsics are emitted.
#shared_overlap = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_overlap = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_no_merge_overlapping_hints
  tt.func public @tdm_no_merge_overlapping_hints(
    %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}
  ) {
    %c_shape = arith.constant 256 : i32
    %c_stride0 = arith.constant 256 : i64
    %c_stride1 = arith.constant 1 : i64
    %c_offset = arith.constant 0 : i32
    %c_pred = arith.constant 1 : i32
    %a_desc = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared_overlap>
    %b_desc = tt.make_tensor_descriptor %arg1, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared_overlap>
    %a_buf = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared_overlap, #smem_overlap, mutable>
    %b_buf = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared_overlap, #smem_overlap, mutable>
    // CHECK-COUNT-2: "llvm.amdgcn.tensor.load.to.lds"
    // CHECK-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %ta = amdg.async_tdm_copy_global_to_local %a_desc[%c_offset, %c_offset] into %a_buf, pred = %c_pred {warp_used_hint = 255 : i32} : !tt.tensordesc<64x64xf16, #shared_overlap> -> !ttg.memdesc<64x64xf16, #shared_overlap, #smem_overlap, mutable>
    %tb = amdg.async_tdm_copy_global_to_local %b_desc[%c_offset, %c_offset] into %b_buf, pred = %c_pred {warp_used_hint = 15 : i32} : !tt.tensordesc<64x64xf16, #shared_overlap> -> !ttg.memdesc<64x64xf16, #shared_overlap, #smem_overlap, mutable>
    tt.return
  }
}

// -----

// Implicit merge negative case: disjoint hints whose UNION is illegal
// (0x03 | 0x10 = 0x13, popcount=3 not a power of two -> not an
// axis-aligned coset).  Each individual hint is verifier-legal but the
// merge analysis declines because the union does not pass the legality
// check.  Two intrinsics are emitted.
#shared_bad_union = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_bad_union = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_no_merge_union_illegal
  tt.func public @tdm_no_merge_union_illegal(
    %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}
  ) {
    %c_shape = arith.constant 256 : i32
    %c_stride0 = arith.constant 256 : i64
    %c_stride1 = arith.constant 1 : i64
    %c_offset = arith.constant 0 : i32
    %c_pred = arith.constant 1 : i32
    %a_desc = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared_bad_union>
    %b_desc = tt.make_tensor_descriptor %arg1, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared_bad_union>
    %a_buf = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared_bad_union, #smem_bad_union, mutable>
    %b_buf = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared_bad_union, #smem_bad_union, mutable>
    // CHECK-COUNT-2: "llvm.amdgcn.tensor.load.to.lds"
    // CHECK-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %ta = amdg.async_tdm_copy_global_to_local %a_desc[%c_offset, %c_offset] into %a_buf, pred = %c_pred {warp_used_hint = 3 : i32} : !tt.tensordesc<64x64xf16, #shared_bad_union> -> !ttg.memdesc<64x64xf16, #shared_bad_union, #smem_bad_union, mutable>
    %tb = amdg.async_tdm_copy_global_to_local %b_desc[%c_offset, %c_offset] into %b_buf, pred = %c_pred {warp_used_hint = 16 : i32} : !tt.tensordesc<64x64xf16, #shared_bad_union> -> !ttg.memdesc<64x64xf16, #shared_bad_union, #smem_bad_union, mutable>
    tt.return
  }
}
