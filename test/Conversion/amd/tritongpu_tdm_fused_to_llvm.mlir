// RUN: triton-opt %s --split-input-file | FileCheck %s --check-prefix=TTG
// RUN: triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=LLVM

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // TTG-LABEL: tdm_manual_hints_stay_separate
  // LLVM-LABEL: tdm_manual_hints_stay_separate
  tt.func public @tdm_manual_hints_stay_separate(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c0 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %desc0_base = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %desc1_base = tt.make_tensor_descriptor %arg1, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %desc0 = amdg.update_tensor_descriptor %desc0_base add_offsets = [%c0, %c0] pred = %pred : !tt.tensordesc<64x64xf16, #shared>
    %desc1 = amdg.update_tensor_descriptor %desc1_base add_offsets = [%c0, %c0] pred = %pred : !tt.tensordesc<64x64xf16, #shared>
    %dst0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %dst1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>

    // User-provided hints on regular copies do not request implicit fusion.
    // Use amdg.async_tdm_fused_copy_global_to_local for manual fusion.
    // TTG-NOT: amdg.async_tdm_fused_copy_global_to_local
    // TTG: amdg.async_tdm_copy_global_to_local
    // TTG-SAME: warp_used_hint = 3 : i32
    // TTG-NOT: amdg.async_tdm_fused_copy_global_to_local
    // TTG: amdg.async_tdm_copy_global_to_local
    // TTG-SAME: warp_used_hint = 12 : i32
    // TTG-NOT: amdg.async_tdm_fused_copy_global_to_local
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %0 = amdg.async_tdm_copy_global_to_local %desc0 into %dst0 {warp_used_hint = 3 : i32} : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = amdg.async_tdm_copy_global_to_local %desc1 into %dst1 {warp_used_hint = 12 : i32} : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}
// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // TTG-LABEL: tdm_explicit_fused
  // LLVM-LABEL: tdm_explicit_fused
  tt.func public @tdm_explicit_fused(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c0 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %desc0_base = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %desc1_base = tt.make_tensor_descriptor %arg1, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %desc0 = amdg.update_tensor_descriptor %desc0_base add_offsets = [%c0, %c0] pred = %pred : !tt.tensordesc<64x64xf16, #shared>
    %desc1 = amdg.update_tensor_descriptor %desc1_base add_offsets = [%c0, %c0] pred = %pred : !tt.tensordesc<64x64xf16, #shared>
    %dst0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %dst1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>

    // TTG: amdg.async_tdm_fused_copy_global_to_local
    // TTG-SAME: warp_used_hints = array<i32: 3, 12>
    // TTG-NOT: amdg.async_tdm_copy_global_to_local
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %0 = amdg.async_tdm_fused_copy_global_to_local %desc0, %desc1 into %dst0, %dst1 {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<64x64xf16, #shared>, !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // TTG-LABEL: tdm_fused_scope_cache_policy
  // LLVM-LABEL: tdm_fused_scope_cache_policy
  tt.func public @tdm_fused_scope_cache_policy(
    %desc0: !tt.tensordesc<64x64xf16, #shared>,
    %desc1: !tt.tensordesc<64x64xf16, #shared>,
    %dst0: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>,
    %dst1: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  ) {
    // TTG: amdg.async_tdm_fused_copy_global_to_local
    // TTG-SAME: cachePolicy = #amdg.cache_policy_gfx12<temporal = none>, scope = 3 : i32
    // LLVM-DAG: %[[AUX:.*]] = llvm.mlir.constant(25 : i32) : i32
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[AUX]])
    %0 = amdg.async_tdm_fused_copy_global_to_local %desc0, %desc1 into %dst0, %dst1 {warp_used_hints = array<i32: 3, 12>, cachePolicy = #amdg.cache_policy_gfx12<temporal = none>, scope = 3 : i32} : !tt.tensordesc<64x64xf16, #shared>, !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#desc = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [32, 64]}>
#alloc = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [256, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // TTG-LABEL: tdm_explicit_fused_padded_row_slices
  // LLVM-LABEL: tdm_explicit_fused_padded_row_slices
  tt.func public @tdm_explicit_fused_padded_row_slices(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_rows = arith.constant 32 : i32
    %c_cols = arith.constant 64 : i32
    %c_stride0 = arith.constant 64 : i64
    %c_stride1 = arith.constant 1 : i64
    %desc0 = tt.make_tensor_descriptor %arg0, [%c_rows, %c_cols], [%c_stride0, %c_stride1] : <f16>, <32x64xf16, #desc>
    %desc1 = tt.make_tensor_descriptor %arg1, [%c_rows, %c_cols], [%c_stride0, %c_stride1] : <f16>, <32x64xf16, #desc>
    // [64:+8] pads each subview origin before adding the per-warp offset:
    // rows 64 and 160 start at elements 4608 and 11520, respectively.
    // LLVM: %[[SMEM:.*]] = llvm.mlir.addressof @global_smem
    // LLVM-DAG: %[[INTERVAL:.*]] = llvm.mlir.constant(6 : i32) : i32
    // LLVM-DAG: %[[PADDING:.*]] = llvm.mlir.constant(3 : i32) : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #alloc, #smem, mutable>
    %dst0 = ttg.memdesc_subslice %alloc[64, 0] : !ttg.memdesc<256x64xf16, #alloc, #smem, mutable> -> !ttg.memdesc<32x64xf16, #alloc, #smem, mutable, 256x64>
    %dst1 = ttg.memdesc_subslice %alloc[160, 0] : !ttg.memdesc<256x64xf16, #alloc, #smem, mutable> -> !ttg.memdesc<32x64xf16, #alloc, #smem, mutable, 256x64>
    // LLVM: %[[CHUNKS0:.*]] = llvm.lshr %[[OFFSET0:.*]], %[[INTERVAL]] : i32
    // LLVM: %[[PAD0:.*]] = llvm.shl %[[CHUNKS0]], %[[PADDING]] : i32
    // LLVM: %[[PADS0:.*]] = llvm.add {{.*}}%[[PAD0]]{{.*}} : i32
    // LLVM: %[[PADDED0:.*]] = llvm.add %[[OFFSET0]], %[[PADS0]] : i32
    // LLVM: %[[BASE0:.*]] = llvm.getelementptr %[[SMEM]][%[[PADDED0]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // LLVM: %[[CHUNKS1:.*]] = llvm.lshr %[[OFFSET1:.*]], %[[INTERVAL]] : i32
    // LLVM: %[[PAD1:.*]] = llvm.shl %[[CHUNKS1]], %[[PADDING]] : i32
    // LLVM: %[[PADS1:.*]] = llvm.add {{.*}}%[[PAD1]]{{.*}} : i32
    // LLVM: %[[PADDED1:.*]] = llvm.add %[[OFFSET1]], %[[PADS1]] : i32
    // LLVM: %[[BASE1:.*]] = llvm.getelementptr %[[SMEM]][%[[PADDED1]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // LLVM: %[[WARP0:.*]] = llvm.getelementptr %[[BASE0]][%{{.*}}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // LLVM: %[[ADDR0:.*]] = llvm.ptrtoint %[[WARP0]] : !llvm.ptr<3> to i32
    // LLVM: llvm.insertelement %[[ADDR0]], %{{.*}}[%{{.*}} : i32] : vector<4xi32>
    // LLVM: %[[WARP1:.*]] = llvm.getelementptr %[[BASE1]][%{{.*}}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // LLVM: %[[ADDR1:.*]] = llvm.ptrtoint %[[WARP1]] : !llvm.ptr<3> to i32
    // LLVM: llvm.insertelement %[[ADDR1]], %{{.*}}[%{{.*}} : i32] : vector<4xi32>
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOT: "llvm.amdgcn.tensor.load.to.lds"
    // TTG: amdg.async_tdm_fused_copy_global_to_local
    // TTG-SAME: warp_used_hints = array<i32: 3, 12>
    %0 = amdg.async_tdm_fused_copy_global_to_local %desc0, %desc1 into %dst0, %dst1 {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<32x64xf16, #desc>, !tt.tensordesc<32x64xf16, #desc> -> !ttg.memdesc<32x64xf16, #alloc, #smem, mutable, 256x64>, !ttg.memdesc<32x64xf16, #alloc, #smem, mutable, 256x64>
    tt.return
  }
}

// -----

#desc = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [1, 64]}>
#alloc = #ttg.padded_shared<[64:+8] {order = [0], shape = [128]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // TTG-LABEL: tdm_fused_partial_selected_buffers
  // LLVM-LABEL: tdm_fused_partial_selected_buffers
  tt.func @tdm_fused_partial_selected_buffers(%desc0: !tt.tensordesc<1x64xf16, #desc>, %desc1: !tt.tensordesc<1x64xf16, #desc>) {
    %c0 = arith.constant 0 : i32
    %alloc0 = ttg.local_alloc : () -> !ttg.memdesc<1x128xf16, #alloc, #smem, mutable>
    %dst0 = ttg.memdesc_index %alloc0[%c0] : !ttg.memdesc<1x128xf16, #alloc, #smem, mutable> -> !ttg.memdesc<128xf16, #alloc, #smem, mutable>
    %alloc1 = ttg.local_alloc : () -> !ttg.memdesc<1x128xf16, #alloc, #smem, mutable>
    %dst1 = ttg.memdesc_index %alloc1[%c0] : !ttg.memdesc<1x128xf16, #alloc, #smem, mutable> -> !ttg.memdesc<128xf16, #alloc, #smem, mutable>
    // TTG: amdg.async_tdm_fused_copy_global_to_local
    // TTG-SAME: warp_used_hints = array<i32: 3, 12>
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOT: "llvm.amdgcn.tensor.load.to.lds"
    amdg.async_tdm_fused_copy_global_to_local %desc0, %desc1 into %dst0, %dst1 {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<1x64xf16, #desc>, !tt.tensordesc<1x64xf16, #desc> -> !ttg.memdesc<128xf16, #alloc, #smem, mutable>, !ttg.memdesc<128xf16, #alloc, #smem, mutable>
    tt.return
  }
}
