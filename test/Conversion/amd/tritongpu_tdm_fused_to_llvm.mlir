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

// Only the member loading into a column subview spreads out its rows: each of
// its 2 warps loads 32 rows of 32 f16 (16 dwords) whose rows are 64 f16 apart,
// so the warps start 32 * 64 = 2^11 elements apart and TDM pads 16 dwords after
// every 16 dwords for that member alone.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // LLVM-LABEL: tdm_fused_column_subview
  tt.func public @tdm_fused_column_subview(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %desc0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x32xf16, #shared>
    %desc1 = tt.make_tensor_descriptor %arg1, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %alloc0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %dst0 = ttg.memdesc_subslice %alloc0[0, 32] : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable, 64x64>
    %dst1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>

    // LLVM-DAG: %[[PAD_MASK:.*]] = llvm.mlir.constant(3145727 : i32) : i32
    // LLVM-DAG: %[[ROW_PAD:.*]] = llvm.mlir.constant(516947968 : i32) : i32
    // LLVM-DAG: %[[WARP_SHIFT:.*]] = llvm.mlir.constant(11 : i32) : i32
    // LLVM: %[[SMEM:.*]] = llvm.mlir.addressof @global_smem
    // LLVM: %[[BASE:.*]] = llvm.getelementptr %[[SMEM]][%{{.*}}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // LLVM: llvm.shl %{{.*}}, %[[WARP_SHIFT]] : i32
    // LLVM: %[[WARP:.*]] = llvm.getelementptr %[[BASE]][%{{.*}}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // LLVM: llvm.ptrtoint %[[WARP]] : !llvm.ptr<3> to i32
    // LLVM: %[[CLEARED:.*]] = llvm.and %{{.*}}, %[[PAD_MASK]] : i32
    // LLVM: llvm.or %[[CLEARED]], %[[ROW_PAD]] : i32
    // LLVM-NOT: %[[ROW_PAD]]
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %0 = amdg.async_tdm_fused_copy_global_to_local %desc0, %desc1 into %dst0, %dst1 {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<64x32xf16, #shared>, !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable, 64x64>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}
