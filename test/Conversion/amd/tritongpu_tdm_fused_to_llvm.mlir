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
