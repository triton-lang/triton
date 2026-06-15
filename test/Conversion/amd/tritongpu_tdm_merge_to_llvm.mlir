// RUN: triton-opt %s --split-input-file --tritonamdgpu-materialize-tdm-merge | FileCheck %s --check-prefixes=MAT,ENABLE-MAT
// RUN: env TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS=1 triton-opt %s --split-input-file --tritonamdgpu-materialize-tdm-merge | FileCheck %s --check-prefixes=MAT,DISABLE-MAT
// RUN: triton-opt %s --split-input-file --tritonamdgpu-materialize-tdm-merge --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=CHECK,ENABLE
// RUN: env TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS=1 triton-opt %s --split-input-file --tritonamdgpu-materialize-tdm-merge --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=CHECK,DISABLE

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // MAT-LABEL: tdm_manual_hints_stay_separate
  // CHECK-LABEL: tdm_manual_hints_stay_separate
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

    // User-provided hints on regular copies no longer request an implicit
    // merge. Use amdg.async_tdm_fused_copy_global_to_local for manual merges.
    // MAT-NOT: amdg.async_tdm_fused_copy_global_to_local
    // MAT: amdg.async_tdm_copy_global_to_local
    // MAT-SAME: warp_used_hint = 3 : i32
    // MAT-NOT: amdg.async_tdm_fused_copy_global_to_local
    // MAT: amdg.async_tdm_copy_global_to_local
    // MAT-SAME: warp_used_hint = 12 : i32
    // MAT-NOT: amdg.async_tdm_fused_copy_global_to_local
    // CHECK: "llvm.amdgcn.tensor.load.to.lds"
    // CHECK: "llvm.amdgcn.tensor.load.to.lds"
    // CHECK-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %0 = amdg.async_tdm_copy_global_to_local %desc0 into %dst0 {warp_used_hint = 3 : i32} : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = amdg.async_tdm_copy_global_to_local %desc1 into %dst1 {warp_used_hint = 12 : i32} : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // MAT-LABEL: tdm_merge_auto_hints
  // CHECK-LABEL: tdm_merge_auto_hints
  tt.func public @tdm_merge_auto_hints(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c0 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %desc_base = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %desc = amdg.update_tensor_descriptor %desc_base add_offsets = [%c0, %c0] pred = %pred : !tt.tensordesc<64x64xf16, #shared>
    %dst_a = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>
    %dst_b = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>

    // Adjacent unhinted copies materialize as one fused op unless disabled.
    // Auto materialization requires consecutive copies; see
    // tdm_auto_hints_skip_interleaved for the non-consecutive case.
    // ENABLE-MAT: amdg.async_tdm_fused_copy_global_to_local
    // ENABLE-MAT-SAME: warp_used_hints = array<i32: 5, 10>
    // ENABLE-MAT-NOT: amdg.async_tdm_copy_global_to_local
    // DISABLE-MAT: amdg.async_tdm_copy_global_to_local
    // DISABLE-MAT: amdg.async_tdm_copy_global_to_local
    // DISABLE-MAT-NOT: amdg.async_tdm_fused_copy_global_to_local
    // ENABLE: "llvm.amdgcn.tensor.load.to.lds"
    // ENABLE-NOT: "llvm.amdgcn.tensor.load.to.lds"
    // DISABLE: "llvm.amdgcn.tensor.load.to.lds"
    // DISABLE: "llvm.amdgcn.tensor.load.to.lds"
    // DISABLE-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %dst0 = ttg.memdesc_index %dst_a[%c0] : !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %dst1 = ttg.memdesc_index %dst_b[%c0] : !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %0 = amdg.async_tdm_copy_global_to_local %desc into %dst0 : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = amdg.async_tdm_copy_global_to_local %desc into %dst1 : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // MAT-LABEL: tdm_auto_hints_skip_interleaved
  // CHECK-LABEL: tdm_auto_hints_skip_interleaved
  tt.func public @tdm_auto_hints_skip_interleaved(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c0 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %desc_base = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %desc = amdg.update_tensor_descriptor %desc_base add_offsets = [%c0, %c0] pred = %pred : !tt.tensordesc<64x64xf16, #shared>
    %dst_a = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>
    %dst_b = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>

    // A memdesc_index between the copies makes them non-consecutive, so auto
    // hint generation leaves them alone -- two intrinsics with or without the
    // env var.  Hoisting the views to fuse this form is a deferred optimization.
    // MAT: amdg.async_tdm_copy_global_to_local
    // MAT: amdg.async_tdm_copy_global_to_local
    // MAT-NOT: amdg.async_tdm_fused_copy_global_to_local
    // CHECK: "llvm.amdgcn.tensor.load.to.lds"
    // CHECK: "llvm.amdgcn.tensor.load.to.lds"
    // CHECK-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %dst0 = ttg.memdesc_index %dst_a[%c0] : !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %0 = amdg.async_tdm_copy_global_to_local %desc into %dst0 : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %dst1 = ttg.memdesc_index %dst_b[%c0] : !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = amdg.async_tdm_copy_global_to_local %desc into %dst1 : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared_inner = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 1, partitionDim = 0, partitionLayout = #shared_inner}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // MAT-LABEL: tdm_auto_hints_skip_partitioned
  // CHECK-LABEL: tdm_auto_hints_skip_partitioned
  tt.func public @tdm_auto_hints_skip_partitioned(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c0 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %desc_base = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <128x16xf16, #partitioned>
    %desc = amdg.update_tensor_descriptor %desc_base add_offsets = [%c0, %c0] pred = %pred : !tt.tensordesc<128x16xf16, #partitioned>
    %dst_a = ttg.local_alloc : () -> !ttg.memdesc<1x128x16xf16, #partitioned, #smem, mutable>
    %dst_b = ttg.local_alloc : () -> !ttg.memdesc<1x128x16xf16, #partitioned, #smem, mutable>

    // Auto hint generation skips partitioned destinations.
    // MAT: amdg.async_tdm_copy_global_to_local
    // MAT: amdg.async_tdm_copy_global_to_local
    // MAT-NOT: amdg.async_tdm_fused_copy_global_to_local
    // CHECK: "llvm.amdgcn.tensor.load.to.lds"
    // CHECK: "llvm.amdgcn.tensor.load.to.lds"
    // CHECK-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %dst0 = ttg.memdesc_index %dst_a[%c0] : !ttg.memdesc<1x128x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>
    %0 = amdg.async_tdm_copy_global_to_local %desc into %dst0 : !tt.tensordesc<128x16xf16, #partitioned> -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>
    %dst1 = ttg.memdesc_index %dst_b[%c0] : !ttg.memdesc<1x128x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>
    %1 = amdg.async_tdm_copy_global_to_local %desc into %dst1 : !tt.tensordesc<128x16xf16, #partitioned> -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>
    tt.return
  }
}
