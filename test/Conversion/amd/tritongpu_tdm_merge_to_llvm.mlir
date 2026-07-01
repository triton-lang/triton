// RUN: env TRITON_HIP_DISABLE_TDM_AUTO_FUSE=0 triton-opt %s --split-input-file --tritonamdgpu-auto-fuse-tdm-copy | FileCheck %s --check-prefixes=TTG,TTG-FUSE
// RUN: env TRITON_HIP_DISABLE_TDM_AUTO_FUSE=1 triton-opt %s --split-input-file | FileCheck %s --check-prefixes=TTG,TTG-NOFUSE
// RUN: env TRITON_HIP_DISABLE_TDM_AUTO_FUSE=0 triton-opt %s --split-input-file --tritonamdgpu-auto-fuse-tdm-copy --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=LLVM,LLVM-FUSE
// RUN: env TRITON_HIP_DISABLE_TDM_AUTO_FUSE=1 triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=LLVM,LLVM-NOFUSE

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

    // User-provided hints on regular copies no longer request implicit fusion.
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
  // TTG-LABEL: tdm_auto_fuse_unhinted
  // LLVM-LABEL: tdm_auto_fuse_unhinted
  tt.func public @tdm_auto_fuse_unhinted(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c0 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %desc_base = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %desc = amdg.update_tensor_descriptor %desc_base add_offsets = [%c0, %c0] pred = %pred : !tt.tensordesc<64x64xf16, #shared>
    %dst_a = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>
    %dst_b = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>

    // Adjacent unhinted copies fuse into one op unless disabled. Destination
    // views may be precomputed before the copy run.
    // TTG-FUSE: amdg.async_tdm_fused_copy_global_to_local
    // TTG-FUSE-SAME: warp_used_hints = array<i32: 5, 10>
    // TTG-FUSE-NOT: amdg.async_tdm_copy_global_to_local
    // TTG-NOFUSE: amdg.async_tdm_copy_global_to_local
    // TTG-NOFUSE: amdg.async_tdm_copy_global_to_local
    // TTG-NOFUSE-NOT: amdg.async_tdm_fused_copy_global_to_local
    // LLVM-FUSE: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-FUSE-NOT: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOFUSE: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOFUSE: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOFUSE-NOT: "llvm.amdgcn.tensor.load.to.lds"
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
  // TTG-LABEL: tdm_auto_fuse_through_memdesc_index
  // LLVM-LABEL: tdm_auto_fuse_through_memdesc_index
  tt.func public @tdm_auto_fuse_through_memdesc_index(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c0 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %desc_base = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %desc = amdg.update_tensor_descriptor %desc_base add_offsets = [%c0, %c0] pred = %pred : !tt.tensordesc<64x64xf16, #shared>
    %dst_a = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>
    %dst_b = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>

    // A memdesc_index between copies is a transparent view op. The fused op is
    // inserted at the last copy so the second view dominates it.
    // TTG-FUSE: amdg.async_tdm_fused_copy_global_to_local
    // TTG-FUSE-SAME: warp_used_hints = array<i32: 5, 10>
    // TTG-FUSE-NOT: amdg.async_tdm_copy_global_to_local
    // TTG-NOFUSE: amdg.async_tdm_copy_global_to_local
    // TTG-NOFUSE: amdg.async_tdm_copy_global_to_local
    // TTG-NOFUSE-NOT: amdg.async_tdm_fused_copy_global_to_local
    // LLVM-FUSE: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-FUSE-NOT: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOFUSE: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOFUSE: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOFUSE-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %dst0 = ttg.memdesc_index %dst_a[%c0] : !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %0 = amdg.async_tdm_copy_global_to_local %desc into %dst0 : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %dst1 = ttg.memdesc_index %dst_b[%c0] : !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = amdg.async_tdm_copy_global_to_local %desc into %dst1 : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // TTG-LABEL: tdm_auto_fuse_skip_barrier
  // LLVM-LABEL: tdm_auto_fuse_skip_barrier
  tt.func public @tdm_auto_fuse_skip_barrier(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c0 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %desc_base = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %desc = amdg.update_tensor_descriptor %desc_base add_offsets = [%c0, %c0] pred = %pred : !tt.tensordesc<64x64xf16, #shared>
    %dst0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %dst1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>

    // Copies with mbarriers are not auto-fuse candidates.
    // TTG: amdg.async_tdm_copy_global_to_local
    // TTG: amdg.async_tdm_copy_global_to_local
    // TTG-NOT: amdg.async_tdm_fused_copy_global_to_local
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %0 = amdg.async_tdm_copy_global_to_local %desc into %dst0 : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = amdg.async_tdm_copy_global_to_local %desc into %dst1, barrier = %bar : !tt.tensordesc<64x64xf16, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared_inner = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 1, partitionDim = 0, partitionLayout = #shared_inner}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // TTG-LABEL: tdm_auto_fuse_skip_partitioned
  // LLVM-LABEL: tdm_auto_fuse_skip_partitioned
  tt.func public @tdm_auto_fuse_skip_partitioned(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c0 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %desc_base = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <128x16xf16, #partitioned>
    %desc = amdg.update_tensor_descriptor %desc_base add_offsets = [%c0, %c0] pred = %pred : !tt.tensordesc<128x16xf16, #partitioned>
    %dst_a = ttg.local_alloc : () -> !ttg.memdesc<1x128x16xf16, #partitioned, #smem, mutable>
    %dst_b = ttg.local_alloc : () -> !ttg.memdesc<1x128x16xf16, #partitioned, #smem, mutable>

    // Auto-fusion skips partitioned destinations.
    // TTG: amdg.async_tdm_copy_global_to_local
    // TTG: amdg.async_tdm_copy_global_to_local
    // TTG-NOT: amdg.async_tdm_fused_copy_global_to_local
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM: "llvm.amdgcn.tensor.load.to.lds"
    // LLVM-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %dst0 = ttg.memdesc_index %dst_a[%c0] : !ttg.memdesc<1x128x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>
    %0 = amdg.async_tdm_copy_global_to_local %desc into %dst0 : !tt.tensordesc<128x16xf16, #partitioned> -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>
    %dst1 = ttg.memdesc_index %dst_b[%c0] : !ttg.memdesc<1x128x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>
    %1 = amdg.async_tdm_copy_global_to_local %desc into %dst1 : !tt.tensordesc<128x16xf16, #partitioned> -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>
    tt.return
  }
}
