// RUN: triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=CHECK,ENABLE
// RUN: env TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS=1 triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=CHECK,DISABLE

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_merge_manual_hints
  tt.func public @tdm_merge_manual_hints(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c0 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %desc0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %desc1 = tt.make_tensor_descriptor %arg1, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %dst0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %dst1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>

    // Adjacent copies with disjoint explicit hints fuse.
    // CHECK: "llvm.amdgcn.tensor.load.to.lds"
    // CHECK-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %0 = amdg.async_tdm_copy_global_to_local %desc0[%c0, %c0] into %dst0, pred = %pred {warp_used_hint = 3 : i32} : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = amdg.async_tdm_copy_global_to_local %desc1[%c0, %c0] into %dst1, pred = %pred {warp_used_hint = 12 : i32} : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_merge_auto_hints
  tt.func public @tdm_merge_auto_hints(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c0 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %dst_a = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>
    %dst_b = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>

    // Canonical memdesc_index/copy pairs get generated hints unless disabled.
    // ENABLE: "llvm.amdgcn.tensor.load.to.lds"
    // ENABLE-NOT: "llvm.amdgcn.tensor.load.to.lds"
    // DISABLE: "llvm.amdgcn.tensor.load.to.lds"
    // DISABLE: "llvm.amdgcn.tensor.load.to.lds"
    // DISABLE-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %dst0 = ttg.memdesc_index %dst_a[%c0] : !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %0 = amdg.async_tdm_copy_global_to_local %desc[%c0, %c0] into %dst0, pred = %pred : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %dst1 = ttg.memdesc_index %dst_b[%c0] : !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = amdg.async_tdm_copy_global_to_local %desc[%c0, %c0] into %dst1, pred = %pred : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared_inner = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 1, partitionDim = 0, partitionLayout = #shared_inner}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_auto_hints_skip_partitioned
  tt.func public @tdm_auto_hints_skip_partitioned(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c0 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <128x16xf16, #partitioned>
    %dst_a = ttg.local_alloc : () -> !ttg.memdesc<1x128x16xf16, #partitioned, #smem, mutable>
    %dst_b = ttg.local_alloc : () -> !ttg.memdesc<1x128x16xf16, #partitioned, #smem, mutable>

    // Auto hint generation skips partitioned destinations.
    // CHECK: "llvm.amdgcn.tensor.load.to.lds"
    // CHECK: "llvm.amdgcn.tensor.load.to.lds"
    // CHECK-NOT: "llvm.amdgcn.tensor.load.to.lds"
    %dst0 = ttg.memdesc_index %dst_a[%c0] : !ttg.memdesc<1x128x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>
    %0 = amdg.async_tdm_copy_global_to_local %desc[%c0, %c0] into %dst0, pred = %pred : !tt.tensordesc<128x16xf16, #partitioned> -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>
    %dst1 = ttg.memdesc_index %dst_b[%c0] : !ttg.memdesc<1x128x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>
    %1 = amdg.async_tdm_copy_global_to_local %desc[%c0, %c0] into %dst1, pred = %pred : !tt.tensordesc<128x16xf16, #partitioned> -> !ttg.memdesc<128x16xf16, #partitioned, #smem, mutable>
    tt.return
  }
}
