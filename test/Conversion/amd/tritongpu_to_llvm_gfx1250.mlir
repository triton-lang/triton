// RUN: split-file %s %t
// RUN: triton-opt %t/base.mlir -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch="gfx1250" | FileCheck %s --check-prefix=GFX1250
// RUN: triton-opt %t/warp-specialize-result.mlir --tritongpu-allocate-warp-groups --allocate-amdgpu-shared-memory=arch=gfx1250 --triton-amdgpu-membar=gfx-arch=gfx1250 --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --triton-amdgpu-convert-warp-specialize-to-llvm=gfx-arch=gfx1250 --canonicalize=region-simplify=disabled | FileCheck %s --check-prefix=WS-RESULT

//--- base.mlir
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4]], warp = [[16, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[1, 0]]}, isTranspose = true, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: wmma_permlane16_swap
  tt.func @wmma_permlane16_swap(%arg0: tensor<32x32xf16, #mma>) {
    // GFX1250-NOT: store
    // GFX1250-NOT: load
    // GFX1250-COUNT-4: llvm.call_intrinsic "llvm.amdgcn.permlane16.swap"
    // GFX1250-NOT: llvm.call_intrinsic "llvm.amdgcn.permlane16.swap"
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf16, #mma> -> tensor<32x32xf16, #linear>
    tt.return
  }
}

// -----

#noncontiguous = #ttg.generic_linear<{register = [[0, 1]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: generic_linear_noncontiguous_fp32_to_bf16
  tt.func @generic_linear_noncontiguous_fp32_to_bf16(%arg0: tensor<8x8xf32, #noncontiguous>) -> tensor<8x8xbf16, #noncontiguous> {
    // GFX1250-NOT: llvm.call_intrinsic "llvm.amdgcn.perm"
    // GFX1250-COUNT-2: llvm.trunc {{.*}} : i32 to i16
    // GFX1250-NOT: llvm.trunc
    // GFX1250-NOT: llvm.call_intrinsic "llvm.amdgcn.perm"
    %0 = tt.fp_to_fp %arg0, rounding = rtz : tensor<8x8xf32, #noncontiguous> -> tensor<8x8xbf16, #noncontiguous>
    tt.return %0 : tensor<8x8xbf16, #noncontiguous>
  }
}

// -----

#partition_aware = #ttg.generic_linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [16, 0], [0, 128]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4]], warp = [[64, 64], [32, 0], [64, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: generic_linear_fp32_to_fp8
  tt.func @generic_linear_fp32_to_fp8(%arg0: tensor<128x256xf32, #partition_aware>) -> tensor<128x256xf8E4M3FN, #partition_aware> {
    // GFX1250-COUNT-16: rocdl.cvt.scalef32.pk8.fp8.f32
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x256xf32, #partition_aware> -> tensor<128x256xf8E4M3FN, #partition_aware>
    tt.return %0 : tensor<128x256xf8E4M3FN, #partition_aware>
  }
}

// -----

#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[1, 0], [2, 0]]}, isTranspose = true, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: reduce_16x16
  tt.func @reduce_16x16(%input: tensor<128x128xf32, #mma>) {
    // GFX1250-COUNT-2: rocdl.permlanex16
    %0 = "tt.reduce"(%input) <{axis = 1 : i32}> ({
      ^bb0(%arg1: f32 , %arg2: f32):
      %2 = "arith.maxnumf"(%arg1, %arg2) : (f32, f32) -> f32
      tt.reduce.return %2 : f32 }) : (tensor<128x128xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
   tt.return
  }
}

// -----

// Test lowering of operations with PartitionedSharedEncodingAttr using padded_shared layout
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#inner_padded = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [16, 16]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_padded}>
#inner_padded_piece = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [4, 16]}>
#partitioned_piece = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_padded_piece}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: partitioned_shared_padded_local_alloc
  tt.func @partitioned_shared_padded_local_alloc(%arg0: tensor<16x16xf16, #blocked>) {
    // GFX1250: llvm.mlir.addressof @global_smem
    // GFX1250-COUNT-4: llvm.store {{.*}} : vector<{{[0-9]+}}xf16>, !llvm.ptr<3>
    %0 = ttg.local_alloc %arg0 : (tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    tt.return
  }

  // GFX1250-LABEL: partitioned_shared_padded_multibuffer_index
  tt.func @partitioned_shared_padded_multibuffer_index(%index: i32) {
    %parent = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #partitioned_piece, #smem, mutable>
    // GFX1250: llvm.mlir.constant(128 : i32)
    // GFX1250: llvm.getelementptr
    %view = ttg.memdesc_index %parent[%index] : !ttg.memdesc<2x16x16xf16, #partitioned_piece, #smem, mutable> -> !ttg.memdesc<16x16xf16, #partitioned_piece, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#inner_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_shared}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: partitioned_shared_multibuffer_subslice
  tt.func @partitioned_shared_multibuffer_subslice() -> tensor<16x16xf16, #blocked> {
    %c0 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<5x16x16xf16, #partitioned, #smem, mutable>
    // GFX1250: [[STAGE_OFFSET:%.*]] = llvm.mlir.constant(256 : i32)
    // GFX1250-COUNT-2: llvm.getelementptr {{.*}}[[STAGE_OFFSET]]
    %1 = ttg.memdesc_subslice %0 [2, 0, 0] : !ttg.memdesc<5x16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<3x16x16xf16, #partitioned, #smem, mutable, 5x16x16>
    %2 = ttg.memdesc_index %1[%c0] : !ttg.memdesc<3x16x16xf16, #partitioned, #smem, mutable, 5x16x16> -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    // GFX1250: llvm.load
    %3 = ttg.local_load %2 : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> tensor<16x16xf16, #blocked>
    tt.return %3 : tensor<16x16xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#inner_padded = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [16, 16]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_padded}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: partitioned_shared_padded_local_load
  tt.func @partitioned_shared_padded_local_load() -> tensor<16x16xf16, #blocked> {
    // Allocate and then load from partitioned shared memory
    // GFX1250: llvm.mlir.addressof @global_smem
    // GFX1250-COUNT-4: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{[0-9]+}}xf16>
    %0 = ttg.local_alloc {allocation.offset = [0 : i32, 65536 : i32, 128 : i32, 65664 : i32]} : () -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    %1 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> tensor<16x16xf16, #blocked>
    tt.return %1 : tensor<16x16xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#inner_padded = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [16, 16]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_padded}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: partitioned_shared_padded_local_store
  tt.func @partitioned_shared_padded_local_store(%arg0: tensor<16x16xf16, #blocked>) {
    // Allocate and then store to partitioned shared memory
    // GFX1250: llvm.mlir.addressof @global_smem
    // GFX1250-COUNT-4: llvm.store {{.*}} : vector<{{[0-9]+}}xf16>, !llvm.ptr<3>
    %0 = ttg.local_alloc {allocation.offset = [0 : i32, 65536 : i32, 128 : i32, 65664 : i32]} : () -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    ttg.local_store %arg0, %0 : tensor<16x16xf16, #blocked> -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: @bf16_mulf
  tt.func @bf16_mulf(%arg0: tensor<64xbf16, #blocked>, %arg1: tensor<64xf8E4M3FN, #blocked>) -> tensor<64xbf16, #blocked> {
    // GFX1250: rocdl.cvt.scale.pk8.bf16.fp8
    // GFX1250: llvm.fmul {{.*}} : vector<2xbf16>
    %0 = tt.fp_to_fp %arg1 : tensor<64xf8E4M3FN, #blocked> -> tensor<64xbf16, #blocked>
    %1 = arith.mulf %arg0, %0 : tensor<64xbf16, #blocked>
    tt.return %1 : tensor<64xbf16, #blocked>
  }

  // GFX1250-LABEL: @bf16_addf
  tt.func @bf16_addf(%arg0: tensor<64xbf16, #blocked>, %arg1: tensor<64xbf16, #blocked>) -> tensor<64xbf16, #blocked> {
    // GFX1250-NOT: llvm.fadd {{.*}} : f32
    // GFX1250: llvm.fadd {{.*}} : vector<2xbf16>
    %0 = arith.addf %arg0, %arg1 : tensor<64xbf16, #blocked>
    tt.return %0 : tensor<64xbf16, #blocked>
  }

  // GFX1250-LABEL: @bf16_subf
  tt.func @bf16_subf(%arg0: tensor<64xbf16, #blocked>, %arg1: tensor<64xbf16, #blocked>) -> tensor<64xbf16, #blocked> {
    // GFX1250-NOT: llvm.fsub {{.*}} : f32
    // GFX1250: llvm.fsub {{.*}} : vector<2xbf16>
    %0 = arith.subf %arg0, %arg1 : tensor<64xbf16, #blocked>
    tt.return %0 : tensor<64xbf16, #blocked>
  }
}

// -----

#blocked8 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // The eight-warp helper uses the caller's precomputed offset.
  // GFX1250-LABEL: llvm.func internal @outlined_indices_8
  // GFX1250: [[WAVE:%.*]] = rocdl.wave.id : i32
  // GFX1250: [[OFFSET:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // GFX1250: [[REL:%.*]] = llvm.sub [[WAVE]], [[OFFSET]] : i32
  // GFX1250: [[MASK:%.*]] = llvm.mlir.constant(7 : i32) : i32
  // GFX1250: llvm.and [[REL]], [[MASK]] : i32
  tt.func private @outlined_indices_8() -> tensor<256xi32, #blocked8> attributes {noinline = true, "ttg.num-warps" = 8 : i32, "ttg.warp-id-offset" = 4 : i32} {
    %range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32, #blocked8>
    tt.return %range : tensor<256xi32, #blocked8>
  }
}

//--- warp-specialize-result.mlir

#all = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#row = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [2, 8], [4, 16]]}, alignment = 8>
#indexed = #ttg.shared_linear<{offset = [[1], [2], [4], [8], [16], [32]]}, alignment = 8, hasIndexPhase = true, indexPhaseMask = 24>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx1250", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A default-region result must carry both the selected base and its phase
  // through warp_yield to the local_load after explicit warp specialization.
  // WS-RESULT-LABEL: llvm.func @warp_specialize_logical_index_result
  // WS-RESULT: llvm.br ^bb{{[0-9]+}}({{.*}} : !llvm.struct<(ptr<3>, i32)>)
  // WS-RESULT: ^bb{{[0-9]+}}(%[[DESC:.*]]: !llvm.struct<(ptr<3>, i32)>):
  // WS-RESULT: %[[BASE:.*]] = llvm.extractvalue %[[DESC]][0]
  // WS-RESULT: %[[PHASE:.*]] = llvm.extractvalue %[[DESC]][1]
  // WS-RESULT: llvm.xor
  // WS-RESULT: %[[PTR:.*]] = llvm.getelementptr inbounds %[[BASE]]
  // WS-RESULT: llvm.load %[[PTR]] : !llvm.ptr<3>
  tt.func @warp_specialize_logical_index_result(%values: tensor<8x64xf32, #all>, %index: i32, %out: tensor<64x!tt.ptr<f32>, #row>) {
    %src = ttg.local_alloc %values : (tensor<8x64xf32, #all>) -> !ttg.memdesc<8x64xf32, #shared, #smem, mutable>
    %view = ttg.warp_specialize()
    default {
      %slice = ttg.memdesc_index %src[%index] : !ttg.memdesc<8x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64xf32, #indexed, #smem, mutable, 8x64>
      ttg.warp_yield %slice : !ttg.memdesc<64xf32, #indexed, #smem, mutable, 8x64>
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> !ttg.memdesc<64xf32, #indexed, #smem, mutable, 8x64>
    %value = ttg.local_load %view : !ttg.memdesc<64xf32, #indexed, #smem, mutable, 8x64> -> tensor<64xf32, #row>
    tt.store %out, %value : tensor<64x!tt.ptr<f32>, #row>
    tt.return
  }
}
