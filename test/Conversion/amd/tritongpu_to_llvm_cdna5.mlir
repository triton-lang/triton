// RUN:  triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch="gfx1250" | FileCheck %s --check-prefix=COMMON
// RUN:  triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch="gfx1250-strict" | FileCheck %s --check-prefix=COMMON
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4]], warp = [[16, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[1, 0]]}, isTranspose = true, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: wmma_permlane16_swap
  tt.func @wmma_permlane16_swap(%arg0: tensor<32x32xf16, #mma>) {
    // COMMON-NOT: store
    // COMMON-NOT: load
    // COMMON-COUNT-4: llvm.call_intrinsic "llvm.amdgcn.permlane16.swap"
    // COMMON-NOT: llvm.call_intrinsic "llvm.amdgcn.permlane16.swap"
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf16, #mma> -> tensor<32x32xf16, #linear>
    tt.return
  }
}

// -----

#noncontiguous = #ttg.generic_linear<{register = [[0, 1]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: generic_linear_noncontiguous_fp32_to_bf16
  tt.func @generic_linear_noncontiguous_fp32_to_bf16(%arg0: tensor<8x8xf32, #noncontiguous>) -> tensor<8x8xbf16, #noncontiguous> {
    // COMMON-NOT: llvm.call_intrinsic "llvm.amdgcn.perm"
    // COMMON-COUNT-2: llvm.trunc {{.*}} : i32 to i16
    // COMMON-NOT: llvm.trunc
    // COMMON-NOT: llvm.call_intrinsic "llvm.amdgcn.perm"
    %0 = tt.fp_to_fp %arg0, rounding = rtz : tensor<8x8xf32, #noncontiguous> -> tensor<8x8xbf16, #noncontiguous>
    tt.return %0 : tensor<8x8xbf16, #noncontiguous>
  }
}

// -----

#partition_aware = #ttg.generic_linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [16, 0], [0, 128]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4]], warp = [[64, 64], [32, 0], [64, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: generic_linear_fp32_to_fp8
  tt.func @generic_linear_fp32_to_fp8(%arg0: tensor<128x256xf32, #partition_aware>) -> tensor<128x256xf8E4M3FN, #partition_aware> {
    // COMMON-COUNT-16: rocdl.cvt.scalef32.pk8.fp8.f32
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x256xf32, #partition_aware> -> tensor<128x256xf8E4M3FN, #partition_aware>
    tt.return %0 : tensor<128x256xf8E4M3FN, #partition_aware>
  }
}

// -----

#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[1, 0], [2, 0]]}, isTranspose = true, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: reduce_16x16
  tt.func @reduce_16x16(%input: tensor<128x128xf32, #mma>) {
    // COMMON-COUNT-2: rocdl.permlanex16
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
#inner_padded = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [4, 16]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_padded}>
#partitioned_piece = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_padded}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: partitioned_shared_padded_local_alloc
  tt.func @partitioned_shared_padded_local_alloc(%arg0: tensor<16x16xf16, #blocked>) {
    // COMMON: llvm.mlir.addressof @global_smem
    // COMMON-COUNT-4: llvm.store {{.*}} : vector<{{[0-9]+}}xf16>, !llvm.ptr<3>
    %0 = ttg.local_alloc %arg0 : (tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    tt.return
  }

  // COMMON-LABEL: partitioned_shared_padded_multibuffer_index
  tt.func @partitioned_shared_padded_multibuffer_index(%index: i32) {
    %parent = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #partitioned_piece, #smem, mutable>
    // COMMON: llvm.mlir.constant(128 : i32)
    // COMMON: llvm.getelementptr
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
  // COMMON-LABEL: partitioned_shared_multibuffer_subslice
  tt.func @partitioned_shared_multibuffer_subslice() -> tensor<16x16xf16, #blocked> {
    %c0 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<5x16x16xf16, #partitioned, #smem, mutable>
    // COMMON: [[STAGE_OFFSET:%.*]] = llvm.mlir.constant(256 : i32)
    // COMMON-COUNT-2: llvm.getelementptr {{.*}}[[STAGE_OFFSET]]
    %1 = ttg.memdesc_subslice %0 [2, 0, 0] : !ttg.memdesc<5x16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<3x16x16xf16, #partitioned, #smem, mutable, 5x16x16>
    %2 = ttg.memdesc_index %1[%c0] : !ttg.memdesc<3x16x16xf16, #partitioned, #smem, mutable, 5x16x16> -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    // COMMON: llvm.load
    %3 = ttg.local_load %2 : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> tensor<16x16xf16, #blocked>
    tt.return %3 : tensor<16x16xf16, #blocked>
  }
}

// -----

#inner_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0], [0, 0]]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_shared}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // The second block basis broadcasts the descriptor. Every CTA in that
  // broadcast group takes the same logical subslice; the subslice does not
  // select a CTA or alter the CGA mapping. The resulting non-square layout
  // still needs to recover and rotate its physical partition.
  // COMMON-LABEL: partitioned_shared_clustered_subslice
  // COMMON: [[PART0:%.*]] = llvm.extractvalue {{.*}}[0]
  // COMMON: [[PART1:%.*]] = llvm.extractvalue {{.*}}[1]
  // COMMON: llvm.insertvalue [[PART1]], {{.*}}[0]
  // COMMON: llvm.insertvalue [[PART0]], {{.*}}[1]
  tt.func private @partitioned_shared_clustered_subslice() -> !ttg.memdesc<2x16xf16, #partitioned, #smem, mutable, 16x16> {
    %parent = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    %view = ttg.memdesc_subslice %parent [2, 0] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<2x16xf16, #partitioned, #smem, mutable, 16x16>
    tt.return %view : !ttg.memdesc<2x16xf16, #partitioned, #smem, mutable, 16x16>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#inner_padded = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [4, 16]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_padded}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: partitioned_shared_padded_local_load
  tt.func @partitioned_shared_padded_local_load() -> tensor<16x16xf16, #blocked> {
    // Allocate and then load from partitioned shared memory
    // COMMON: llvm.mlir.addressof @global_smem
    // COMMON-COUNT-4: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{[0-9]+}}xf16>
    %0 = ttg.local_alloc {allocation.offset = [0 : i32, 65536 : i32, 128 : i32, 65664 : i32]} : () -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    %1 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> tensor<16x16xf16, #blocked>
    tt.return %1 : tensor<16x16xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#inner_padded = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [4, 16]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_padded}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: partitioned_shared_padded_local_store
  tt.func @partitioned_shared_padded_local_store(%arg0: tensor<16x16xf16, #blocked>) {
    // Allocate and then store to partitioned shared memory
    // COMMON: llvm.mlir.addressof @global_smem
    // COMMON-COUNT-4: llvm.store {{.*}} : vector<{{[0-9]+}}xf16>, !llvm.ptr<3>
    %0 = ttg.local_alloc {allocation.offset = [0 : i32, 65536 : i32, 128 : i32, 65664 : i32]} : () -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    ttg.local_store %arg0, %0 : tensor<16x16xf16, #blocked> -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: @bf16_mulf
  tt.func @bf16_mulf(%arg0: tensor<64xbf16, #blocked>, %arg1: tensor<64xf8E4M3FN, #blocked>) -> tensor<64xbf16, #blocked> {
    // COMMON: rocdl.cvt.scale.pk8.bf16.fp8
    // COMMON: llvm.fmul {{.*}} : vector<2xbf16>
    %0 = tt.fp_to_fp %arg1 : tensor<64xf8E4M3FN, #blocked> -> tensor<64xbf16, #blocked>
    %1 = arith.mulf %arg0, %0 : tensor<64xbf16, #blocked>
    tt.return %1 : tensor<64xbf16, #blocked>
  }

  // COMMON-LABEL: @ocp_fp8_to_f32
  tt.func @ocp_fp8_to_f32(%arg0: tensor<64xf8E4M3FN, #blocked>, %arg1: tensor<64xf8E5M2, #blocked>) {
    // COMMON: rocdl.cvt.scale.pk8.f32.fp8
    %0 = tt.fp_to_fp %arg0 : tensor<64xf8E4M3FN, #blocked> -> tensor<64xf32, #blocked>
    // COMMON: rocdl.cvt.scale.pk8.f32.bf8
    %1 = tt.fp_to_fp %arg1 : tensor<64xf8E5M2, #blocked> -> tensor<64xf32, #blocked>
    tt.return
  }

  // COMMON-LABEL: @bf16_addf
  tt.func @bf16_addf(%arg0: tensor<64xbf16, #blocked>, %arg1: tensor<64xbf16, #blocked>) -> tensor<64xbf16, #blocked> {
    // COMMON-NOT: llvm.fadd {{.*}} : f32
    // COMMON: llvm.fadd {{.*}} : vector<2xbf16>
    %0 = arith.addf %arg0, %arg1 : tensor<64xbf16, #blocked>
    tt.return %0 : tensor<64xbf16, #blocked>
  }

  // COMMON-LABEL: @bf16_subf
  tt.func @bf16_subf(%arg0: tensor<64xbf16, #blocked>, %arg1: tensor<64xbf16, #blocked>) -> tensor<64xbf16, #blocked> {
    // COMMON-NOT: llvm.fsub {{.*}} : f32
    // COMMON: llvm.fsub {{.*}} : vector<2xbf16>
    %0 = arith.subf %arg0, %arg1 : tensor<64xbf16, #blocked>
    tt.return %0 : tensor<64xbf16, #blocked>
  }

  // COMMON-LABEL: @f32_fma
  tt.func @f32_fma(%arg0: tensor<64xf32, #blocked>, %arg1: tensor<64xf32, #blocked>, %arg2: tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked> {
    // COMMON-NOT: llvm.intr.fma({{.*}}) : (f32, f32, f32) -> f32
    // COMMON: llvm.intr.fma({{.*}}) : (vector<2xf32>, vector<2xf32>, vector<2xf32>) -> vector<2xf32>
    %0 = math.fma %arg0, %arg1, %arg2 : tensor<64xf32, #blocked>
    tt.return %0 : tensor<64xf32, #blocked>
  }

  // COMMON-LABEL: @bf16_fma
  tt.func @bf16_fma(%arg0: tensor<64xbf16, #blocked>, %arg1: tensor<64xbf16, #blocked>, %arg2: tensor<64xbf16, #blocked>) -> tensor<64xbf16, #blocked> {
    // COMMON-NOT: llvm.intr.fma({{.*}}) : (bf16, bf16, bf16) -> bf16
    // COMMON: llvm.intr.fma({{.*}}) : (vector<2xbf16>, vector<2xbf16>, vector<2xbf16>) -> vector<2xbf16>
    %0 = math.fma %arg0, %arg1, %arg2 : tensor<64xbf16, #blocked>
    tt.return %0 : tensor<64xbf16, #blocked>
  }
}

// -----

#blocked8 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // The eight-warp helper uses the caller's precomputed offset.
  // COMMON-LABEL: llvm.func internal @outlined_indices_8
  // COMMON: [[WAVE:%.*]] = rocdl.wave.id : i32
  // COMMON: [[OFFSET:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // COMMON: [[REL:%.*]] = llvm.sub [[WAVE]], [[OFFSET]] : i32
  // COMMON: [[MASK:%.*]] = llvm.mlir.constant(7 : i32) : i32
  // COMMON: llvm.and [[REL]], [[MASK]] : i32
  tt.func private @outlined_indices_8() -> tensor<256xi32, #blocked8> attributes {noinline = true, "ttg.num-warps" = 8 : i32, "ttg.warp-id-offset" = 4 : i32} {
    %range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32, #blocked8>
    tt.return %range : tensor<256xi32, #blocked8>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Only the thread that acquires the lock, lane 0 of warp 0, may release it.
  // COMMON-LABEL: @experimental_lock_release
  // COMMON: [[WAVE:%.*]] = rocdl.wave.id : i32
  // COMMON: llvm.cond_br %arg1, ^[[PRED:bb[0-9]+]], ^[[END:bb[0-9]+]]
  // COMMON: ^[[PRED]]:
  // COMMON: rocdl.s.barrier
  // COMMON: [[LANE0:%.*]] = llvm.icmp "eq" {{%.*}}, {{%.*}} : i32
  // COMMON: [[WARP0:%.*]] = llvm.icmp "eq" [[WAVE]], {{%.*}} : i32
  // COMMON: [[OWNER:%.*]] = llvm.and [[LANE0]], [[WARP0]] : i1
  // COMMON: llvm.cond_br [[OWNER]], ^[[RELEASE:bb[0-9]+]], ^[[JOIN:bb[0-9]+]]
  // COMMON: ^[[RELEASE]]:
  // COMMON-NEXT: llvm.atomicrmw xchg %arg0, {{%.*}} syncscope("agent") release
  // COMMON-NEXT: llvm.br ^[[JOIN]]
  // COMMON: ^[[JOIN]]:
  // COMMON-NEXT: llvm.br ^[[END]]
  // COMMON-NOT: llvm.atomicrmw
  tt.func private @experimental_lock_release(%lock: !tt.ptr<i32>, %pred: i1) {
    tti.experimental_lock_release %lock, %pred : !tt.ptr<i32>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: llvm.func @cooperative_atomic_i32
  tt.func @cooperative_atomic_i32(%src: !tt.ptr<i32>, %dst: !tt.ptr<i32>, %offsets: tensor<32xi32, #blocked>) {
    // One element per lane is a 32x4B line. Accesses are relaxed;
    // acquire/release order the operation as for tt.atomic_load/store. The
    // address is the scalar base plus a zero-extended 32-bit byte offset.
    // COMMON: %[[SCOPE:.*]] = llvm.mlir.metadata_as_value #llvm.md_node<#llvm.md_string<"agent">>
    // COMMON: %[[ORDER:.*]] = llvm.mlir.constant(0 : i32)
    // COMMON: %[[BYTES:.*]] = llvm.mul {{.*}} overflow<nuw> : i32
    // COMMON: %[[WIDE:.*]] = llvm.zext %[[BYTES]] : i32 to i64
    // COMMON: %[[ADDR:.*]] = llvm.getelementptr inbounds %{{.*}}[%[[WIDE]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    // COMMON: llvm.call_intrinsic "llvm.amdgcn.cooperative.atomic.load.32x4B"(%[[ADDR]], %[[ORDER]], %[[SCOPE]]) : (!llvm.ptr<1>, i32, !llvm.metadata) -> i32
    // COMMON: llvm.fence syncscope("agent") acquire
    %value = amdg.cooperative_atomic_load acquire, gpu, %src[%offsets] : !tt.ptr<i32> -> tensor<32xi32, #blocked>
    // COMMON: llvm.fence syncscope("agent") release
    // COMMON: llvm.call_intrinsic "llvm.amdgcn.cooperative.atomic.store.32x4B"({{.*}}) : (!llvm.ptr<1>, i32, i32, !llvm.metadata) -> ()
    amdg.cooperative_atomic_store release, gpu, %value, %dst[%offsets] : !tt.ptr<i32> -> tensor<32xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: llvm.func @cooperative_atomic_i64
  tt.func @cooperative_atomic_i64(%src: !tt.ptr<i64>, %dst: !tt.ptr<i64>, %offsets: tensor<32xi32, #blocked>) {
    // One 8-byte element per lane is a 16x8B line. Relaxed accesses add no
    // fence or barrier.
    // COMMON-NOT: {{llvm.fence|rocdl.s.barrier}}
    // COMMON: %[[VALUE:.*]] = llvm.call_intrinsic "llvm.amdgcn.cooperative.atomic.load.16x8B"({{.*}}) : (!llvm.ptr<1>, i32, !llvm.metadata) -> vector<2xi32>
    // COMMON: llvm.bitcast %[[VALUE]] : vector<2xi32> to i64
    // COMMON-NOT: {{llvm.fence|rocdl.s.barrier}}
    %value = amdg.cooperative_atomic_load relaxed, cta, %src[%offsets] : !tt.ptr<i64> -> tensor<32xi64, #blocked>
    // COMMON: llvm.call_intrinsic "llvm.amdgcn.cooperative.atomic.store.16x8B"({{.*}}) : (!llvm.ptr<1>, vector<2xi32>, i32, !llvm.metadata) -> ()
    // COMMON-NOT: {{llvm.fence|rocdl.s.barrier}}
    // COMMON: llvm.return
    amdg.cooperative_atomic_store relaxed, cta, %value, %dst[%offsets] : !tt.ptr<i64> -> tensor<32xi64, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: llvm.func @cooperative_atomic_bf16_lines
  tt.func @cooperative_atomic_bf16_lines(%src: !tt.ptr<bf16>, %dst: !tt.ptr<bf16>, %offsets: tensor<4x64xi32, #blocked>) {
    // Eight BF16 per lane are 16 bytes: four 8x16B lines per instruction.
    // COMMON: llvm.mlir.metadata_as_value #llvm.md_node<#llvm.md_string<"">>
    // COMMON: %[[VALUE:.*]] = llvm.call_intrinsic "llvm.amdgcn.cooperative.atomic.load.8x16B"({{.*}}) : (!llvm.ptr<1>, i32, !llvm.metadata) -> vector<4xi32>
    // COMMON: llvm.bitcast %[[VALUE]] : vector<4xi32> to vector<8xbf16>
    %value = amdg.cooperative_atomic_load acquire, sys, %src[%offsets] : !tt.ptr<bf16> -> tensor<4x64xbf16, #blocked>
    // COMMON: llvm.call_intrinsic "llvm.amdgcn.cooperative.atomic.store.8x16B"({{.*}}) : (!llvm.ptr<1>, vector<4xi32>, i32, !llvm.metadata) -> ()
    amdg.cooperative_atomic_store release, sys, %value, %dst[%offsets] : !tt.ptr<bf16> -> tensor<4x64xbf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: llvm.func @cooperative_atomic_mask_across_axes
  tt.func @cooperative_atomic_mask_across_axes(
      %src: !tt.ptr<i32>,
      %dst: !tt.ptr<i32>,
      %offsets: tensor<16x8xi32, #blocked>,
      %active: i32,
      %other: tensor<16x8xi32, #blocked>) {
    %four = arith.constant dense<4> : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %line = arith.divsi %rows, %four : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %limit = tt.splat %active : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %enabled = arith.cmpi slt, %line, %limit : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %expanded = tt.expand_dims %enabled {axis = 1 : i32} : tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi1, #blocked>
    %mask = tt.broadcast %expanded : tensor<16x1xi1, #blocked> -> tensor<16x8xi1, #blocked>
    // The mask is constant over a line spanning four logical rows.
    // COMMON: llvm.cond_br {{.*}}, ^[[LOAD:bb[0-9]+]], ^[[MERGE:bb[0-9]+]]({{.*}} : vector<4xi32>)
    // COMMON: ^[[LOAD]]:
    // COMMON: %[[VALUE:.*]] = llvm.call_intrinsic "llvm.amdgcn.cooperative.atomic.load.8x16B"
    // COMMON: llvm.br ^[[MERGE]](%[[VALUE]] : vector<4xi32>)
    // COMMON: ^[[MERGE]](%{{.*}}: vector<4xi32>):
    %value = amdg.cooperative_atomic_load relaxed, sys, %src[%offsets], %mask, %other : !tt.ptr<i32> -> tensor<16x8xi32, #blocked>
    // COMMON: llvm.cond_br {{.*}}, ^[[STORE:bb[0-9]+]], ^[[END:bb[0-9]+]]
    // COMMON: ^[[STORE]]:
    // COMMON: llvm.call_intrinsic "llvm.amdgcn.cooperative.atomic.store.8x16B"
    // COMMON: llvm.br ^[[END]]
    amdg.cooperative_atomic_store relaxed, sys, %value, %dst[%offsets], %mask : !tt.ptr<i32> -> tensor<16x8xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 32, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: llvm.func @cooperative_atomic_uniform_mask
  tt.func @cooperative_atomic_uniform_mask(%src: !tt.ptr<i32>, %offsets: tensor<4x32x4xi32, #blocked>, %active: i1) {
    // A mask uniform over the whole tensor guards every instruction with one
    // branch.
    %mask = tt.splat %active : i1 -> tensor<4x32x4xi1, #blocked>
    // COMMON: llvm.cond_br {{.*}}, ^[[LOAD:bb[0-9]+]], ^[[MERGE:bb[0-9]+]]
    // COMMON: ^[[LOAD]]:
    // COMMON-COUNT-4: llvm.call_intrinsic "llvm.amdgcn.cooperative.atomic.load.8x16B"
    // COMMON-NEXT: llvm.br ^[[MERGE]]
    // COMMON-NOT: llvm.cond_br
    // COMMON: llvm.return
    %value = amdg.cooperative_atomic_load relaxed, sys, %src[%offsets], %mask : !tt.ptr<i32> -> tensor<4x32x4xi32, #blocked>
    tt.return
  }
}
