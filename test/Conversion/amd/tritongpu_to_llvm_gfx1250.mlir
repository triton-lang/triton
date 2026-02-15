// RUN:  triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch="gfx1250" | FileCheck %s --check-prefix=GFX1250
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

#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[1, 0], [2, 0]]}, isTranspose = true, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: reduce_16x16
  tt.func @reduce_16x16(%input: tensor<128x128xf32, #mma>) {
    // GFX1250-COUNT-2: llvm.call_intrinsic "llvm.amdgcn.permlane16.swap"
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
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: partitioned_shared_padded_local_alloc
  tt.func @partitioned_shared_padded_local_alloc(%arg0: tensor<16x16xf16, #blocked>) {
    // GFX1250: llvm.mlir.addressof @global_smem
    // GFX1250-COUNT-4: llvm.store {{.*}} : vector<{{[0-9]+}}xf16>, !llvm.ptr<3>
    %0 = ttg.local_alloc %arg0 : (tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    tt.return
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
