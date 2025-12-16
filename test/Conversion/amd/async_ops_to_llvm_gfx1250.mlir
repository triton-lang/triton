// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx1250 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_copy_with_swizzle
  tt.func public @async_copy_with_swizzle(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg2: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // We need the splat to allow the AxisAnalysis to work during lowering
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    // Each thread needs to load 8 elements and we load 1 (sizePerThread) per global.load.lds
    // CHECK-COUNT-8: llvm.amdgcn.global.load.async.to.lds.b32
    // CHECK-NOT: llvm.amdgcn.global.load.async.to.lds
    %2 = ttg.async_copy_global_to_local %1, %arg2 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_load_strided_into_lds_with_swizzle
  tt.func public @async_load_strided_into_lds_with_swizzle(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // Each thread loads 256 contiguous bits so we split into 2 128bit loads. This was not possible on GFX9
    // CHECK-COUNT-2: llvm.amdgcn.global.load.async.to.lds.b128
    // CHECK-NOT: llvm.amdgcn.global.load.async.to.lds
    %6 = ttg.async_copy_global_to_local %arg0, %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_copy_with_swizzle
  tt.func public @async_copy_with_swizzle(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg2: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // We need the splat to allow the AxisAnalysis to work during lowering
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    // Each thread needs to load 8 elements and we load 1 (sizePerThread) per global.load.lds
    // CHECK-COUNT-8: llvm.amdgcn.global.load.async.to.lds.b32
    // CHECK-NOT: llvm.amdgcn.global.load.async.to.lds
    %2 = ttg.async_copy_global_to_local %1, %arg2 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Broadcast to all CTAs so we should just see 15 (0b1111) as the broadcast mask since we have 4 CTAs per CGA
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0], [0, 0]]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CGALayout = [[0, 0], [0, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_load_multicast_to_all_ctas
  tt.func public @async_load_multicast_to_all_ctas(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // CHECK: %[[GROUP_MASK:.*]] = llvm.mlir.constant(15 : i32) : i32
    // CHECK: llvm.amdgcn.cluster.load.async.to.lds{{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[GROUP_MASK]]

    %6 = ttg.async_copy_global_to_local %arg0, %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// 8 CTAs, 2 multicast groups of 4 CTAs each. Each group is strided by 1 so the base mask should be 0b1010101 (85) and the non free mask is -7 (~0b110)
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0], [0, 0], [0, 0]]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CGALayout = [[1, 0], [0, 0], [0, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_load_multicast_to_half_ctas
  tt.func public @async_load_multicast_to_half_ctas(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // CHECK: %[[CTA_ID:.*]] = {{.*}}llvm.amdgcn.cluster.workgroup.id.x
    // CHECK: %[[NON_FREE_BITS:.*]] = llvm.mlir.constant(-7 : i32) : i32
    // CHECK: %[[SHIFT_AMOUNT:.*]] = llvm.and %[[CTA_ID]], %[[NON_FREE_BITS]]
    // CHECK: %[[GROUP_MASK:.*]] = llvm.mlir.constant(85 : i32) : i32
    // CHECK: %[[CTA_MASK:.*]] = llvm.shl %[[GROUP_MASK]], %[[SHIFT_AMOUNT]]
    // CHECK: llvm.amdgcn.cluster.load.async.to.lds{{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[CTA_MASK]]
    %6 = ttg.async_copy_global_to_local %arg0, %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// 16 CTAs, 8 multicast groups of 2 CTAs each, each group is strided by 8 so the base mask should be 0b100000001 (257) and the non free mask is -9 (~0b1000)
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1], [0, 2], [0, 4], [0, 0]]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CGALayout = [[0, 1], [0, 2], [0, 4], [0, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 16 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_load_multicast_group_of_2_strided_by_8
  tt.func public @async_load_multicast_group_of_2_strided_by_8(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // Skip the first cluster id because it's emitted for address calculation
    // CHECK: %[[CTA_ID:.*]] = {{.*}}llvm.amdgcn.cluster.workgroup.id.x
    // CHECK: %[[NON_FREE_BITS:.*]] = llvm.mlir.constant(-9 : i32) : i32
    // CHECK: %[[SHIFT_AMOUNT:.*]] = llvm.and %[[CTA_ID]], %[[NON_FREE_BITS]]
    // CHECK: %[[GROUP_MASK:.*]] = llvm.mlir.constant(257 : i32) : i32
    // CHECK: %[[CTA_MASK:.*]] = llvm.shl %[[GROUP_MASK]], %[[SHIFT_AMOUNT]]
    // CHECK: llvm.amdgcn.cluster.load.async.to.lds{{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[CTA_MASK]]
    %6 = ttg.async_copy_global_to_local %arg0, %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// 16 CTAs split into 16 multicast groups so we should not emit cluster load since we do not share any data
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1], [0, 2], [0, 4], [0, 8]]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CGALayout = [[0, 1], [0, 2], [0, 4], [0, 8]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 16 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_load_multi_cta_but_not_data_sharing
  tt.func public @async_load_multi_cta_but_not_data_sharing(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // CHECK-NOT: llvm.amdgcn.cluster.load.async.to.lds
    // CHECK: llvm.amdgcn.global.load.async.to.lds.b64
    // CHECK-NOT: llvm.amdgcn.cluster.load.async.to.lds
    %6 = ttg.async_copy_global_to_local %arg0, %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test with linear layout as src layout
// 16 CTAs, 8 multicast groups of 2 CTAs each, each group is strided by 8 so the base mask should be 0b100000001 (257) and the non free mask is -9 (~0b1000)
#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[0, 0], [0, 0], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [[0, 4], [0, 8], [0, 16], [0, 0]], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CGALayout = [[0, 1], [0, 2], [0, 4], [0, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 16 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_load_multi_cta_linear_layout
  tt.func public @async_load_multi_cta_linear_layout(%arg0: tensor<32x32x!tt.ptr<f32>, #linear> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // Skip the first cluster id because it's emitted for address calculation
    // CHECK: %[[CTA_ID:.*]] = {{.*}}llvm.amdgcn.cluster.workgroup.id.x
    // CHECK: %[[NON_FREE_BITS:.*]] = llvm.mlir.constant(-9 : i32) : i32
    // CHECK: %[[SHIFT_AMOUNT:.*]] = llvm.and %[[CTA_ID]], %[[NON_FREE_BITS]]
    // CHECK: %[[GROUP_MASK:.*]] = llvm.mlir.constant(257 : i32) : i32
    // CHECK: %[[CTA_MASK:.*]] = llvm.shl %[[GROUP_MASK]], %[[SHIFT_AMOUNT]]
    // CHECK: llvm.amdgcn.cluster.load.async.to.lds{{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[CTA_MASK]]
    %6 = ttg.async_copy_global_to_local %arg0, %arg1 : tensor<32x32x!tt.ptr<f32>, #linear> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test async_copy_local_to_global - basic case
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_copy_local_to_global_basic
  tt.func public @async_copy_local_to_global_basic(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                                   %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    // Each thread stores 8 elements with 32-bit stores
    // CHECK-COUNT-8: llvm.amdgcn.global.store.async.from.lds.b32
    // CHECK-NOT: llvm.amdgcn.global.store.async.from.lds
    %2 = amdg.async_copy_local_to_global %arg1, %1 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Test async_copy_local_to_global with larger vector size
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_copy_local_to_global_vec128
  tt.func public @async_copy_local_to_global_vec128(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                                    %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // Each thread stores 8 elements (256 bits), split into 2 128-bit stores
    // CHECK-COUNT-2: llvm.amdgcn.global.store.async.from.lds.b128
    // CHECK-NOT: llvm.amdgcn.global.store.async.from.lds
    %2 = amdg.async_copy_local_to_global %arg1, %arg0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Test async_copy_global_to_local with padded shared layout
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[8:+4] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_copy_global_to_local_padded
  tt.func public @async_copy_global_to_local_padded(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                                    %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    // Each thread loads 8 elements with 32-bit loads
    // CHECK-COUNT-8: llvm.amdgcn.global.load.async.to.lds.b32
    // CHECK-NOT: llvm.amdgcn.global.load.async.to.lds
    %2 = ttg.async_copy_global_to_local %1, %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test async_copy_local_to_global with padded shared layout
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[8:+4] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_copy_local_to_global_padded
  tt.func public @async_copy_local_to_global_padded(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                                    %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    // Each thread stores 8 elements with 32-bit stores
    // CHECK-COUNT-8: llvm.amdgcn.global.store.async.from.lds.b32
    // CHECK-NOT: llvm.amdgcn.global.store.async.from.lds
    %2 = amdg.async_copy_local_to_global %arg1, %1 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Test that minInterval limits vectorization for async_copy_global_to_local
// sizePerThread = [1, 4] would normally allow 128-bit (4 x f32) loads,
// but minInterval = 2 limits to 64-bit (2 x f32) loads
// Layout covers 32x16, tensor is 32x32, so 2 repetitions in dim1
// Each thread handles 1*4*1*2 = 8 elements -> 4 x 64-bit loads
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[2:+2] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_copy_global_to_local_padded_limited_vec
  tt.func public @async_copy_global_to_local_padded_limited_vec(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                                                %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // minInterval=2 limits vectorization to 2 elements (64 bits)
    // Each thread handles 8 elements -> 4 x 64-bit loads
    // CHECK-COUNT-4: llvm.amdgcn.global.load.async.to.lds.b64
    // CHECK-NOT: llvm.amdgcn.global.load.async.to.lds
    %2 = ttg.async_copy_global_to_local %arg0, %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test that minInterval limits vectorization for async_copy_local_to_global
// sizePerThread = [1, 4] would normally allow 128-bit (4 x f32) stores,
// but minInterval = 2 limits to 64-bit (2 x f32) stores
// Layout covers 32x16, tensor is 32x32, so 2 repetitions in dim1
// Each thread handles 1*4*1*2 = 8 elements -> 4 x 64-bit stores
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[2:+2] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_copy_local_to_global_padded_limited_vec
  tt.func public @async_copy_local_to_global_padded_limited_vec(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                                                %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // minInterval=2 limits vectorization to 2 elements (64 bits)
    // Each thread handles 8 elements -> 4 x 64-bit stores
    // CHECK-COUNT-4: llvm.amdgcn.global.store.async.from.lds.b64
    // CHECK-NOT: llvm.amdgcn.global.store.async.from.lds
    %2 = amdg.async_copy_local_to_global %arg1, %arg0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
